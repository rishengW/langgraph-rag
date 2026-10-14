# Full Sweep 13 — 分区 A8：MCP 出站客户端 + FastAPI 前端 + 配置 + 部署

> 只信代码。每条结论带 `文件:行号`。只读，未读 `.env`（真实密钥）。未发网络请求、未跑测试、未装依赖、未执行 docker。

**两条核心结论：**
1. `src/backend/adapters/mcp_client/security.py` 里那套 SSRF 防护是**真货**（四层 + 进程白名单，工程质量很高），但它在应用层**零调用点、零 connector 实现**——是死代码，今天在生产上保护不了任何东西。
2. **生产硬化是净零**：`config/production.yaml` 的 4 个键与 `config/default.yaml` 逐一相同；真正让生产 fail-closed 的是 `RAG_ENV=production` 这一个环境变量（`auth.py:27,45-51`），不是任何配置文件。

---

## 1. 逐文件说明

### MCP 出站适配器 `src/backend/adapters/mcp_client/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `__init__.py` | 76 | 纯 re-export。docstring 自曝：`"This package is not imported or started by the inbound MCP or FastAPI entry points"`（:3-5） |
| `security.py` | 591 | **本分区最强的文件**。fail-closed 端点策略 + DNS 钉扎 + 重定向复验 + 进程 argv 白名单 + 有界子进程清理 |
| `config.py` | 481 | 闭合 JSON schema 解析器（version=1），全部硬上限在 `:20-37` |
| `runtime.py` | 623 | `ManagedOutboundMCPProvider`：有界连接/调用/重连/关闭，快照不可变，观察事件无参 |
| `lifecycle.py` | 88 | provider 绑定 + 凭据解析，明确"构造 provider 不得做 I/O"（:27-31） |

### MCP 目录 `src/backend/mcp/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `__init__.py` | 137 | re-export；导出 `DisabledOutboundMCPProvider`（:49,83） |
| `catalog.py` | 807 | 快照校验；schema 硬上限 `:38-45`（**部分读取**） |
| `catalog_lifecycle.py` | 185 | 有界 cleanup（默认 5s / 硬上限 60s，`:26-27`），`setattr` 打补丁替换基类 `_close_provider_quietly`（:58-62, 174-178） |
| `models.py` | 271 | 冻结 dataclass；`ToolProvider` Protocol `:166-176` |
| `observability.py` | 507 | 有界脱敏事件导出；`**读取 1-140 行**` |
| `policy.py` | 1002 | `ToolExecutionPipeline`；`ToolExecutionLimits` `:130-140`（**部分读取**） |
| `providers.py` | 502 | 本地 provider；`DisabledOutboundMCPProvider` `:359-395`（**部分读取**） |
| `secrets.py` | 149 | `SecretReference` / `ResolvedSecret` / `SecretResolver` / `EnvironmentSecretProvider` |
| `telemetry.py` | 135 | 无参审计事件 + 聚合指标 |

### 前端 `src/frontend/api/` 与 `src/frontend/chat/`

| 文件 | 行数 | 说明 |
|---|---|---|
| `api/__init__.py` | 1 | 空 |
| `api/auth.py` | 83 | bearer token，`hmac.compare_digest`，环境门控 fail-closed |
| `api/dependencies.py` | 237 | `app.state` 初始化、CORS、readiness、有界资源关闭（5s） |
| `api/errors.py` | 73 | RAGError→HTTP 状态码映射 |
| `api/streaming.py` | 29 | SSE 帧序列化 |
| `api/amap_proxy.py` | 289 | **本分区唯一做对的出站代理**：固定主机 + 路径白名单 + `allow_redirects=False` + 逐 IP 限流 |
| `api/models/chat_models.py` | 111 | `extra="forbid"` 请求模型 |
| `chat/api.py` | 1261 | 全部路由 + 生命周期 + lifespan 清理 |
| `chat/main.py` | 295 | `serve` / `chat` CLI；`CHAT_API_HOST`/`CHAT_API_PORT` |
| `chat/graph.py` | 48 | 废弃 shim → `src/backend/graph/builder` |
| `chat/memory_hooks.py` | 227 | 记忆抽取钩子，全程 best-effort `except Exception` |
| `chat/nodes.py` | 85 | 废弃 shim |
| `chat/sessions.py` | 24 | 废弃 shim |
| `chat/state.py` | 8 | 废弃 shim |
| `chat/uploads.py` | 627 | 上传白名单/落地/删除 |
| `chat/static/index.html` | ~960 | **仅 grep 关键点** |
| `chat/static/script.js` | 2065 | 前端逻辑 |

### 配置 / 部署 / 工具

| 文件 | 行数 | 说明 |
|---|---|---|
| `config/settings.py` | 290 | 纯 dataclass，全部默认值 |
| `config/loader.py` | 756 | 自研扁平 YAML 解析器 + CLI>env>YAML>内置 优先级 |
| `config/__init__.py` | 32 | re-export |
| `deployment/topology.py` | 342 | 单实例门控 + 共享状态能力证明 |
| `deployment/release_gate.py` | 450 | 发布证据校验（**只验形状**） |
| `deployment/__init__.py` | 39 | re-export |
| `utils/retry.py` | 264 | 4 种重试封装 |
| `utils/networking.py` | 58 | SSL 旁路开关 + DashScope base URL |
| `utils/urls.py` | 22 | URL CSV 归一化 |

---

## 2. 🛡️ SSRF 防护链与可达性

### 2.1 四层各拦什么

**第 1 层 — URL 语法与 scheme（`security.py:140-180` `parse_https_endpoint`）**

```python
_MAX_ENDPOINT_CHARS = 2_048          # :23
_MAX_REDIRECT_ORIGINS = 16           # :24
_MAX_REDIRECTS = 5                   # :25
_METADATA_HOSTS = frozenset({        # :30-37
    "metadata", "metadata.google.internal",
    "metadata.azure.internal", "instance-data.ec2.internal"})
_PROHIBITED_HOST_SUFFIXES = (".internal", ".local", ".localhost")  # :38
```

拦：长度 >2048、控制字符、空白、反斜杠（:143-150）；`scheme.lower() != "https"`（:156-157，即 **http:// 直接拒绝**）；无 netloc/hostname（:158-159）；**userinfo 凭据**（:160-161）；**query 与 fragment**（:162-163）；端口越界（:164-165）。`_normalize_host`（:499-508）剥尾点、拒 `%`（v6 zone）、IDNA 归一、转小写。`_host_is_always_prohibited`（:511-516）拦 metadata 主机名、`localhost`、三个后缀。字面 IP 走 `_literal_address` + `_is_always_prohibited`（:170-172）。

**第 2 层 — DNS 解析 + 地址判定（`security.py:265-294` `_resolve` → `:547-563` `_validate_address`）**

```python
def _is_always_prohibited(address: IPAddress) -> bool:      # :537-544
    return (address.is_loopback or address.is_link_local
            or address.is_multicast or address.is_unspecified
            or address.is_reserved)

if _is_always_prohibited(address): raise ...                 # :553-554
if address.is_global: return                                # :555-556
if address.is_private and private_approval is not None      # :557-562
   and private_approval.permits(host, address): return
raise OutboundSecurityError("...resolved to a prohibited address")
```

拦：回环 / 链路本地 / 多播 / unspecified / reserved（**全部拒绝，无豁免开关**）。私有地址只有走 `PrivateEndpointApproval`（:86-137）才放行，且该审批自身受限：approval_id ≤128（:100-106）、**必须给出精确主机名**（:107-109）、CIDR 前缀下限 v4=/8 v6=/32（:116-118）、整个网段必须全私有且两端点都不得是 always-prohibited（:119-126）、`permits` 要求**主机名与地址同时命中**（:134-137）。resolver 抛 `OSError`/`UnicodeError` 一律压成脱敏错误（:268-273）。

**第 3 层 — 重定向逐跳独立复验（`security.py:238-251` `validate_redirect`）**

```python
if isinstance(redirect_number, bool) or not isinstance(redirect_number, int) \
   or redirect_number < 1 or redirect_number > self._max_redirects:
    raise OutboundSecurityError("Outbound MCP redirect limit exceeded")   # :241-247
parsed = parse_https_endpoint(location)                                    # :248
if parsed.origin not in self._allowed_origins:                             # :249-250
    raise OutboundSecurityError("Outbound MCP redirect target is not allowlisted")
return await self._resolve(parsed)                                         # :251  重新 DNS + 重新判地址
```

`_allowed_origins` = 初始 origin ∪ 显式配置的 `allowed_redirect_origins`（:224）；后者必须经 `parse_https_origin` 强制 `path == "/"`（:183-189），≤16 且唯一（:210-216）；`max_redirects` 构造期上限 5（:204-209），runtime 侧 connector 以 `follow_redirects=False` 调用（`runtime.py:463-467`），把 `Location` 作为 `OutboundRedirect` 抛回 provider 逐跳复验（:468-474）。

**第 4 层 — 连接对端钉扎（`security.py:253-263` `validate_connection_peer`）**

```python
address = _parse_address(peer_address)
_validate_address(endpoint.origin.host, address, private_approval=self._private_approval)
if str(address) not in endpoint.addresses:
    raise OutboundSecurityError("Outbound MCP connection peer was not pre-resolved")
```

关掉了 DNS-rebinding 窗口。runtime 侧再加一道：`result.effective_url != target.url or result.peer_address is None` → 拒绝（`runtime.py:477-480`）；stdio connector 若回传任何网络元数据也被拒（`runtime.py:493-498`）。

**附加：进程/stdio 白名单（`security.py:297-342` `ApprovedProcessTemplate` + `:345-376` `ExecutableAllowlist` + `:396-406` `_spawn_process`）**

绝对路径可执行文件（:310-312）、参数 ≤64 且每项 ≤4096 字符（:26-27, :313-322）、cwd 必须绝对（:323-325）、环境 ≤64 项且 key 不得含 `=`（:326-338）；`create_subprocess_exec` 直接 exec，**无 shell、无 PATH 查找、不继承环境**（:396-406）；启动前 `resolve(strict=True)` 并要求 `resolved == template.executable`（堵符号链接逃逸）+ `is_file()` + `os.X_OK`（`SecureProcessLauncher.start` :471-481）；托管生产默认禁 stdio，需显式审批（`ExecutableAllowlist.resolve` :371-372）。

### 2.2 可达性验证 —— **确认是死代码**

`grep -rn "ManagedOutboundMCPProvider|load_outbound_mcp_settings" E:/langgraph-rag` 全仓库结果，**只有包内自引用 + 测试 + 文档**：

| 符号 | 出现位置 |
|---|---|
| `ManagedOutboundMCPProvider` | `mcp_client/__init__.py:21,52`（re-export）；`runtime.py:95,524,552,617`（定义/工厂）；`tests/test_outbound_mcp_security.py:20,309,356,391,434`；`docs/test-suite-map.md` |
| `load_outbound_mcp_settings` | `mcp_client/config.py:236,480`（定义）；`__init__.py:12,71`（re-export）；`tests/test_outbound_mcp_security.py:29,176`；`tests/test_outbound_mcp_config.py:17,49,56,68,71,92,108` |

**应用层调用点数：0。**

第二击更硬——`OutboundConnector` / `ConnectorFactory` / `connect_http` / `connect_stdio` / `build_managed_outbound_providers` 的 grep 结果**全部落在 `src/backend/adapters/mcp_client/` 内部**（`runtime.py:73,76,84,92,101,111,112,463,490,515,518,616,619,622` 与 `__init__.py:20,23,26,49,55,70`）。

> **全仓库不存在任何 `OutboundConnector` 的实现类。** 而 `ManagedOutboundMCPProvider.__init__` 在 `runtime.py:111-112` 强制 `isinstance(connector, OutboundConnector)` 否则 `TypeError`。也就是说：即使有人调了工厂，也因为没有 connector 实现而无法构造。这条路连半通都没通。

活着的替身是 `src/backend/mcp/providers.py:359-395`：

```python
class DisabledOutboundMCPProvider:
    """Explicit future seam that cannot activate outbound MCP in Phase 3."""
    def __init__(self, *, enabled: bool = False) -> None:
        if enabled:
            raise ConfigurationError("Outbound MCP providers are disabled in this release")
```

它只被 `mcp/__init__.py:49,83` 导出，**应用代码从不实例化**；真实工具集来自 `default_provider_tools` / `default_provider_entries`（`providers.py:398-435`，经 `graph/builder.py:645` 走 `InjectedToolProvider`）。

作者自己写了两处诚实声明：
- `src/backend/adapters/mcp_client/__init__.py:3-5`："This package is not imported or started by the inbound MCP or FastAPI entry points."
- `docs/outbound-mcp-configuration.md:3`："It is not imported or started by the FastAPI process."

**结论**：46 个测试用例（`test_outbound_mcp_security.py`、`test_outbound_mcp_config.py`）在测一套没有任何生产调用点的防御。它质量高，但**当前不提供任何实际保护**。真正会打到网络的出站路径（`web_fetch_tool.py`、`summarize_tool.py`、`_http.py`）不经过它——见兄弟分区 A4 报告 `docs/exploration/full-sweep-13/a4-net-tools.md`。

---

## 3. ⚙️ 配置漂移表

### 3.1 端口绑定：**声明面与生效面完全分离**

| 变量名 | 声明处 | 是否被读取 | 实际生效的替代 |
|---|---|---|---|
| `settings.api_host` | `settings.py:45` = `"127.0.0.1"`；`config/default.yaml:24`；`SETTING_ENV_NAMES` `loader.py:41` ← `API_HOST` | **否。全仓库唯一命中是 `tests/test_config.py:399` 的断言** | 无 |
| `settings.api_port` | `settings.py:46` = `8000`；`config/default.yaml:25`；`loader.py:42` ← `API_PORT` | **否。同上，只被 `tests/test_config.py:399` 断言** | 无 |
| `CHAT_API_HOST` | `main.py:43-47` argparse 默认 `"127.0.0.1"` | **是**（`os.getenv` 直接读） | — |
| `CHAT_API_PORT` | `main.py:48-53` argparse 默认 `"8001"` | **是**（`os.getenv` 直接读，`int()`） | — |
| `create_app(api_host, api_port)` | `api.py:619-621` 默认 `"127.0.0.1"`, `8001` | 是（签名默认值，被 `main.py:102` 覆盖） | — |

**`CHAT_API_HOST` / `CHAT_API_PORT` 不在 `SETTING_ENV_NAMES`（`loader.py:23-201`）里**，所以它们根本不进 `Settings`，`Settings.api_host/api_port` 也永远没人读。

**三个互相不同的默认端口**：
- `8000` — `settings.py:46` + `config/default.yaml:25` + `.env.example:24 API_PORT=8000`（**死**）
- `8001` — `api.py:621` + `main.py:51` + `Dockerfile:36,41` + `docker-compose.yml:9,11,21`（**活**）

`.env.example:21-24` 注释说 `API_HOST`/`API_PORT` 是 "Bind address and port for the API server"，**这是错的**——改它不会改变监听地址。README.md:226-227 倒是有记录这个分裂，但 `.env.example` 没有。

### 3.2 `production.yaml` 净硬化：**0**

| 键 | `config/production.yaml` | `config/default.yaml` | 净变化 |
|---|---|---|---|
| `api_key` | `""` (:4) | `""` (:23) | 无 |
| `cors_allow_origins` | `[]` (:5) | `[]` (:26) | 无 |
| `web_search_enabled` | `true` (:6) | `true` (:58) | 无 |
| `web_search_verify_ssl` | `true` (:7) | `true` (:75) | 无 |

4/4 与 default 逐字相同。**`config/staging.yaml` 同样净零**（`api_key:""` / `cors_allow_origins:[]` / `web_search_enabled:true`，全部等同 default）。

生产 fail-closed 的真实来源是环境变量 `RAG_ENV`：
- `auth.py:27` `_PROTECTED_ENVIRONMENTS = frozenset({"production", "staging"})`
- `auth.py:45-51` 若 `expected_key` 为空且 `_active_environment()` 命中 → `503 SERVICE_UNAVAILABLE`
- `loader.py:614-616` `_environment_config_file()` 读 `RAG_ENV`（默认 `development`，:21）
- `docker-compose.yml:16` 硬编码 `RAG_ENV: ${RAG_ENV:-production}`

`production.yaml:1-2` 注释写 "Set API_KEY, DASHSCOPE_API_KEY, and CORS_ALLOW_ORIGINS in the environment" —— 这句是对的（确实靠环境变量），但文件本身提供的硬化贡献是 0。

### 3.3 `docker-compose.yml` 的 `.env` 读取链：**三重隔断**

1. `docker-compose.yml` **没有 `env_file:`**（全文件 4-30 行只有 `environment:` 白名单，:13-21）。
2. `.dockerignore:15-16` 显式排除 `.env` 和 `.env.*`，所以镜像里根本没有 `.env`。
3. 进程内 `load_settings()` 确实调用 `load_dotenv(".env")`（`loader.py:692-699`），但容器里找不到文件。

**净效果**：仓库根 `.env` 在 Docker 路径下**永远不会被读取**。只有 `${DASHSCOPE_API_KEY:-}`、`${API_KEY:-}`、`${CORS_ALLOW_CHIGINS:-}`、`${RAG_ENV:-production}`（`docker-compose.yml:13-18`）这几个由**宿主机 shell** 展开的值能进容器——而 shell 展开并不读仓库根的 `.env`（除非宿主 shell 自己 source 了）。非容器本地运行才是真的读 `.env`。

### 3.4 三方默认值不一致

| 键 | `settings.py`（代码） | `config/default.yaml` | `.env.example` |
|---|---|---|---|
| `web_search_js_fallback_enabled` | `False` (:116) | **`true`** (:80) | `false` (:174) |
| `file_read_enabled` | `False` (:151) | `false` (:101) | **`true`** (:233) |
| `weather/stock/currency/wikipedia/directions/map/math/statistics/linalg/number_theory/datetime/summarize_url/web_fetch`（13 个） | 全部 `False` (:138-150) | 全部 `false` (:88-100) | **全部 `true`** (:203-228) |

`config/default.yaml:80` 会把代码默认 `False` 覆盖成 `true`；而 `.env.example:174` 写的是 `false`。**同一个开关三个说法。**

`.env.example:198-202` 的注释确实承认 "Every flag in this section defaults to false in code ... The values below are the fully-enabled configuration" —— 诚实，但这是个陷阱：一次 `cp .env.example .env` 就会同时打开 13 个工具族 + `FILE_READ_ENABLED=true`（:233）+ `WEB_FETCH_ENABLED=true`（:228），后者是兄弟分区认定的裸奔 SSRF 入口。

### 3.5 依赖锁漂移

`requirements.txt:49` 有 `reportlab==4.4.9`（PDF 水印需要，README.md:183 也提到），但 `pyproject.toml:7-38` 的 `[project].dependencies` **没有 reportlab**。Docker/CI 走 `requirements.txt`（`Dockerfile:8`、`ci.yml:38`）能装上；`pip install .` 装不上。

---

## 4. 🔐 认证 / 上传 / 发布闸门 / CI 面

### 4.1 认证面 `src/frontend/api/auth.py`（83 行）

密钥来源（:13-16）：
```python
def _configured_api_key(request: Request) -> str:
    settings = getattr(request.app.state, "config", None)
    settings_key = getattr(settings, "api_key", "") if settings is not None else ""
    return (settings_key or os.getenv("API_KEY", "")).strip()
```
`settings.api_key` 本身由 `SETTING_ENV_NAMES` 的 `"api_key": "API_KEY"`（`loader.py:40`）经 `_apply_env_overrides`（`loader.py:655-659`）填入；YAML 里 4 个配置文件全部是 `api_key: ""`。

比较方式（:53-59）——**常量时间，做对了**：
```python
header = request.headers.get("Authorization", "")
scheme, _, supplied_key = header.partition(" ")
is_valid = (
    scheme.lower() == "bearer"
    and bool(supplied_key)
    and hmac.compare_digest(supplied_key.strip(), expected_key)
)
```
只支持单个静态 bearer 密钥；**无密钥轮换、无多密钥、无过期、无 scope/role**。`Principal` 完全由服务端配置给出（:19-24），不接受调用方身份头。

**默认空口令行为**（:45-52）：
- `expected_key == ""` 且 `RAG_ENV ∈ {production, staging}` → `503`（fail-closed）
- 否则 → **直接 `return _configured_principal(request)`，完全无认证**

`_active_environment()`（:30-31）默认 `"development"`，与 `loader.py:21 DEFAULT_ENVIRONMENT` 一致。**所以仓库的默认状态是无认证的开放服务**——`SECURITY.md:51-55` 明确承认了这一点。

`require_admin_api_key`（:75-80）在无密钥时返回 **404** 而非 403，用来隐藏端点存在性；用在 `GET /metrics`（`api.py:1255`）和 `GET /admin/health/dependencies`（`api.py:768-775`）。

**跨站请求保护**
- **没有 CSRF 机制**（无 CSRF token、无 `SameSite` 控制）。
- 唯一的结构性防线在 `dependencies.py:27-33`：`allow_credentials=False`（跨源不带 cookie）、`allow_methods=["GET","POST","DELETE","OPTIONS"]`、`allow_headers=["Authorization","Content-Type"]`。
- **矛盾点**：`allow_credentials=False` 却显式放行 `Authorization` 头（:32）。由于凭据是 header 而非 cookie，`allow_credentials=False` 对 header 凭据不构成任何限制——真正决定跨源可否发凭据的是浏览器策略；一旦有人在前端脚本里带 `Authorization`，`allow_origins` 白名单之外的来源会被浏览器拦，白名单之内的来源可以带着密钥跨域调。`allow_headers` 放行 `Authorization` 让这条路径成立。
- CORS 关闭条件：`cors_allow_origins` 为空时**整个中间件不装**（`dependencies.py:22-27`），此时跨源请求由浏览器 CORS 预检自然拒绝。

**未认证路由清单**（`api.py`）：`GET /` (:749)、`GET /health` (:756)、`GET /ready` (:762)、`GET /chat/config` (:777-784)、`GET /_AMapService/{path}` (:794-832)。全部 chat/文件端点都挂 `PrincipalDep = Annotated[Principal, Depends(require_principal)]`（:165）。

`GET /chat/config` 是**无认证的**，返回 `build_amap_client_config(settings)`（`amap_proxy.py:103-122`），其中含 `amap_js_api_key`。这是设计意图（`SECURITY.md:29-34`），但该密钥因此对任何能访问该端口的浏览器可见。

### 4.2 AMap 代理 `amap_proxy.py` —— 本分区唯一做对的出站代理

固定主机常量 `restapi.amap.com` / `webapi.amap.com`（:18-19），路径白名单 `_ALLOWED_REST_PATHS = {"v3/iasdkauth", "v3/config/district"}` + `_ALLOWED_WEB_PATH_PREFIXES = ("v4/map/styles",)`（:31-39）；`validate_amap_proxy_path`（:125-146）拒空白/长度>512/二次解码不一致/`\`、`:`、`//`/空段/`.`/`..`/非法字符；`allow_redirects=False`（:200）；调用方传入的 `jscode` 被丢弃后由服务端注入（:177-180）；响应 `Content-Type` 走白名单，未知一律 `application/octet-stream`（:40-46, :272-276）；响应上限 5,000,000 字节（:24, :250-269）。

**限流缺陷**：`api.py:806` 用 `request.client.host`（原始 socket 对端）做限流键，**不读 `X-Forwarded-For`**。放在反代后面所有客户端坍缩成反代 IP 的同一个 60/分钟桶；`request.client is None` 时全部落到 `"unknown"` 单桶。另外 `ClientRateLimiter.allow`（`amap_proxy.py:87-100`）先 `setdefault` + `append`，**只在 `len(self._hits) > 4096` 之后才驱逐过期键**；窗口只有 60 秒，若攻击者在 60 秒内制造大量新鲜唯一键，`_hits` 在该窗口内可无界增长（每键一个 `deque`）。`client_key` 长度本身无上限。

### 4.3 上传面 `uploads.py`

白名单 `ALLOWED_UPLOAD_SUFFIXES`（:21-53），34 个后缀。**包含可执行内容**：`.sh`、`.bash`（可直接执行的 shell 脚本）、`.php`、`.rb`、`.lua`、`.hs`、`.pl`、`.jl`、`.r`、`.rs`、`.go`、`.sql`、`.m`、`.groovy`、`.swift`、`.zip`（zip 可装任何东西）、`.pptx`/`.pdf`/`.docx`/`.xlsx`（OOXML/二进制压缩包，历史上是宏与 XML 注入载体）。**无 MIME 嗅探、无内容检查、无压缩包解包限制。** `.env` **不在**列表内（好）。

落地目录 `session_upload_dir`（:74-79）= `Path(settings.file_read_root).expanduser() / "chat_uploads" / sanitize_filename(thread_id)`；`file_read_root` 默认 `"."`（`settings.py:152`）→ 进程 CWD。

大小上限：`settings.file_read_max_bytes`，默认 **5,000,000**（`settings.py:153`、`config/default.yaml:103`）。内存版 `save_upload` 在 :550-554 检查；流式版 `save_upload_stream` 在 :574-596 逐块累计检查（块 1MB，:569）。

文件名消毒 `sanitize_filename`（:479-490）：`Path(name).name` → `.strip().strip(".")` → `_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")` 替换为 `_` → `[:200]`。穿越安全、无 shell 元字符。

**三个实际缺口：**
1. **静默覆盖**：`save_upload` 用 `target_path.write_bytes(content)`（:557），`save_upload_stream` 用 `temp_path.replace(target_path)`（:601）。**两者都不检查目标是否存在**。`api.py:1027-1031` 的注释直接把这当特性："Per-name locks preserve deterministic overwrite order when one multipart request contains duplicate filenames"。这与全仓库"绝不覆盖上传源文件"的叙事（针对编辑器输出文件）在同一模块里并存，容易让读者以为上传也不覆盖。
2. **单请求文件数无上限**：`semaphore = asyncio.Semaphore(min(4, max(1, len(files))))`（`api.py:1030`）只限并发为 4，`len(files)` 本身不限。每文件上限 5MB，所以一个 multipart 请求可以写任意多的磁盘量。
3. **临时文件泄漏**：`save_upload_stream`（:602-609）只在 `UploadError` 和 `OSError` 时 `unlink`。`BaseException`（`asyncio.CancelledError`、`KeyboardInterrupt`）会留下 `root/chat_uploads/.chat-upload-*.tmp`。该临时文件在会话目录**上一层**（:583-587，故意避开 turn 期间的上传发现），但它仍在 `file_read_root` 内，因此对文件读取工具可寻址（`list_session_uploads` 只遍历会话目录，:100-107，所以不会被列出）。

下载路径本身是干净的：`api.py:494-523` 要求 `sanitize_filename(filename) == filename`（:502-503）、后缀白名单（:505-507）、`target.parent != base` 拒绝（:515-516）、媒体类型按后缀映射且缺省 `application/octet-stream`（:126-158）。

### 4.4 发布闸门 `release_gate.py`

**`secret_references` 只验结构，从不解析标识符。**

```python
# release_gate.py:201-222
def _validate_secret_references(value: object) -> tuple[SecretReference, ...]:
    raw_references = _array(value, label="secret_references",
                           minimum=1, maximum=MAX_SECRET_REFERENCES)   # 1..32
    references: list[SecretReference] = []
    seen: set[tuple[str, str]] = set()
    for raw_reference in raw_references:
        try:
            reference = SecretReference.from_config(raw_reference)     # ← 仅结构
        except (ConfigurationError, TypeError, ValueError):
            raise ReleaseGateError("secret_references must contain only closed Secret_Reference objects") from None
        identity = (reference.provider, reference.identifier)
        if identity in seen:
            raise ReleaseGateError("secret_references must not contain duplicates")
        seen.add(identity)
        references.append(reference)
    return tuple(references)
```

`SecretReference.from_config`（`secrets.py:36-55`）只做：字段闭合（拒未知键）、字符串类型、`_PROVIDER_PATTERN = ^[a-z][a-z0-9_-]{0,63}$`（:14）、identifier 1..512 字符、无首尾空白、无控制字符。**没有 `SecretResolver`、没有 `EnvironmentSecretProvider`、没有任何环境变量查找**——`release_gate.py:15` 只 import 了 `SecretReference`。

`_safe_reference`（:131-141）只按 `_SAFE_REFERENCE_PATTERN = ^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,255}$` + 拒 `://` + 禁首尾空白放行。

**后果可以直接演示**：`config/production-rehearsal.example.json:19-21` 自带
```json
{ "provider": "env", "identifier": "MCP_API_KEY" }
```
而 `MCP_API_KEY` 在全仓库任何 `SETTING_ENV_NAMES`（`loader.py:23-201`）里都不存在，没有任何代码读它。闸门**愉快地接受**这个永远解析不出值的引用。同理 `APPROVED_DOCS_MCP_TOKEN`（`config/outbound-mcp.example.json:11-13`）。

**闸门确实校验了什么**：
- 文件 ≤ `MAX_EVIDENCE_BYTES = 64*1024`（:27, :381-382），UTF-8 JSON
- 顶层 11 字段闭合精确匹配（:316-334）
- `schema_version == 1`（:335-336）、`environment` casefold 后必须 `"production"`（:337-339）
- 镜像摘要 `^sha256:[0-9a-f]{64}$` 且 release ≠ rollback（:33-34, :150-153, :341-347）
- 拓扑必须通过 `validate_single_instance_deployment(environment="production", ...)`（:181-198）——多 worker / 多副本直接拒绝（`topology.py:294-322`，:309-310 只在 `environment == "production" and is_scaled` 时校验）
- backup/restore：两个 SHA-256 必须**相等**（:239-246），`writers_quiesced is True`、`isolated_restore is True`
- drills：**恰好 8 个场景**、唯一、`passed is True`、带时区时间戳、未来时间最多超出 `_FUTURE_CLOCK_SKEW = 5 min`（:30, :175-178）、陈旧度默认 7 天 / 硬上限 30 天（:25-26, :309-310）

**不校验的**：演练是否真发生、备份产物是否存在、SHA-256 是否对应真实文件、镜像摘要是否在 registry 里存在、`secret_references` 能否解析、`state_volume_ref` 是否存在、`operator_ref` 是否是有权操作者。**它是一个纯形状校验器**，8 个演练 + 8 个字段全部可以照 `config/production-rehearsal.example.json` 抄占位符替换后通过。

`topology.py` 的共享状态能力证明（:199-291）设计得严密：8 项能力全部要求**代码提供的 `implementation` 标识符 + `configured is True` + `validator() is True`**，validator 异常被压成能力名（:278-282）。但 `api.py:682` 的 lifespan 只调 `validate_single_instance_deployment()` **不带 `shared_state_capabilities` 参数**，所以生产单实例路径根本走不到这段逻辑——它是为将来的扩容路径预留的死分支。

### 4.5 CI 面 `.github/workflows/ci.yml`（60 行）

触发：push 到 `main`/`production` + 所有 PR（:6-11）；`cancel-in-progress: true`（:16-18）。

步骤：
1. `actions/checkout@v4`（:26-27）
2. `actions/setup-python@v5`，`python-version: "3.11"`，`cache: pip`（:29-33）
3. `pip install -r requirements.txt -r requirements-dev.txt`（:35-38）
4. **`pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70`**（:40-41）—— 唯一真正会 fail 的测试闸门，覆盖率下限 70%
5. `ruff check .`（:43-44）
6. `ruff format --check .`（:46-47）
7. `mypy src/`（:49-50）
8. **`pip-audit`，`continue-on-error: true`**（:52-59）

```yaml
      - name: Audit dependencies
        # Known-vulnerable pins in the locked runtime stack are tracked
        # separately; keep this advisory until the pins are upgraded so the
        # gate does not mask test/lint failures.
        continue-on-error: true
        run: |
          python -m pip install pip-audit
          python -m pip audit -r requirements.txt -r requirements-dev.txt --progress-off
```

**被废掉的部分**：只有 pip-audit 一个步骤被 `continue-on-error: true` 废成纯咨询，**它永远不能让构建变红**。

**注释里的"tracked separately"指向不存在的东西**：`git ls-files | grep -iE 'depend|audit|vuln|cve'` 只返回 `docs/archive/memory-notes/new-files-audit.md`、`src/frontend/api/dependencies.py`、`tests/test_api_dependencies.py`；仓库根目录无任何此类文件。`.gitignore` 也没有对应规则去隐藏它。**这是一个没有 referent 的注释。** 另外 `pip-audit` 本身**未固定版本**（临时 `pip install`），`requirements-dev.txt:5-13` 固定了 ruff/mypy 等却不含 pip-audit。

`requirements-dev.txt` 只有 8 个包（pytest、pytest-cov、pytest-timeout、ruff==0.8.6、mypy>=1.14,<2、pre-commit、httpx、hypothesis、types-PyYAML）。

`.pre-commit-config.yaml`：pre-commit-hooks v5.0.0（check-yaml、check-toml、end-of-file-fixer、trailing-whitespace）、ruff-pre-commit v0.8.6（`--fix` + format）、mirrors-mypy v1.14.1（`pass_filenames: false`，`args: [src/]`）。**没有 secret 扫描 hook，没有 audit hook。** 且 pre-commit 的 mypy 钉死 v1.14.1，而 requirements-dev 是 `>=1.14,<2`，两者会漂移。

---

## 5. 🧮 公式与阈值清单

### 5.1 退避曲线 `src/utils/retry.py`

| 函数 | 行号 | 公式 | 抖动 | 上限 |
|---|---|---|---|---|
| `invoke_with_retry` | :80 | `base_delay * (2**attempt) + (os.urandom(1)[0] / 256)` | **有**，加性 `[0,1)` 秒 | **无** |
| `call_with_retry` | :116 | `base_delay * (2**attempt)` | **无** | **无** |
| `dashscope_call_with_retry` | :163 | `base_delay * (2**attempt)` | **无** | **无** |
| `remove_tree_with_retry` | :246 | 固定 `delay`（默认 1.0），另每轮 :231 `sleep(0.1)` | 无 | 无（线性常数） |

默认值：`max_retries=3`、`base_delay=1.0`（:58-59, :97-98, :138-139）；`remove_tree_with_retry` 为 `max_retries=10`、`delay=1.0`（:181-182）。四处都有 `max_retries = max(1, max_retries)` 钳制（:64, :105, :144, 以及 remove_tree 无钳制）。**三种 LLM 重试都没有上限封顶**——`dashscope_max_retries` 可从配置任意调大（`settings.py:289`，`loader.py` 只 `max(1, parsed)`，:499-504），退避也随之线性翻倍。`invoke_with_retry` 的抖动是加到指数项上的**加性**抖动，不是 full-jitter。

`is_retryable_connection_error`（:31-51）是**对异常消息做子串匹配**：`("SSL","CERTIFICATE","EOF","CONNECTION","MAX RETRIES","TIMEOUT","REMOTE END","TEMPORARILY UNAVAILABLE","429","THROTTLING","RATE LIMIT")`。任何异常消息里出现 "timeout" 就会被重试，脆弱但不构成安全问题。

`RETRYABLE_HTTP_STATUSES`（:22-28）= 429/500/502/503/504。

**出站 MCP 重连退避**（`runtime.py:389-392`）——这一条是封顶的：
```python
delay = min(self._settings.reconnect_backoff_seconds * (2**attempt), 5.0)
if delay > 0:
    await asyncio.sleep(delay)
```
且只在 `retry_safe=True` 时真正重试，否则重建连接后直接抛 "invocation was not retried"（:396-399）。

### 5.2 超时

| 项 | 值 | 行号 |
|---|---|---|
| DashScope 请求超时 | 120s | `settings.py:288`、`config/default.yaml:220`、`.env.example:73` |
| DashScope 最大重试 | 3 | `settings.py:289` |
| `page_load_timeout` | 15s | `settings.py:267` |
| `page_load_max_concurrency` | 8 | `settings.py:270` |
| web_search provider / api / deadline | 8s / 20s / 30s | `settings.py:94-96` |
| AMap 代理超时 | 10s，运行时 `max(1, int(...))` | `amap_proxy.py:154,195`；`settings.py:105` |
| AMap 代理限流 | 60 req / 60s，追踪客户端 ≤4096 | `amap_proxy.py:25-27` |
| 工具执行 deadline | 30s（允许 0<d≤300） | `policy.py:134,151-153` |
| 工具并发 | 8 / per-principal 8 / per-tenant 8（各 1..64） | `policy.py:135-137,154-166` |
| 工具速率 | 600/min（principal 与 tenant 同值） | `policy.py:138-139` |
| provider cleanup 超时 | 默认 5.0s，硬上限 60.0s | `catalog_lifecycle.py:26-27` |
| app 资源关闭超时 | 5.0s | `dependencies.py:116` |
| shutdown 各阶段 | `settings.shutdown_timeout_seconds / 5` | `runtime.py:318` |
| `close_lifecycle_resource` | 5.0s | `dependencies.py:116` |
| Docker HEALTHCHECK | interval 30s / timeout 5s / start-period 30s / retries 3 | `Dockerfile:38-39` |
| 发布证据陈旧度 | 默认 7 天，硬上限 30 天；时钟偏移容忍 5 min | `release_gate.py:25-26,30,309-310` |
| 观察 duration_ms | ≤ 86,400,000（一天） | `observability.py:68`、`runtime.py:365` |

### 5.3 大小上限

| 项 | 值 | 行号 |
|---|---|---|
| 上传/文件读取 | **5,000,000 B** | `settings.py:153`、`uploads.py:550,574` |
| 上传块大小 | 1 MiB | `uploads.py:569` |
| 上传文件名 | 200 字符 | `uploads.py:490` |
| 单请求文件数 | **无上限** | `api.py:1030`（只限并发 4） |
| AMap 响应 | 5,000,000 B | `amap_proxy.py:24` |
| AMap 路径 / 查询 | 512 / 4096 字符 | `amap_proxy.py:22-23` |
| AMap 分块读取 | 64 KiB | `amap_proxy.py:255` |
| 前端下载卡片校验 | 100,000,000 B（客户端，与服务端 5MB 不一致但服务端兜底） | `script.js:26` |
| 出站 MCP 配置文档 | 65,536 B | `mcp_client/config.py:21` |
| 出站 MCP 请求 / 结果 | 65,536 / 262,144 B（硬上限 262,144 / 1,048,576） | `mcp_client/config.py:31-32` |
| 出站 MCP 服务器数 / 重定向源 / 重连 | 32 / 16 / 3 次 | `mcp_client/config.py:22-25` |
| 工具 schema | 65,536 B / 深 16 / 属性 256 / enum 256 / 白名单条目 256 | `catalog.py:40-44` |
| 工具名 / 描述 | 128 / 4,096 字符 | `catalog.py:38-39` |
| 发布证据 | 64 KiB；secret reference 256 字符 / 32 条 | `release_gate.py:27-29` |
| 观察记录 | 4,096 B / 字段 128 字符 | `observability.py:64-65` |
| 审计字段 / duration | 128 字符 / 86,400,000 ms | `telemetry.py:47,60` |
| provider 名 / health detail | 128 / 256 字符 | `models.py:19,129` |
| schema 冻结深度 | 64 | `models.py:18` |

### 5.4 配额默认值 `settings.py:54-68`

| 维度 | principal | tenant |
|---|---|---|
| requests/min | 120 | 600 |
| concurrent calls | 4 | 16 |
| searches/min | 30 | 120 |
| tokens/min | 250,000 | 1,000,000 |
| tool calls/min | 240 | 1,000 |
| retries/min | 600 | 3,000 |
| cost units/min | 500,000 | 2,000,000 |

`quota_max_tracked_identities` = 4,096（:68）；loader 侧上限为 100,000（`loader.py:219`），并发上限 1,024（:207-208）。

### 5.5 端口与监听

| 项 | 值 | 行号 |
|---|---|---|
| **生效** `CHAT_API_HOST` / `CHAT_API_PORT` | `127.0.0.1` / `8001` | `main.py:45,51` |
| `create_app` 默认 | `127.0.0.1` / `8001` | `api.py:620-621` |
| `Dockerfile` EXPOSE / CMD | 8001 / 8001 | `Dockerfile:36,41` |
| `docker-compose` command / ports / `CHAT_API_PORT` | 8001 / 8001:8001 / "8001" | `docker-compose.yml:9,11,21` |
| `RAG_WORKER_COUNT` / `RAG_REPLICA_COUNT`（镜像内硬编码） | 1 / 1 | `Dockerfile:18-19` |
| **死** `Settings.api_host` / `api_port` | `127.0.0.1` / **8000** | `settings.py:45-46` |
| 工作器计数别名 | `RAG_WORKER_COUNT`、`WEB_CONCURRENCY`、`UVICORN_WORKERS` | `topology.py:11-15` |

---

## 6. 发现与矛盾

### A. 死配置 / 死代码

1. **`settings.api_host`、`settings.api_port` 完全死亡。** 声明在 `settings.py:45-46`，由 `config/default.yaml:24-25` 与 `loader.py:41-42`（`API_HOST`/`API_PORT`）填充，但**运行时无一读取**。全仓库唯一 `.api_port` 命中是 `tests/test_config.py:399` 的断言。改 `API_HOST`/`API_PORT` 不会改变监听地址。
2. **`mcp_client` 整套（security/config/lifecycle/runtime 共 1,843 行）无应用层调用点，且 `OutboundConnector` 无任何实现类。** 证据见 §2.2。
3. **`DisabledOutboundMCPProvider`（`providers.py:359-395`）只被导出，无人实例化。**
4. **`topology.py:199-291` 的共享状态能力证明永不执行**：`api.py:682` 调 `validate_single_instance_deployment()` 不传 `shared_state_capabilities`，而该分支只在 `environment=="production" and is_scaled`（:309-310）时进入——生产被拓扑门强制单实例，所以永远走不到。
5. **`api.py:688-691` 构造了 `Principal(...)` 却丢弃返回值**（无接收者）。

### B. 文档与代码不符

6. **`SECURITY.md:61-63` 过时且与 `ARCHITECTURE.md:139` 直接冲突。** SECURITY.md 说 "Dependency scanning is not configured yet. Before production use, add Dependabot or a `pip-audit` workflow"；但 `pip-audit` 已存在于 `ci.yml:52-59`，`ARCHITECTURE.md:139` 也明确写 "Dependency scanning is covered by the advisory `pip audit` step in CI."。
7. **`SECURITY.md:65` 引用 `SSL_FIX.md`，该文件不存在。** 既不在工作树也不在 git（`git ls-files` 无命中），且 `.gitignore:53` 主动忽略它。死引用。
8. **`docs/production-operations-runbook.md:18-34` 的发布闸门命令在干净 checkout 上必然失败。** :24 引用 `tests/test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph`，而该文件已被删除（`tests/test_rag_application_service.py` → MISSING）。pytest 对不存在的路径以退出码 4 结束。死测试 ID 只存活在 `.pytest_cache/v/cache/nodeids`（15,138 行，含 `test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph` 等 3 条）。
9. **`ci.yml:53-55` 的 "tracked separately" 没有 referent。** 仓库内无任何依赖漏洞跟踪文件。
10. **`README.md:30` 把 `adapters/mcp_client/` 标注为 "(config, lifecycle, runtime, SSRF guards)"**，读起来像生效中的防护，没有任何"未接入"提示。对照 `mcp_client/__init__.py:3-5` 与 `docs/outbound-mcp-configuration.md:3` 的诚实表述。
11. **`README.md:226-227`** 已记录 `API_HOST`/`API_PORT` 与 `CHAT_API_PORT` 的分裂，但 `.env.example:21-24` 未记录，且 `.env.example` 只文档化了**死的那对**，完全没提 `CHAT_API_HOST`/`CHAT_API_PORT`。
12. **`docs/production-topology.md:24` 与 `:53` 是本区唯一诚实的文档**：明确写 "the current legacy loaders cannot pin those validated addresses to later connections or independently enforce this policy on every redirect. Do not claim outbound-fetch DNS-rebinding resistance." 这条与实际代码一致——**真正会打网络的出站路径不经过 `mcp_client`**（兄弟分区 A4 已确认 `web_fetch_tool.py`/`summarize_tool.py` 只 `startswith("http")` 就 `fetch_pages`）。

### C. 净零硬化

13. **`config/production.yaml` 相对 `config/default.yaml` 的净硬化 = 0**（4/4 键逐字相同，见 §3.2）。**`config/staging.yaml` 同样净零。** 生产 fail-closed 的唯一来源是 `docker-compose.yml:16` 的 `RAG_ENV=production` → `auth.py:45-51`。
14. **`docker-compose.yml` 无 `env_file:`**，叠加 `.dockerignore:15-16` 排除 `.env`，仓库根 `.env` 在容器路径下永远不被读取（`loader.py:699` 的 `load_dotenv` 在容器内找不到文件）。
15. **`.env.example` 是一个"全开"模板**：`WEATHER/STOCK/CURRENCY/WIKIPEDIA/DIRECTIONS/MAP/MATH/STATISTICS/LINALG/NUMBER_THEORY/DATETIME/SUMMARIZE_URL/WEB_FETCH` 全部 `=true`（:203-228）+ `FILE_READ_ENABLED=true`（:233），而代码与 YAML 默认全是 `false`（`settings.py:138-150`、`config/default.yaml:88-100`）。`.env.example:198-202` 承认这是"fully-enabled configuration"，但 `cp .env.example .env` 一次操作即打开 13 个工具族 + 5MB 上传面 + 用户可控 URL 抓取面。

### D. 注释/叙事与实现不符

16. **`config/default.yaml:80` `web_search_js_fallback_enabled: true`** 覆盖代码默认 `False`（`settings.py:116`），而 `.env.example:174` 写 `false`。**同一开关三个说法。**
17. **上传"绝不覆盖"叙事在上传本身不成立。** `uploads.py:557`（`write_bytes`）与 `uploads.py:601`（`replace`）都会静默覆盖同名文件；`api.py:1027-1031` 的注释把这当特性写出来。"never overwrite" 的叙事只针对编辑器**输出**文件。
18. **`SECURITY.md:57-59` 说 CORS "Keep the allowed-origin list scoped to known clients"**，但 `dependencies.py:27-33` 在 `cors_allow_origins` 为空时**直接不装中间件**，此时并没有"作用域"可言；且 `allow_credentials=False`（:30）与 `allow_headers=["Authorization"]`（:32）并存，对 header 凭据不构成限制。
19. **`requirements.txt:49 reportlab==4.4.9` 不在 `pyproject.toml:7-38` 的依赖列表里**，`pip install .` 与 Docker/CI 环境不一致。

### E. 真实安全问题（按严重度）

20. **`networking.py:14-26` `configure_ssl_from_env` 是生产可达的全局 TLS 旁路。** `DISABLE_SSL_VERIFY=true` → 清空 `REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE` 并 monkeypatch `ssl._create_default_https_context = ssl._create_unverified_context`（进程级、全局）。它在 `src/backend/graph/nodes/common.py:41` 于 import 期被调用，**没有任何 `RAG_ENV != production` 门控**。`SECURITY.md:65-67` 说 "Disabling SSL verification should remain a local workaround only, not a production default"，但代码没有任何机制阻止生产这么设。
21. **`networking.py:28-36` `parse_dashscope_base_url` 接受 `http://`**，运维可把携带 API key 的 DashScope 端点指向明文 HTTP，无环境门控。
22. **前端完全没有认证能力**（`script.js:1165-1189` 的 `apiPost`/`apiGet`/`apiDelete` 从不设置 `Authorization`，:1484/:1729/:1795 的原生 `fetch` 同样没有）。**这与生产强制 `API_KEY`（`auth.py:45-51`）直接冲突**：一旦设置 `API_KEY`，随附静态 UI 的每一个调用（chat/config/sessions/upload/history）都返回 401，UI 100% 失效。生产部署要么放弃 UI，要么放弃认证。
23. **`index.html:866-868` 从 `cdn.jsdelivr.net` 加载 KaTeX 0.16.11（CSS + JS + auto-render）无 SRI、无 `crossorigin`。** CDN 被攻破即在聊天页面执行任意 JS。同页还从服务端拉 `amap_js_api_key`（`api.py:777-784` 无认证 → `script.js:488-502` → :530-557 注入 `<script src="...&key=...">`）。
24. **AMap 限流按原始 socket 对端 IP 计**（`api.py:806`，不读 `X-Forwarded-For`）：反代后面所有客户端共用一个 60/分钟桶；`request.client` 为 None 时全部落到 `"unknown"` 单桶。`ClientRateLimiter`（`amap_proxy.py:87-100`）在驱逐前先 append，60 秒窗口内可被新鲜唯一键撑大。
25. **上传无文件数上限**（`api.py:1030` 只限并发 4）× 5MB/文件 = 单请求可写任意磁盘量。**上传类型含 `.sh`/`.bash`（可直接执行）与 `.zip`**，无 MIME/内容检查（`uploads.py:21-53`）。
26. **`release_gate.py:201-222` `secret_references` 只验结构不验可解析性**，可永久接受无法解析的标识符——仓库自带的 `config/production-rehearsal.example.json:19-21` 就贡献了一个任何代码都不读的 `MCP_API_KEY`。
27. **`pip-audit` 被 `continue-on-error: true` 废成纯咨询**（`ci.yml:56`），且未固定版本；其"tracked separately"注释无任何 referent。

---

## 7. 未读到的文件

**分区内未完整读取（仅 grep 常量/片段）：**
- `src/backend/mcp/catalog.py`（807 行）——只读了 :38-55 常量与 grep 片段
- `src/backend/mcp/policy.py`（1002 行）——只读了 :130-167 `ToolExecutionLimits`、:326-341 限流分支、:497-512 超时租约、grep 片段
- `src/backend/mcp/observability.py`（507 行）——只读 :1-140（常量与 `ObservationEvent` 校验头部）
- `src/backend/mcp/providers.py`（502 行）——只读 :1-120、:341-435、grep 命中
- `src/frontend/chat/static/index.html`（~960 行）——只 grep `<script>`/`<link>`/`fetch`/`token` 关键行

**分区内完全未读：**
- `CONTRIBUTING.md`
- `docs/graphs.md`
- `docs/tool-catalog.md`（仅 grep 命中）
- `docs/production-topology.md`（仅 grep 命中）
- `docs/test-suite-map.md`（仅 grep 命中）
- `docs/workflow-graphs.md`（仅 grep 命中）
- `ARCHITECTURE.md`（只读 :125-141）

**旁读到的他区文件（用于交叉验证，非本区审计对象）：**
- `docs/exploration/full-sweep-13/a4-net-tools.md`（兄弟分区报告，grep 摘录）
- `README.md` 相关行（grep 摘录）
- `src/backend/graph/builder.py`、`src/backend/graph/nodes/common.py`（grep 命中，用于确认调用关系）
- `src/backend/web_search/*`（grep 命中）
- `.kiro/specs/*`、`docs/archive/*`（grep 命中，仅作历史参考，不当作现行规范）

---

## 附录：关键行号索引

| 主题 | 定位 |
|---|---|
| SSRF 第 1 层 URL/scheme | `security.py:140-180` |
| SSRF 第 2 层 DNS/地址 | `security.py:265-294`、`:537-563` |
| SSRF 第 3 层 重定向 | `security.py:238-251`、`runtime.py:463-474` |
| SSRF 第 4 层 对端钉扎 | `security.py:253-263`、`runtime.py:477-481` |
| 私有网段窄豁免 | `security.py:86-137` |
| 进程白名单 | `security.py:297-376`、`:396-406`、`:460-481` |
| 出站死代码证据 | `mcp_client/__init__.py:3-5`、`providers.py:359-395`、`docs/outbound-mcp-configuration.md:3` |
| 认证 | `auth.py:13-16`、`:27-31`、`:45-66`、`:75-80` |
| CORS | `dependencies.py:22-33` |
| 死配置端口 | `settings.py:45-46`、`main.py:43-53`、`api.py:619-621` |
| 生产净零硬化 | `config/production.yaml:1-7` vs `config/default.yaml:23,26,58,75` |
| compose 无 env_file | `docker-compose.yml:4-30`、`.dockerignore:15-16`、`loader.py:699` |
| 上传白名单/覆盖/无上限 | `uploads.py:21-53`、`:557`、`:601`、`api.py:1030` |
| 发布闸门 secret 校验 | `release_gate.py:201-222`、`secrets.py:36-55` |
| CI 咨询化审计 | `ci.yml:52-59` |
| TLS 全局旁路 | `networking.py:14-26`、调用点 `graph/nodes/common.py:41` |
| 明文 HTTP 端点 | `networking.py:28-36` |
| 退避公式 | `retry.py:80`、`:116`、`:163`、`:246`；封顶 `runtime.py:389-392` |
| 工具执行上限 | `policy.py:130-167` |
| schema 上限 | `catalog.py:38-45` |
| 拓扑单实例门 | `topology.py:294-322` |
| 前端无认证 | `script.js:1165-1189` |
| 无 SRI CDN | `index.html:866-868` |
| 发布闸门命令引用已删测试 | `docs/production-operations-runbook.md:24` |
| SECURITY.md 过时 | `SECURITY.md:61-63` vs `ci.yml:52-59`、`ARCHITECTURE.md:139` |
| SECURITY.md 死引用 | `SECURITY.md:65`、`.gitignore:53` |
