# full-sweep-13 — 全仓库文件与公式审计（2026-10-06）

分支 `production`。8 个分区并行子代理 + 主代理补测。全程**静态阅读**：未发任何 HTTP 请求、未执行任何 RCE/注入探针、未跑测试套件、未装依赖、未读写 `.chroma/`/`chat_uploads/`/真实 SQLite、未读 `.env`（仅 `.env.example`）。

## 报告清单

| 分区 | 报告 | 规模 | 状态 |
|---|---|---|---|
| A0 测试套件 | `a0-test-suite.md` | 191 行 | 主代理自查（子代理未覆盖） |
| A1 数学/表达式/**方程** | `a1-math-formula.md` | 390 行 | 主代理自救 + 子代理补充 |
| A2 文件 IO + 配置格式工具 | `a2-file-io-tools.md` | 410 行 | 子代理续命后完成 |
| A3 编程语言工具家族 | `a3-language-tools.md` | 35 行 | ⚠️ **仅骨架**，429 限流死亡 |
| A4 联网/地图/行情/摘要 | `a4-net-tools.md` | 859 行 | 子代理完成（最详） |
| A5 Graph + 应用层 + LLM | `a5-graph-app-llm.md` | 主代理自救 | 仅并发/配额深挖；**图拓扑未覆盖** |
| A6 记忆 + 会话 + 安全 | `a6-memory-sessions-security.md` | 901 行 | 子代理完成（最大） |
| A7 Web 搜索 + RAG | `a7-websearch-rag.md` | 248 行 | 主代理自救，仅公式面 |
| A8 MCP + 前端 + 配置部署 | `a8-mcp-frontend-config.md` | 587 行 | 子代理完成 |

⚠️ **覆盖率诚实声明**：A3（40 个语言工具文件）与 A5 的图拓扑/提示词/metrics/sanitize 部分**未覆盖**。各报告末尾均有"未读到 / 不确定"章节逐条列出。

---

## 五条最重要的结论（跨分区）

### 1. 表达式解析是活的 RCE，且修复不在当前分支 —— 注释撒谎

`math_tool.py:149-157` 的 `_parse` 只传 `transformations` 和 `evaluate=True`，**没有 `local_dict`/`global_dict`**，而注释写着 *"parse_expr is SymPy's safe parser: no Python eval of arbitrary code"*。

静态取证 sympy 源码（`.venv/Lib/site-packages/sympy/parsing/sympy_parser.py:1050-1059` → `:1081`）：`global_dict is None` 时 sympy 主动 `exec('from sympy import *')` **并把 `vars(builtins)` 里所有 `BuiltinFunctionType` 注入 eval 命名空间**，包括 `__import__`。然后裸 `eval`。

记忆记录的修复手法（`exec("from sympy import *", _ns); _ns["__builtins__"]={}` + `global_dict=_ns`）**在当前分支不存在**。⚠️ 版本漂移：`pyproject.toml:35` 钉 `sympy==1.14.0`，实际安装 **1.13.1**。

第二条通道：`linalg_tool.py:163` 把字符串单元格交给 `sympy.sympify`（其 docstring 自警"uses eval, shouldn't be used on unsanitized input"），由 `matrix`/`matrix_b`/`vector` 三字段共同通向。`sympify` 签名无 `global_dict` 但有 `locals` —— 未验证的修复路线。

方程面 8 个测试**全是功能测试，零恶意输入测试**。

### 2. SSRF 四层防护是真代码，但是死代码

`mcp_client/security.py` 的防护工程质量很高（HTTPS-only、拒 userinfo/query/fragment、DNS+地址判定拒绝回环/链路本地/多播/reserved、重定向逐跳复验、对端钉扎防 DNS-rebinding、进程 argv 白名单 + 无 shell exec）。

**但 `OutboundConnector` / `ConnectorFactory` 全仓库不存在任何实现类**，而 `runtime.py:111-112` 强制 `isinstance(connector, OutboundConnector)` 否则 `TypeError`。这条路连半通都没通。替身是 `providers.py:359-395` 的 `DisabledOutboundMCPProvider`，只被导出、无人实例化。

**真正打网络的出站路径不经过它**：`web_fetch_tool.py:55` / `summarize_tool.py:68` 的全部校验只有 `startswith(("http://","https://"))`。全仓库 grep `169.254|127.0.0.1|::1|is_private` 在 `tools/` 与 `web_search/` **零命中**。

### 3. 一个卡住的 turn 可以占死整个租户的并发

`chat_service.py:69-70`：**配额租约在 turn 锁之前取得**。而 `_acquire_lock`（`:154-161`）用 `asyncio.to_thread(lock.acquire)`（无超时）+ `asyncio.shield`，且 `except CancelledError: await worker` —— **取消一个请求必须先赢到那把锁，取消无法放弃卡住的锁获取**。

`Lock` Protocol（`:146-151`）契约里根本没有 timeout 参数。`concurrent_calls` 默认 4（`quotas.py:91-95`），租约在锁等待期间全程持有（`quotas.py:227-231`，仅 `_release` 递减）。

组合后果：会话 A 的一个卡住 turn → 后续针对 A 的请求全部阻塞在锁上且各占一个槽 → 4 个槽占满 → 第 5 个请求在 `quotas.py:213-217` 拿不到槽就抛 `QuotaExceededError`，**连锁都没碰到**。同租户其它会话一起饿死。`QuotaExceededError` 有 5 个抛出点，调用方无法区分"我速率超限"和"我前面有人等死锁"。

流式路径（`stream_turn:77-89`）同顺序且更差：它是**同步 `def`**，生成器未被 `aclose()` 时租约泄漏到 GC。

### 4. 发布闸门与硬化都是"看起来在管"

- `production.yaml` 的 4 个键与 `default.yaml` **逐一相同，净硬化 = 0**。真正 fail-closed 的是 `docker-compose.yml:16` 的 `RAG_ENV=production`。
- `docker-compose.yml` **无 `env_file:`**，叠加 `.dockerignore:15-16`，仓库根 `.env` 在容器内**永远不被读取**。
- `API_HOST`/`API_PORT` 是死配置：加载进 Settings 但运行时无一读取；绑定实际用 `CHAT_API_HOST`/`CHAT_API_PORT`，而这俩**不在** `SETTING_ENV_NAMES` 里、根本不进 Settings。三个互不相同的默认端口 8000（死）/8001（活）。`.env.example:21-24` 只文档化了死的那对。
- `release_gate.py:201-222` 对 `secret_references` **只验结构**：仓库自带的 `production-rehearsal.example.json:19-21` 贡献了 `{"provider":"env","identifier":"MCP_API_KEY"}`，而 `MCP_API_KEY` 全仓库无人读，闸门愉快接受。
- `pip-audit` 被 `ci.yml:52-59` 的 `continue-on-error: true` 废成纯咨询，且注释称"tracked separately"但**该跟踪文件不存在**。
- **发布 runbook 命令在干净 checkout 上必然失败**：`production-operations-runbook.md:24` 引用 `tests/test_rag_application_service.py`，**该文件不存在**（A0 与 A8 独立发现并互验）。

### 5. 两处"静默测试丢失"

1. `test_excel_create.py:19-25` 的 `artifact_runtime` fixture：`node` 不在 PATH 或 `node_modules/@oai/artifact-tool` 缺失即**整文件 skip**。该依赖**未在任何 requirements 里声明**。本机实测：node 在 PATH 但 `artifact-tool` 缺失 → **Excel 创建的公式注入守卫在本机一次都没真正执行过**。
2. 测试共 **1113 个函数 / 81 个文件**；静态导入分析显示 117 个 src 模块无直接导入，其中确认有实质测试的已剔除，`_files.py`（路径解析/blocklist）与 `_source_edit.py`（补丁算法）为**零直接测试**的高置信度空洞。

---

## 公式面清单（"方程"主题汇总）

| 位置 | 公式 |
|---|---|
| `math_tool.py:149-157` | `_parse` → `parse_expr` **无沙箱**（RCE 根因） |
| `reputation.py:154-170` | `reputation_delta`：0.5 中点分段线性，`[0.5,1]→[0,+10]`、`[0,0.5]→[0,-10]`；⚠️ `unreachable` 计入失败分母 |
| `semantic.py:105-118` | `semantic_bonus`：`[threshold,1.0]`→`[0,25]` 线性，只加不减，地板 0 |
| `recency.py:28-78` | 时效阶梯 `+18/+12/+4/-4`、单年命中/冲突 `+20/-60`、未来 `-12`、老年 `-min(18, 4+3y)` |
| `document_quality.py:305-309` | `_recency_score`：`1.0 - age/window`，超窗硬断崖 0；⚠️ 未来日期得满分 |
| `quotas.py:266-281` | `ceil(len/4)` token 估算（中文高估 4-8×）；`max(1, tokens + searches*100 + tool_calls*10 + retries*5)` |
| `excel_edit.py:184-193` | 公式守卫：拦 `|`（旧式 DDE）；⚠️ 无长度上限；⚠️ 放行 `WEBSERVICE`/`WEBAPI`/`HYPERLINK`/`DDE()` |
| `excel_create.py:64-73` | 公式守卫：**无 `|` 检查** |
| `csv_edit.py` | **零公式注入守卫**（1041 行无 `startswith`） |
| `retry.py:80,116,163` | `base_delay * 2**attempt`；仅 `:80` 有抖动（`+ urandom(1)[0]/256`），**三处均无封顶** |
| `document_quality.py:238` | 关键词闸门 `len(token) >= 3` —— ⚠️ **对中文基本失效**，闸门静默全放行 |

---

## 记忆修正（本轮）

1. **`memory-store-quarantine`**：索引行已修正。两条路径必须分开——路径 A（JSON 语法损坏）read() **确实** `os.replace` 挪走文件（`store.py:264-266` → `_quarantine:283-313`，本轮已亲自验证 `:297`）；路径 B（schema 版本不可读，`serialization.py:74-80`）不挪、不备份、就地覆盖。**"read() 不动磁盘"只对路径 B 成立。** 记忆文件正文本就是对的，错的只是索引行措辞。
2. **`sympy-parse-expr-rce`**：索引行已修正——修复**不在当前分支**；安装 sympy 是 1.13.1 而非 1.14.0；`sympify` 有 `locals` 参数（可能可用作 linalg 修法，未验证）。

---

## 关于本轮的方法学与局限

- **8 个并发子代理中 3 个死于 429 限流**（A3、A5、A7 首轮），其中 A2 经 `SendMessage` 续命成功。这是既有记忆记录的"并发上限不存在、代理在第一次工具调用前就死"的再次实证——但本轮补充：**限流打死的是 SendMessage 通道本身**，Bash 分类器也被限流，只有 Read/Grep/Glob 持续可用。自救策略（亲自读关键文件）因此成为唯一可靠路径。
- **交叉验证成功的三处**：runbook 死测试引用（A0 与 A8 独立发现）；`quotas.py:271` token 估算（A1 与主代理独立发现）；memory store 双路径（A6 与记忆文件互相印证，且 A6 的误判源于索引行而非代码）。
- **交叉验证推翻的一处**：A1 初稿判断 `excel_create.validate_sheet` 接受 `=` 前缀值是公式注入漏洞；A2 读 `excel_create.mjs:64` 证明 `range.values` 把 `=` 前缀字符串落为**纯文本**，已降级为校验不对称。
- 各分区报告的"未读到 / 不确定"章节是本审计最重要的产出之一——**明确标注什么没被验证，比假装覆盖了全部更有价值**。
