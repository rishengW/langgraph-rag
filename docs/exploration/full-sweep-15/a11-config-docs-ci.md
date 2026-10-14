# A11 — 配置 / 文档 / CI-CD / 容器 / 运维资产审查

- 轮次：full-sweep-15（2026-10-08），分支 `production`
- 执行者：**主代理自救**（子代理 a11 因安全分类器 429 限流三次未能派出）
- 方法：静态阅读 + grep 交叉验证 + 一次只读 `python -I -c` 版本实测。**未发任何网络请求，未启动服务器，未跑测试套件。**
- 纪律：全文结论均以「实际代码/文件 + 行号」佐证，不采信注释、docstring、README、docs/ 与 `.kiro/specs/` 的任何说法 —— 恰恰相反，这些文档本身就是本轮的审查对象。
- ⚠️ `.env`、`debug.log`、`install_log.txt`、`chat_uploads/` 本轮**未读内容**（本轮派单未完成该部分），密钥泄漏排查是本轮的明确空缺。

---

## 0. 本轮最重要的跨格结论：生产容器下的攻击面默认不可达

这是本轮最有价值的发现，因为它**重新定性**了 A1（sympy RCE）、A5（公式注入）两格的全部结论。

`config/default.yaml` 里 **90+ 个 `*_enabled` 开关全部默认为 `false`**，包括：

```yaml
file_read_enabled: false        # 整个文件工具生态的总闸
math_enabled: false             # A1 的 sympy RCE 入口
statistics_enabled: false
linalg_enabled: false
number_theory_enabled: false
summarize_url_enabled: false
memory_enabled: false
memory_extraction_enabled: false
c_edit_enabled: false
word_edit_enabled: false
excel_create_enabled: false
excel_edit_enabled: false
csv_edit_enabled: false
typescript_edit_enabled: false
json_edit_enabled: false
...（另 20+ 个语言/格式编辑开关，全部 false）
```

而三条路径**都没有打开其中任何一个**：

1. `config/production.yaml` 只有 4 个键（`api_key`、`cors_allow_origins`、`web_search_enabled`、`web_search_verify_ssl`），不覆盖任何 `*_enabled`。
2. `docker-compose.yml:12-22` 的 `environment:` 块显式传入的只有 8 个变量：`DASHSCOPE_API_KEY`、`RAG_ENV`、`API_KEY`、`CORS_ALLOW_ORIGINS`、`CHROMA_DIR`、`CHAT_API_HOST`、`CHAT_API_PORT`。**没有一个是 `*_enabled`。**
3. `.env.example` 里 **`grep -c "_enabled"` 返回 0** —— 工具开关在整个环境文档体系里完全不存在。

**推论**：生产容器部署（`docker-compose`）下，A1 的 sympy RCE、A5 的 Excel/CSV 公式注入、全部编辑工具的攻击面**默认都不可达**。它们只在**本地开发**（用真实 `.env` 打开开关）下可达。

⚠️ 但这不等于"安全"：
- `.env`（9KB，仓库根，gitignored）是打开这些开关的实际机制，本轮未读其内容 → **无法排除本地开发环境已打开**；
- `file_read_root: "."`（default.yaml）意味着一旦打开，工具根就是**工作目录**；
- `docker-compose.yml:20` 硬编码 `CHAT_API_HOST: 0.0.0.0` —— 容器对外监听所有网卡，一旦有人用本地 `.env` 起服务并暴露，攻击面即刻激活。

---

## 1. 历史结论复核表（8 条）

| # | 历史结论 | 复核结果 | 证据 |
|---|---|---|---|
| 1 | `API_HOST`/`API_PORT` 是死配置，实际绑定用 `CHAT_API_HOST`/`CHAT_API_PORT` | ✅ **确认** | `src/config/loader.py:41-42` 映射进 Settings；`src/frontend/chat/main.py:45,51` 实际读取 `CHAT_API_*`；`uvicorn.run` 在 `:105` 用 `args.host`/`args.port`。全仓无 `API_HOST`/`API_PORT` 运行时读取点 |
| 2 | `production.yaml` 净硬化 = 0 | ✅ **确认** | `production.yaml` 仅 4 键，逐字对比 `default.yaml`：`api_key: ""` ✓、`cors_allow_origins: []` ✓、`web_search_enabled: true` ✓、`web_search_verify_ssl: true` ✓ —— **全部相同** |
| 3 | `docker-compose.yml` 无 `env_file:` | ✅ **确认** | 只有 `environment:` 块（`:12`），无 `env_file:` |
| 4 | `release_gate.py` 对 `secret_references` 只验结构 | ✅ **确认** | `release_gate.py:201-222`：`_array(..., minimum=1, maximum=MAX_SECRET_REFERENCES)` + `SecretReference.from_config` + 去重。**不校验 identifier 能否解析** |
| 5 | `pip-audit` 被 `continue-on-error: true` 掏空 | ✅ **确认** | `ci.yml:60` `continue-on-error: true`；注释称 "tracked separately" 但**无 `dependabot.yml`、无审计追踪文件**（Glob 零命中） |
| 6 | `.gitignore` 无 `.corrupt-*` 规则 | ✅ **确认** | `grep corrupt .gitignore` 零命中 |
| 7 | runbook 引用已删除的 `test_rag_application_service.py` | ✅ **确认且扩散** | `docs/production-operations-runbook.md` 的 Automated rehearsal gate 引用该文件；**`docs/test-suite-map.md:243` 也引用它** → 两处悬空引用 |
| 8 | `pyproject.toml` 钉 `sympy==1.14.0`，实际装 `1.13.1` | ❌ **推翻（漂移已消除）** | `pyproject.toml:35` 与 `requirements.txt:50` 均为 `sympy==1.14.0`；`python -I -c "import sympy; print(sympy.__version__)"` → **`1.14.0`** |

**关于第 8 条的推论**：历史记录（2026-10-06）与本轮（2026-10-08）仅隔两天，`sympy` 已被升级到 1.14.0。这意味着 **A1 那套基于 sympy 1.13.1 行号的 `parse_expr` RCE 取证全部需要在新版本里重做** —— A1 子代理正在做这件事。版本漂移消除不等于漏洞消除：机制（`global_dict is None` 时注入 `vars(builtins)`）是解析器设计层面的，不随小版本改变。

---

## 2. CI 流水线拆解（`.github/workflows/ci.yml`）

单 job `quality`，7 个 step：

| Step | 命令 | 真门禁？ | 备注 |
|---|---|---|---|
| checkout | `actions/checkout@v4` | — | `@v4` 钉大版本，不钉 SHA |
| setup-python | `actions/setup-python@v5` | — | Python **3.11**，`cache: pip` |
| install | `pip install -r requirements.txt -r requirements-dev.txt` | — | 无 `--require-hashes` |
| **Run tests** | `pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70` | ✅ **真门禁** | 覆盖率下限 70%，会因失败而失败 |
| **Run Ruff** | `ruff check .` | ✅ **真门禁** | 无 `continue-on-error` |
| **Ruff format** | `ruff format --check .` | ✅ **真门禁** | 同上 |
| **mypy** | `mypy src/` | ✅ **真门禁** | 同上 |
| **Audit deps** | `pip install pip-audit && pip audit -r ...` | ❌ **掏空** | `continue-on-error: true`（`:60`） |

**如实记录正面部分**：tests/ruff/format/mypy 四步都是真实 enforced 的，没有 `|| true`、没有注释掉、没有被跳过。`--cov-fail-under=70` 是一个真实生效的覆盖率闸门。这是本仓库 CI 里**确实管用**的部分。

**唯一被掏空的是 `pip-audit`**，且其注释自陈原因：

> Known-vulnerable pins in the locked runtime stack are tracked separately; keep this advisory until the pins are upgraded so the gate does not mask test/lint failures.

这个理由本身不荒谬（避免阻断 CI），但「tracked separately」的追踪物**在仓库里不存在**：无 `dependabot.yml`、无 `.github/audit*.json`、无任何 pin 例外清单。结论：一个自陈"另有跟踪"的豁免，指向一个不存在的跟踪物。

### ⚠️ 新发现 A：`release_gate` 完全不在 CI 里

`grep -rn "release_gate" .github/ pyproject.toml .pre-commit-config.yaml` → **零命中**。

发布闸门（`src/deployment/release_gate.py`，含 secret 引用校验、备份恢复证据校验、拓扑校验、8 个演练场景校验）是一个**纯手动运维流程**，由 `docs/production-operations-runbook.md` 的 "Automated rehearsal gate" 章节描述、由运维人员手动执行。**CI 对此零把关**。

含义：一个绕过 runbook 的部署可以直接上线，CI 不会拦下任何一项 release 证据缺失。这比"release_gate 校验不严"严重得多 —— 它连"运行过"这件事都没被任何自动化保证。

---

## 3. 文档与代码落差表（核心产出）

| 文档声称 | 实际 | 证据 |
|---|---|---|
| `docs/tool-catalog.md` 是工具目录 | **不含任何工具条目**，只有 5 个标题：`Catalog lifecycle` / `Providers and graph consistency` / `Execution policy` / `Telemetry` | 实际讲 `ToolCatalogSnapshot` 架构设计；`src/backend/tools/` 实测 `def build_*` **99 个**。文件名与内容完全不符 |
| `SECURITY.md:61-63`「Dependency scanning is **not configured yet**. Before production use, add Dependabot or a `pip-audit` workflow」 | CI **已经有** `pip-audit` workflow（`:58-63`），只是被 `continue-on-error` 掏空 | 文档已过时，但**方向比文档说的更糟**：读 SECURITY.md 的人会以为完全没有依赖扫描，实际有一个**看似在跑但永不失败**的 —— 这比"没有"更危险，因为它制造虚假的安心感 |
| `SECURITY.md:65`「See `SSL_FIX.md`」 | `SSL_FIX.md` **不存在** | Glob `SSL_FIX*` 零命中 |
| `runbook` Automated gate 引用 `test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph` | 该文件**不存在** | `ls` 确认；`docs/test-suite-map.md:243` 也引用同一文件 → 两处悬空引用，干净 checkout 下该 gate 必然硬失败 |
| `.env.example` 文档化绑定地址/端口（`:21-24`） | 文档化的是**死配置** `API_HOST=127.0.0.1` / `API_PORT=8000`；实际生效的 `CHAT_API_HOST`（默认 127.0.0.1）/`CHAT_API_PORT`（默认 **8001**）在 `.env.example` 里**完全不存在** | 新细节：`.env.example` 零 `CHAT_API_*` 条目 |
| `.env.example` 作为配置参考 | `grep -c "_enabled" .env.example` → **0**；`config/default.yaml` 有 90+ 个 `*_enabled` | 工具开关在整个环境文档体系里缺席 |

---

## 4. 容器与构建审查（Dockerfile / .dockerignore / docker-compose）

**如实记录的正面部分**（本仓库 Dockerfile 质量高于其 Python 层）：

- ✅ **非 root 运行**：`RUN useradd --create-home --shell /usr/sbin/nologin appuser` + `chown -R appuser:appuser /app` + `USER appuser`
- ✅ **不是 `COPY . .`**：只 `COPY config/ config/` 和 `COPY src/ src/`，`.env` **不可能**进镜像
- ✅ **双阶段构建**：`builder` 阶段装依赖到 `/opt/venv`，`runtime` 只拷贝 venv —— 构建工具不留在运行镜像
- ✅ **依赖层缓存**：`COPY requirements.txt .` 单独一层再 `pip install`
- ✅ **`EXPOSE 8001` 与实际绑定一致**（`main.py` 默认 `CHAT_API_PORT=8001`，`docker-compose.yml:21` 也是 8001）—— 与第 1 条里"三个不同默认端口"的死配置问题形成对照，容器路径是干净的
- ✅ **`HEALTHCHECK` 有 timeout**：`--interval=30s --timeout=5s --start-period=30s --retries=3`，探 `http://127.0.0.1:8001/health`
- ✅ **`PYTHONDONTWRITEBYTECODE=1`**、`PYTHONUNBUFFERED=1`
- ✅ `.dockerignore` 排除了 `.env`、`.env.*`、`.venv/`、`.chroma/`、`__pycache__/`、`*.pyc` —— 与 Dockerfile 的显式 COPY 构成**双重保护**

**需要收紧的两处**：

1. `FROM python:3.11-slim` **未钉 digest**（只有 tag）。CI 用 `actions/checkout@v4`、`setup-python@v5` 同样只钉大版本。镜像可复现性依赖上游 tag 不被重新指向 —— 与 runbook 要求的"immutable image digests"（`:9`）自相矛盾。release evidence 里要求 `image_digest: sha256:...`，但构建侧没有任何机制保证用的是不可变 digest。
2. `docker-compose.yml:16` 的 `RAG_ENV: ${RAG_ENV:-production}` 是**唯一 fail-closed 点**（历史结论正确）。一旦有人为调试设 `RAG_ENV=development`，`require_principal` 失效，而 `CHAT_API_HOST: 0.0.0.0` 意味着服务直接暴露到所有网卡。**这两行相邻且互相放大**，建议在 compose 里拒绝 `RAG_ENV=development` + `CHAT_API_HOST=0.0.0.0` 的组合。

---

## 5. 本轮明确空缺（诚实声明）

子代理 a11 三次因安全分类器限流未能派出，本文件由主代理在剩余限流窗口内完成，以下部分**未完成**：

- ❌ `.env`（9KB）内容未读 → **密钥泄漏排查未做**（本地 `.env` 是否打开了工具开关、是否含密钥，均未确认）
- ❌ `debug.log`（16KB）、`install_log.txt`（57KB）未读 → 密钥泄漏排查未做
- ❌ `chat_uploads/` 未列目录（可能有用户上传数据）
- ❌ `README.md`（72KB）抽查未做
- ❌ `ARCHITECTURE.md`、`CONTRIBUTING.md`、`docs/production-topology.md` vs `topology.py` 一致性未做
- ❌ `.kiro/specs/` 三个 spec 与实现对照未做（含历史确认的 R5.7 vs R5 自相矛盾）
- ❌ `SECURITY.md:32-34` 对 `/_AMapService` 的五项防护声明（relative path 校验 / 固定上游 / 覆盖调用方 security code / 拒绝重定向 / 限 query 与 response 大小）未做代码验证 —— 该项已划归 A9 子代理
- ❌ `requirements.txt` 与 `pyproject.toml` 之间的**其它**依赖漂移未逐条比对（只查了 sympy，且该条已消除）
- ❌ `benchmarks/mandarin_search/` 未审

---

## 6. 发现清单（按严重度）

### [高] CI 对 release 证据零把关
`release_gate` 不在 `.github/workflows/`、`pyproject.toml`、`.pre-commit-config.yaml` 里出现。发布闸门是纯手动流程，CI 不拦下任何一项证据缺失。
→ 建议：在 CI 或预发布流水线里加一步 `python -m src.deployment.release_gate --evidence <path>`（需确认实际 CLI 入口），至少对 release evidence 的**结构**做自动校验。

### [高] `SECURITY.md` 的依赖扫描声明已反向失真
`SECURITY.md:61-63` 说"尚未配置，请添加 pip-audit workflow"，实际 `ci.yml:58-63` 已有一个带 `continue-on-error: true` 的 `pip-audit`。文档制造了与事实相反的图景：读者以为没有扫描，实际有一个**永不失败**的扫描。
→ 建议：要么删掉 `continue-on-error` 让 `pip-audit` 成为真门禁，要么在 SECURITY.md 里如实写明"已配置但为 advisory，追踪物位于 <实际路径>"。

### [中] 工具开关在环境文档体系里完全缺席
`.env.example` 零 `_enabled` 条目，而 `config/default.yaml` 有 90+ 个。唯一能让工具生效的机制（本地 `.env`）没有任何文档。这既是"默认关闭"的安全红利，也意味着**任何一次配置事故都无法从文档回溯**。
→ 建议：在 `.env.example` 增加一节，至少列出「生产必须保持 false 的开关」与「打开后会引入的攻击面」。

### [中] 绑定端口文档化的是死的那一对
`.env.example:23-24` 文档化 `API_HOST`/`API_PORT`（8000，死配置），实际生效的 `CHAT_API_HOST`/`CHAT_API_PORT`（默认 8001）**零文档化**。
→ 建议：删掉死的那对，补上活的那对；或删掉 `loader.py:41-42` 的映射让 Settings 不再持有死字段。

### [中] 两处悬空引用同一已删除测试
`docs/production-operations-runbook.md`（Automated rehearsal gate）与 `docs/test-suite-map.md:243` 都引用 `test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph`。任何干净 checkout 下手动执行该 gate 必然硬失败。
→ 建议：确认该测试是被合并、被改名还是被删除；`git log --diff-filter=D` 可定位。

### [中] `docs/tool-catalog.md` 名不副实
99 个 `build_*` 注册函数，文档里零条目。名为 catalog 的文档不含任何 catalog 内容。
→ 建议：要么补一个可生成的工具清单（从注册表导出），要么重命名为 `tool-catalog-architecture.md`。

### [中] 镜像可复现性与 runbook 自身要求冲突
`FROM python:3.11-slim` 只钉 tag，CI 的 `actions/*@v4`/`@v5` 同病。runbook 要求 immutable image digests，release evidence 要求 `image_digest: sha256:...`，但构建侧无任何机制保证。

### [低] `SECURITY.md:65` 悬空引用 `SSL_FIX.md`
该文件不存在。

### [低] `.dockerignore` 的 `.env.*` 过于宽
会连带排除 `.env.example`（无害）与任何 `.env.production` 类文件。无害但语义不精确。

---

## 7. 与其它分区的交叉引用

- **A1（方程/数值工具）的 sympy RCE**：本轮推翻了其"版本漂移"前提（sympy 已是 1.14.0），且提供可达性限定（生产容器默认 `math_enabled: false`）。A1 子代理负责在新版本里重做机制取证。
- **A5（办公/数据格式）的公式注入**：同样受 `excel_create_enabled: false` 等开关约束。
- **A9（MCP/前端/部署）**：`amap_proxy.py` 的五项防护声明验证归 A9；`release_gate` 的逐项拆解归 A9（本轮只做了 secret_references 与"不在 CI 里"两点）。
- **A10（测试覆盖）**：`test_excel_create.py` 的 node_modules 条件跳过、`.pytest_cache` 里的死 node ID 归 A10。本轮的"两处悬空引用"与 A10 的"已删除测试"是同一根因。
