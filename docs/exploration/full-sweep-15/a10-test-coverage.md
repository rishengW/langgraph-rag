# A10 — 测试套件覆盖审计（full-sweep-15）

> 审计对象：`tests/`（82 个 .py 文件）对 `src/` 的覆盖缺口，以及「测试静默失效」机制。
> 方法：只读 `src/`/`tests/`，唯一允许的测试运行为 `pytest --collect-only`。
> 所有结论均来自 grep + 实际代码阅读；注释、docstring、docs/ 一律不作为证据。
> 环境：`E:\langgraph-rag`，branch `production`，`.venv` Python 3.11.9。
> 采集命令：`.venv\Scripts\python.exe -m pytest tests --collect-only -q`（**无导入错误，exit 0**）。

---

## 0. 三个数字对比（核心产出）

| 口径 | 数值 | 说明 |
|---|---|---|
| `.pytest_cache/v/cache/nodeids` 中 `tests/` 的历史 node ID | **1494** | 缓存最后一次完整记录（cache 文件本身 15136 条，其余 13642 条是 `.venv` 里的 sympy 测试） |
| 当前 `--collect-only` 实际可收集项 | **1438** | 81 个 test 文件（`conftest.py` 无测试项），收集阶段 **零导入错误** |
| 源码里 `def test_` 定义数 | **1113** | 未展开 parametrize；69 个 `@pytest.mark.parametrize` 标记把 1113 个函数放大到 1438 个节点 |

**差值 1494 − 1438 = 56 个节点永久消失**，构成如下：

- **32 个**：来自 5 个**已从磁盘删除**的测试文件（见 §1）。
- **24 个**：来自 **12 个仍然存在**的测试文件，其内部用例被删减（见 §1.3）。

---

## 1. 已删除测试的完整清单

### 1.1 「node ID 里存在、磁盘上已不存在」的 5 个文件（共 32 个死 node ID）

| 已删除测试文件 | 死 node ID 数 | git 删除提交 |
|---|---|---|
| `tests/test_mcp_server.py` | 12 | `51a3ece change` |
| `tests/test_api_errors.py` | 9 | `ec42b41 delete-qa` |
| `tests/test_api_security_readiness.py` | 5 | `ec42b41 delete-qa` |
| `tests/test_api_streaming.py` | 3 | `ec42b41 delete-qa` |
| `tests/test_rag_application_service.py` | 3 | `3206639 refactor` |
| **合计** | **32** | |

这 5 个文件**全部曾进入 git**（`git log --diff-filter=D --name-only` 可复现），5 个都已删除。

### 1.2 第 6 个 pyc 孤儿：`test_temp_web_fetch_events`

`tests/__pycache__/test_temp_web_fetch_events.cpython-311-pytest-8.4.2.pyc` 存在，对应 `.py` 不存在。

- `git log --all --name-only -- '*test_temp_web_fetch_events*'` → **空**（从未进入 git）。
- `grep -c test_temp_web_fetch_events .pytest_cache/v/cache/nodeids` → **0**（也从未进缓存）。

即：这是一个**纯工作区残留**，连 pytest 缓存都不知道它存在过。现存 `tests/test_web_fetch_events.py` 是另一个文件（由 `ba03a4b web-search-display` 引入）。

### 1.3 存活文件里的用例漂移（12 个文件，−29 项）

缓存记录数 vs 当前可收集数：

| 文件 | 缓存 | 现在 | 差 |
|---|---|---|---|
| `tests/test_deployment_topology.py` | 36 | 31 | −5 |
| `tests/test_lightweight_graph_web_answer_integration.py` | 31 | 26 | −5 |
| `tests/test_api_dependencies.py` | 10 | 7 | −3 |
| `tests/test_agent_tools.py` | 42 | 40 | −2 |
| `tests/test_code_edit.py` | 36 | 34 | −2 |
| `tests/test_config.py` | 110 | 109 | −1 |
| `tests/test_compat.py` | 2 | 1 | −1 |
| `tests/test_conditional_expansion_nodes.py` | 38 | 37 | −1 |
| `tests/test_graph_state_and_nodes.py` | 21 | 20 | −1 |
| `tests/test_planning_reflection.py` | 11 | 10 | −1 |
| `tests/test_tool_catalog_policy.py` | 39 | 38 | −1 |
| `tests/test_web_search_providers.py` | 68 | 67 | −1 |
| **合计** | | | **−29** |

**这一项是历史审查完全没有记录的**：死掉的用例不只在已删除的文件里，还散落在 12 个存活文件中。`test_compat.py` 从 2 项掉到 1 项尤其可疑（`src/_compat.py` 仍在）。

---

## 2. pyc 孤儿复核（精确计数）

扫描方式：`glob('**/__pycache__/*.pyc')` 排除 `.venv`/`.git`，对每个 pyc 取
`basename` 去掉 `.pyc` 后按 `\.cpython-\d+` 截断得到模块名，再检查同目录下 `<模块名>.py` 是否存在。
（未使用字符类正则，避免了历史审查记录里提到的扫描坑。）

| 区域 | 孤儿 pyc | 孤儿模块 |
|---|---|---|
| 仓库根 `__pycache__/` | **28** | 28（`essay_content_c{1,2,3,4,56,7,89,10..16}` + `essay_en_c{...}`，各 14 个） |
| `src/` | **8** | 8 |
| `tests/` | **11** | 6 |
| **合计（项目内，不含 .venv）** | **47** | **42** |

项目内 pyc 总数 446，其中孤儿 47。

`src/` 的 8 个孤儿模块：

| 孤儿 pyc | 含义 |
|---|---|
| `src/__pycache__/{draw_graph, embeddings, web_search}.cpython-314.pyc` | 根级 shim 已被删除；注意解释器是 **CPython 3.14**（项目锁 `<3.13`），说明曾用非项目解释器跑过 |
| `src/backend/application/__pycache__/rag_service.cpython-311.pyc` | `rag_service.py` 已删 |
| `src/backend/web_search/__pycache__/tool.cpython-311.pyc` | `tool.py` 已删 |
| `src/backend/tools/__pycache__/{code_edit, code_file, web_search}.cpython-311.pyc` | `web_search.py` 已改名为 `live_web_search.py` |

`tests/` 的 11 个孤儿 pyc 对应 6 个模块（部分模块同时有 `cpython-311` 与 `cpython-311-pytest-8.4.2` 两份）：
`test_api_errors`、`test_api_security_readiness`、`test_api_streaming`、`test_mcp_server`、`test_rag_application_service`、`test_temp_web_fetch_events`。

---

## 3. `.gitignore` 规则现状

- `.gitignore:13` → `.pytest_cache/`。**已确认忽略**。因此 §1.1 的 32 个死 node ID 与 §1.3 的 29 项漂移**对 CI 完全不可见**：任何干净 checkout 都不会带这份缓存。
- **`.gitignore` 中不存在 `.corrupt-*` 规则**。全文件通读无 `corrupt` 字样。因此 memory store 隔离出的 `.corrupt-<ts>` 文件不会被忽略（会被 git 看见）。
- 附带发现：`.gitignore` 忽略 `benchmarks/**/runs.jsonl`、`benchmarks/**/judgments.jsonl`、`benchmarks/**/runs/`，但**不忽略** `benchmarks/queries.json`。所以 `test_mandarin_search_benchmark.py:70` 的注释「benchmarks/ is gitignored」**不准确**——只有运行产物被忽略，查询清单 `queries.json` 是可追踪的。

---

## 4. 静默失效测试清单（最有价值的产出）

### 4.1 环境/数据依赖型跳过（当前环境即触发）

| 位置 | 机制 | 影响面 | 当前状态 |
|---|---|---|---|
| `tests/test_excel_create.py:19-25` | fixture `artifact_runtime`：`shutil.which("node") is None` 或 `node_modules/@oai/artifact-tool` 不存在时 `pytest.skip(...)` | 该文件 9 个收集项中 **3 个**（`:51`、`:98`、`:164`）以 fixture 参数形式依赖它 | **正在生效**：仓库根无 `node_modules`（`Glob node_modules/@oai/*` 无结果） |
| `tests/test_powerpoint_edit.py:20` | **模块级** `pptx = pytest.importorskip("pptx")` | 整个文件 5 项 | 依赖 `python-pptx` |
| `tests/test_word_edit.py:20` | **模块级** `docx = pytest.importorskip("docx")` | 整个文件 11 项 | 依赖 `python-docx` |
| `tests/test_web_search_pdf_loader.py:8` | **模块级** `pytest.importorskip("pypdf")` | 整个文件 8 项 | 依赖 `pypdf` |
| `tests/test_chat_upload.py:37,409,479` | 函数级 `importorskip("pptx"/"docx"/"openpyxl")` | 仅相关用例（23 项中部分） | 依赖可选包 |
| `tests/test_mandarin_search_benchmark.py:69-72` | `@pytest.mark.skipif(not DEFAULT_QUERY_SET.exists(), ...)` | 该文件 6 项中的 1 项 | 依赖本地 `benchmarks/queries.json` 是否存在 |

**关键区分**：前 4 条里，`test_powerpoint_edit.py` / `test_word_edit.py` / `test_web_search_pdf_loader.py` 是**模块级** `importorskip`，一旦缺包则**整个文件从收集阶段就消失**，且在 `--collect-only` 输出中不留痕迹（只表现为该文件计数偏低）。而 `test_excel_create.py` 是 **fixture 级** `skip`，收集阶段照常显示 9 项，**运行时**才报 skip —— 这是历史审查记录中提到的「第二种静默丢失机制」，两种机制的失败信号形态完全不同。

### 4.2 捕获过宽

| 位置 | 写法 | 评估 |
|---|---|---|
| `tests/test_memory_store.py:612, 654, 669, 686, 707` | `pytest.raises(Exception)` | **实际有约束**：每处随后都断言异常消息子串（`"directory"`、`"not writable"`、`"filesystem reported an error"`），`:617` 还额外断言 `type(...).__name__ == "MemoryWriteError"`。捕获过宽但被后续断言收紧，**不构成真实失效** |
| `tests/test_tool_catalog_policy.py:337` | `pytest.raises(Exception, match="disabled")` | 同上，`match` 已收紧 |
| `tests/test_sessions.py:591` | `except BaseException as exc:` | 局部 try，下一行即断言；注释标 `# pragma: no cover` |

**结论**：本仓库**没有**「`pytest.raises(Exception)` 裸捕导致任何异常都算通过」的真实案例。历史审查清单里的这一项在本仓库不成立。

### 4.3 over-mock：把被测对象整体换掉

`monkeypatch.setitem(sys.modules, <真实模块名>, <ModuleType stub>)` 模式，被测模块的真实实现从未执行：

| 位置 | 被替换的真实模块 |
|---|---|
| `tests/test_rag_interfaces.py:27-29` | 整个 `dashscope`（stub 只设了 `base_http_api_url = ""`） |
| `tests/test_rag_interfaces.py:47-63` | `langchain_huggingface` + `langchain_huggingface.embeddings` |
| `tests/test_rag_interfaces.py:580` | `chromadb.api.shared_system_client` |
| `tests/test_web_fetch_events.py:48-53` 与 `:135-148` | `src.backend.web_search.content_fetcher`（stub 只有 3 个属性） |
| `tests/test_lightweight_graph_web_answer_integration.py:24-35`（并被 15 处调用） | `content_fetcher` + `prompt_builder` |
| `tests/test_answer_sanitizer.py:119-120` | `content_fetcher` + `prompt_builder` |

其中 `test_rag_interfaces.py:26-43` 的 `test_dashscope_embedding_model_satisfies_protocol` 值得单独点名：它把 `dashscope` 换成空壳后**只断言 `isinstance(embeddings, EmbeddingModel)` 和 `model_name`** —— 即 DashScope embedding 的真实请求/重试/超时/批处理路径**完全未被执行**，只验证了协议结构（类型断言，非值断言）。

`tests/test_graph_builder.py:277-281`、`:333-337` 把 `build_text_file_tool`/`build_word_tool`/`build_excel_tool`/`build_pdf_tool`/`build_*_edit_tools` 全部换成 `lambda: object()` 或 `lambda: []`。这是**接线测试**的合理做法，不判定为失效；但它意味着这些 builder 的真实性能从 `test_graph_builder.py` 得到任何保证——它们必须靠各自的专用测试文件。

### 4.4 全局状态与文件系统污染

未发现真实污染。已核实的两处 autouse fixture 都做正确清理：

- `tests/test_memory_integration.py:24-28`：`reset_store_cache()` 前后各调一次（清理进程内 store 缓存，防跨测试泄漏）。
- `tests/test_web_search_providers.py:43-49`：清理 `discovery_module._provider_cooldowns` / `_provider_failure_counts` 两个模块级字典。

未发现：`@pytest.mark.skip`（无条件跳过）为 **0 处**；`assert ... or True`、`assert True`、空断言为 **0 处**；`try/except Exception: pass` 包住测试主体为 **0 处**（`grep` 到的 `try:` 均为局部、且后接断言）。`os.environ` 赋值全部走 `monkeypatch.setenv`，无裸写。`monkeypatch.chdir` 均指向 `tmp_path`。

### 4.5 CI 层吞失败

| 位置 | 内容 | 判定 |
|---|---|---|
| `.github/workflows/ci.yml:56` | `continue-on-error: true`（pip-audit 步骤） | **唯一吞失败的写法**。该步骤永不能让构建变红 |
| `.github/workflows/ci.yml:58` | `python -m pip install pip-audit`（**未固定版本**） | `requirements-dev.txt:5-13` 固定了 pytest/ruff/mypy/httpx/hypothesis，独缺 pip-audit |
| `.github/workflows/ci.yml:53-55` | 注释称漏洞「tracked separately」 | **该跟踪文件不存在**。`grep -ri 'pip-audit\|audit\|cve\|dependabot'` 只命中 `SECURITY.md`、本 CI 文件、`docs/archive/**`、`docs/exploration/**`（本仓库的自述）。仓库根无任何此类文件，`.gitignore` 也无对应规则 → 无 referent 的注释 |
| `.github/workflows/ci.yml:41` | `python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70` | **闸门是真的**，ruff(`:44`)、ruff format(`:47`)、mypy(`:50`) 也真 enforced。但覆盖率地板只有 **70%**，且 `pyproject.toml:61` 的 `skip_covered = true` 让报告只列未全覆盖文件——单模块 0% 覆盖不会被突出显示 |

CI 的 pytest 调用**没有** `-x`、`|| true`、`|| exit 0`，也**没有** `-m 'not integration'`——即集成测试也在 CI 里跑。`pyproject.toml:44-48` 的 `addopts` 含 `--timeout=120`（`pytest-timeout` 已在 `requirements-dev.txt:7`），无 `--strict-markers` 之外的宽松项。

---

## 5. 覆盖映射表：src/ → tests/

### 5.1 语言对（`<lang>_edit.py` / `<lang>_file.py`）—— 精确计数

- `<lang>_edit.py` 模块：**38** 个；`<lang>_file.py` 模块：**35** 个；**合计 73 个模块**，归入 **39 个语言组**（含 `_source_edit` 这个私有引擎）。
- **有测试的模块：51 / 73（69.9%）**
- **零测试的模块：22 / 73**

零测试的 22 个模块（= 10 个语言组完全无测试 + 2 个部分缺测）：

```
_source_edit            pdf_edit
c_edit     c_file       toml_edit  toml_file
env_edit   env_file     xml_edit   xml_file
html_edit  html_file    ini_edit   ini_file
java_edit  java_file    java_script_edit javascript_file
python_edit python_file properties_edit properties_file
```

按组看（39 组）：

- **23 组 edit+file 双模块都有测试**：excel, go, groovy, haskell, json, julia, latex, log, lua, markdown, matlab, php, prolog, r, ruby, rust, shell, sql, swift, text, typescript, word, yaml
- **2 组部分覆盖**：`csv`（只有 `csv_edit`）、`pdf`（只有 `pdf_file`）
- **3 组单模块已覆盖**：jsonl、powerpoint、zip
- **10 组完全零测试**：c, env, html, ini, java, javascript, properties, python, toml, xml
- **1 组特殊**：`_source`（只有私有引擎 `_source_edit.py`）

### 5.2 `test_code_edit.py` 的 16 语言表——缺 4 个

`tests/test_code_edit.py:27-64` 的 `LANGUAGES` 表含 16 个语言，覆盖 r、rust、go、sql、php、ruby、latex、prolog、haskell、lua、julia、shell、matlab、groovy、swift、log。
经 `grep 'build_(c|java|javascript|python)_edit_tools' tests/` 验证：**c / java / javascript / python 这 4 个编辑模块在整个测试目录里 0 次引用**。
即主代理 a3 的发现**成立**：这 4 个语言是 38 个 `_edit.py` 里唯一既不在 16 语言表、又没有独立测试文件的（c/java/javascript/python 共 8 个模块全部零测试）。

### 5.3 任务点名的模块逐一核验

| 模块 | 测试状态 |
|---|---|
| `math_tool.py` | **有真实逻辑测试**（`test_agent_tools.py:166-207, 536-602`：limit/series/derivative/integral/equation/solve）。但**注入面 0 覆盖**：`__import__`/`os.system`/`subprocess`/`exec`/`eval`/`doctest` 关键字在该文件中出现 **0 次** |
| `linalg_tool.py` | 有（`test_agent_tools.py:209-238`），无安全用例 |
| `statistics_tool.py` | 有（`:191-207`） |
| `number_theory_tool.py` | 有（`:250-283`） |
| `excel_create.py` | `test_excel_create.py` 5 个函数 / 9 项，其中 3 项被 fixture skip；**无任何 DDE `|` / HYPERLINK / WEBSERVICE / WEBAPI / IMPORTXML 用例** |
| `excel_edit.py` | `test_excel_edit.py` 26 项；**有** DDE 拦截用例（`:355-359`，`=cmd|'/c calc'!A1`）。但同样**无** HYPERLINK/WEBSERVICE/WEBAPI/IMPORTXML |
| `csv_edit.py` | `test_csv_edit.py` 12 项；**无**任何公式注入用例（`HYPERLINK\|WEBSERVICE\|WEBAPI\|IMPORTXML\|DDE` 关键字 0 命中） |
| `web_fetch_tool.py` | **零测试**（`grep web_fetch_tool\|direct_fetch\|web_fetch_url tests/` → 0 命中） |
| `summarize_tool.py` | `test_summarize_url_tool.py` 4 项，URL 校验只测了 `ftp://`（`:20-22`）；`file://`/`gopher://`/`dict://`/localhost/RFC1918/元数据 IP **全部未测**；`fetch_pages` 全程被 mock |
| `_files.py` | **读路径有覆盖**（`test_file_tools.py:268-277` 测 `.env` 拦截），但 `resolve_safe_path` 在 tests/ 中仅出现在 `test_zip_file.py:108` 的**注释**里。`list_safe_directory` **0 命中**；`credentials`/`id_rsa`/`id_ed25519`/`.pem`/`.key`/`_BLOCKED` 等块名单**在 tests/ 中 0 命中**（唯一命中是 `test_amap.py:86` 与 outbound MCP 无关） |
| `_http.py` | **零测试**（`request_json` 仅 1 个函数，tests/ 无任何引用） |
| `list_files.py` | `test_list_files_tool.py` 12 项。**测了**隐藏项跳过（`:57-65`）与 `.env` 作为**目标路径**被拒（`:111-117`），但**从未断言块名单文件的名字/字节数不会出现在目录列表输出里**——这正是块名单不对称漏洞的测试缺口 |
| `chat_service.py` | **★ 零测试**（详见 §5.4） |
| `quotas.py` | 有（`test_resource_ownership_quotas.py` 16 项），但 `Lock`/`deadlock`/`starv`/`shield`/`cancel` 关键字 **0 命中** → 锁顺序问题未被测 |
| `serialization.py` | 有（`test_memory_store.py:153-154, 180-305`） |
| `store.py` | 有（7 个测试文件引用，`test_memory_store.py` 72 项） |
| `web_search/*` | 覆盖最好的一层：13 个测试文件，含 providers/discovery/evidence/recency/reputation/semantic/claim_consensus/authority/discovery_blending/query_constraints/page_structure/pdf_loader/lightweight_primitives。**例外**：`playwright_loader.py` **0 引用** |
| `frontend/api/*` | 有 `test_api_auth.py`(8)、`test_api_dependencies.py`(7)、`test_amap_proxy.py`(24)、`test_chat_upload.py`(23)、`test_chat_turn_inputs.py`(4)。**例外**：`models/chat_models.py` **0 引用** |
| `deployment/release_gate.py` | `test_production_release_gate.py` 16 项，覆盖较好；`:216` 甚至断言 runbook 含 `python -m src.deployment.release_gate`（但**没有**断言 rehearsal gate 里引用的测试文件是否存在） |

### 5.4 最高优先级单项缺口：`chat_service.py`

`grep 'ChatApplicationService\|_acquire_lock\|_turn_charge\|_operation_charge\|src\.backend\.application\.chat_service' tests/` → **No matches**。

`src/backend/application/chat_service.py` 定义 `ChatApplicationService`（`:32`）、`Lock` 协议（`:146`）、`_acquire_lock`（`:154`）、`_turn_charge`（`:164`）、`_operation_charge`（`:177`），**在整个 tests/ 目录里 0 次引用**。

而 `tests/test_chat_application_services.py` 这个名字具有误导性：它 import 的是 `SessionLifecycleService` 和 `TurnExecutionService`（`tests/test_chat_application_services.py:10-17`），**不是** `ChatApplicationService`。

未被测的具体代码路径包括：

- `:69-70` — `with self._quotas.acquire(...)` **先于** `await _acquire_lock(session.turn_lock)`（配额先于锁的顺序问题）
- `:74-75` — `finally: session.turn_lock.release()`
- `:77-89` — `stream_turn` / `_stream_turn` 的同一顺序问题（流式路径）
- `:57-58` — `start()` 的配额持有期包住整个 lifecycle start
- `:45-46` — `quotas or QuotaManager()` 与 `retry_reservation` 默认值

### 5.5 memory store 双路径覆盖不对称

- **路径 A（JSON 语法损坏 → quarantine）覆盖充分**：`test_memory_store.py:434-446`（隔离 + 文件名去冒号 + 日志）、`:449-457`（3 次隔离不撞名）、`:327-330`（5 种非法 JSON 全 raise `CorruptDocumentError`）。
- **路径 B（不可读 schema 版本 → `MemoryDocument.empty()`、不隔离、文件原地保留）在 store 层零覆盖**：`test_memory_store.py:301-305` 的 `test_unreadable_version_yields_no_records` 只在 **`parse_document`** 层用 `[2, 99, None, "1", 1.5, True, 0, -3]` 验证过。**没有任何测试调用 `store.read()` 并断言文件仍在磁盘上、未被隔离**——即「下一次 `save()` 原地覆盖 = 数据不可逆丢失」这一行为完全没有测试守卫。

---

## 6. 完全零测试的 src 模块（非语言类）

223 个 `src/**/*.py` 中，以下模块在 tests/ 中**没有任何引用**（`grep -w <模块名>` 0 命中）：

| 模块 | 备注 |
|---|---|
| `src/backend/application/chat_service.py` | ★ 见 §5.4，含锁顺序问题 |
| `src/backend/tools/web_fetch_tool.py` | SSRF 裸露面 |
| `src/backend/tools/_http.py` | `request_json`，全仓仅 1 个函数 |
| `src/backend/tools/_file_cache.py` | 私有缓存层 |
| `src/backend/tools/_source_edit.py` | 私有引擎（38 个 `_edit.py` 的共用底座） |
| `src/backend/mcp/catalog_lifecycle.py` | |
| `src/backend/sessions/isolation.py` | |
| `src/backend/web_search/playwright_loader.py` | JS 回退加载器 |
| `src/frontend/api/models/chat_models.py` | API 响应模型 |
| `src/utils/networking.py` | 真实出站 HTTP 层（tests 中 0 直接引用；所有测试都把 `fetch_pages` mock 掉） |

外加 §5.1 列出的 22 个语言模块。

**合计：32 个 src 模块在 tests/ 中零引用**（22 个语言类 + 10 个非语言类）。

---

## 7. `conftest.py` 深读

`tests/conftest.py` 共 50 行，**结构干净**：

- `pytest_collection_modifyitems`（`:12-23`）：按文件名给 `*integration*` 文件加 `integration` 标记。影响面：`test_memory_extraction_integration.py`、`test_memory_integration.py`、`test_lightweight_graph_web_answer_integration.py` 三个文件。**注意 CI 没有 `-m 'not integration'`，所以这些文件照常执行**——它们不依赖网络（用 `monkeypatch` 换掉 `content_fetcher`），因此「integration」标记在当前 CI 里形同虚设。
- `mock_settings`（`:26-33`）、`isolated_settings`（`:36-41`）、`ai_message`（`:44-49`）：纯工厂，无全局副作用，全部指向 `tmp_path`。
- **无 `autouse` fixture**（仅 `test_memory_integration.py:24` 与 `test_web_search_providers.py:43` 两个文件级 autouse，见 §4.4）。
- **无全局 monkeypatch、无真实网络、无真实 API key**（所有 key 均为字面量 `"test-key"`/`"x"`）。
- 依赖：`from src.config import Settings`、`from langchain_core.messages import AIMessage`。后者是**唯一的集合阶段外部依赖**——用系统 `python`（非 `.venv`）跑会 `ModuleNotFoundError: No module named 'langchain_core'` 并杀掉全部 81 个文件（本审计即由此确认；改用 `.venv\Scripts\python.exe` 后收集成功）。

---

## 8. 历史结论复核表

| # | 历史结论 | 复核结果 |
|---|---|---|
| 1 | 全仓 41 个孤儿模块 / 47 个 pyc（tests 6、src 8、根目录 27 个 `essay_*`） | **部分推翻，95% 正确**。pyc 数 **47 完全正确**；孤儿**模块数是 42 不是 41**；tests 6 模块 ✓、src 8 模块 ✓、根目录 `essay_*` 是 **28 不是 27**（`essay_content_*` 14 + `essay_en_*` 14）。差的那 1 个正是计数误差来源 |
| 1b | 其中只有 5 个曾进过 git | **确认**。`git log --diff-filter=D` 显示 5 个测试文件均被提交过（`ec42b41`、`51a3ece`、`3206639`） |
| 1c | `test_temp_web_fetch_events` 从未进过 git | **确认**。`git log --all -- '*test_temp_web_fetch_events*'` 为空；且它也不在 nodeids 缓存里（`grep -c` = 0）——比历史记录说的还彻底 |
| 2 | 32 个死 node ID 只存在于 `.pytest_cache/v/cache/nodeids`，而 `.gitignore:13` 忽略它 | **确认，且行数已变**。死 node ID 恰为 **32**（12+9+5+3+3）；`.gitignore` 忽略 `.pytest_cache/` 的规则现在在**第 11 行**（不是 13 行） |
| 2b | （历史未记录）存活文件里还有 29 项用例漂移，涉及 12 个文件 | **新发现**，见 §1.3 |
| 3 | `runbook:24` 引用已删除的 `test_rag_application_service.py`，干净 checkout 下 release gate 硬失败 | **确认**。`docs/production-operations-runbook.md:24` = `tests/test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph`；该函数曾存在于被删文件的第 124 行（`git show 3206639^:tests/test_rag_application_service.py` 可查），现已不存在 |
| 3b | `docs/test-suite-map.md:243` 也引用同一文件，构成两处悬空引用 | **需澄清**：`test-suite-map.md:243` 的原文是「`tests/test_rag_application_service.py` **was deleted** in step with its subject」——这是**对删除行为的描述**，不是可执行引用，**不构成第二处悬空引用**。真正会让命令失败的只有 runbook:24 一处 |
| 4 | 扫描坑：正则字符类必须含 `-`，且要扫描仓库根 | **确认必要性**。根目录确实有 28 个 `essay_*` pyc，`src/` 之外不扫就漏掉 28/47。本次改用「basename 去 `.pyc` 后按 `\.cpython-\d+` 截断」的方式，未依赖字符类正则 |
| 5 | `test_excel_create.py:19-25` 在 `node_modules` 不存在时跳过整个文件（第二种静默丢失机制） | **行号确认，措辞需修正**。`:19-25` 是 fixture `artifact_runtime`，`:23-24` 条件为 `node is None or not (modules/"@oai"/"artifact-tool").is_dir()`。**但它不跳过整个文件**：该文件 9 个收集项中只有 3 个（`:51`、`:98`、`:164`）依赖此 fixture，其余 6 项照常运行。`node_modules` 当前确实不存在 → 这 3 项**正在被静默跳过**。「跳过整个文件」是**模块级** `importorskip` 的行为（`test_powerpoint_edit.py:20`、`test_word_edit.py:20`、`test_web_search_pdf_loader.py:8`），两种机制不应混为一谈 |
| 6 | `pip-audit` 被 `continue-on-error: true` 掏空，追踪文件被 gitignore 且不存在 | **确认**，另补两点：`ci.yml:58` 的 `pip-audit` **未固定版本**（`requirements-dev.txt` 固定了其余全部工具却独缺它）；「tracked separately」注释**无 referent**（全仓无对应文件，`.gitignore` 也无对应规则） |
| — | 历史清单中的「`pytest.raises(Exception)` 捕得太宽」 | **本仓库不成立**。5 处（`test_memory_store.py:612,654,669,686,707`）+ 1 处（`test_tool_catalog_policy.py:337`）全部被后续消息子串断言或 `match=` 收紧。见 §4.2 |

---

## 9. 附：三数字与关键结论速查

- 历史 node ID **1494** → 当前可收集 **1438**（81 文件，零导入错误）→ 源码 `def test_` **1113**
- 消失 56 项 = 32（5 个已删文件）+ 24（12 个存活文件内部删减；实际逐项核对为 −29，其中 `test_config.py` 等个别文件有新增用例抵消）
- 语言对：**51/73 模块有测试（69.9%）**，22 个零测试，10 个语言组完全无测试
- 完全零引用的 src 模块：**32 个**
- 最大单项缺口：**`src/backend/application/chat_service.py` 在 tests/ 中 0 引用**，而 `test_chat_application_services.py` 测的是另外两个 service
- 静默失效：**1 处当前正在生效的 fixture skip**（`test_excel_create.py` 3/9 项）、**3 个模块级 `importorskip`**、**1 处数据依赖 skipif**、**1 处 CI `continue-on-error`**、**6 处 `sys.modules` 整体替换型 over-mock**
