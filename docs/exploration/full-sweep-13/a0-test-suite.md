# A0 — 测试套件分区（full-sweep-13）

> 负责人：主代理（子代理分区未覆盖测试，由主代理自查）
> 根目录：`E:\langgraph-rag`

## 0. 规模

- **81 个测试文件**，**1113 个 `def test_` 函数**（`grep -c "^def test_"` 合计）。
- pytest 配置在 `pyproject.toml:40-52`：`testpaths=["tests"]`、`pythonpath=["."]`、`addopts=["-q","--tb=short","--timeout=120"]`、注册 `integration` / `slow` 两个 marker。
- 自动标记机制在 `tests/conftest.py:12-23`：文件名含 `integration` 即加 `integration` marker，无需手动维护。
- **重要发现**：`pytest.mark.integration` / `pytest.mark.slow` 的**显式装饰器在 tests/ 下零命中**（仅 `conftest.py` 自身出现）——也就是说 `slow` marker 从未被使用过，`integration` 全靠文件名约定。这是"声明了但没在用"的 marker。

### 测试密度最高的 10 个文件

| 测试数 | 文件 |
|---:|---|
| 67 | `tests/test_web_search_providers.py` |
| 61 | `tests/test_memory_mutations.py` |
| 48 | `tests/test_config.py` |
| 42 | `tests/test_memory_store.py` |
| 40 | `tests/test_agent_tools.py` |
| 34 | `tests/test_tool_catalog_policy.py` |
| 30 | `tests/test_memory_tools.py` |
| 29 | `tests/test_conditional_expansion_nodes.py` |
| 27 | `tests/test_memory_transcript.py` |
| 26 | `tests/test_lightweight_graph_web_answer_integration.py` |

测试重心明显在 **memory（7 个文件、226 个）**、**web_search（14 个文件、约 200 个）**、**config**。

---

## 1. 覆盖度对照：模块 → 测试

方法：先取 `tests/*.py` 里所有 `src.*` 导入（直接导入），再与 `src/` 下 203 个非 `__init__` 模块做差集。

**⚠️ 方法论警告（必须先读）**：直接导入差集得到 **117 个"无直接导入"模块，但这**不等于** 117 个无测试**。大量模块经父包属性访问，例如：

- `tests/test_agent_tools.py:9-34` 只写 `from src.backend.tools import build_math_tool, ...`，但实际覆盖了 `math_tool.py`、`linalg_tool.py`、`number_theory_tool.py`、`statistics_tool.py`、`weather.py`、`currency.py`、`wikipedia_tool.py`、`stock.py`、`datetime_tool.py`、`directions.py`、`map_tool.py` 以及 15 个 `*_file.py`。
- `tests/test_sessions.py` / `test_resource_ownership_quotas.py` 经 `src.backend.security` 包间接覆盖 `quotas.py`、`identity.py`。
- `tests/test_outbound_mcp_security.py` 经 `src.backend.adapters.mcp_client` 包间接覆盖 `security.py`。

因此 117 这个数字**只能当上限**，不能当结论。下面按可信度分层。

### 1.1 确实有实质测试、但容易被误判为空洞

已逐一确认测试存在且非空壳（断言具体行为）：

| 模块 | 测试证据 |
|---|---|
| `math_tool.py` | `test_agent_tools.py:536-605`，8 个测试：导数 / 不定积分 / 定积分 / 解方程 / 含等号解方程 / 数值求值 / 拒绝自由符号 / 错误返回字符串 |
| `linalg_tool.py` | `test_agent_tools.py:208-244`，4 个：行列式 / 解方程组 / 奇异矩阵 / 形状不匹配 |
| `number_theory_tool.py` | `test_agent_tools.py:250-288`，5 个 |
| `statistics_tool.py` | `test_agent_tools.py:191-206`，2 个 |
| `weather.py` / `currency.py` / `wikipedia_tool.py` / `stock.py` / `datetime_tool.py` / `directions.py` / `map_tool.py` | `test_agent_tools.py:19-534`，全部有注入式 fake requester 测试 |

**关键观察**：`test_agent_tools.py:587-591` 的 `test_math_tool_evaluate_rejects_free_symbols` 传 `expression="x + 1"`，只断言返回里含 `"unknown symbol"`。**它测的是"表达式求值时未绑定自由符号会报错"，不是测代码注入防御**。整个 tests/ 目录里没有任何测试用 `os.system`、`__import__`、`exec(`、`parse_expr` 之外的构造器去探测表达式沙箱。也就是说：**方程面有功能测试，没有恶意输入测试**。

### 1.2 高置信度的真实覆盖空洞

以下模块在 `tests/` 下**既无直接导入，也无父包属性引用**，可判为真空洞（用 `grep -E "(^|[^a-zA-Z0-9_])NAME([^a-zA-Z0-9_]|$)"` 在导入其父包的测试文件中二次确认过）：

**后端核心（最应担心的）**
- `src/backend/application/chat_service.py` —— 配额/turn 锁顺序所在，**零直接测试**
- `src/backend/application/turn_execution.py`
- `src/backend/application/session_lifecycle.py`
- `src/backend/application/errors.py`、`src/backend/application/models.py`
- `src/backend/sessions/checkpoint.py`、`isolation.py`、`models.py`、`registry.py`、`sqlite.py`、`storage.py`（注：`test_sessions.py` 测的是 `src.frontend.chat.sessions` 包装层，底层实现未被导入）
- `src/backend/graph/nodes/decompose.py`、`expand.py`、`fallback_answer.py`（注：`test_fallback_answer.py` 只 2 个测试且只导入包）
- `src/backend/mcp/catalog_lifecycle.py`、`models.py`、`policy.py`、`secrets.py`、`telemetry.py`
- `src/deployment/topology.py`
- `src.frontend.api.errors`、`models.chat_models`、`streaming`、`src.frontend.chat.main`、`src.frontend.chat.nodes`
- `src._compat`、`src.errors`
- `src/utils/networking.py`

**文件工具引擎（安全关键路径）**
- `src/backend/tools/_files.py` —— 路径解析/blocklist 所在，**零直接测试**
- `src/backend/tools/_source_edit.py` —— 补丁应用算法，**零直接测试**
- `src/backend/tools/_file_cache.py`
- `src/backend/tools/_http.py`

**Web 搜索（3 个 provider + 2 个加载器）**
- `src/backend/web_search/api_providers.py`、`baidu.py`、`bing.py`、`duckduckgo.py`、`discovery.py`、`playwright_loader.py`

> `test_web_search_providers.py`（67 个测试，全库最大）测的是 `src.backend.web_search` + `common.py` + `query_prep.py`，**不含** baidu/bing/duckduckgo 三个具体 provider。

### 1.3 文件工具家族的测试覆盖形状

`tests/test_agent_tools.py:9-34` 一次性导入 16 个 `build_*_file_tool`，`tests/test_code_edit.py` 导入 4 个，`test_file_tools.py` 导入 9 个。但**逐一确认**：这些测试只验证**构建器能返回一个带正确 name 的工具对象**（元数据冒烟测试），**没有任何测试实际调用这些工具的 invoke 去读写一个真实文件并校验输出内容**。

语言类 `_edit.py`（约 20 个）中，**只有 3 个有专属测试文件**：
- `test_text_edit.py`（14）
- `test_markdown_edit.py`（14）
- `test_typescript_edit.py`（16）
- `test_json_edit.py`（9）、`test_csv_edit.py`（12）、`test_yaml_edit.py`（12）、`test_excel_edit.py`（13）、`test_word_edit.py`（11）、`test_powerpoint_edit.py`（5）

**完全没有专属测试的 `_edit.py`**：`c_edit`、`go_edit`、`groovy_edit`、`haskell_edit`、`java_edit`、`javascript_edit`、`julia_edit`、`latex_edit`、`lua_edit`、`matlab_edit`、`php_edit`、`prolog_edit`、`r_edit`、`ruby_edit`、`rust_edit`、`shell_edit`、`sql_edit`、`swift_edit`、`html_edit`、`xml_edit`、`toml_edit`、`ini_edit`、`properties_edit`、`env_edit`、`log_edit` —— **25 个**。

---

## 2. ⚠️ 两种"静默测试丢失"机制（实测确认）

这是本分区的核心发现。测试套件在特定环境下会**静默变成更小的套件，且不会有任何失败信号**。

### 2.1 机制一：环境依赖的整文件跳过（`pytest.skip` 在 fixture 里）

`tests/test_excel_create.py:19-25`：

```python
@pytest.fixture
def artifact_runtime() -> tuple[str, str]:
    node = shutil.which("node")
    modules = Path("node_modules").resolve()
    if node is None or not (modules / "@oai" / "artifact-tool").is_dir():
        pytest.skip("artifact-tool Node runtime is unavailable")
    return node, str(modules)
```

- 该 fixture 被文件内**全部 5 个测试函数**引用（`:51-54` 等），任一失败即整个文件 skip。
- **本机实测**：`node` 在 PATH，但 `node_modules/@oai/artifact-tool` **不存在** → `test_excel_create.py` 整文件 skip。
- 后果：`excel_create.py` 的公式注入守卫（`validate_sheet` 对 `=`/`|` 前缀单元格值的处理）**在本机一次都没真正执行过**。而它正是 `excel_edit.py:174-193` 有守卫、但 `excel_create.py` 没有同类守卫的那块代码。
- 这类 skip 在 CI 报告里显示为 `skipped`，容易被当成"环境限制"忽略。

同类 `importorskip`（缺失即 skip，非整文件但同样静默）：
- `tests/test_powerpoint_edit.py:20` → `pptx`
- `tests/test_word_edit.py:20` → `docx`
- `tests/test_web_search_pdf_loader.py:8` → `pypdf`
- `tests/test_chat_upload.py:37,409,479` → `pptx` / `docx` / `openpyxl`
- `tests/test_mandarin_search_benchmark.py:69` → `@pytest.mark.skipif(...)`

注意：`pyproject.toml:31-35` 已声明 `python-docx`、`python-pptx`、`pypdf`、`openpyxl` 为项目依赖，所以这些 `importorskip` 在正常安装后应全部通过——**只有 `node_modules/@oai/artifact-tool` 这个依赖没有在任何 requirements 里声明**，是唯一真正会静默失效的。

### 2.2 机制二：runbook 引用的测试文件已不存在（发布闸门硬失败）

`docs/production-operations-runbook.md:19-24` 定义发布闸门命令，其中 `:24` 引用：

```
tests/test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph
```

**实测：`tests/test_rag_application_service.py` 在仓库中不存在。**

逐项核对 runbook 引用的 9 个测试目标：

| 引用 | 存在？ |
|---|---|
| `test_deployment_topology.py::test_production_accepts_one_worker_and_replica_with_matching_aliases` | ✅ |
| `test_deployment_topology.py::test_replica_count_rejects_scaled_production_and_names_local_blockers` | ✅ |
| `test_deployment_topology.py::test_production_launch_configuration_declares_one_instance` | ✅ |
| `test_rag_application_service.py::test_failed_rebuild_restores_previous_global_graph` | ❌ **文件不存在** |
| `test_provider_publication.py::test_required_failure_is_atomic_and_retains_previous_generation` | ✅ |
| `test_provider_publication.py::test_optional_failure_omits_all_dependent_tools_and_degrades_readiness` | ✅ |
| `test_sessions.py::test_sqlite_storage_round_trips_metadata` | ✅ |
| `test_sessions.py::test_registry_restores_session_from_metadata` | ✅ |
| `test_sessions.py::test_sqlite_memory_saver_restores_graph_state_after_reopen` | ✅ |

后果：**runbook 里的发布闸门命令在任意干净 checkout 上必然失败**（pytest 找不到节点），且失败信息是"file not found"而不是任何有意义的断言失败。文档声称的"发布前必须跑"这一步实际上从未能在干净环境跑通。

---

## 3. 测试基建观察

- `tests/conftest.py:26-41`：只有 `mock_settings`（用 `tmp_path` 隔离 chroma 目录、`dashscope_max_retries=1`）和 `isolated_settings`（`dataclasses.replace` 包装器）两个 fixture。测试隔离靠 fixture 而非进程级。
- `tests/conftest.py:12-23` 的 `pytest_collection_modifyitems` 是唯一的全局钩子。
- 外部依赖一律用**注入式 fake**（`requester=requester`、`FakeTicker`、`tmp_path`），全库无真实网络调用——这一点做得很干净。
- `tests/test_chat_static_amap.py`（11 个测试）和 `tests/test_compat.py`（1 个）不导入任何 `src.*`：前者测前端静态文件内容，后者测 `src/_compat.py`。
- `src/backend/memory/watermark.py` 有**两个**测试文件（`test_memory_watermark.py` 8 个 + 经 `test_memory_extraction*.py` 间接覆盖），是本库覆盖最密的单一模块之一。

---

## 4. 汇总发现

| # | 发现 | 位置 | 严重度 |
|---|---|---|---|
| 1 | runbook 发布闸门引用不存在的测试文件，干净 checkout 必然失败 | `docs/production-operations-runbook.md:24` | **高** |
| 2 | `test_excel_create.py` 因 `node_modules/@oai/artifact-tool` 缺失整文件静默 skip，本机实测即触发 | `tests/test_excel_create.py:19-25` | **高** |
| 3 | 该 `artifact-tool` 依赖未在任何 requirements 声明，是唯一真实可失效的外部依赖 | `pyproject.toml:7-38` | 中 |
| 4 | 方程面（`math_tool`）有 8 个功能测试但**零恶意输入测试**，沙箱只测"自由符号报错" | `tests/test_agent_tools.py:536-605` | **高** |
| 5 | `_files.py`（路径解析/blocklist）与 `_source_edit.py`（补丁算法）零直接测试 | `src/backend/tools/` | **高** |
| 6 | `chat_service.py`（配额/turn 锁顺序）、`sessions/sqlite.py`、`web_search/{baidu,bing,duckduckgo}.py` 无直接测试 | 见 §1.2 | 中 |
| 7 | 25 个语言 `_edit.py` 完全没有专属测试文件 | `src/backend/tools/*_edit.py` | 中 |
| 8 | `pytest.mark.slow` marker 声明但全库零使用 | `pyproject.toml:51` | 低 |
| 9 | `test_web_search_providers.py`（67 测试，全库最大）不覆盖任何具体 provider 实现 | `tests/test_web_search_providers.py` | 中 |

---

## 5. 覆盖方法学与局限

- 覆盖判定基于**静态导入分析**（直接 import + 父包属性引用二次确认），未执行测试套件。因此"有测试"只表示存在导入关系与断言，不代表测试当前通过。
- "零直接测试"已二次确认过父包属性引用路径，但仍可能通过 `importlib`、字符串路径、或 `monkeypatch` 间接覆盖——本报告标记为"高置信度空洞"而非"绝对未覆盖"。
- 未读 `.pytest_cache/v/cache/nodeids`（被 `.gitignore:13` 忽略），其中可能残留已删除测试的 node ID，是"曾存在过哪些测试"的唯一线索。
- 未运行 `pytest`，故上述 skip 行为的结论来自**源码静态判定 + 本机 node/artifact-tool 存在性实测**，而非实际 test run 输出。
