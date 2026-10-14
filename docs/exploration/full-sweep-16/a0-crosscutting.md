# Full Sweep 16 — a0 横切扫描（配置装载链 / 静默失败面 / 测试可见性）

- 日期：2026-10-09
- 范围：由主代理亲自做的仓库级 grep / 静态提取，不依赖任何子代理
- 方法：所有结论均为 `[实测]`（脚本提取或直接读文件），无一来自注释或文档

---

## 0. 一条主线：这批"新功能"在部署产物里全部是死代码

这是本轮最重的一条，链路上每一环都已实测验证。

### 0.1 38 个工具开关在 settings.py 里默认全为 `False` `[实测]`

`src/config/settings.py` 共声明 179 个字段。其中以下**全部默认 `False`**：

```
c_edit_enabled, csv_edit_enabled, env_edit_enabled, excel_create_enabled,
excel_edit_enabled, go_edit_enabled, groovy_edit_enabled, haskell_edit_enabled,
html_edit_enabled, ini_edit_enabled, java_edit_enabled, javascript_edit_enabled,
json_edit_enabled, jsonl_edit_enabled, julia_edit_enabled, latex_edit_enabled,
log_edit_enabled, lua_edit_enabled, markdown_edit_enabled, matlab_edit_enabled,
pdf_edit_enabled, php_edit_enabled, powerpoint_edit_enabled, prolog_edit_enabled,
properties_edit_enabled, python_edit_enabled, ruby_edit_enabled, rust_edit_enabled,
shell_edit_enabled, sql_edit_enabled, swift_edit_enabled, text_edit_enabled,
toml_edit_enabled, typescript_edit_enabled, word_edit_enabled, xml_edit_enabled,
yaml_edit_enabled
```

以及 `file_read_enabled = False`（这是几乎所有编辑工具的前置条件）。

默认值为 `True` 的 enabled 类开关**只有 7 个**：`document_quality_filter_enabled`、
`memory_auto_recall_enabled`、`web_search_enabled`、`web_search_domain_reputation_enabled`、
`web_search_structure_filter_enabled`、`planning_enabled`(False，不在列) …
实际 `config/default.yaml` 里 `enabled: true` 的只有 4 项（58/80/187/203 行）。

### 0.2 部署路径上没有任何一层能把它打开 `[实测]`

**`config/production.yaml` 全文件只有 1 个 `_enabled` 键**（第 6 行 `web_search_enabled: true`），
其余全部回落 `config/default.yaml`，而后者把 38 个编辑开关逐一写成 `false`。

**`docker-compose.yml` 没有 `env_file:`**。它只显式转发 4 个变量（第 13–18 行）：
`DASHSCOPE_API_KEY` / `RAG_ENV` / `API_KEY` / `CORS_ALLOW_ORIGINS`，外加写死
`CHROMA_DIR` / `CHAT_API_HOST` / `CHAT_API_PORT`。**38 个工具开关一个都没转发。**

**`Dockerfile` 第 27–28 行只 COPY `config/` 和 `src/`，从不 COPY `.env`。**

### 0.3 运行时确实会读 `.env`，但读不到 `[实测]`

`src/config/loader.py`：
- `:697` 声明优先级 `CLI > env > YAML > built-in default`
- `:699` `load_dotenv(env_file)`，默认 `env_file=".env"` —— **相对 CWD**
- `:655-657` `_apply_env_overrides` 逐个 `os.getenv(env_name)`

容器内 CWD 下没有 `.env`（compose 不注入、镜像不 COPY），`load_dotenv` 对不存在的文件
**静默 no-op**，没有任何日志。于是容器内所有 38 个开关取 YAML 值 = `False`。

### 0.4 结论：最近 4 个 commit 交付的功能在两个方向上都是死的 `[实测]`

最近提交：`fa7c64f` (.whl 归档)、`fcfbcd5` (INI 工具)、`bc42eba` (Java .properties 工具)、
`b07dbf3`/`8b93091` (TOML/XML/.env 编辑器文档)。

- **部署方向**：`INI_EDIT_ENABLED` / `PROPERTIES_EDIT_ENABLED` 既不在 `production.yaml`、
  也不在 `compose` 的 `environment:` → 容器内这两个工具族**完全不存在于工具目录**。
  代码合入了，用户却永远调不到。
- **连开发者本机都没开**：仓库根的 `.env`（本地开发实际用的那个，gitignored）里
  `grep INI_EDIT_ENABLED` 和 `grep PROPERTIES_EDIT_ENABLED` **均为空** → 走 settings 默认
  `False` → **新加的两个工具族在作者的机器上也是关的**。
  对照：`.env` 里其余 49 个 `*=true` 一应俱全。
  也就是说最近两次 feature 提交（INI、properties）是"写了代码、改了 `.env.example` 文档、
  但连 `.env` 自己都忘了加"。

**建议的最小修法**（任一即可，我未改动）：
1. 在 `docker-compose.yml` 加 `env_file: - .env`（最简单，但需确认 `.env` 不该进镜像的
   安全立场）；或
2. 把工具开关统一收进 `config/production.yaml`，让"启用哪些工具"成为配置而不是环境变量；或
3. 至少把 `INI_EDIT_ENABLED=true` / `PROPERTIES_EDIT_ENABLED=true` 补进 `.env.example`
   对应的部署清单 —— 目前它们在 `.env.example` 里，但没有任何部署文档提到要设。

---

## 1. `getattr(settings, ...)` 静默失败面：45+ 处，横跨 9 个模块 `[实测]`

记忆里的"坑"原本被记录为只存在于 `_source_edit.py:179`（20 个语言共享）。
实测后**这是一个全仓库问题，范围大得多**：

| 位置 | 处数 | 拼错的后果 |
|---|---|---|
| `src/backend/web_search/discovery.py` | 9 | 静默取默认 → 搜索预算/并发/fanout 悄悄变成默认值 |
| `src/backend/web_search/factory.py` | 17 | 静默取默认 → **provider api_key 变空串**，provider 静默下线 |
| `src/backend/graph/builder.py` | 11 | 静默取默认 → planning/reflection/critic 阈值悄悄复位 |
| `src/frontend/api/dependencies.py` | 4 | `llm_provider` 拼错 → 静默回落 `dashscope`，**换错模型供应商** |
| `src/frontend/api/auth.py` | 3 | `api_key` 拼错 → 返回 `""` |
| `src/frontend/api/amap_proxy.py` | 3 | 静默禁用地图代理 |
| `src/backend/tools/map_tool.py` / `directions.py` | 4 | `amap_web_service_key` 变空串 |
| `src/backend/graph/nodes/common.py:83` | 1 | persona 拼错 → 静默回落 `"trump"` |
| `src/backend/web_search/{semantic,reputation}.py` | 3 | 语义过滤/域名信誉静默关掉 |

对比：`_source_edit.py:179` 用的是 `settings.file_read_enabled and getattr(settings, config.flag_name)`
—— 前半句是**直接属性访问**（拼错会 `AttributeError`，会崩，好），后半句是 `getattr`（拼错静默）。
同一行里两种风格并存，说明这个模式是被复制扩散的，不是有意设计。

**修法建议**：全仓库把 `getattr(settings, "<flag>", <default>)` 改成直接属性访问。
`settings.py` 的字段是静态声明的（179 个，pydantic），不存在"可能没有"的情形，
`getattr` 在这里**纯粹是在掩盖拼写错误**。唯一例外是 `factory.py` / `dependencies.py` 这类
接受任意 `config` 对象的函数，那里的 `getattr` 是合理的（参数类型不是 settings）。

---

## 2. 一个错误的注释，指向真实的 RCE `[实测]`

`src/backend/tools/math_tool.py:152`：

```python
# parse_expr is SymPy's safe parser: no Python eval of arbitrary code.
```

这条注释**是错的**，而且就是它让这条 RCE 链藏了多轮审查（记忆 `sympy-parse-expr-rce` 已实测确认）。
`parse_expr` 在 `global_dict is None` 时会注入 `vars(builtins)`（含 `__import__`）。
第 153 行的调用只传了 `transformations` + `evaluate`，**没有 `local_dict` / `global_dict`**
→ 命中不安全分支。

另注：记忆记载 `pyproject.toml` 钉 `sympy==1.14.0` 而实际装的是 1.13.1 ——
本轮未重复验证版本漂移，留给文件级代理。

`src/backend/tools/linalg_tool.py:163` 是第二个入口：`sympy.sympify(value)` 不带 `locals=`。

---

## 3. 测试可见性：6 个静默跳过点 + 5 个指向已删除文件的死 nodeid `[实测]`

### 3.1 静默跳过（全仓库穷举，仅 6 处）

| 位置 | 机制 | 触发条件 |
|---|---|---|
| `tests/test_excel_create.py:22-24` | `pytest.skip("artifact-tool Node runtime is unavailable")` | `node_modules` 不存在 → **整个文件跳过** |
| `tests/test_word_edit.py:20` | `importorskip("docx")` | 未装 python-docx |
| `tests/test_powerpoint_edit.py:20` | `importorskip("pptx")` | 未装 python-pptx |
| `tests/test_chat_upload.py:37,409,479` | `importorskip("pptx"/"docx"/"openpyxl")` | 未装对应包 |
| `tests/test_web_search_pdf_loader.py:8` | `importorskip("pypdf")` | 未装 pypdf |
| `tests/test_mandarin_search_benchmark.py:69` | `@pytest.mark.skipif(...)` | 环境标记 |

注意：`test_excel_create.py:19-25` 用 `skip` 而不是 `importorskip`，跳过条件检查的是
**目录存在性**（`Path("node_modules")` 相对 CWD）→ 在不同 checkout 目录下结论不同，
且**不会在报告里显示为 import 缺失**。这是记忆"两种静默测试丢失机制"中环境依赖那种的实体。

### 3.2 `.pytest_cache/v/cache/nodeids` 里的死引用 `[实测]`

缓存共 15136 条 nodeid，其中属于本项目 `tests/` 的 1494 条，
**5 个指向已不存在的测试文件**：

```
tests/test_api_errors.py
tests/test_api_security_readiness.py
tests/test_api_streaming.py
tests/test_mcp_server.py
tests/test_rag_application_service.py
```

`lastfailed` 里另有 6 条。`.pytest_cache/` 被 `.gitignore:13` 忽略 →
**这些丢失对任何 CI 都不可见**。记忆记载 `docs/production-operations-runbook.md:24`
引用了 `test_rag_application_service.py` —— 该文件确实已死，
所以发布手册的发布闸门在任何干净 checkout 上都会硬失败。

---

## 4. `.gitignore` 缺口 `[实测]`

`.gitignore` 全文 78 行，**没有 `.corrupt-*` 规则**。

`src/backend/memory/store.py` 的 `_quarantine` 会把 JSON 语法损坏的记忆文件
`os.replace` 成 `.corrupt-<ts>`（记忆 `memory-store-quarantine` 已验证路径 A）。
后果：这些隔离文件既不会被提交（好），也不会在 `git status` 里被解释为有意的产物（坏），
而且**没有任何清理策略**——它们会永久留在记忆目录里，且因为没进 ignore 规则，
任何一次 `git add .` 都会把损坏的记忆内容（可能含个人上下文）拉进版本库。

对比同文件其他运行时产物都有规则：`.chroma/`(22)、`memory/`(30)、`chat_uploads/`(33)。
缺的就是 `.corrupt-*`。spec 声称此项已完成（`.kiro/specs/chat-long-term-memory/design.md`），
属于文档-现实漂移。

---

## 5. `.env` ↔ `.env.example` 漂移 `[实测]`

- `.env` 127 个键，`.env.example` 182 个键。
- **只在 `.env.example`、`.env` 没有的：41 个键**，其中包括全部 `QUOTA_*`（14 个）、
  全部 web search provider key（`SERPER`/`BRAVE`/`TAVILY`/`BING_SEARCH_*`）、
  `INI_EDIT_ENABLED`、`PROPERTIES_EDIT_ENABLED`、`PDF_EDIT_ENABLED`、`WEB_FETCH_ENABLED`、
  `CHUNK_CONTEXT_*`(4)、`MEMORY_*`(9)、`AGENT_PERSONA_STYLE`、`API_HOST`/`API_PORT`。
- **只在 `.env`、`.env.example` 没有的：1 个键** —— `HF_HUB_OFFLINE`。
  这是唯一一个本地有、模板没有的，说明模板没有反向同步。

其中 `API_HOST`/`API_PORT` 出现在 `.env.example` 但按记忆记载是**死代码**
（真正绑定用 `CHAT_API_HOST`/`CHAT_API_PORT`）—— 模板在教用户设两个不起作用的变量。
本轮未复核该条，留给后端主干代理。

---

## 5.5 CI 只有一个真实闸门，且 release_gate 从未接入 `[实测]`

`.github/workflows/ci.yml` 全文 60 行，共 6 个 step。逐条判定：

| Step | 是否真实闸门 | 备注 |
|---|---|---|
| pytest `--cov-fail-under=70` | ✅ 是 | 覆盖率下限 70% |
| `ruff check .` | ✅ 是 | |
| `ruff format --check .` | ✅ 是 | |
| `mypy src/` | ✅ 是 | |
| `pip audit` | ❌ **被架空** | 第 56 行 `continue-on-error: true` |
| `release_gate.py` | ❌ **从未接入** | 全文 grep 无任何引用 |

两个具体问题：

**(a) pip-audit 的"单独跟踪"承诺无处可寻 `[实测]`**
第 53–55 行注释声称：
> Known-vulnerable pins in the locked runtime stack are tracked separately; keep this
> advisory until the pins are upgraded so the gate does not mask test/lint failures.

但工作流里**没有任何 SARIF 输出、没有 `pip-audit -f sarif`、没有 upload-artifact、
没有指向跟踪文件的任何路径**。"tracked separately" 在这个仓库里指不到任何实体。
也就是说：依赖审计失败既不会红灯，也没有留痕。
（记忆记载"跟踪文件被 gitignore 且不存在"——比这更进一步：这里连跟踪机制的痕迹都没有。）

**(b) `src/deployment/release_gate.py`（450 行）从未在 CI 里运行 `[实测]`**
这是验证生产拓扑 / secret reference / 镜像摘要的发布闸门，
CI 里没有任何 step 调用它。若它是有意由发布手册人工触发，则应在文档里写明；
结合第 3.2 节（runbook 引用的测试文件已死），这条链很可能从未真正跑通过。
需后端主干代理确认 `release_gate` 的预期调用方。

另注：`pytest` 只输出 `term` 覆盖率报告，不产出 XML，无 coverage 上传 →
本地 `.coverage` 是唯一的覆盖率实体，且被 `.gitignore:10` 忽略。

---

## 6. 覆盖自述

本文件覆盖：`.gitignore`、`.env`、`.env.example`、`docker-compose.yml`、`Dockerfile`、
`config/default.yaml`、`config/production.yaml`、`src/config/settings.py`（字段静态提取）、
`src/config/loader.py`（装载链）、`.pytest_cache/v/cache/{nodeids,lastfailed}`、
`tests/`（跳过点穷举）、`src/backend/tools/math_tool.py`（第 152 行注释 + 调用参数）。

未覆盖（分派给子代理）：各工具实现细节、图执行、记忆/会话/安全、web_search 包、
`config/staging.yaml`/`config/development.yaml`/`outbound-mcp.example.json`、
`docs/` 全部、`benchmarks/`、前端静态资源。
