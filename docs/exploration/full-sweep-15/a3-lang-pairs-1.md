# A3 语言读写工具对（第一批）同构性审计

- 日期：2026-10-08
- 分支：production
- 范围：`src/backend/tools/` 下 11 个 `<lang>_edit.py` + 11 个 `<lang>_file.py`，外加 `text_edit.py`、`text_file.py`、`_source_edit.py`、`_files.py`
- 语言集合：c, go, groovy, haskell, java, javascript, julia, lua, matlab, php, python
- 方法：逐个读源码 + grep 调用点/注册点/测试点。**未**信任任何注释或 docstring，未执行被测代码，未跑测试套件。

## 0. 总结论（一句话）

这 22 个新文件本身**没有独立漏洞**：它们是逐字同构的薄壳，把全部逻辑委托给 `_source_edit.py`，而该引擎**完全不含** `eval` / `exec` / `compile()` / `subprocess` / `os.system`。任务预设的三个攻击面（python 编辑→执行链、`node --check`、`.m` 内塞 shell 命令）**经 grep 验证均不存在**。真正的问题集中在**测试盲区、门控不对称、和 841 行重复实现**上。

---

## 1. 横向对比表

列缩写：**引擎** = 调用的 `_source_edit.py` 入口（`N` = 绕过引擎自己实现）；**Exec** = 是否存在 eval/exec/compile/subprocess/os.system；**后缀** = 是否真正按扩展名校验（Y=是 / N=仅靠 LLM 传的字符串）；**语法校验** = 写入前是否有语言级语法检查；**原子写** = 临时文件 + `os.replace`；**大小** / **编码** / **换行**。

### 1.1 引擎与共享模块

| 文件 | 引擎 | Exec | 后缀 | 语法校验 | 原子写 | 大小 | 编码 | 换行 |
|---|---|---|---|---|---|---|---|---|
| `_source_edit.py` | — （引擎本体） | N | `resolve_safe_path(expected_suffixes=config.suffixes)` :584-589 / :1036-1041 | `line_validator`，本批 11 语言全为 `None` :77 | `mkstemp`+fsync+回读+`os.replace` :839-865 | `_MAX_LINE_CHARS` 50_000 / `_MAX_OPERATIONS` 100 / `max_bytes` | strict UTF-8，拒 NUL/BOM-16-32/控制符 :612-663 | LF/CRLF 保留，拒混合与裸 CR :646-651 |
| `_files.py` | — （路径核心） | N | `resolved.suffix.lower()` 比对 :73-77 | N/A | N/A | `stat().st_size > max_bytes` :79-81 | 不处理（只给 Path） | 不处理 |
| `text_edit.py` | **N — 841 行独立重实现** | N | `_SUFFIXES=(".txt",)` 硬编码 :37 | N | 同构 `mkstemp`+`os.replace` :721-747 | 同值常量 :38-41 | 同构 strict :513-562 | 同构 :545-550 |
| `text_file.py` | **N — 自实现** | N | `_SUFFIXES=(".txt",".md",".log",".csv")` :14 | N/A | N/A | `max_bytes` 经 `resolve_safe_path` | **`errors="replace"` :92** | 直通 |

### 1.2 本批 11 个 `_edit.py`（全部逐字同构，仅 4 个字面量不同）

| 文件 | 引擎 | Exec | 后缀 | 语法校验 | 原子写 | 大小 | 编码 | 换行 | 唯一差异（配置字段） |
|---|---|---|---|---|---|---|---|---|---|
| `c_edit.py` :21-29,40 | `build_source_edit_tools` | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".c",".h")` |
| `go_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".go",)` |
| `groovy_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".groovy",)` |
| `haskell_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".hs",)` |
| `java_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".java",)` |
| `javascript_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".js",".mjs",".cjs")` |
| `julia_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".jl",)` |
| `lua_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".lua",)` |
| `matlab_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".m",)` |
| `php_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".php",)` |
| `python_edit.py` :21-29,40 | 同上 | N | Y | N | Y | Y | strict | 保留 | `suffixes=(".py",)` |

### 1.3 本批 11 个 `_file.py`（全部逐字同构，仅 4 个字面量不同）

| 文件 | 引擎 | Exec | 后缀 | 语法校验 | 原子写 | 大小 | 编码 | 换行 | 唯一差异 |
|---|---|---|---|---|---|---|---|---|---|
| `c_file.py` :14-22,28 | `build_source_file_tool` | N | Y | N/A | N/A | Y | **`errors="replace"`** | 直通 | `suffixes=(".c",".h")` |
| `go_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".go",)` |
| `groovy_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".groovy",)` |
| `haskell_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".hs",)` |
| `java_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".java",)` |
| `javascript_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".js",".mjs",".cjs")` |
| `julia_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".jl",)` |
| `lua_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".lua",)` |
| `matlab_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".m",)` |
| `php_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".php",)` |
| `python_file.py` :14-22,28 | 同上 | N | Y | N/A | N/A | Y | 同上 | 直通 | `(".py",)` |

### 1.4 惯例合规性

**24 个文件全部遵守「每语言独立 `<lang>_edit.py` + `<lang>_file.py`」惯例，没有残缺对。** 11 个语言全部成对存在，且 22 个构建函数在 `src/backend/tools/__init__.py`（:3,4,36-41,44-47,58-60,72-73,85-86,104-105,116-117）与 `src/backend/mcp/providers.py`（读 :189-218；写 :244,288-332）两处都有注册。**无死代码、无注册了但不可达的工具。**

---

## 2. 不一致清单（偏离多数写法的文件及风险）

| # | 偏离对象 | 多数写法 | 偏离写法 | 风险 |
|---|---|---|---|---|
| 1 | `text_edit.py`（841 行） | 其余 11 个 `_edit.py` 各 48 行，一行委托引擎 | **完整复制引擎全部逻辑**，未 import `_source_edit` | 两处实现可漂移；已实际漂移（见 #3、#4） |
| 2 | `text_file.py` 的 `_SUFFIXES` | 与配套 `_edit` 共用同一组后缀 | `_SUFFIXES=(".txt",".md",".log",".csv")`，而 `text_edit.py` 只有 `(".txt",)` | 读 4 种、写 1 种，能力不对称（设计意图，非漏洞） |
| 3 | 11 个 `_file.py` 的 `flag_name` 字段 | `_edit.py` 的 `flag_name` 被 `build_source_edit_tools` 读取（:179） | **`build_source_file_tool` 从不读取 `config.flag_name`**（:994-1024 只用 `mime_type`/`tool_prefix`/`suffixes`/`label`） | 字段是死元数据，且造成门控不对称（见发现 2） |
| 4 | 读侧 vs 写侧编码错误处理 | 写侧 `errors="strict"`（`_source_edit.py:631`、`text_edit.py:530`） | 读侧 `errors="replace"`（`_source_edit.py:1053`、`text_file.py:92`） | 同一文件「读得出、编辑被拒」，见发现 4 |
| 5 | `tests/test_code_edit.py` 的 `LANGUAGES` 表 | 16 个语言逐一覆盖 | **缺 c、java、javascript、python** | 见发现 1（最刺眼） |
| 6 | 写出路径是否校验敏感文件名 | 读侧 `resolve_safe_path` 强制 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES` | **写侧 `_reserve_output_path` / `_reserve_created_output_path`（:804-825, 881-912）不校验** | 低：后缀被强制，实际无法构造出被禁名，但纵深防御缺失 |

**其余 22 个文件之间不存在任何偏差。** 逐行对比确认：11 个 `_edit.py` 的唯一差异是 `label`/`suffixes`/`creation_suffix`/`tool_prefix`/`mime_type`/`flag_name`/`temp_prefix` 七个字面量；11 个 `_file.py` 同理。行号结构完全对齐（配置 :14-22，构建函数 :25-28，`__all__` :31-34）。

---

## 3. 预设攻击面复核：三个都不存在

任务要求重点核查的三个假设，grep 结果全部为**否**：

1. **`python_edit.py` 编辑 `.py` 后被别的工具执行？** — 否。全仓 `src/backend/tools/` 中唯一 `subprocess` 是 `excel_create.py:10`（xlsx 生成，不在本批范围）。`grep -rniE 'subprocess|Popen|os\.system|os\.popen'` 在整个 backend/frontend 只命中 `excel_create.py` 与 MCP 客户端自身的进程清理。**没有任何工具执行 `.py` 文件**，故不存在代码执行链。
2. **`javascript_edit.py` 的 `node --check`？** — 否。`javascript_edit.py` 全文 48 行，只有配置字典 + 一行委托，不含任何子进程调用。
3. **`matlab_edit.py` 在 `.m` 里塞 shell 命令？** — 否。`.m` 内容只经过 `_decode_text`/`decode_source` 的文本校验（拒控制字符）后原样写为下载物，无人执行。

**推论**：这三个工具的威胁模型应重新定位为「下载物投毒」（用户下载后在自己机器上运行），而非「服务端 RCE」。引擎的设计（session-scoped、copy-on-write、仅产物下载）与之自洽。

---

## 4. 发现清单

### [中] 发现 1：c、java、javascript、python 四个语言对零测试覆盖

**证据**：`tests/test_code_edit.py:27-64` 的 `LANGUAGES` 表列出 16 个语言：r, rust, go, sql, php, ruby, latex, prolog, haskell, lua, julia, shell, matlab, groovy, swift, log。**不含 c、java、javascript、python。**
全仓 `grep -rn "build_c_edit_tools|build_java_edit_tools|build_javascript_edit_tools|build_python_edit_tools|build_c_file_tool|build_java_file_tool|build_javascript_file_tool|build_python_file_tool" tests/ --include=*.py` **零命中**（本批 11 个 `_file.py` 读工具同样零命中）。

**触发条件**：任何对这四个语言对的配置手误（后缀写错、flag 名错拼）都不会被 CI 抓到。
**影响**：本批 11 个 `_edit` 中 7 个有测试（go/groovy/haskell/julia/lua/matlab/php），4 个无测试；11 个 `_file` 读工具全部无测试。而 `text_edit.py` 有 `tests/test_text_edit.py` 400+ 行覆盖，`text_file.py` 在 `test_file_tools.py` 有覆盖 — **覆盖强度与本批 22 个新文件严重倒挂**。
**建议修法**：把这 4 个语言补进 `LANGUAGES` 表即可零成本获得往返 + 遍历拒绝的覆盖（表驱动的 3 个参数化测试会自动套用）；并为 `_file.py` 读工具补一条「后缀拒绝 + 越界拒绝」用例。
**注意**：这 4 个文件与已测的 7 个**逐字同构**，所以当前实际风险很低 — 这是覆盖债，不是活跃漏洞。

### [中] 发现 2：读工具门控依赖调用方，构建函数自身无防线

**证据**：`build_source_edit_tools` 自带门控（`_source_edit.py:179`）：
```python
if not (settings.file_read_enabled and getattr(settings, config.flag_name)):
    return []
```
而 `build_source_file_tool`（`_source_edit.py:994-1024`）**无任何门控**，无条件构造工具：
```python
def build_source_file_tool(settings: Settings, config: SourceEditConfig) -> BaseTool:
    root = Path(settings.file_read_root)
    max_bytes = settings.file_read_max_bytes
    ...
    return StructuredTool.from_function(...)
```
实际防线在 `src/backend/mcp/providers.py:186-187`：
```python
if not settings.file_read_enabled:
    return ()
```
**触发条件**：任何新增的 `build_source_file_tool` 调用点忘记做 `file_read_enabled` 检查。
**影响**：`file_read_enabled=False` 的关停语义对编辑工具是绝对的，对读工具是「靠调用方记得」。当前唯一调用点合规，故为纵深防御缺口而非可利用漏洞。附带问题：设置 `python_edit_enabled=False` 会关掉 `edit_python_file`，但 `read_python_file` **仍然构建** — 配置表面与实际能力不一致（与 MEMORY.md 中「config surface vs effective drift」是同一模式）。
**建议修法**：把 `file_read_enabled` 检查下沉进 `build_source_file_tool`，让构建函数自洽。

### [中] 发现 3：`text_edit.py` 是 841 行的完整重复实现，与引擎已发生实际漂移

**证据**：`text_edit.py` 不 import `_source_edit`，而是自带 `TextEditOperation`/`_TextDocument`/`_decode_text`/`_plan_operations`/`_apply_operations`/`_write_text_atomically`/`_reserve_output_path`/`_output_stem`/`_prepare_session_directory`（:72-826）。与 `_source_edit.py` 对应逻辑逐行比对：`_decode_text`(text_edit:513) vs `decode_source`(_source_edit:612)、`_plan_operations`(565) vs `_plan_operations`(674)、`_write_text_atomically`(711) vs `_write_source_atomically`(828) — 逻辑等价，但有两处已分叉：

1. `text_edit.py:306-308` 的 `create_text_file` **不调用行校验器**（引擎侧 `create_source_file` 在 `_source_edit.py:368` 调 `_validate_document_lines`）。本批语言 `line_validator=None`，故当前无行为差异 — 但一旦未来给某语言加校验器，`text_edit` 与 `_source_edit` 就会分叉出真实行为差。
2. 错误消息、日志前缀、artifact 构造各自一份。

**触发条件**：在 `_source_edit.py` 修任何安全 bug（例如锚点匹配或大小限制），`text_edit.py` 不会继承。
**影响**：引擎是单点修复，text 编辑是第二份需要独立同步的副本。
**建议修法**：把 text 改造成 `_edit.py` 那种薄壳（`SourceEditConfig(label="Text", suffixes=(".txt",), creation_suffix=".txt", tool_prefix="text", mime_type=TEXT_MIME_TYPE, flag_name="text_edit_enabled", temp_prefix=".text_write-")`），删除约 600 行重复代码。

### [低] 发现 4：读侧 `errors="replace"`、写侧 `errors="strict"`，同一文件「读得出但编辑被拒」

**证据**：
- 写侧严格：`_source_edit.py:631` `text = payload.decode("utf-8", errors="strict")`，且 :628 拒 NUL、:643 拒控制字符。
- 读侧宽容：`_source_edit.py:1053` `resolved.read_text(encoding="utf-8", errors="replace")`；`text_file.py:92` 同样。

**触发条件**：用户上传一个含非法 UTF-8 字节或 UTF-16 BOM 的 `.py` 文件。
**影响**：`read_python_file` 会返回满屏 `U+FFFD` 替换字符（BOM-16 甚至不在 `_BLOCKED` 列表内，会被当成普通字节读出），而 `inspect_python_file` 会拒绝。LLM 拿到的是**无法用于编辑的乱码**，且 `errors="replace"` 会静默丢失字节，没有任何「此文件不受支持」的提示。这不是提权漏洞，但会让模型基于损坏内容生成错误产物。
**建议修法**：读侧对 BOM-16/32 与非 UTF-8 显式报错，与写侧口径统一；或至少在返回头里标注「含无效 UTF-8，已替换」。

### [低] 发现 5：22 处 `X_CONFIG` 重复定义 + 4 个死字段

**证据**：每个 `_edit.py` 和 `_file.py` 各自独立实例化同一个 `SourceEditConfig`（例如 `python_edit.py:21-29` 与 `python_file.py:14-22` 内容完全相同）。
**影响**：改后缀要改 22 处。当前无漂移，但这是本批文件最大的可维护性债。
**死字段**：`_file.py` 里声明的 `flag_name`、`temp_prefix`、`mime_type` 在 `build_source_file_tool` 中**从未被读取**（该函数只用 `label`/`suffixes`/`tool_prefix`），是误导性元数据。
**建议修法**：把每个语言的 config 收敛到单一来源（例如 `<lang>_config.py` 或在 `_source_edit.py` 里建配置表），`_edit.py`/`_file.py` 都从同一处取。

### [低] 发现 6：写出路径不校验敏感文件名/后缀

**证据**：`resolve_safe_path` 在 `_files.py:65-77` 强制 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES` 与后缀白名单；但写侧 `_reserve_created_output_path`（`_source_edit.py:804-825`）与 `_reserve_output_path`（:881-912）只做 `_SAFE_NAME.sub("_", ...)` 字符清洗，不查黑名单。
**影响**：实际危害极小 —— 后缀被强制替换为 `config.creation_suffix`，且 `requested.strip().strip(".")`（:798, :921）剥掉了前导点，无法构造出 `.env`/`id_rsa` 之类的输出名，且输出只落在 session 目录内。属纵深防御缺口。
**建议修法**：写出前复用同一套 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES` 检查，让读写两侧口径一致。

### [低] 发现 7：原子写无目录 fsync，且 mkstemp 的 0600 权限被保留到产物

**证据**：`_source_edit.py:839-865` / `text_edit.py:721-747`：`mkstemp` → 写 → `fsync(stream.fileno())` → 回读校验 → `os.replace(temp_path, target)`。
**影响**：(a) 只 fsync 了文件，没 fsync 父目录，掉电时 `os.replace` 本身可能丢失（耐久性，非安全）；(b) `mkstemp` 建的是 0600，`os.replace` 后产物保留 0600，多用户部署下 HTTP 服务账户可能读不到已发布文件（功能性 bug，Windows 上无感）。
**建议修法**：`os.replace` 后 `os.fsync(dir_fd)`；产物按预期 umask 设置权限。

---

## 5. `_source_edit.py` 自身攻击面：锚点/定位匹配

**核心结论：本引擎不存在「锚点 + 正则」这类可被构造性输入利用的定位机制。** 任务假设的四个子问题逐一核查：

| 假设的利用方式 | 实际机制 | 是否可利用 |
|---|---|---|
| 锚点含正则元字符 | **无锚点、无正则**。定位是 `line_index`（int，`ge=0`）+ `expected_text` 精确字符串比较（`_plan_operations` :702 `document.lines[index] != operation.expected_text`） | 否 |
| 锚点跨文件 | 单文件文档对象 `_SourceDocument`，一次一批操作；`_plan_operations` 先全部校验再 `_apply_operations`（:684-711 / :714-740） | 否 |
| 空锚点 | 无锚点概念。`expected_text=""` 是合法输入，且精确匹配空行是正确语义（不产生歧义） | 否 |
| 锚点歧义导致误改 | 按行号定位，无歧义 | 否 |
| `replace_all` 爆炸半径 | **无 `replace_all` 参数**（`SourceEditOperation` :98-149 只有 `action`/`line_index`/`expected_text`/`new_text`） | 否 |

**已确认的护栏**：
- 同一 `line_index` 不能重复定位（:692-694）→ 无单点重复写。
- 越界行号拒绝（:696-701）。
- `expected_text` 必须精确匹配（:702-703）→ 天然乐观锁，文件被改动则整批失败。
- 失败时**不做任何修改**（:705-710 先收集全部问题再 raise）。
- `max_length=_MAX_LINE_CHARS`(50_000) 由 Pydantic 在 :117, :127 强制；`_MAX_OPERATIONS=100` 在 :324 强制；`_MAX_OUTPUT_VARIANTS=100` 在 :812, :897 限制枚举爆炸半径。
- 换行注入被 Pydantic 拦截（:145-148 拒绝 `\r`/`\n`）→ 单行操作无法写入多行，`_decode` 侧再拦裸 CR 与混合换行。
- `new_text or ""`（:726, :728, :731）用 falsy 折叠，`""` → `""`，无行为差异。

**路径穿越复核（这是引擎真正的攻击面）**：
- `_resolve_session_source`（:548-601）对相对路径尝试 `session_root/candidate` 与 `file_root/candidate`，然后 **`_is_within(resolved, session_resolved)`**（:580）二次收敛到 session 目录；绝对路径只试一次并同样收敛。
- `resolve()` 会跟随符号链接，故 session 内指向外部的软链会被 `_is_within` 拒绝。
- `../thread-b/secret.sql` 与绝对路径两种穿越方式均有测试：`tests/test_code_edit.py:159-186`，断言返回 `"outside this chat session"` 且**不产生输出文件**（:186）。
- `thread_id` 无法穿越：`_THREAD_ID_PATTERN = ^[A-Za-z0-9._-]{1,64}$`（:47）拒绝含 `/` 的值；而 `".."` 虽能通过该正则，但 `session_upload_dir`（`src/frontend/chat/uploads.py:74-79`）经 `sanitize_filename`（:479-490，`Path(name).name` + `strip().strip(".")`）会把它消成空串并回退到 `"session"`。**链路闭环，不可逃逸。**
- 写出侧 `_reserve_output_path` 额外拒绝覆盖源文件（:900 `if candidate.resolve() == resolved_source: continue`），`_prepare_session_directory`（:766-782）禁止输出目录等于 file_root 本身。

---

## 6. 「编辑工具能否写入任意路径」

**不能。** 三条链路均收敛：

1. **创建**：`_creation_stem`（:785-801）先 `Path(filename).name` 剥掉目录，再 `_SAFE_NAME.sub("_", ...)` 把 `/`、`\\`、空格等替换为 `_`，最后强制替换为白名单后缀。输出只落在 `_prepare_session_directory` 校验过的 session 目录内。
2. **编辑输出名**：`_output_stem`（:915-924）同样 `Path(output_name).name` + 字符清洗 + `strip(".")`。`output_name="../../etc/passwd"` → `.name`=`"passwd"` → 写成 session 内的 `passwd.py`。
3. **所有写入都经 `resolve_safe_path` 或等价的 `_is_within` 收敛**；没有一处自己拼路径绕过 `_files.py`。

---

## 7. 注册点与可达性

| 位置 | 内容 |
|---|---|
| `src/backend/tools/__init__.py:3-4,36-41,44-47,58-60,72-73,85-86,104-105,116-117` | 22 个构建函数全部导出 |
| `src/backend/tools/__init__.py:317-373` | `__all__` 全部列出 |
| `src/backend/mcp/providers.py:186-225`（`DocumentToolProvider`） | 11 个读工具，先查 `file_read_enabled` |
| `src/backend/mcp/providers.py:230-345`（`SessionEditingToolProvider`） | 11 个写工具（各展开为 create/inspect/edit 三个），全部传 `session_root` + `thread_id` |
| `src/config/settings.py:219-242` | 11 个 `*_edit_enabled` 标志齐全，默认全 `False` |
| `src/config/loader.py:149-162` | 11 个环境变量映射齐全（`PYTHON_EDIT_ENABLED` 等） |

**结论：无死代码，无「注册了但不可达」的工具。** 11 个语言对的读工具由 `file_read_enabled` 单独门控，写工具额外受 `<lang>_edit_enabled` 门控。

**附带发现（超出本批文件但相邻）**：`src/frontend/chat/uploads.py` 的 edit-note 提示串只对 text/markdown/typescript/json/jsonl/yaml/r/rust/go/sql/php/ruby/haskell/lua/julia/shell/matlab/groovy 等生成，`python_edit_enabled`、`java_edit_enabled`、`javascript_edit_enabled`、`c_edit_enabled` **未出现在 `uploads.py` 的参数与 `api.py` 的透传中**（grep 验证）。这意味着即使开启 `PYTHON_EDIT_ENABLED=true`，上传提示里也不会告诉模型有 `create_python_file` 可用 — 工具在工具列表里，但提示层失声。属提示层配置漂移，不是安全问题。

---

## 8. 与已知问题的关系

- `_files.py:15-26` 的 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES` 缺口（缺 `id_dsa`/`id_ecdsa`/`authorized_keys`/`.asc`/`.keystore`/`.jks`/`.npmrc`/`.git-credentials`/`config.gpg`）已按 MEMORY.md 记录复现确认；本批新增读工具不扩大该集合，因为新增后缀（`.py`/`.js`/`.java`/`.m`/`.php`/`.lua`/`.jl`/`.hs`/`.groovy`/`.go`/`.c`/`.h`）与敏感名交集为空。
- `read_source_file`（`_source_edit.py:1046-1054`）与 `read_text_file`（`text_file.py:85-93`）**共用 `parser_key="utf8-text-v1"`**，同一文件经两个工具读只解码一次。`PARSED_FILE_CACHE`（`_file_cache.py:26-58`）有 `max_entries=128` 上限与 digest 缓存，**无无界增长 DoS**。

---

## 9. 建议优先级

1. **P2**：把 c/java/javascript/python 补进 `tests/test_code_edit.py:27-64` 的 `LANGUAGES` 表（一次改动，零成本覆盖 4 个语言对 + 读工具）。
2. **P2**：`build_source_file_tool` 内建 `file_read_enabled` 门控，消除「靠调用方记得」。
3. **P3**：读侧编码错误处理与写侧对齐（发现 4）。
4. **P3**：把 `text_edit.py` 降为薄壳（发现 3），或删除重复实现。
5. **P4**：把 22 处 `X_CONFIG` 收敛为单一来源，删掉 `_file.py` 的 3 个死字段（发现 5）。
6. **P4**：写出侧复用敏感名检查（发现 6）；补目录 fsync 与产物权限（发现 7）。
