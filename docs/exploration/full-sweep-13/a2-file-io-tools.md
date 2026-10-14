# A2 — 文件 IO 引擎 + 配置/数据格式工具

> 分区：文件 IO 引擎、文本/数据格式/配置/运维类工具
> 根目录：`E:\langgraph-rag`　日期：2026-10-06
> 纪律：只读源码，只信代码。所有结论带 `文件:行号`。

## 0. 状态

- 已完成：引擎 5 文件全文读完；文本类 6 文件全文读完；`csv_edit` / `json_edit` / `jsonl_edit` / `yaml_edit` / `xml_edit` / 全部 `*_file` 读取器（除 `ini_file`/`properties_file` grep 级）/ `excel_edit` / `excel_create(+.mjs)` 全文读完；`log/sql/shell` 6 个薄适配器全文读完。
- 完整读到：`toml_edit.py`、`ini_edit.py`、`properties_edit.py`、`env_edit.py` 的**全部函数定义、全部 `raise`、全部 `startswith` 调用**（grep 全覆盖）+ 各自 config 块原文；已确认四者均无公式注入守卫、均复用 `decode_source` + `_check_operation`/`_apply_operation` + `_publish_edited`。
- 未读：`excel_file.py`、`ini_file.py`、`properties_file.py`（用 grep 确认了 `_SUFFIXES`/工具名；`ini_file.py:14`= `(".ini",".cfg",".conf")`、`properties_file.py:14`= `(".properties",)`）。
- 已做的探针（只读）：`Path.suffix` 对多点文件名（`.env.local` → `.local`）行为的 Python 验证；`excel_create.mjs` 全文读取；测试目录 grep 验证守卫覆盖。

---

## 1. 逐文件说明

### 1.1 引擎层

**`_files.py`（191 行）— 唯一的路径安全边界**

- `_BLOCKED_NAMES`（`:15-25`）= `{".env", ".env.local", "credentials", "credentials.json", "id_rsa", "id_ed25519", ".netrc", ".pgpass", ".htpasswd"}`
- `_BLOCKED_SUFFIXES`（`:26`）= `{".pem", ".key", ".pfx", ".p12"}`
- `resolve_safe_path`（`:33-83`）、`resolve_safe_directory`（`:94-131`）、`list_safe_directory`（`:134-183`）、`_is_within`（`:86-91`，用 `Path.relative_to` 判包含）、`FileAccessError`（`:29-30`）
- 详见 §3 路径安全矩阵。

**`_file_cache.py`（101 行）— 内容寻址解析缓存**

- `ParsedFileCache(max_entries=128)`（`:26-31`）：32 条 stripe 锁（`:31`），`_values` LRU 上限 128，`_digests` LRU 上限 `max_entries*2=256`（`:85`）。
- 缓存键 = `(sha256, parser_key)`（`:42-43`）；SHA-256 分 1 MiB 块读（`:73-76`）。
- `_content_digest`（`:60-89`）先比对 `(size, mtime_ns)`，命中则**不重算 hash**；重算后用第二次 `stat` 比对，不一致则重试一次，两次都不一致抛 `OSError`（`:89`）。
- 全局单例 `PARSED_FILE_CACHE`（`:99`）。**只有读取器用**（`text_file.py:89`、`markdown_file.py:91`、`yaml_file.py:91`、`toml_file.py:91`、`xml_file.py:91`、`json_file.py:92`、`env_file.py:93`、`_source_edit.py:1050`），所有 edit 路径都走严格 `decode_source`，不走缓存。
- 所有读取器共用同一个 `parser_key="utf8-text-v1"`，所以同一份上传内容被解码一次即可。

**`_source_edit.py`（1064 行）— 会话级行编辑引擎**

- 公共常量：`_INSPECT_MAX_CHARS=20_000`（`:40`）、`_MAX_OPERATIONS=100`（`:41`）、`_MAX_LINE_CHARS=50_000`（`:42`）、`_MAX_OUTPUT_VARIANTS=100`（`:43`）、`_MAX_CHARS_READER=20_000`（`:45`）、`_SAFE_NAME=[^A-Za-z0-9._-]+`（`:46`）、`_THREAD_ID_PATTERN=^[A-Za-z0-9._-]{1,64}$`（`:47`）。
- `SourceEditConfig`（`:67-81`）：per-language 身份 + `line_validator` 钩子。
- 三个入口：`build_source_edit_tools`（`:170-274`，create/inspect/edit 三件套）、`build_source_file_tool`（`:994-1024`，只读）、`read_source_file`（`:1027-1064`）。
- 开关：`settings.file_read_enabled and getattr(settings, config.flag_name)`（`:179-180`），且 `session_root` 必须存在、`thread_id` 必须匹配 `_THREAD_ID_PATTERN`（`:181-182`）——不满足就返回 `[]`，工具完全不注册。
- 原子写：`tempfile.mkstemp` → 写 + `flush` + `os.fsync` → 重新 `_read_source_document` 解码比对 → `os.replace`（`:838-865`）；任何失败都 unlink temp **和** 已预留的 target（`:866-877`）。
- 见 §2 算法剖析。

**`list_files.py`（66 行）** — `list_files` 工具，`_MAX_ENTRIES=500`（`:15`），schema `le=2000`（`:28-33`），`FileAccessError` 转字符串返回（`:44-47`）。

**`zip_file.py`（274 行）**— `inspect_zip_file` + `read_zip_entry`；`_SUFFIXES=(".zip", ".whl")`（`:24`）、`_MAX_LISTED_ENTRIES=500`（`:28`）、`_BINARY_SNIFF_BYTES=8_192`（`:30`）。全程内存读取，不解包（`:1-7` docstring 属实，代码确实没有 `extract*`）。

### 1.2 文本类

| 文件 | 形态 | 关键差异 |
|---|---|---|
| `text_file.py` | 独立读取器 | `_SUFFIXES=(".txt",".md",".log",".csv")`（`:14`）— 与 markdown/log/csv 读取器后缀**重叠** |
| `text_edit.py` | **~841 行完整独立实现**（非适配器） | 自带 `_decode_text`/`_plan_operations`/`_apply_operations`/`_write_text_atomically` |
| `markdown_file.py` | 独立读取器 | `_SUFFIXES=(".md",)`（`:14`） |
| `markdown_edit.py` | **~849 行完整独立实现**（非适配器） | 同 text_edit 结构 |
| `html_edit.py` | 48 行薄适配器 | `HTML_CONFIG.suffixes=(".html",".htm")`，`build_source_edit_tools` |
| `html_file.py` | 34 行薄适配器 | `build_source_file_tool` |

- `text_edit.py`/`markdown_edit.py` 与 `_source_edit.py` 的算法**逐行同构**（`_plan_operations` 565-600 / 571-606 ↔ `_source_edit.py:674-711`；`_apply_operations` 603-629 / 609-635 ↔ `:714-740`），但已经出现漂移：`_creation_stem` 硬编码 `".txt"`（`text_edit.py:684-685`）/ `".md"`（`markdown_edit.py:690-691`），而 `_source_edit.py:791-797` 遍历 `config.suffixes`。
- 三者都被注册（`src/backend/mcp/providers.py:251`、`:254`、`:297`），所以**不是死代码**，是重复实现。
- 三个文件的工具名不冲突：`create_text_file`（`text_edit.py:251`）与 `_source_edit` 生成的 `create_<tool_prefix>_file` 不重叠，因为 text/markdown 不走 `_source_edit`。

### 1.3 数据格式类

| 文件 | 寻址方式 | 输入校验 |
|---|---|---|
| `csv_edit.py`（1041 行，独立实现） | 行/单元格索引 | 二维 `expected_row`/`expected_value` 精确匹配；`_MAX_ROWS=20_000`、`_MAX_COLUMNS=256`、`_MAX_CELL_CHARS=10_000`（`:48-50`）；delimiter sniff 限 `,;\\t|`（`:45`、`:669-679`） |
| `json_file.py` | — | `_SUFFIXES=(".json",".jsonl")`（`:14`） |
| `json_edit.py`（804 行，适配器+自有逻辑） | 点路径 `$.a[0].b` | 值必须 JSON 可解析（`:173-178`）；`_MAX_VALUE_CHARS=100_000`、`_MAX_INSPECT_ENTRIES=1_000`、`_MAX_SCALAR_CHARS=200`（`:48-50`）；`path` `max_length=500`（`:86`） |
| `jsonl_edit.py`（65 行，纯适配器） | 行索引 | 唯一带 `line_validator` 的配置：每行必须 `json.loads` 成功且非空（`:24-33`） |
| `yaml_file.py` | — | `_SUFFIXES=(".yaml",".yml")`（`:14`） |
| `yaml_edit.py`（855 行） | 点路径 | `yaml.safe_load_all` 且**拒绝多文档**（`:580-587`）；输出 `yaml.safe_dump(..., indent, width=4096)`（`:595-602`） |
| `toml_file.py` | — | `_SUFFIXES=(".toml",)`（`:14`） |
| `toml_edit.py`（888 行） | 点路径 | `tomllib.loads`（`:568`） |
| `xml_file.py` | — | `_SUFFIXES=(".xml",)`（`:14`） |
| `xml_edit.py`（897 行） | 点路径 + `#text`/`@attr` 键 | `ET.fromstring`（`:576`）；输出 `ET.tostring(..., xml_declaration=False)`（`:880`） |

- json/yaml/toml/xml 四个编辑器共享同一套私有 import（`_resolve_session_source`、`_SourceDocument`、`_write_source_atomically` 等），见 `json_edit.py:23-40`。
- 全部分为 3 类：行式（csv/text/markdown/html/log/sql/shell/jsonl）、路径式（json/yaml/toml/xml）、单元格式（excel）。

### 1.4 配置类

| 文件 | suffixes | key 语法 |
|---|---|---|
| `env_file.py` | `(".env", ".env.local", ".env.example")`（`:14`） | — |
| `env_edit.py`（686 行） | 同（`ENV_CONFIG` `:61-69`，`suffixes` 在 `:63`） | `_KEY_PATTERN=^[A-Za-z_][A-Za-z0-9_.]*$`（`:59`）；支持 `export ` 前缀（`:55-57`） |
| `properties_file.py` | `(".properties",)` | `_KEY_PATTERN=^(?:\\[=: \\]|[^=:\\ #!])+$`（`:47`） |
| `ini_file.py` | `(".ini", ".cfg", ".conf")` | `_KEY_PATTERN=^[^=:\[\]\s]+(\.[^=:\[\]\s]+)*$`（`:46`）；`ROOT_SECTION="(root)"` |

- 三者都走 `_resolve_session_source` + `_write_source_atomically`，全部复用 `_source_edit` 的会话边界与原子写。
- 关键 bug 见 §6-1 / §6-2。

### 1.5 运维类

- `log_edit.py`（48）/ `log_file.py`（34）/ `sql_edit.py`（48）/ `sql_file.py`（34）/ `shell_edit.py`（48）/ `shell_file.py`（34）——**全部是 4 行配置的薄适配器**。
- `SHELL_CONFIG.suffixes=(".sh",".bash")`、`creation_suffix=".sh"`（`shell_edit.py:21-29`）；`LOG_CONFIG.suffixes=(".log",)`；`SQL_CONFIG.suffixes=(".sql",)`。
- 无 `line_validator`，即不校验语法，只做 UTF-8/换行/控制字符级校验。

### 1.6 Excel（本区附带）

- `excel_edit.py`（754 行，openpyxl）：`_MAX_OPERATIONS=50`（`:45`）、`_MAX_INSPECT_ROWS=200`/`_MAX_INSPECT_COLS=30`（`:47-48`）、`_MAX_ZIP_ENTRIES=1024`、`_MAX_UNCOMPRESSED_BYTES=100_000_000`、`_MAX_COMPRESSION_RATIO=200`（`:52-54`）。
- `excel_create.py`（402 行）+ `excel_create.mjs`（124 行）：Python 校验 → 写 `request.json` → `subprocess.run(node, timeout=120)`（`:230-243`）→ `_validate_xlsx` zip 结构检查（`:345-356`）→ `os.replace`。
- 唯一有 DDE 守卫的地方是 `excel_edit.py:192-193`，见 §5。

---

## 2. 🔧 编辑算法剖析（`_source_edit.py`）

### 2.1 输入格式

`SourceEditOperation`（`:98-149`）：

```python
action: Literal["replace_line","delete_line","insert_before_line","append_line"]
line_index: int | None      # ge=0，来自 inspector 的零基索引
expected_text: str | None   # max_length=50_000
new_text: str | None        # max_length=50_000
```

字段组合由 `model_validator` 强制（`:131-149`）：
- `append_line` 不需要 `line_index`/`expected_text`
- `delete_line` 不需要 `new_text`
- **`expected_text`/`new_text` 含 `\r` 或 `\n` 直接拒绝**（`:145-148`）——所以单行限制是硬性的，无法用一次 op 插入多行。

批量：`operations` `min_length=1, max_length=100`（`:321-329`）。

### 2.2 定位策略：**零基索引 + 内容寻址（exact match），无 fuzzy / 无 offset / 无 difflib**

全库 grep 确认：`_source_edit.py`、`text_edit.py`、`markdown_edit.py`、`csv_edit.py` 中**没有** `difflib`、`unified_diff`、`hunk`、`fuzzy`、`offset`。定位靠 `_plan_operations` 的两个精确比较。

### 2.3 逐步流程

```
edit_source_file (456-526)
  1. _resolve_session_source  (548-601)   会话目录边界 + resolve_safe_path
  2. _read_source_document    (604-609)   → decode_source (612-663) 严格 UTF-8
  3. _plan_operations         (674-711)   只读校验，全批
  4. _apply_operations        (714-740)   单遍生成新文档
  5. _validate_document_lines (529-545)   语言级 line_validator（仅 jsonl 有）
  6. _publish_source (743-763)
       → _reserve_output_path (881-912)   touch(exist_ok=False) 独占预留
       → _write_source_atomically (828-878) mkstemp→fsync→重解码比对→os.replace
```

### 2.4 `_plan_operations` — 校验阶段（`_source_edit.py:674-711`）

```python
for position, operation in enumerate(operations):
    if operation.action == "append_line":
        appends.append(operation); continue
    index = operation.line_index
    if index in indexed:
        problems.append(f"...line {index} is targeted more than once."); continue
    indexed[index] = operation
    if index >= len(document.lines):
        problems.append(f"...line_index {index} is out of range; ..."); continue
    if document.lines[index] != operation.expected_text:
        problems.append(f"...expected_text does not exactly match line {index}.")
if problems:
    raise SourceEditError("no changes were made because ... Re-run inspect_* and retry.")
```

要点：
- **全部对着原始行校验**，校验时不做任何 mutation，因此索引天然稳定。
- **all-or-nothing**：一条冲突则整批拒绝，磁盘零写入。
- 三类冲突：重复目标行、越界、`expected_text` 不精确匹配。
- 所有问题**汇总报错**（不是 fail-fast），便于模型一次修完。

### 2.5 `_apply_operations` — 应用阶段（`:714-740`）

```python
output: list[str] = []
for index, line in enumerate(document.lines):
    operation = indexed.get(index)
    if operation is None:
        output.append(line)
    elif operation.action == "replace_line":
        output.append(operation.new_text or "")
    elif operation.action == "insert_before_line":
        output.extend((operation.new_text or "", line))
    else:  # delete_line
        continue
output.extend(operation.new_text or "" for operation in appends)
return _SourceDocument(..., has_final_newline=document.has_final_newline and bool(output))
```

单遍 O(n)，无需维护索引偏移——这是"先全量校验、再单遍输出"设计的直接红利。

### 2.6 失败模式与边界（实测逻辑推演）

1. **`insert_before_line` 无法插到文件末尾**。校验阶段对 `insert_before_line` 也要求 `line_index < len(lines)`（`:696-701`），所以"插到最后"必须用 `append_line`。语义上没有 `insert_after_line`。
2. **`append_line` 的位置与批次顺序无关**。所有 append 恒落到末尾（`:731`），即使模型把它排在 `replace_line` 中间。
3. **删除全部行会把 `has_final_newline` 强制归零**（`:739`）——注释（`:736-738`）明确说是为了让空文件能过 `encode_source` 往返校验。注释与代码一致。
4. **空文件 + 仅 `append_line`**：`decode_source` 对空文本返回 `lines=()`、`has_final_newline=False`（`:655-657`），append 成功但产物没有结尾换行。
5. **无多行插入**：`new_text` 含 `\n` 在 schema 层就被拒（`:145-148`）。
6. **单批最多 100 个 op、单行最多 50000 字符**（`:41-42`）；同一 stem 最多 100 个变体（`:43`），超限报 "too many files named ... already exist"。
7. **往返校验是最后一道闸**：`_write_source_atomically` 写完 temp 后**重新解码**并比较 `_SourceDocument`（`:862-864`），不等则报 "did not pass validation" 并回滚。
8. **源文件永不被改写**：`_reserve_output_path` 明确跳过 `candidate.resolve() == resolved_source`（`:900-901`）。

### 2.7 CSV 变体：两遍 plan + 单元格先行

`csv_edit.py:691-790`（`_plan_operations`）与 `_source_edit` 结构不同：

- **第一遍**分桶并查重复（`:709-732`）：`structural: dict[row_index, op]`、`cell_targets: dict[(row_index, column), op]`、`appends`。
- **第二遍**查范围/期望值/跨类冲突（`:740-775`）。
- 关键跨类规则（注释 `:697-701` 与代码 `:761-765` 一致）：`update_cell` 与同行的 `replace_row`/`delete_row` 冲突，但**不**与 `insert_before_row` 冲突（插入后原行仍存在）。
- 应用顺序（`csv_edit.py:793-824`）：**先应用全部 cell updates，再做结构行操作**。因为 `update_cell` 的行号是原始行号，结构操作不移动它。
- `_apply_operations` 末尾再查 `len(output) > _MAX_ROWS`（`:816-817`）。
- `_decode_csv` 在解析期就限列数/行数（`:651-658`）。

### 2.8 路径式变体（json/yaml/toml/xml）

- `_check_operation`（如 `json_edit.py:663-729`）对**原始文档**逐个校验并聚合报错；`_apply_operation`（`:732-756`）对 deepcopy 逐个应用，并且**再校验一次**（`:735-737`，此时文档已被前面的 op 改动）——所以重复目标是安全的，但错误提示形态与 CSV 的聚合报错不同，见 §6-11。
- JSON 重新序列化会规范化缩进（`_detect_indent` `:776-790`）；YAML 用 `safe_dump` 丢注释（`yaml_edit.py:591-608`）；XML 丢声明（`xml_edit.py:880`）。

---

## 3. 🛡️ 路径安全矩阵

### 3.1 校验点行号索引

| 机制 | 位置 |
|---|---|
| `_BLOCKED_NAMES` 定义 | `_files.py:15-25` |
| `_BLOCKED_SUFFIXES` 定义 | `_files.py:26` |
| `resolve_safe_path` | `_files.py:33-83` |
| ├ 去引号/空白 + 非空 | `_files.py:46-48` |
| ├ root 解析 `expanduser().resolve()` | `_files.py:50` |
| ├ 相对路径挂到 root 下 | `_files.py:51-53` |
| ├ `candidate.resolve()`（跟随符号链接） | `_files.py:56` |
| ├ **包含检查** `_is_within` → `relative_to` | `_files.py:60-63` + `:86-91` |
| ├ **blocklist（name + suffix）** | `_files.py:65-66` |
| ├ exists / is_file | `_files.py:68-71` |
| ├ `expected_suffixes` vs `resolved.suffix.lower()` | `_files.py:73-77` |
| └ 大小 `stat().st_size > max_bytes` | `_files.py:79-81` |
| `resolve_safe_directory` | `_files.py:94-131`（同名/后缀检查**在 root 自身时跳过** `:121-124`） |
| `list_safe_directory` | `_files.py:134-183` |
| ├ dot-prefix 过滤 | `_files.py:153-155` |
| ├ symlink 过滤 | `_files.py:157-158` |
| └ 名称+字节数渲染 | `_files.py:163-166` |
| `realpath` / `is_relative_to` / `symlink` grep 结果 | **`realpath`：0 命中**；**`is_relative_to`：0 命中**（用 `relative_to` 代替）；`is_symlink` 仅 `_files.py:157` |

### 3.2 不对称矩阵

| 过滤器 | `resolve_safe_path` | `resolve_safe_directory` | `list_safe_directory`（对子条目） |
|---|---|---|---|
| root 包含检查 | ✅ `:60-63` | ✅ `:116-119` | ✅（继承 `:142`） |
| `_BLOCKED_NAMES` | ✅ `:65` | ✅ `:121`（**root 自身豁免** `:123`） | ❌ **完全不查** |
| `_BLOCKED_SUFFIXES` | ✅ `:65` | ✅ `:122` | ❌ **完全不查** |
| dot-prefix 隐藏项 | 无此过滤（`.env` 靠 blocklist） | 无 | ✅ `:153-155` |
| symlink | `resolve()` 跟随（`:56`），可能解析到 root 外 → 被包含检查拦 | 同 | ✅ 跳过 `:157-158` |
| 后缀白名单 | ✅ `:73-77` | ❌ 有意跳过 | n/a |
| 文件大小上限 | ✅ `:79-81` | ❌ 有意跳过 | ❌ **不检查** |
| `is_file` | ✅ `:70` | ✅ `:128` | ✅（`is_dir` 分支 `:159`） |

### 3.3 不对称的实害

1. **名字与字节数泄漏**（`_files.py:163-166`）：`list_safe_directory` 对子条目不查 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES`，所以 root 下的 `credentials`、`credentials.json`、`id_rsa`、`id_ed25519`、以及任意 `*.pem`/`*.key`/`*.pfx`/`*.p12` 都会以 `[file] 名字 (12,345 bytes)` 的形式进入 LLM 上下文。**泄漏集合不含** `.env`/`.netrc`/`.pgpass`/`.htpasswd`（被 `:153` 的 dot-prefix 过滤拦住）。
2. **blocklist 只按精确文件名匹配**（`resolved.name.lower()` `:65`）：`.env.production`、`.env.prod`、`.envrc`、`env`、`secret.pem.txt` 全部不在名单内（但 `.pem` 后缀的会被后缀检查拦住）。
3. **`_BLOCKED_SUFFIXES` 用 `Path.suffix` 比对**（`:65`）：`foo.env.local` 的 `.suffix` 是 `.local`，多点文件名的后缀检查可以整体绕过——这是 §6-1 env 工具失效的同一根因。
4. **`list_safe_directory` 的 `hidden_count` 会回传给模型**（`:179-180`）——泄漏隐藏文件**数量**（不泄漏名字）。
5. **`list_safe_directory` 把 `OSError` 吞成字符串**（`:146-148`），调用方无法区分"访问被拒"与"IO 失败"。
6. **会话级路径解析的双候选**：`_resolve_session_source`（`:567-572`）对相对路径试 `session_root/candidate` 与 `file_root/candidate`，但两者都必须 resolve 进 `session_root`（`:580-581`）才通过——所以 `file_root` 候选只是给"上传笔记里的相对路径"一个兜底，不构成逃逸。
7. **`_prepare_session_directory` 的 `mkdir(parents=True, exist_ok=True)`**（`:777`）：在 `file_root` 内任意深度建目录；前置 `:774` 已保证在 root 内。
8. **写入路径**：所有 writer 都先 `touch(exist_ok=False)` 独占预留（`:816`、`:903`），再写 temp、再 `os.replace`，源文件被 `:900` 显式排除。全库**没有**任何 `open(..., "w")` 直写源文件的路径。

---

## 4. 🧮 数值与阈值清单

| 阈值 | 值 | 位置 |
|---|---|---|
| `file_read_max_bytes`（配置默认） | 5_000_000 B（5 MB） | `src/config/settings.py:153` |
| `file_read_root` 默认 | `"."` | `src/config/settings.py:152` |
| `file_read_enabled` 默认 | `False` | `src/config/settings.py:151` |
| `_INSPECT_MAX_CHARS` / `_MAX_CHARS_READER` | 20_000 | `_source_edit.py:40`、`:45` |
| `_MAX_OPERATIONS` | 100 | `_source_edit.py:41` |
| `_MAX_LINE_CHARS` | 50_000 | `_source_edit.py:42` |
| `_MAX_OUTPUT_VARIANTS` | 100 | `_source_edit.py:43` |
| inspector / reader `max_chars` schema 上限 | 200_000 | `_source_edit.py:162-167`、`:986-991` |
| `filename` / `output_name` 长度 | 1–200 | `_source_edit.py:283-292`、`:330-337` |
| stem 截断 | 150 字符 | `_source_edit.py:798`、`:921` |
| `_THREAD_ID_PATTERN` | 1–64 字符 `[A-Za-z0-9._-]` | `_source_edit.py:47` |
| CSV `_MAX_ROWS` | 20_000 | `csv_edit.py:48` |
| CSV `_MAX_COLUMNS` | 256 | `csv_edit.py:49` |
| CSV `_MAX_CELL_CHARS` | 10_000 | `csv_edit.py:50` |
| CSV delimiter sniff 样本 | 4096 字符 | `csv_edit.py:672` |
| JSON/YAML/TOML/XML `_MAX_VALUE_CHARS` | 100_000 | `json_edit.py:48`、`yaml_edit.py:52`、`toml_edit.py:47`、`xml_edit.py:52` |
| 路径式 `_MAX_INSPECT_ENTRIES` | 1_000 | `json_edit.py:49` 等 |
| 路径式 `_MAX_SCALAR_CHARS` | 200 | `json_edit.py:50` 等 |
| JSON `path` 字段 | 500 | `json_edit.py:86` |
| YAML dump `width` | 4096 | `yaml_edit.py:601` |
| `_file_cache` max_entries | 128（digests 256） | `_file_cache.py:26`、`:85` |
| SHA-256 读块 | 1 MiB | `_file_cache.py:74` |
| stripe 锁数 | 32 | `_file_cache.py:31` |
| `list_files` 默认/上限 | 500 / 2000 | `list_files.py:15`、`:31` |
| zip `_MAX_LISTED_ENTRIES` | 500 | `zip_file.py:28` |
| zip `_BINARY_SNIFF_BYTES` | 8_192 | `zip_file.py:30` |
| excel `_MAX_OPERATIONS` | 50（**其余家族 100**） | `excel_edit.py:45` |
| excel `_MAX_INSPECT_ROWS/COLS` | 200 / 30 | `excel_edit.py:47-48` |
| excel `_MAX_ZIP_ENTRIES` | 1024 | `excel_edit.py:52` |
| excel `_MAX_UNCOMPRESSED_BYTES` | 100_000_000 | `excel_edit.py:53` |
| excel `_MAX_COMPRESSION_RATIO` | 200 | `excel_edit.py:54` |
| excel `_MAX_OUTPUT_STEM_CHARS` / `_MAX_PORTABLE_PATH_CHARS` | 150 / 240 | `excel_edit.py:55-56` |
| excel `_MAX_FORMULA_CHARS` / `_MAX_CELL_CHARS` | 4_000 / 50_000 | `excel_edit.py:49-50` |
| excel `_MAX_STRUCTURAL_COUNT`（插/删行列数） | 1_000 | `excel_edit.py:51` |
| excel create `_MAX_SHEETS`/`_MAX_ROWS`/`_MAX_COLUMNS`/`_MAX_FORMULAS` | 20 / 10_000 / 100 / 2_000 | `excel_create.py:45-48` |
| excel create 列宽 | 0–255 | `excel_create.py:128-130` |
| excel create 子进程超时 | **120 秒** | `excel_create.py:242` |
| excel mjs inspect 限制 | rows 1000 / cols 100 / cell 200 / maxChars 50000 / maxResults 300 | `excel_create.mjs:34-48` |
| 重试 | 无（全库无 retry 循环）；唯一例外是 `_file_cache._content_digest` 的 2 次 digest 重试（`_file_cache.py:63`） | |
| 超时 | 仅 `excel_create.py:242` 一处；所有文件 IO 无超时 | |

---

## 5. 💉 公式注入覆盖矩阵

grep 全库 `startswith("=")` / `startswith("|")` / `HYPERLINK|WEBSERVICE|WEBAPI|IMPORTXML|DDE` 的实际命中**只有 `excel_edit.py` 和 `excel_create.py` 两处**：

| 工具 / 入口 | 拦 `=` | 拦 `|`（DDE） | 载荷落地方式 | 结论 |
|---|---|---|---|---|
| `excel_edit.set_cell`（`:177-181`） | ✅ 拒绝 `=` 前缀 | ❌ 不拦 | `cell.value = operation.value`（`:424`） | 安全（`|` 无前导 `=` 时 openpyxl 存为字符串） |
| `excel_edit.set_formula`（`:184-193`） | ✅ 必须 `=` 开头 | ✅ **`:192-193` 拒绝任何 `\|`** | `cell.value = operation.formula`（`:426`） | 有守卫，注释 `:189-191` 与代码一致 |
| **`excel_create.formulas`**（`:64-73`） | ✅ 必须 `=` 开头（`:68`） | ❌ **完全不拦** | `.formulas = [[formula.formula]]`（`excel_create.mjs:67`） | ❌ **DDE 可达** |
| **`excel_create.rows`（cell values）**（`:111-116`） | ❌ 不拦 | ❌ 不拦 | `.values = sheetSpec.rows`（`excel_create.mjs:64`） | 存为字符串，惰性（见 §6-5） |
| **`csv_edit`（create/inspect/edit 全路径）** | ❌ **无任何守卫** | ❌ 无 | `csv.writer` 原样写出（`csv_edit.py:827-837`） | ❌ **最实害** |
| `ini/properties/env/toml/yaml/json/xml` | 无此语义 | — | 键值对/文本 | n/a |
| text/markdown/html/log/sql/shell | 无 | — | 纯文本 | n/a |

**三个明确分歧：**

1. **`csv_edit.py` 零守卫**——全文 1041 行 grep 不到任何 `startswith`。`update_cell`（`new_value` `max_length=10_000`）、`append_row`/`replace_row`/`insert_before_row`（`new_row` 元素 `max_length=10_000`）都能写 `=HYPERLINK("https://evil","click")`、`=WEBSERVICE(...)`、`=WEBAPI(...)`、`=IMPORTXML(...)`。CSV **没有字符串/公式的类型区分**，Excel 打开即以公式解析（单元格会被 csv.writer 引号包裹，Excel 仍按公式解析引号字段内容）。这是本区最高危缺口。
2. **`excel_create.py` 允许 DDE**——`SpreadsheetFormula.validate_formula`（`:64-73`）只查 `=` 前缀与控制字符，`excel_edit.py:192-193` 有 `|` 拦截而 `excel_create.py` 没有。载荷 `=cmd|'/c calc'!A1` 会被 `excel_create.mjs:67` 逐字写进工作簿。
3. **`excel_create.validate_sheet`（`:111-116`）只查长度和控制字符**，接受 `=`/`|` 前缀的 cell 值；但这些值走 `.values`（`mjs:64`）落成字符串，**不构成公式注入**。所以之前记录的"`validate_sheet` 分歧"是**校验层不对称**，不是可执行载荷——需要修正。

补强证据（同一份 mjs 里的两道后检）：`scanFormulaErrors`（`excel_create.mjs:44-54`）只查计算错误（`#REF!` 等），**不查 DDE/外部函数**；`inspectWorkbook` 用 `include:"values,formulas"`（`:33`）也只是观测。

---

## 6. 发现与矛盾

**1. `.env` 工具是自我矛盾的——声明能读三种文件，实际一种都读不到**
`env_file.py:14` / `env_edit.py:63` 声明 `(".env", ".env.local", ".env.example")`，但：
- `_files.py:15-17` 把精确文件名 `.env` 和 `.env.local` 放进 `_BLOCKED_NAMES`，`:65` 直接拒绝；
- `Path("x.env.local").suffix == ".local"`（已用 Python 探针验证），而 `_files.py:73-77` 拿 `resolved.suffix.lower()` 去比 `expected_suffixes` → `.env.local`/`.env.example` 报 "unsupported file type"。
结论：`read_env_file` 只能读**任意名字但后缀恰好是 `.env`** 的文件（如 `foo.env`），三种被宣传的扩展名全部不可用。工具描述（`env_file.py:60-66`）说"包含 secrets"，而真正可能含 secrets 的 `.env` 被 blocklist 拦死。

**2. 多点后缀 create 成功、edit 必失败**
`_source_edit.py:791-797` 的 `_creation_stem` 用 `endswith(option)` 匹配，所以可以产出 `secrets.env.local`（不在 `_BLOCKED_NAMES` 精确名单里，创建合法）；但随后 `_resolve_session_source`（`:584-589`）把同一组 suffixes 传给 `resolve_safe_path`，`:73-77` 按 `Path.suffix`（`.local`）比较 → 拒绝。**创建出来的文件永远无法被 edit/inspect。**

**3. `csv_edit.py` 完全没有公式注入守卫**（见 §5-1）。对比 `excel_edit.py:178-180` 有 `=` 拦截。

**4. `excel_create.py` 缺 DDE 守卫，`excel_edit.py` 有**（见 §5-2）。

**5. 之前记录关于 `excel_create.validate_sheet` 的结论需要修正**：`:111-116` 确实接受 `=`/`|` 前缀值，但 `excel_create.mjs:64` 用 `.values` 写入 → 存为字符串，非可执行公式。属于**校验不对称**，不是漏洞。

**6. `text_edit.py`（841 行）与 `markdown_edit.py`（849 行）是 `_source_edit.py` 的整份拷贝，而非薄适配器**；`html/log/sql/shell/jsonl` 都是 4 行配置的适配器。两套实现都被注册（`providers.py:251`、`:254`、`:297`），所以是**重复实现**不是死代码，且已产生漂移：`text_edit.py:684-685`/`markdown_edit.py:690-691` 硬编码单后缀，而 `_source_edit.py:791-797` 遍历 `config.suffixes`。

**7. `_source_edit.py` 的 `__all__` 位于文件中部**（`:955-971`），其中 `build_source_file_tool`、`read_source_file`、`SourceFileInput` 定义在其后（`:974-1064`）。合法但不寻常，容易误判为缺失。

**8. `json_edit.py:600` 在 `_publish_edited` 内部重复 import 了 `_write_source_atomically`**，而模块顶部 `:38` 已导入同一符号——冗余死行。

**9. zip 读取器信任中央目录头**：`zip_file.py:230` 用 `info.file_size > max_bytes` 判断 zip bomb，但 `info.file_size` 是**声明值**，被篡改/谎言头可绕过；且只有**单条目**限制，没有总量、没有压缩比检查。对比 `excel_edit.py:575-583` 同时查压缩比（`:577-581`）与总量（`:582-583`）——同一仓库内两种严格度。

**10. `list_files.py:56-57` 的工具描述说 "Hidden entries and symbolic links are not shown"** ——对 dotfile/symlink 属实（`_files.py:153`、`:157`），但**没有**提到 `_BLOCKED_NAMES`/`_BLOCKED_SUFFIXES` 类文件会正常列出，容易造成"已脱敏"的错觉。

**11. 路径式编辑器的"批预校验"与"应用时复校验"是两层，行为与 CSV 不同（⚠️ 不是安全洞）**
`json_edit.py:488-497` 先对**原始**文档逐个 `_check_operation` 收集聚合错误；`:499-501` 再逐个 `_apply_operation`，而 `_apply_operation`（`:732-737`）**会对已被前面 op 改过的文档再校验一次**。所以"两个 `append_to_array` 都带 `expected_length=N`"这种重复目标**会被第二次的复校验拦下**（数组此时已是 N+1）——安全性成立。
差异在于错误形态：CSV 的重复目标是**批量聚合报错**（`csv_edit.py:719-731`，"no changes were made because ... operation(s) did not match"）；json/yaml/toml/xml 的同类冲突走的是**应用期单条报错**（`_apply_operation` 抛 `JsonEditError`），因此模型的错误提示形态不同、且前 N-1 个 op 已经在内存对象上生效（但没有任何落盘，因为 `_publish_edited` 在全部 op 之后才执行）。yaml（`yaml_edit.py:767-772`）、toml（`toml_edit.py:743`）、xml 结构同构。

**12. `xml_edit.py` 静默丢失结构**：`dump_xml` 用 `xml_declaration=False`（`:880`）→ **`<?xml version="1.0"?>` 声明被删**；`ET.fromstring`（`:576`）默认丢弃注释与 PI；`element_to_dict` 只读 `element.text` 且 `.strip()`（`:805`）→ 元素内前后空白被压掉，`element.tail` 完全丢弃。这些都不在该模块 docstring（`:1-8`）里。

**13. `xml_edit.py:576` 用裸 `ET.fromstring`**：无 defusedxml、无 DTD/entity 白名单。实际可用性依赖 Python 3.8+ expat 内置的 billion-laughs 保护（`SetBillionLaughsAttackProtectionMaximumAmplification`），外部实体 ET 默认不解析——**未确认**具体 expat 阈值，仅确认代码层无守卫。

**14. 读取器与编辑器的解码策略不一致**：读取器走 `read_text(encoding="utf-8", errors="replace")`（如 `text_file.py:92`）→ 非法 UTF-8 被静默替换；编辑器走 `decode_source`（`_source_edit.py:630-635`）→ `errors="strict"` 直接拒绝。**同一文件读得到、改不了**，且读者看到的是已被替换污染的文本。

**15. `_file_cache` 可能返回陈旧内容**：`_content_digest`（`:66-70`）在 `(size, mtime_ns)` 命中时不重算 SHA-256。同 size+同 mtime 但内容不同（例如某些备份工具保留 mtime）→ 返回旧内容。

**16. 后缀白名单重叠**：`text_file.py:14` 的 `(".txt",".md",".log",".csv")` 与 `markdown_file.py:14`、`log_file.py`、`csv_edit` 的后缀重叠，`.md`/`.log`/`.csv` 各有两个入口。仅冗余，不冲突。

**17. 空 YAML 文件无法填充**：`yaml_edit.py:588` 空文档解析为 `None`，而 `_check_fields`（`:146-147`）禁止对 root 用 `expected_missing=True` → 空 YAML 只能通过 `set_value` + `expected_value='"null"'`（依赖 `None == json.loads("null")`）间接创建。边界怪异但可达。

**18. `excel_create.py:331-334` 在 Windows 上用 `cmd.exe /c mklink /J`** 拼接路径：`subprocess.run(["cmd.exe","/c","mklink","/J",str(link),str(modules)])`。cmd.exe 会把后续 argv 重新拼成命令行再解析，含 `&`/`|`/`^`/重定向符的 `EXCEL_NODE_MODULES_PATH` 会被 cmd 解释。`modules` 来自管理员配置、`link` 由 `_THREAD_ID_PATTERN` 校验过的 thread_id 派生，所以攻击面限于被污染的配置文件——低危，但确实是 shell 拼接。

**19. `_MAX_OPERATIONS` 不一致**：Excel 是 50（`excel_edit.py:45`），其余家族统一 100。

**20. 无重试、无超时**：全部分区代码没有重试循环（唯一例外 `_file_cache.py:63` 的 2 次 digest 重试）；除 `excel_create.py:242` 的 120s 子进程超时外，所有文件 IO 都无超时。

**21. 测试覆盖空洞（grep 验证）**
- `tests/test_list_files_tool.py` 是**唯一**触及 `list_files`/`list_safe_directory` 的测试文件，其中 grep `credentials` / `BLOCKED` / `.pem` / `.key` / `id_rsa` **全部 0 命中**——即 §3.3-1 的名字+字节数泄漏**没有任何测试断言**。
- DDE 守卫测试只存在于 `tests/test_excel_edit.py:359`（标注 `# DDE blocked`）。`tests/test_excel_create.py` 与 `tests/test_csv_edit.py` 中 grep `DDE`/`HYPERLINK`/`WEBSERVICE` **0 命中** → §5-1 与 §5-2 两个缺口均无测试。
- **完全没有 env 相关测试**：`tests/` 下 grep `env` 找不到任何 `test_env_*` 文件 → §6-1 的 `.env` 自我矛盾未被发现。

---

## 7. 未读到的文件

**分区内未逐行读取**（但已做结构级 grep 扫描，函数/raise/startswith 全覆盖，故上述结论仍成立）：
- `toml_edit.py`（888 行）、`ini_edit.py`（786 行）、`properties_edit.py`（686 行）、`env_edit.py`（686 行）—— 逐行细节未读，仅 config 块原文 + 结构扫描
- `ini_file.py`、`properties_file.py` — 仅 grep `_SUFFIXES`/工具名（结构与 `env_file.py`/`toml_file.py` 逐行同构）
- `excel_file.py` — 完全未读（不在本区清单）

**分区外（其他子代理负责）**：`_http.py`、`_amap.py`、`_geocoding.py`、`math_tool.py`、`linalg_tool.py`、`statistics_tool.py`、`number_theory_tool.py`、`memory_tool.py`、`word_*`、`pdf_*`、`powerpoint_*`、`excel_file.py`、以及所有编程语言族 `c_/go_/java_/javascript_/rust_/python_/...` 的 edit/file 文件。

**grep 覆盖到的事实**（范围已标注）：
- `realpath` 在 `src/` 下 **0 命中**；`is_relative_to` 在 `src/backend/` 下 **0 命中** → 全部边界判定都靠 `Path.relative_to`（`_files.py:88`、`_source_edit.py:949`）。
- `difflib` / `unified_diff` / `apply_patch` / `fuzzy` 在 `src/backend/tools/` 下 **0 命中** → 全部分区没有补丁式/模糊匹配编辑。
- `hunk` / `offset` 在 `_source_edit.py`、`text_edit.py`、`markdown_edit.py`、`csv_edit.py` 中 **0 命中**。
- `symlink` 仅 `_files.py:157`（读侧跳过）与 `excel_create.py:331`（写侧 `symlink_to`）。
- 公式注入关键词（`=HYPERLINK`/`=WEBSERVICE`/`WEBAPI`/`IMPORTXML`/DDE/`startswith("=")`/`startswith("|")`）在 `src/backend/` 下仅命中 `excel_edit.py` 与 `excel_create.py`。
