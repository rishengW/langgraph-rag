# A2 / full-sweep-16 — 语言工具对 + 结构化编辑器审计

- 日期：2026-10-09
- 分支：production
- 范围：22 对语言/结构化编辑器（20 语言 + jsonl + typescript）+ 共享引擎 `_source_edit.py` + 注册总表 + 相关测试
- 纪律：全部结论以代码 file:line 佐证，未采信注释/docstring；实测用 `_probe_lang1.py` / `_probe_lang2.py`

---

## 0. 一句话总结

引擎本身是干净的（无 eval/subprocess、路径收敛完整、原子写完备），但存在一个**新发现的高价值缺陷**：`env_edit.py` 声明支持三种后缀 `.env` / `.env.local` / `.env.example`，然而 `resolve_safe_path` 只对**最后一节**扩展名做匹配，导致 `x.env.local` 与 `x.env.example` **能创建、读不回、改不回**（round-trip 断裂）；同时对 `.env` 与 `.env.local` **恰好**被 `_BLOCKED_NAMES` 挡住，所以"敏感"的反而安全，"不那么敏感"的 `.env.example` 反而被写死。另外旧记忆里的"20 个语言对字节一致"是错的（每一对都有 7 处字面量差异），`_source_edit.py:179` 的 `getattr` 也**不是**"静默空列表"而是 `AttributeError`。

---

## 1. 覆盖清单

### 1.1 实际读取的源码文件（行数按 `wc -l`）

| 文件 | 行数 | 类型 |
|---|---|---|
| `_source_edit.py` | 1064 | 共享引擎 |
| `_files.py` | 192 | 路径解析 |
| `c_edit.py` … `sql_edit.py`（21 对 + `jsonl_edit.py`） | 每对 47 + 33 或 48 + 34 | 薄适配器 |
| `typescript_edit.py` + `typescript_file.py` | 869 + 113 | 自研 |
| `text_edit.py` + `text_file.py` | 841 + 110 | 自研 |
| `markdown_edit.py` + `markdown_file.py` | 849 + 112 | 自研 |
| `yaml_edit.py` + `yaml_file.py` | 855 + 112 | 自研 |
| `toml_edit.py` + `toml_file.py` | 888 + 112 | 自研 |
| `xml_edit.py` + `xml_file.py` | 897 + 112 | 自研 |
| `json_edit.py` + `json_file.py` | 804 + 113 | 自研 |
| `ini_edit.py` + `ini_file.py` | 786 + 114 | 自研（本轮新增） |
| `env_edit.py` + `env_file.py` | 686 + 114 | 自研 |
| `properties_edit.py` + `properties_file.py` | 686 + 112 | 自研 |
| `jsonl_edit.py` | 65 | 薄适配器 + line_validator |
| `__init__.py` | 428 | 注册总表 |

### 1.2 20 语言对的 md5 比对（实测）

任务假设"19 个字节一致的 48 行适配器"。**实测不成立**：

- `md5sum *.py` 显示**每一个** `<lang>_edit.py` / `<lang>_file.py` 的 md5 都不同（21/22 各不相同）。
- 逐文件 diff（如 `diff c_edit.py go_edit.py`）证实：结构**逐行相同**，但每个文件都有**恰好 7 处字面量差异**：
  1. 模块 docstring 里的语言名（1 行）
  2. `X_CONFIG` 变量名
  3. `label=` 值
  4. `suffixes=` 元组
  5. `creation_suffix=`
  6. `tool_prefix=`
  7. `mime_type=`
  8. `flag_name=`
  9. `temp_prefix=`
  10. `build_X_edit_tools` 函数名
  11. `__all__` 两项
- 即：**结构逐字一致，字面量按语言替换**。所以"字节一致"应改为"结构一致，语言字面量差异"。

| 语言 | 后缀 | 创建后缀 |
|---|---|---|
| c | `.c`, `.h` | `.c` |
| go | `.go` | `.go` |
| groovy | `.groovy` | `.groovy` |
| haskell | `.hs` | `.hs` |
| html | `.html`, `.htm` | `.html` |
| java | `.java` | `.java` |
| javascript | `.js`, `.mjs`, `.cjs` | `.js` |
| julia | `.jl` | `.jl` |
| latex | `.tex` | `.tex` |
| log | `.log` | `.log` |
| lua | `.lua` | `.lua` |
| matlab | `.m` | `.m` |
| php | `.php` | `.php` |
| prolog | `.pl` | `.pl` |
| python | `.py` | `.py` |
| r | `.r` | `.r` |
| ruby | `.rb` | `.rb` |
| rust | `.rs` | `.rs` |
| shell | `.sh`, `.bash` | `.sh` |
| sql | `.sql` | `.sql` |
| swift | `.swift` | `.swift` |

（`jsonl_edit.py` 也是薄适配器，附加一个 `line_validator=_validate_json_line` 钩子，仓库里**唯一**在用的验证器，见 `_source_edit.py:535-545`。）

**结论**：仓库中 **22 对**（21 语言 + jsonl）是共享引擎的薄适配器；只有 **10 对**（typescript / text / markdown / yaml / toml / xml / json / ini / env / properties）是自研的完整实现。这**修正**了记忆"19 identical"的错误表述。

### 1.3 测试文件行数

`test_code_edit.py` 255 行、`test_typescript_edit.py` 529、`test_text_edit.py` 449、`test_markdown_edit.py` 449、`test_yaml_edit.py` 319、`test_json_edit.py` 265、`test_file_tools.py` 402、`test_list_files_tool.py` 130。

---

## 2. `_source_edit.py` 引擎的真实行为语义

### 2.1 公开操作

三个工具（`:236-274` 装配）：

| 工具 | 参数 | 语义 |
|---|---|---|
| `create_<prefix>_file` | `filename`, `content` | 新建 session 目录内文件，不覆盖；`_creation_stem` 强制后缀；`_reserve_created_output_path` 用 `touch(exist_ok=False)` 原子占名，最多枚举 100 个变体 |
| `inspect_<prefix>_file` | `path`, `max_chars`(1..200_000, default 20_000) | 只读，先 `_resolve_session_source` 再全量 `read_bytes` → `decode_source` → 逐行 JSON 转义 |
| `edit_<prefix>_file` | `path`, `operations[1..100]`, `output_name?` | 校验通过后写入新文件（不覆盖源），返回 artifact |

### 2.2 `SourceEditOperation` 的四种动作（`:101-149`）

- `replace_line`（需 `line_index` + `expected_text` + `new_text`）
- `delete_line`（需 `line_index` + `expected_text`）
- `insert_before_line`（需 `line_index` + `expected_text` + `new_text`）
- `append_line`（只 `new_text`）

Pydantic `model_validator` 强制单行（`:145-148` 拒 `\r`/`\n`）、`line_index ge=0`、`max_length=50_000` 字节。

### 2.3 参数校验（严格且完备）

- **线程**：`_THREAD_ID_PATTERN = ^[A-Za-z0-9._-]{1,64}$`（`:47`），`:181, :358` 双重校验。
- **文件路径**：`resolve_safe_path`（`_files.py:33-83`）强制 5 层：非空、根内、非黑名单、非黑名单后缀、大小≤`file_read_max_bytes`。
- **解码**（`decode_source:612-663`）：拒 UTF-16/32 BOM、NUL、C0 控制字符（除 `\t\r\n`）、DEL 0x7F、0xFFFE/0xFFFF、裸 CR、CRLF/LF 混用。**读侧不经过这里**（见 §4 编码问题）。
- **锚点**：`line_index` 越界（`:696-701`）、重复定位（`:692-694`）、`expected_text` 不精确匹配（`:702-703`）都收集后**整批 raise**。
- **批量大小**：`_MAX_OPERATIONS=100`（`:41`, `:323`）、`_MAX_OUTPUT_VARIANTS=100`（`:43`, `:812, :897`）。

### 2.4 原子性

`_write_source_atomically:828-878`：
1. `mkstemp(prefix=temp_prefix, suffix="<lang>.tmp", dir=target.parent)` — 同目录临时文件
2. `write → flush → os.fsync(fd)` — 只 fsync 文件，**不 fsync 父目录**（sweep-15 已记录）
3. 回读 `validated != document` → 回滚
4. `os.replace(temp, target)` — POSIX 原子
5. 异常时清 target + temp

**无 chmod**：`mkstemp` 建 0600，产物保留 0600（sweep-15 已记录，Windows 无感）。

### 2.5 幂等性

**是**：所有失败路径都**不产生任何修改**（`:705-710` 先全部校验再应用；`:866-877` 回滚 target 与 temp）。成功路径**从不修改源文件**，只新建 `.edited<suffix>`。

### 2.6 错误模型

- `create_source_file` / `edit_source_file` 捕获 `(OSError, UnicodeError, SourceEditError)` 返回**字符串内容**（不抛），`artifact=None`
- `inspect_source_file` / `read_source_file` 返回字符串内容（不抛）
- 所有对外工具**永不抛异常**——错误都进入 tool response 的 `.content`

### 2.7 关键脚枪（复核并修正）

**`:179` 门控的真实行为**：
```python
if not (settings.file_read_enabled and getattr(settings, config.flag_name)):
    return []
```

Python 的 `getattr` 无默认值**会抛 `AttributeError`**。实测验证（`Settings(dashscope_api_key='dummy')` + `getattr(s, 'typo')`）：抛 `AttributeError: 'Settings' object has no attribute 'totally_made_up_flag'`。

所以真实行为是**双模态**：
- `file_read_enabled=False`（默认）→ 短路，`getattr` 不执行 → 静默返回 `[]`（与合法的"未启用"无法区分，无告警）
- `file_read_enabled=True` + flag 名错拼 → **抛 AttributeError**，向上冒泡到 `providers.py:288-332`，可能挂掉整个工具构建

sweep-15 记录的"拼错 flag 静默空列表"**只在 `file_read_enabled=False` 时成立**；开启文件读取后的错拼是**响亮的**（会崩）。

**21 个薄适配器共享这个行为**（c/go/groovy/haskell/html/java/javascript/julia/latex/log/lua/matlab/php/prolog/python/r/ruby/rust/shell/sql/swift 加 jsonl）。自研的 10 个编辑器**不用** `getattr`——例如 `env_edit.py:224` 直接写 `settings.env_edit_enabled`，`text_edit.py:202` 写 `settings.text_edit_enabled`。

---

## 3. 结构化编辑器的独有危险面

### 3.1 `.env` 编辑器（`env_edit.py`，本轮重点）

`ENV_CONFIG.suffixes = (".env", ".env.local", ".env.example")`（`:63`），`creation_suffix = ".env"`。

**新发现缺陷：round-trip 断裂**（`_probe_lang2.py` 实测确认）：

| 操作 | `prod.env` | `.env` | `x.env.local` | `x.env.example` |
|---|---|---|---|---|
| `create_env_file` | ✅ 成功 | ❌ "a usable .env filename is required"（`_creation_stem:798` 把 `""` 剥空） | ✅ 成功 | ✅ 成功 |
| `read_env_file` | ✅ 可读 | ❌ `resolve_safe_path` 拒 `.env`（`_files.py:65`） | ❌ "unsupported file type '.local'" | ❌ "unsupported file type '.example'" |
| `inspect_env_file` | ✅ | ❌ 同上 | ❌ 同上 | ❌ 同上 |
| `edit_env_file` | ✅ 可改 | ❌ 同上 | ❌ 同上 | ❌ 同上 |

根因：`resolve_safe_path:73-77` 用 `resolved.suffix.lower() not in expected_suffixes`，而 `Path("x.env.local").suffix` 返回 `.local`（**只取最后一段**），与 `expected_suffixes` 里的 `.env.local` 永不相等。

结果：
- 敏感文件名 `.env` / `.env.local` **碰巧**被 `_BLOCKED_NAMES` 挡住（`_files.py:15-26`），是安全冗余
- 名义上应支持读的 `x.env.local` / `x.env.example` **能建不能读**——LLM 拿到 `Created x.env.local` 却永远读不回来
- `prod.env` / `secrets.env` / `credentials.env` 都能读能改，但 `read_text_file` 只接受 `.txt/.md/.log/.csv`（`text_file.py:14`）会拒——**env 编辑器是这些文件唯一的读入口**（这修正了 MEMORY.md 的暗示，不是"env 编辑器被启用才读得到"，而是 `file_read_enabled=true` 就够，见 `providers.py:186-220`）

**修复方向**：要么把 `suffixes` 收敛为 `(".env",)`，要么把 `expected_suffixes` 匹配改成"任一段扩展名匹配"（即 `suffixes=("env", "env.local", "env.example")` 用自定义解析）。

### 3.2 YAML 编辑器

- 用 `yaml.safe_load_all`（`yaml_edit.py:580`）——**安全加载**，拒 `!!python/*` 标签（`:570-588`）
- 拒多文档流（`:583-587`）
- 序列化用 `yaml.safe_dump(..., sort_keys=False, default_flow_style=False)`（`:595-603`）——**保留键顺序**
- **注释全部丢失**：走 parse → dict → dump，注释不在 AST 里
- 完整走 `_source_edit` 的原子写与 session scope

### 3.3 TOML 编辑器

- 用 stdlib `tomllib.loads`（`toml_edit.py:568`）——**安全，仅读**
- 序列化走自己的 emitter（未走 tomllib，因为 stdlib 只读）
- 同 YAML 一样丢注释、保留键顺序

### 3.4 XML 编辑器

- 用 `xml.etree.ElementTree`（`xml_edit.py:20`）——**安全，不解析 DTD、不解析实体扩展**
- 用 `dict_to_element` / `element_to_dict` 双向转换（`__init__.py:153-156` 导出）
- 同名多元素通过 `[0]`/`[1]` 索引区分

### 3.5 JSON 编辑器

- `json.loads` / `json.dumps`（stdlib）——安全
- 用 JSON Pointer 风格路径 `$.users[0].name`
- 保留键顺序（Python dict 有序）
- 丢注释（JSON 无注释）

### 3.6 INI 编辑器（本轮新增）

- 自定义正则 `_SECTION_PATTERN` / `_LINE_PATTERN`（`ini_edit.py:42-46`）
- `suffixes=(".ini", ".cfg", ".conf")`（`:50`），全部单段扩展名，round-trip 完好
- 注释（`#` / `;`）与空行**保留**——因为 `set_value` 只重写单行（`:176-178` docstring 也写"formatting is preserved"，**代码与文档一致**）
- `expected_value` 精确匹配保证乐观锁

### 3.7 `.properties` 编辑器

- 结构类似 INI，但**无 section**
- 单后缀 `.properties`（`properties_edit.py:51`）
- 注释/空行保留

### 3.8 `.env` / INI / properties 共有的"值"注入面

三者都允许任意字符串作为 value。因为不经过 shell，没有命令注入。但注意 `_MAX_VALUE_CHARS = 100_000`（`env_edit.py:49`, `ini_edit.py:36`, `properties_edit.py:37`）——比行级 `_MAX_LINE_CHARS=50_000` 大一倍。这是设计意图（值可能含 base64 之类长串）。

### 3.9 全部结构化编辑器都共享的护栏

- 全走 `_source_edit` 的 `decode_source`（strict UTF-8、拒 NUL/控制字符/BOM-16/32、拒裸 CR 与混合换行）
- 全走 `_source_edit` 的 `_write_source_atomically`
- 全走 `_source_edit` 的 `_resolve_session_source`（session scope）
- 全部 session-scoped（`_THREAD_ID_PATTERN` 校验 thread_id）
- 全部 copy-on-write（不修改上传的源文件）
- **读侧仍用 `errors="replace"`**（见 §4）

### 3.10 会话作用域（无泄漏）

grep 全仓 `session_root is None` 检查——所有自研编辑器都在 `build_X_edit_tools` 里显式检查（`text_edit.py:204`、`markdown_edit.py:205`、`typescript_edit.py:208`、`json_edit.py:260`、`yaml_edit.py:265`、`toml_edit.py:261`、`xml_edit.py:266`、`env_edit.py:226`、`properties_edit.py:215`、`ini_edit.py:222`、`csv_edit.py:264`、`excel_edit.py:252`、`excel_create.py:161`、`word_edit.py:351`、`powerpoint_edit.py:121`）；所有薄适配器由 `_source_edit.py:181` 检查。**没有**遗漏（这修正了 MEMORY.md 里 `pdf_edit.py:130-141` 是唯一遗漏的记录——pdf 属于 office 家族，本轮不复审但保留结论）。

### 3.11 路径逃逸复核

- 相对路径：`_resolve_session_source:568-572` 尝试 `session_root/candidate` 与 `file_root/candidate`，然后 `_is_within(resolved, session_resolved)` 二次收敛
- 绝对路径：只试一次，同样收敛
- 符号链接：`resolve()` 跟随，`_is_within` 拒外部
- 写出侧：`_reserve_output_path:900` 拒覆盖源文件；`_prepare_session_directory:774-775` 禁 session_root == file_root
- `thread_id`：`_THREAD_ID_PATTERN` 拒 `/`；`".."` 虽能通过正则但被 `session_upload_dir` 的 `sanitize_filename` 消空回退（sweep-15 已复核）

**结论**：路径逃逸链路闭环。

---

## 4. 读侧 `errors="replace"` / 写侧 `errors="strict"`（复核，仍然成立）

实测 grep 11 处 `errors="replace"`：
- `_source_edit.py:1053`（`read_source_file`）
- `text_file.py:92`
- `markdown_file.py:94`
- `typescript_file.py:95`
- `json_file.py:95`
- `yaml_file.py:94`
- `toml_file.py:94`
- `xml_file.py:94`
- `env_file.py:96`
- `ini_file.py:96`
- `properties_file.py:94`

写/编辑侧全部经 `decode_source`（`_source_edit.py:631`）用 `errors="strict"`，且额外拒 NUL/BOM-16-32/控制字符。

**行为差**：含非法 UTF-8 字节或 UTF-16 BOM 的文件 → `read_X_file` 返回满屏 `U+FFFD` 替换字符（无错误提示），而 `edit_X_file` / `inspect_X_file` 直接拒绝。模型基于损坏内容生成错误产物。11 个 reader 共享这个坑。

---

## 5. 工具注册表真相

### 5.1 `__init__.py:246` 与 `__all__` 的幽灵导出

**实测确认**：`__init__.py:424-425` 的 `__all__` 里包含 `"read_zip_entry"` 与 `"inspect_zip_file"`，但 `:246` 只从 `.zip_file` 导入 `ZipEntryInput, ZipFileInput, build_zip_tools`。

实测：
```
> from src.backend.tools import *
AttributeError: module 'src.backend.tools' has no attribute 'read_zip_entry'
```

**修正**：记忆写的是"ImportError"，实测是 **`AttributeError`**（Python 3.12+ 对 `__all__` 里的未定义名字直接抛 AttributeError，不是 ImportError）。`import src.backend.tools`（不带 *）仍能成功，177/179 项能访问。

### 5.2 当前配置下实际生效的工具

- 默认所有 `*_edit_enabled=False`（`config/default.yaml:101-177`、`.env.example:231-318`）
- 唯一开启的开关：`.env.example:233 FILE_READ_ENABLED=true`（**唯一默认 true 的**），但 `FILE_READ_ROOT=.`（等于仓库根！）
- 生产 yaml（`config/production.yaml`）本轮 grep 未查到任何 `*_edit_enabled` 覆盖——与 MEMORY.md 记录的"production.yaml 净零加固"一致
- 所以**默认部署下 33 个编辑器全关**，只有 `read_text_file` / `read_md_file` 等 31 个读工具在 `FILE_READ_ENABLED=true` 下活着
- 但 `providers.py:186` 只按 `file_read_enabled` 门控 31 个读工具（包括 `read_env_file`），意味着默认配置下 `.env` 后缀文件**在 22 种语言里唯一的路径可读**是 env 编辑器（尽管 env_edit 是关的，read_env_file 是开的）——这个"读得到 env 但改不了 env"的错位在默认配置下就已经存在

### 5.3 flag 存在性复核

37 个 `*_edit_enabled` 字段在 `settings.py:157-242` 全部定义；37 个对应环境变量在 `.env.example:241-318` 全部列出；37 个在 `config/default.yaml:101-177` 全部默认 false。**无 flag 缺失**，无错拼。

---

## 6. 测试盲区

### 6.1 `test_code_edit.py:27-64` 的 `LANGUAGES` 表缺 5 对

表内 16 对：r, rust, go, sql, php, ruby, latex, prolog, haskell, lua, julia, shell, matlab, groovy, swift, log
**缺**：c, java, javascript, python, **html**

html 是新发现遗漏——sweep-15 的清单里没有 html 也漏掉了。**5 对零覆盖**。

### 6.2 `_file.py` 读工具全部零覆盖

`build_c_file_tool` / `build_go_file_tool` / … 22 个读工具**零测试**。

### 6.3 结构化编辑器测试缺口

- **零测试**覆盖 `.env` 的 round-trip 断裂（`x.env.local` / `x.env.example` 能建不能读）
- **零测试**覆盖 `env_edit_enabled=false` 时读工具的行为
- **零测试**覆盖 `yaml_edit` 的注释丢失
- **零测试**覆盖 `env_edit` 的 `_LINE_PATTERN` 对 `#` 内联注释的处理（`:55-58` 明确写了"inline comments only count when unquoted"，但代码里没看到相应逻辑——需进一步验证）
- **零测试**覆盖多文档 YAML 拒绝（`:583-587`）

### 6.4 环境依赖式静默跳过（本批无）

grep `skip|xfail|importorskip|skipif|pytestmark` 命中 0 处。**本批 8 个测试文件都是硬跑**，没有静默跳过。
- 对比：`test_excel_create.py:24` 有 `pytest.skip("artifact-tool Node runtime is unavailable")`，但那是 excel_create，不在本批范围

---

## 7. 对旧报告的校正清单

| 旧结论 | 校正 |
|---|---|
| **13/14 轮**："20 语言对是薄适配器，无独立漏洞" | ✅ 保留（结论对，但漏了 html 也是薄适配器，应为 21+jsonl） |
| **15 a3**："`_source_edit.py:179` 拼错 flag 名会静默返回空工具列表" | ❌ **部分错**。只当 `file_read_enabled=False` 时短路才静默；开启时 `getattr` 抛 `AttributeError` 向上冒泡 |
| **15 a3 发现 5**："22 处 `X_CONFIG` 重复定义" | ✅ 修正为 22 处（含 swift 与 html）|
| **15 a3 发现 3**：`text_edit.py` 是 841 行完整重复 | ✅ 仍然成立 |
| **15 a4**："10 个 `_edit.py` 全部零内容过滤，只有 jsonl 有 `line_validator`" | ✅ 仍然成立（`_source_edit.py:535-545` + 全仓 grep `line_validator` 只有 `jsonl_edit.py:44` 一个用户）|
| **15 a4**："typescript 是未迁移的引擎孪生副本" | ✅ 仍然成立 |
| **MEMORY.md**："19 个字节一致的 48 行适配器" | ❌ **错**。**每一对 md5 都不同**，因为每对都有恰好 11 处语言字面量差异。修正为"22 对结构逐行一致、字面量替换" |
| **MEMORY.md**："`tools/__init__.py:424-425` 幽灵导出……`from src.backend.tools import *` 会 ImportError" | ⚠️ **部分对**：确实是幽灵导出，但实测抛的是 **`AttributeError`** 不是 `ImportError` |
| **MEMORY.md**："`file_read_root=.` 时 MEMORY.md/SOUL.md 等可读" | ⚠️ 需补充：这不是因为 `env_edit` 开了，是因为**默认 `file_read_enabled=true`** 就足以让 31 个读工具全部构造，其中任一都能读到后缀在 `_SUFFIXES` 白名单里的文件（如 `.md` 就是 SOUL/USER/MEMORY 的后缀） |
| **15 a4 发现 4**："log inspect 先全量构造再截断，内存放大数十倍" | ✅ 仍然成立（`_source_edit.py:438-452` 全量 `json.dumps` → `"\n".join` → 再 `body[:limit]` 截断） |
| **15 a3 发现 6**："写出路径不校验敏感文件名" | ✅ 仍然成立；且本轮补充发现：`.env` 恰好被 `_creation_stem` 的 `strip(".")` 空校验挡住，但 `prod.env` 依然可写可读 |

---

## 8. 无法判定的问题

1. **`env_edit.py` 的 `_LINE_PATTERN` 是否真的处理内联注释**：`:55-58` 的注释写"inline comments only count when unquoted"，但 grep 不到相关实现（可能是靠"value up to end of line"的自然解析，也可能没实现）——需要单独构造 `"FOO=bar # comment"` 与 `"FOO=\"bar # cmt\""` 两种 case 做实测才能定论
2. **`xml_edit.py` 是否真的防 XXE**：只用 `ElementTree`，默认不启用外部实体解析（`xml.etree.ElementTree` 本身不解析外部实体，且 stdlib 已禁用），静态看是安全的，但没有测试断言
3. **`yaml_edit.py` 的 `safe_dump` 对含非 ASCII 字符的行为**：`allow_unicode=True` 保留原字符；但含 C0 控制字符以外的 Unicode（如 C1、BOM）的行为需要实测
4. **`json_edit.py` 的浮点数精度**：Python 的 `json.dumps` 用 `repr` 保留精度，但反序列化再序列化可能因 32 位 float 转换丢精度——无测试

---

## 9. 优先级建议

1. **P1**：修 env 编辑器的 round-trip 断裂（`x.env.local` / `x.env.example` 能建不能读）
2. **P1**：补 c/java/javascript/python/html 五对到 `test_code_edit.py:27-64`
3. **P2**：修 `__init__.py:424-425` 幽灵导出（删掉或补上 `read_zip_entry` / `inspect_zip_file` 导入）
4. **P2**：给 22 个读工具各补一条"后缀拒绝 + 越界拒绝"用例
5. **P2**：加测试锁定 `.env` round-trip（`.env` / `prod.env` / `x.env.local` / `x.env.example` 各一次）
6. **P3**：读侧统一用 `decode_source` 严格解码，消除 11 处 `errors="replace"`
7. **P3**：把 `_source_edit.py:179` 的 `getattr` 换成 `hasattr + settings.<flag>` 的直接访问（与 `text_edit.py:202` 一致），或者加显式默认值 `getattr(settings, flag_name, False)`
8. **P3**：给 `_source_edit.py:438-452` 的 inspect 加流式化（内存峰值 5MB → 20KB）
9. **P4**：合并 22 处 `X_CONFIG` 到单一来源表（可维护性）

---

## 附录 A：md5 实测（21 对 `<lang>_edit.py`）

```
0a3aeaa44a5b22291436e3fed5b160d7  java_edit.py
173d1c939fead30e3a1c80b31f7e0fc5  c_edit.py
1d87c0579048e4ed2ee59c71cdc8c417  prolog_edit.py
253b65f27424735c4cf38e540df05edf  ruby_edit.py
34df563a6eb2a082a3b36adf1eabf8e7  r_edit.py
5eeb6694b94eed0061dbabd9681600bd  shell_edit.py
68bc6d45b1e78eca11314c62d7c17bda  lua_edit.py
7940c0a9a3dd6a204dbba64f251ca9c7  swift_edit.py
7967c80b0a42893e50af000aaaba11b3  javascript_edit.py
941a260b01102197aa06117726dc9fc2  haskell_edit.py
9a4bd87041f4f51645e19c7d362be15b  go_edit.py
9c0d402dc34708c783712b40a6009084  groovy_edit.py
9c7ab7c8c9ed34fe1f559513ed0008e2  html_edit.py
a26ea6962de805a773e413a7ec6d65c3  julia_edit.py
aad2037b93e7a1c154b126599507be13  matlab_edit.py
c73e1dfa17f85d25a2e3242ede23836c  sql_edit.py
dd628fd3471464e6a67abb5d67f4b708  rust_edit.py
e7b90a21e8f4f5e3df419ea7551df44c  python_edit.py
f316169cab925285ace242c9e958bfbb  php_edit.py
f8961c34124bff7308c23c09175b0afd  log_edit.py
fb8379ec5970cc211e76d2afbedd499e  latex_edit.py
```

21 个文件 21 个不同 md5。**没有一个字节一致**。

---

## 附录 B：`_probe_lang1.py` 输出

```
CREATE result: Created prod.env in this chat session. The file is available to download.
file exists: True
READ result: Contents of prod.env (28 chars):

FOO=bar
DB_URL=postgres://x
INSPECT result: .env file prod.env:
Encoding: UTF-8
Newline: LF
Final newline: yes
KEYS:
FOO = bar
DB_URL = postgres://x
EDIT result: Applied 1 edit(s) (set_value) to prod.env. The original file is unchanged; the edited copy was saved as prod.edited.env and is available to download.
READ ../prod.env: Could not read .env file: access denied: '../prod.env' is outside the allowed directory.
```

## 附录 C：`_probe_lang2.py` 输出

```
CREATE .env: Could not create .env file: a usable .env filename is required.
READ .env: Could not read .env file: access denied: '.env' is a protected file.
INSPECT .env: Could not inspect .env file: access denied: '.env' is a protected file.
--- Now create foo.env.local ---
CREATE x.env.local: Created x.env.local in this chat session. The file is available to download.
READ x.env.local: Could not read .env file: unsupported file type '.local'; expected one of: .env, .env.local, .env.example.
CREATE x.env.example: Created x.env.example in this chat session. The file is available to download.
READ x.env.example: Could not read .env file: unsupported file type '.example'; expected one of: .env, .env.local, .env.example.
```

关键：**能建的读不回**。这是 env 编辑器唯一的功能性 bug。
