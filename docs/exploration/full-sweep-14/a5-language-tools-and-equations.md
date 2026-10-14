# 子代理 5：编程语言工具家族 + 数学/方程/科学计算

- 源码文件：47 个（40 语言工具 + 5 数学 + 2 辅助）/ 实际读取完成：0（进行中）
- 测试文件：2 个 / 实际读取完成：0（进行中）
- 未能读取：待补充

> ⚠️ 本轮（full-sweep-14）是 full-sweep-13 的补齐轮。13 轮本分区因 429 限流只留下 35 行骨架。
> ⚠️ 任务清单标称「42 个文件 = 21 对」，但实际列出的清单是 **40 个文件 = 20 对**（c、go、groovy、haskell、java、javascript、julia、latex、lua、matlab、prolog、php、python、r、ruby、rust、shell、sql、swift、typescript）。本仓库 `src/backend/tools/` 实测确实只有这 20 个语言家族成对存在。**这是任务描述本身的数字偏差，不是仓库缺文件。**
> ⚠️ 纪律：全文结论均以「实际代码 + 行号」佐证，不采信注释/docstring/设计文档。

---

## 0. 本轮新增发现（`[NEW]` = 与 13 轮记录区分）

见文末「本轮新增发现」清单。

---

## 1. 数学 / 方程 / 科学计算（5 个文件）—— 最高优先级，已读

### 1.1 `math_tool.py`（266 行）—— 已完整读取

**职责**：SymPy 后端符号数学工具，对外 LangChain 工具名 `solve_math`。支持 `derivative` / `integral` / `simplify` / `solve` / `evaluate` / `limit` / `series`（`:12-20`）。

**公开符号**：
- `MathInput`（`:23-73`）—— Pydantic schema，8 个字段全为字符串型自由文本，除 `order: int` 有 `ge=1, le=20`（`:64-69`）外**没有任何长度上限/字符白名单**。
- `build_math_tool(_settings) -> BaseTool`（`:76-114`）
- `solve_math(*, operation, expression, variable="x", lower_bound=None, upper_bound=None, point=None, order=6, direction="+") -> str`（`:117-177`）
- 私有：`_do_derivative :180`、`_do_integral :185`、`_do_solve :205`、`_do_limit :218`、`_do_series :233`、`_do_evaluate :248`

**核心漏洞（已 13 轮记录，本轮逐行复核通过）**：

`_parse` 定义于 `:151-157`：
```python
def _parse(text: str) -> Any:
    # parse_expr is SymPy's safe parser: no Python eval of arbitrary code.
    return parse_expr(
        text,
        transformations=transformations,
        evaluate=True,
    )
```
- **`:152` 的注释是错的**：没传 `local_dict` / `global_dict`。sympy 在 `global_dict is None` 时主动 `exec("from sympy import *")` 并把 `vars(builtins)` 中的 builtin 函数（含 `__import__`）注入 eval 命名空间 → 裸 `eval` → **真实 RCE**。机制静态取证见 `.venv/Lib/site-packages/sympy/parsing/sympy_parser.py`。
- 可执行 payload 形态：`expression = "__import__('os').system('whoami')"`；更干净的：`"Integer(1).__class__.__mro__"` 之类遍历。

**a) `_parse` 全部调用点逐一点名（11 处调用，4 个用户可控字段，全部 RCE）**：

| 行号 | 调用 | 用户可控字段 | RCE |
|---|---|---|---|
| `:163` | `expr = _parse(raw_expr)` | `expression` | ✅ |
| `:162`→`:208` | `sympy.Eq(parse(lhs_text), parse(rhs_text))` | `expression` 按 `=` 切两半 | ✅ ×2 |
| `:162`→`:210` | `sympy.Eq(parse(raw_expr), 0)` | `expression` | ✅ |
| `:167`→`:196` | `low = parse(str(lower_bound))` | `lower_bound` | ✅ |
| `:167`→`:197` | `high = parse(str(upper_bound))` | `upper_bound` | ✅ |
| `:171`→`:227` | `target = parse(point_text)` | `point`（limit） | ✅ |
| `:173`→`:242` | `around = parse(point_text)` | `point`（series） | ✅ |

另外两处**只传递 `_parse` 引用**（`:162`、`:167`、`:171`、`:173`），不是调用点，但导致上述 7 处间接调用全部继承同一漏洞。

**关键澄清（与 13 轮「所有 `_parse` 调用点都是 RCE」的表述对齐）**：`variable` **不**经过 `_parse` —— `:160` 直接 `sympy.Symbol(variable.strip() or "x")`，只影响符号名，不构成 RCE 通道。`direction` 在 `:228` 只与字面量 `"-"` 比较。**所以 RCE 面 = 4 个字段：`expression`、`lower_bound`、`upper_bound`、`point`。**

**错误处理**：`:176-177` 是 `except Exception as exc: return f"Math evaluation failed for {raw_expr!r}: {exc}"` —— **吞掉所有异常并回显**。副作用：RCE 已经执行完了才到这里，异常处理对安全毫无保护作用；但它会把 `os.system()` 的返回值/报错回传给 LLM，可构成信息回读（blind RCE → semi-blind）。

**版本漂移（已记录，本轮复核）**：`pyproject.toml` 钉 `sympy==1.14.0`，实际安装 **1.13.1**。→ 需在本轮核实（见 §4d）。

**修复手法（已记录，本轮复核可行性）**：
```python
_ns = {}
exec("from sympy import *", _ns)
_ns["__builtins__"] = {}
parse_expr(text, transformations=transformations, evaluate=True, global_dict=_ns)
```
`_ns["__builtins__"] = {}` 那行**必需**：sympy 注入的是 builtin *函数*（`__import__` 作为普通键），不是 `__builtins__` 这个 *键*，所以 `__import__` 仍可通过 `globals()["_ns"]["__import__"]` 逃逸，除非显式清空。

### 1.2 `linalg_tool.py`（233 行）—— 已完整读取

**职责**：精确线性代数，工具名 `linear_algebra`。操作：`determinant` / `inverse` / `transpose` / `multiply` / `add` / `eigenvalues` / `rank` / `solve`（`:12-21`）。

**公开符号**：`LinearAlgebraInput`（`:24-47`）、`build_linalg_tool`（`:50-79`）、`linear_algebra`（`:82-129`）、`_LinAlgError`（`:132`）、私有 `_parse_matrix :136`、`_to_number :157`、`_require_square :167`、`_do_multiply :173`、`_do_add :185`、`_do_solve :194`、`_format_eigenvalues :215`、`_render :221`。

**⚠️ 与 13 轮记录的一处重要修正**：13 轮把这条通道描述为「字符串单元格交给 `sympy.sympify`」。实测代码路径是**两跳**：

1. `_parse_matrix:141` 先做 `data = ast.literal_eval(text)` —— **`ast.literal_eval` 本身安全**，只做字面量解析，不执行代码。
2. **但 `literal_eval` 允许字符串字面量作为元素**。所以 `matrix='[["os.system(1)"]]'` 会成功解析成 `[['os.system(1)']]`（内含一个 `str`）。
3. `_parse_matrix:152` 逐单元格调 `_to_number`，`:162-163`：
   ```python
   if isinstance(value, str):
       return sympy.sympify(value)
   ```
   → 字符串单元格 → `sympy.sympify` → `eval` → **RCE**。

**三个字段 → `_to_number:163`（全部 RCE）**：
- `matrix`：`:102` `_parse_matrix(sympy, matrix, "matrix")`
- `matrix_b`：`:176`（`_do_multiply`）、`:188`（`_do_add`）
- `vector`：`:197`（`_do_solve`）

**`sympify` 签名（本轮需用 Grep 核实真实安装的 1.13.1）**：预期 `sympify(a, locals=None, convert_xor=True, strict=False, rational=False, errors='raise', **kwargs)` —— 无 `global_dict` 但**有 `locals`**，所以 `sympy.sympify(value, locals=_ns)` 是候选修复路线（13 轮标记「未验证」，本轮见 §4e）。

**`_to_number:158-159` 有个小防护**：显式拒绝 `bool`（`isinstance(value, bool)` → `ValueError`），避免 `True`/`False` 被 sympy 当符号。但 `isinstance(True, int)` 为真，所以必须靠这行先挡住 —— 写法正确。

**错误处理**：`:126-129` 双 `except`（`_LinAlgError` 与裸 `Exception`），同样**吞异常回显**，RCE 后无防护。

**其它观察**：
- `_do_solve:197-199` 对 `vector` 若解析出多列则 `.T` 转置，是合理容错。
- `_do_solve:205-206` 奇异矩阵提前返回。
- `:161` `float` → `sympy.nsimplify(value)`，会把 `0.1` 变 `1/10`，是有意的精确化。
- **无数组大小上限**（对比 `statistics_tool` 的 `_MAX_VALUES=100_000`）→ `matrix='[[1,' * 10**7` 类的输入可造成 CPU/内存 DoS。`eigenvalues` 对大矩阵尤其慢。

### 1.3 `statistics_tool.py`（140 行）—— 已完整读取

**职责**：描述性统计，工具名 `compute_statistics`。**单一字段** `numbers: str`。

**安全面结论：无 RCE 通道。**
- 输入解析是**纯正则提取**：`_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")`（`:14`），`_parse_numbers:87-95` 用 `findall` + `float(token)`。**没有 `eval` / `exec` / `sympify` / `parse_expr` / `compile` / `__import__`**。
- 只依赖标准库 `statistics` 模块。

**发现的问题（非安全，正确性/健壮性）**：
- `_fmt:119-122`：`if value == int(value)` —— 若 `value` 是 `inf` 或 `nan`，`int(value)` 抛 `OverflowError`。触发路径：`numbers='1e309'` → `float('1e309') == inf` → `sum` 或 `fmean` 得到 `inf` → `_fmt(inf)` 崩溃，且崩溃在 `compute_statistics:66` 的 `_fmt(total)`，**不在任何 try 块内**（`:51-84` 整个函数无 try/except）。所以 `compute_statistics('1e309')` 会向上抛异常。
- `_parse_numbers:88-95` 从任意文本里抽数字：`'abc 3 def 4'` → `[3.0, 4.0]`。对 LLM 工具是可接受的设计，但意味着**非数字内容被静默丢弃**，不报错。
- `int` 上界：`_MAX_VALUES = 100_000`（`:15`），`compute_statistics:55-56` 检查 —— 有上限，好。
- `_fmt:122` 用 `:.6g`，会丢精度（`1.23456789` → `1.23457`）。
- `_mode:103`：`len(modes) == len(set(values))` 判「无众数」，对全重复输入 `[5,5]` 判为有众数 `5`，正确。
- `_quartiles:109` `len(values) < 2` 返回 None，`quantiles` 对 n=2 会抛 `StatisticsError` 被 `:114` 捕获 → None。OK。

### 1.4 `number_theory_tool.py`（164 行）—— 已完整读取

**职责**：整数理论，工具名 `number_theory`。操作：`factorize` / `is_prime` / `gcd` / `lcm` / `prime_factors` / `next_prime` / `to_base`（`:11-19`）。

**安全面结论：无 RCE 通道。** 所有输入是 `int` 类型（`number: int`、`second_number: int | None`、`base: int`，`:33-46`），Pydantic 在工具层就已强类型化，字符串 payload 无法进入 `sympy` 解析路径。全文只调用 `factorint` / `primefactors` / `isprime` / `nextprime` / `gcd` / `lcm`，**没有任何 eval 家族函数**。

**发现问题**：
- **CPU DoS（非 RCE）**：`base` 有 `ge=2, le=36`（`:43-44`）+ 运行时双重检查 `:145-146`，但 `number` **没有上界**。`sympy.factorint(n)` 对一个 200 位的半素数会挂死整个 worker（ECM 算法耗时指数级）；`nextprime(n)` / `isprime(n)` 对超大整数同样慢。这是资源型 DoS。
- `:101` `n = int(number)`：若 Pydantic 层被绕过（直接调用函数），`int("1e309")` 抛 `ValueError`，被 `:118` 吞掉 —— 无 RCE，只是错误处理宽松。
- `_do_to_base:149-156` 手写 divmod，负数正确处理（`:150` 取符号，`:151` 取绝对值），正确。
- `_do_pair:136-140` `second_number is None` → `_NumberError`，正确。

### 1.5 `datetime_tool.py` —— **尚未读取**（本轮第一批待办）

---

## 2. 编程语言工具家族（20 对 / 40 文件）—— 进行中

（待补充：配对一致性审计、扩展名清单与冲突、校验强度分级）

---

## 3. 测试覆盖 —— 待核实

（待补充：`test_code_edit.py`、`test_typescript_edit.py` 内容；是否还有其他语言工具测试；数学面 8 个测试的断言清单）

---

## 4. 专项核实清单

- [ ] 4a. `math_tool` 所有 `_parse` 调用点 → **已完成，见 §1.1（11 处调用 / 4 个字段全 RCE）**
- [ ] 4b. `statistics` / `number_theory` / `datetime` 是否有 eval 家族通道 → statistics ✅ 无、number_theory ✅ 无、datetime ⏳ 待读
- [ ] 4c. `doctest('x')` 是否 spawn 子进程 → ⏳ 待核实
- [ ] 4d. sympy 版本漂移（pyproject 钉 1.14.0 / 实装 1.13.1）→ ⏳ 待核实
- [ ] 4e. `sympify` 签名与 `locals=` 修复可行性 → ⏳ 待 Grep sympy 源码
- [ ] 4f. 第三条公式通道（latex 渲染 / 表达式拟合 / 多项式求根）→ ⏳ 待 Grep
- [ ] 4g. 方程面 8 个测试是否全是功能测试 → ⏳ 待核实

---

## 5. 测试丢失问题（`__pycache__` 孤儿核实）

（待补充：Glob 当前真实数字 vs 13 轮记录的「41 孤儿模块 / 47 pyc」）

---

## 本轮新增发现（`[NEW]`）

（待补充）
