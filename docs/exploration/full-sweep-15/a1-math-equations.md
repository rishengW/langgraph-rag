# A1 — 方程求解与数值计算工具全家桶（full-sweep-15）

> 本轮**做了实测**，与 full-sweep-13 的「只静态阅读、零探针」不同。
> 所有「已执行 / 未执行」结论都来自真实运行结果，不是静态推断。
>
> **运行环境（已实测）**
> - `sympy 1.14.0` on `python 3.11.9`（`.venv/Lib/site-packages`，`sympy-1.14.0.dist-info`）
> - 复现命令：`.venv/Scripts/python.exe -I docs/exploration/full-sweep-15/_probe_a1.py`
> - 完整输出：`docs/exploration/full-sweep-15/_probe_a1_out.txt`
> - 副作用策略：payload 只做两件事——向 stdout 打哨兵字符串、touch 一个 `%TEMP%` 下的 marker 文件。不写仓库、不联网、无破坏性命令。marker 每轮运行前清空、每次探测前后独立路径 + 预清理，杜绝跨轮假阳性。

---

## 0. 最重要的一条

### 0.1 RCE 已实测确认（当前 `production` 分支，未修复）

`src/backend/tools/math_tool.py:151-157`：

```python
    def _parse(text: str) -> Any:
        # parse_expr is SymPy's safe parser: no Python eval of arbitrary code.
        return parse_expr(
            text,
            transformations=transformations,
            evaluate=True,
        )
```

实测：把 `__import__('os').system('echo RCE_PROOF > <marker>')` 作为表达式喂进去，**marker 文件被创建，`system()` 返回 0**。7 个入口全部打通：

| 入口 | 结果 |
|---|---|
| `evaluate` + `expression` | ✅ 执行 |
| `solve` + `expression`（lhs = rhs 两侧） | ✅ 执行 |
| `integral` + `upper_bound` | ✅ 执行 |
| `series` + `point` | ✅ 执行 |
| `limit` + `point` | ✅ 执行 |

`linalg_tool.py` 的第二条通道同样打通：`matrix` / `matrix_b` / `vector` 三个字段，4 种操作路径，**全部执行**。

### 0.2 历史报告给的「已验证有效」修复方案，实测是**不完整的**

历史记忆（及 full-sweep-13 §7/§7.5）给出并声称有效的修法：

```python
ns = {}
exec("from sympy import *", ns)
ns["__builtins__"] = {}
parse_expr(text, transformations=T, global_dict=ns)   # math_tool
sympy.sympify(value, locals=ns)                        # linalg_tool
```

本轮实测推翻其中两处：

| 修法 | 直连 `__import__` payload | `nsimplify("payload")` | `compile("payload",0,"exec")` |
|---|---|---|---|
| `parse_expr(global_dict=ns, __builtins__={})` | ✅ 挡住 | ❌ **仍执行** | ❌ **仍执行** |
| `sympy.sympify(value, locals=ns)` | ❌ **仍执行** | ❌ 仍执行 | ❌ 仍执行 |
| `sympy.sympify(value, strict=True)` | ✅ 挡住 | ✅ 挡住 | ✅ 挡住 |
| `parse_expr` + **名称白名单** token 转换 | ✅ 挡住 | ❌ **仍执行** | ❌ 仍执行 |

根因：sympy 的构造函数会**用完全不带命名空间的 `sympify` 重新解析字符串实参**。例如 `.venv/Lib/site-packages/sympy/simplify/simplify.py:1451`：

```python
    expr = sympify(expr).xreplace({
        Float('inf'): S.Infinity,
        Float('-inf'): S.NegativeInfinity,
    })
```

`nsimplify` 是 `from sympy import *` 导出的名字，因此在任何沙箱命名空间里都是「可见的合法函数」，而它内部那一次 `sympify` 调用**根本不继承外层命名空间**——`global_dict=None` → sympy 注入 `vars(builtins)` → `__import__` 可用。

**结论：sympy 的解析器不是一个安全边界。** 任何「只限制命名空间」的方案都不成立。可靠修法见 §8。

---

## 1. 历史结论复核表

| # | 历史结论 | 结论 | 证据 |
|---|---|---|---|
| 1 | `parse_expr` 无 `local_dict`/`global_dict` → RCE | ✅ **确认，且已实测** | `math_tool.py:151-157`；探针 A/H/H2/H3/H4/H5 |
| 2 | 位置在 `math_tool.py:153-157` | ✅ **确认，行号未漂移** | `:153-157` 即 `parse_expr` 调用体；`_parse` 定义在 `:151`，误导注释在 `:152` |
| 3 | `math_tool.py:152` 的注释「parse_expr is SymPy's safe parser」是假的 | ✅ **确认** | 注释原文与实测行为直接矛盾 |
| 4 | 修复不在当前 `production` 分支 | ✅ **确认** | 全文 grep 仅 `math_tool.py:140,153` 两处 `parse_expr`，均无命名空间参数 |
| 5 | 全仓库 `_parse` 调用点均为 RCE | ✅ **确认** | `_parse` 被调 7 处：`:163, :196, :197, :208, :210, :227, :242`，全部 LLM 可控 |
| 6 | linalg 有 3 条通道（`matrix`/`matrix_b`/`vector` → `_to_number`） | ✅ **确认，且已实测** | `linalg_tool.py:102, :176, :188, :197` → `:136` `_parse_matrix` → `:157-164` `_to_number`；探针 H6/H7/H8/H9 |
| 7 | `_to_number` 在 `:163` | ✅ **确认，行号未漂移** | `:163` 即 `return sympy.sympify(value)` |
| 8 | ⚠️ **历史遗漏**：linalg 通道要求 payload 必须是**字符串字面量** | 🔧 **新增限定** | `_parse_matrix:141` 先跑 `ast.literal_eval`；裸写 payload 被拒，写成 `'[[1,2],[3,"payload"]]'` 才放行。探针 H6 第一版即因此踩空 |
| 9 | 修复技巧：`global_dict=ns` + **必须** `ns["__builtins__"] = {}` | ⚠️ **部分推翻** | 见下 |
| 9a | 前半：`global_dict=ns` 确实能挡住直连 `__import__` | ✅ **确认** | 探针 B/D/A2 |
| 9b | 后半理由：「sympy 只注入内置函数、不注入 `__builtins__` 键，缺这行 `__import__` 仍能经 `globals()` 逃逸」 | ❌ **理由不成立** | 探针 D：`ns` 由 `exec("from sympy import *", ns)` 构造但**不设** `__builtins__={}`，payload 一样没有执行。真正生效的机制是 sympy 在 **token 层**用 `_token_callable`（`sympy_parser.py:45-55`）检查 `local_dict`/`global_dict` 的**键**；`__import__` 不是键 → 被 `auto_symbol` 改写成 `Symbol`，根本走不到 Python 的 eval 查找 |
| 9c | 该修复方案「已验证有效」 | ❌ **不完整** | 探针 N：`nsimplify("payload")` 与 `compile("payload",0,"exec")` 两种 payload 在 FIXED-config 下**依然执行** |
| 10 | `sympify` 签名是 `sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None)` | ✅ **确认，行号一致** | `sympy/core/sympify.py:124` |
| 11 | `sympify` 没有 `global_dict` 但有 `locals`，所以 `sympy.sympify(value, locals=_ns)` 可能是 linalg 可行修法（历史标为「未验证」） | ❌ **推翻（已验证：无效）** | 探针 G/P：`sympy.sympify(payload, locals=ns)` **仍然执行代码**，返回 0。`locals` 只影响 `local_dict`，而 `sympify` 内部调 `parse_expr(a, local_dict=locals, ...)` 时 `global_dict` 仍为 `None`（`sympify.py:507`）→ 照样注入 `vars(builtins)` |
| 12 | `doctest('x')` 会 fork 子进程 | ⚠️ **部分推翻** | 实测 `sympy.doctest('x')` 在 **当前进程内**跑完整个 doc-test 套件：`476 passed, 3 skipped, in 9.94 seconds`（另一次带缓存 10.8s）。不是 fork。但签名确有 `subprocess` 参数。且 `parse_expr('doctest')` 返回 `sympy.core.cache.LazyFunction`（**可调用**）→ 仍然是一条可达的 CPU 耗尽通道 |
| 13 | 版本漂移：`pyproject.toml` 钉 `sympy==1.14.0`，实际装 `1.13.1` | ❌ **推翻** | `pyproject.toml:35` = `sympy==1.14.0`；`.venv/Lib/site-packages/sympy-1.14.0.dist-info` 存在，`import sympy` → **`1.14.0`**。`1.13.1` 只存在于**全局解释器** `C:\Users\OMEN\AppData\Local\Programs\Python\Python311\Lib\site-packages`，不是应用运行时 |
| 13b | 漂移导致的历史行号引用不可靠 | ✅ **确认无影响** | 我核对的机制行号在 **1.13.1 与 1.14.0 两个版本完全一致**：`sympy_parser.py:1050-1058`（`global_dict is None` 分支）、`:1081`（`eval_expr`）、`sympify.py:124`/`:507` |
| 14 | F6：`statistics_tool` 无 inf/NaN 防护 → `_fmt` 抛 `OverflowError` 且 `compute_statistics` 无 try/except | ✅ **确认，且已实测** | 探针 L/L2/L3/L4：`compute_statistics("1e999")` 端到端抛 `OverflowError: cannot convert float infinity to integer` |
| 15 | F7：`linalg_tool` 无任何矩阵维度上限，无超时 | ✅ **确认** | `_parse_matrix`（`linalg_tool.py:136-154`）只查非空与锯齿行，无 `rows`/`cols` 上限 |
| 16 | F8：`datetime_tool.py:178` 用 `abs(delta.days)` 输出 total days，自相矛盾 | ✅ **确认**（历史示例数值有误，见 §4.7） | `:178` 原文 `f"(total {sign}{abs(delta.days)} days)"` |
| 17 | §7.5「`locals=_ns` 是首选方案，比改调 `parse_expr` 更省事」 | ❌ **推翻** | 见 #11。首选方案实际无效；有效的是 `strict=True`（但会拒绝 `"sqrt(2)/2"` 这类合法字符串）或彻底拒绝字符串单元格 |
| 18 | §6-7：全部符号运算无超时/复杂度限制 | ✅ **确认，且已实测出量级** | 探针 K5：`sympy.factorint` 对 160-bit 半素数 **25 秒未返回**（子进程被杀）。实测 `solve(x**5-5*x+12)` 1.91s、`simplify(cos^10+sin^10)` 0.28s。整个仓库无 deadline 机制 |

**汇总：确认 12 条 / 推翻 5 条（#9b、#9c、#11、#13、#17）/ 部分推翻 1 条（#12）/ 新增限定 1 条（#8）。**

---

## 2. 🧨 两条 RCE 通道（已实测）

### 2.1 通道 A：`math_tool._parse`

`math_tool.py:149-157`：

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

既不传 `local_dict` 也不传 `global_dict`，两者都落到 sympy 的 `None` 分支（`.venv/Lib/site-packages/sympy/parsing/sympy_parser.py:1050-1058`）：

```python
    if global_dict is None:
        global_dict = {}
        exec('from sympy import *', global_dict)

        builtins_dict = vars(builtins)
        for name, obj in builtins_dict.items():
            if isinstance(obj, types.BuiltinFunctionType):
                global_dict[name] = obj          # 含 __import__ / eval / exec / open / compile
        global_dict['max'] = Max
        global_dict['min'] = Min
```

随后 `:1081` → `eval_expr(code, local_dict, global_dict)` → 裸 `eval`。

**机制实测（探针 A2）**：`parse_expr("__import__")` 在默认配置下返回 `type=builtins.builtin_function_or_method`、`value='<built-in function __import__>'` —— sympy 真的把内置函数注入了。在沙箱命名空间下同一输入返回 `sympy.core.symbol.Symbol` —— 被改写成符号。

**入口实测（全部 ✅ 执行）**：

```
H  evaluate, expression=payload          → "… = 0"          （system() 的返回码）
H2 solve,  equation 两侧都是 payload       → "No solution found …"
H3 integral, upper_bound=payload          → "integral of x dx from 0 to 0 = 0"
   ↑ 注意：payload 的返回值 0 被当成积分上限，积分照样算完——无报错泄露
H4 series,  point=payload                 → "series of exp(x) around x=__import__(…)…"
H5 limit,   point=payload                 → "limit of 1/x as x->__import__(…) = oo"
```

`H3` 值得单独点出：**payload 的副作用已经发生，但工具返回的是一条完全正常的积分结果**，不会在日志里留下任何异常痕迹。这是「静默 RCE」，比报错更糟。

`_parse` 的 7 个调用点全部通向 LLM 可控字段：

| 行 | 调用 | 字段 |
|---|---|---|
| `:163` | `_parse(raw_expr)` | `expression` |
| `:196` | `parse(str(lower_bound))` | `lower_bound` |
| `:197` | `parse(str(upper_bound))` | `upper_bound` |
| `:208` | `parse(lhs_text)` / `parse(rhs_text)` | `expression` 按 `=` 切分 |
| `:210` | `parse(raw_expr)` | `expression` |
| `:227` | `parse(point_text)` | `point`（limit） |
| `:242` | `parse(point_text)` | `point`（series） |

### 2.2 通道 B：`linalg_tool._to_number`

`linalg_tool.py:157-164`：

```python
def _to_number(sympy: Any, value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid matrix entries")
    if isinstance(value, (int, float)):
        return sympy.nsimplify(value) if isinstance(value, float) else sympy.Integer(value)
    if isinstance(value, str):
        return sympy.sympify(value)        # ← 用户字符串 → sympify，无命名空间
    raise ValueError(f"unsupported entry {value!r}")
```

`sympify` 自带警告（`.venv/Lib/site-packages/sympy/core/sympify.py:137-139`）：

> .. warning:: Note that this function uses ``eval``, and thus shouldn't be used on unsanitized input.

**但关键细节历史报告漏了**：`_parse_matrix`（`:136-154`）先用 `ast.literal_eval`（`:141`）解析矩阵结构。`ast.literal_eval` 会拒绝裸写 payload，所以 payload **必须写成字符串字面量**：

```python
linear_algebra(operation="rank", matrix="[[1, 2], [3, \"__import__('os').system(…)]]")
```

`ast.literal_eval` 接受它（就是一个 str），随后 `_to_number` 把这个 str 交给 `sympy.sympify`。

**实测（全部 ✅ 执行）**：

```
H6 matrix  cell → 'rank = 2'
H7 vector entry → 'solution x = [-10, 15/2]'
H8 matrix_b cell → 'product =\n[1, 2]\n[3, 4]'
H9 transpose path → 'transpose =\n[1, 3]\n[2, 0]'
```

三条字段通道：`matrix`（`:102`）、`matrix_b`（`:176` multiply / `:188` add）、`vector`（`:197` solve）。全部通向 `_parse_matrix` → `_to_number` → `sympify`。

`sympy` 自身的签名确认（`sympify.py:124` / `:507`）：

```python
def sympify(a, locals=None, convert_xor=True, strict=False, rational=False, evaluate=None):
    ...
    expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
```

`global_dict` 参数根本不存在，且内部调用不传 `global_dict` → 永远落到注入内置函数的分支。

---

## 3. 🛑 历史「已验证有效」修复方案的实测判决

### 3.1 矩阵总表

`S = __import__('os').system('echo RCE_PROOF > <marker>')`

| payload \ 配置 | A：当前代码（无命名空间） | B：`global_dict=ns` + `__builtins__={}` | C：仅 `global_dict={'__builtins__': {}}` | D：`global_dict=ns`，**不设** `__builtins__` |
|---|---|---|---|---|
| `S`（直连） | ✅ **执行** | blocked | blocked | blocked |
| `eval(S)` | ✅ **执行** | blocked | — | blocked |
| `exec(S)` | ✅ **执行** | blocked | — | blocked |
| `compile(S,0,"exec")` | blocked（`TypeError: expected str…not Zero`） | ✅ **执行** | — | ✅ **执行** |
| `nsimplify(S)` | ✅ **执行** | ✅ **执行** | — | — |
| `Mul(1, S)` / `Add(1, S)` | blocked（`SympifyError`） | blocked | — | — |
| `Integer(S)` | blocked（`ValueError`） | blocked | — | — |
| `Symbol(S)` | blocked（变成符号名） | blocked | — | — |
| `(1, S, 2)` | blocked（保留字符串） | blocked | — | — |
| `().__class__.__base__.__subclasses__()` | blocked（返回 `'type'`） | blocked | — | blocked |

**读法**：

1. **B 挡得住直连 `__import__`，但挡不住 `compile(...)` 和 `nsimplify(...)`。** 历史记忆把 B 描述为「已验证有效」是过度乐观。
2. **C 单独用会破坏正常功能**：`parse_expr` 在仅有 `{'__builtins__': {}}` 时连 `Symbol` 都解析不出来，`parse_expr('sin(x)')` 抛 `NameError: name 'Function' is not defined`。必须先 `exec("from sympy import *", ns)` 填充名字空间，再清空 builtins。
3. **D（不设 `__builtins__={}`）居然和 B 表现一致** —— 这是推翻历史结论 #9b 的直接证据。生效的是 sympy 的 **token 层** `_token_callable`（`sympy_parser.py:45-55`）：它只查 `local_dict`/`global_dict` 的**键**。`__import__` 不是键 → `auto_symbol` 把它变成 `Symbol('__import__')` → eval 时 `Symbol('__import__')('os')` 得到字符串 `os` → `.system` → `AttributeError: 'str' object has no attribute 'system'`。**Python 层面的内置函数查找从未被触发**，与 `__builtins__` 键无关。
4. `A2` 补证：`parse_expr('__import__')` 在默认配置返回真 `builtins.__import__`，在 B 配置返回 `sympy.core.symbol.Symbol` —— 与上表完全自洽。

### 3.2 为什么 B 挡不住 `nsimplify(...)` —— 这是本轮最有价值的发现

`nsimplify` 是 `from sympy import *` 导出的名字，因此在任何沙箱命名空间里都是**可见的、可被 `_token_callable` 判定为可调用的合法函数**。它内部（`.venv/Lib/site-packages/sympy/simplify/simplify.py:1451`）：

```python
    expr = sympify(expr).xreplace({
        Float('inf'): S.Infinity,
        Float('-inf'): S.NegativeInfinity,
    })
```

这次 `sympify` 调用**不继承外层命名空间** → `global_dict=None` → 注入 `vars(builtins)` → `__import__` 可用。

**定位证据（探针）**：把 marker 预清理后逐步执行，`stringify_expr` 之后 marker 不存在，`eval(code, ns)` 之后 marker 出现 —— 执行发生在 eval 阶段，且发生在**命名空间已受限**的情况下。即绕过路径是 `parse_expr` → `eval` → `nsimplify` → `sympify` → `parse_expr(global_dict=None)`。

同一类函数（被导出 + 内部裸 `sympify`）理论上不止 `nsimplify` 一个；本轮只验证了 `nsimplify` 一例。

### 3.3 为什么名称白名单也不行

我实现了一个 token 层名称白名单转换函数（`ALLOWED_CALLS = {sin, cos, exp, sqrt, …}`，非白名单 NAME 标记为 `null` 符号），且**插在 `standard_transformations` 之前**运行。结果：

- 直连 `__import__`、`compile(...)`、`Mul/Add/Integer/Symbol/Tuple` 全部 blocked ✅
- 正常数学全部正常：`x**2 + 3*x`、`sin(x)*exp(x)`、`sqrt(2)/2`、`2*x+1`、`log(x)/x**2`、`1/3`、`x**2 - 4` 全部解析正确 ✅
- **`nsimplify("payload")` 仍然执行** ❌

原因：sympy 的 `Symbol` 实例**本身是可调用的**（`callable(Symbol('f'))` 为真，`Symbol('f')('x')` 会构造 `Function('f')('x')`），而 sympy 的构造函数会对实参做 `sympify`。所以「把名字变成符号」并不能阻止字符串被二次解析。

### 3.4 哪些修法**真的有效**（已验证）

| 修法 | 直连 payload | `nsimplify` payload | 副作用 |
|---|---|---|---|
| `sympy.sympify(value, strict=True)` | ✅ blocked（`SympifyError`） | ✅ blocked | **会拒绝 `"sqrt(2)/2"` 这类合法字符串单元格**（探针 G2/G3 显示不带 `strict` 时合法字符串正常；`strict=True` 时按设计拒绝） |
| 彻底删除 `linalg_tool._to_number` 的 `isinstance(value, str)` 分支 | ✅ | ✅ | 数值矩阵本就不该有字符串条目，同时修掉一个正确性问题 |
| Python 侧 AST 白名单预检（拒绝 `Attribute` / `Subscript` / 非白名单 `Call` / `str` `Constant` / `Starred` / `Lambda`） | ✅ 11/11 攻击载荷全拒 | ✅ 全拒 | 需对 `solve` 的 `lhs = rhs` **每一侧**分别检查，不能整体检查（见 §8.3） |

**AST 白名单预检的实测（探针 R，已验证）**：

- 10 个合法数学表达式 **10/10 放行**：`x**2 + 3*x`、`sin(x)*exp(x)`、`sqrt(2)/2`、`2*x+1`、`log(x)/x**2`、`1/3`、`x**2 - 4`、`x**5 - 5*x`、`2*x + 1`、`5`
- 11 个攻击载荷 **11/11 拒绝**：直连 `__import__`、`compile(...)`、`nsimplify(...)`、`Mul/Add/Integer/Symbol(...)`、`().__class__.__base__.__subclasses__()`、`open('/etc/passwd').read()`、`exec('import os')`、`__builtins__`、列表推导

---

## 4. 逐文件说明（9 个文件全部读完）

### 4.1 `math_tool.py`（266 行）
工具名 `solve_math`（`:104`）。7 个操作（`_OPERATIONS` `:12-20`）。`MathInput`（`:23-73`）：`expression` 仅 `min_length=1`（`:35-37`），`lower_bound`/`upper_bound`/`point` 均为 `str | None` 无上限，`order` 有 `ge=1, le=20`（`:64-69`），`direction` 无枚举约束（`:70-73`，`_do_limit:228` 只判 `"-"`，其余一律按 `"+"`）。
- **致命**：`_parse`（`:151-157`），见 §2.1。
- 异常全部吞成字符串返回（`:176-177`），不抛错、不记日志 —— 对安全审计不利。
- `order` 下限被无条件覆盖（`:243` `n = max(1, int(order))`）；Pydantic 已限 int，故此分支仅防御性。
- `_do_solve`（`:205-215`）按**第一个** `=` 切分（`:206-208`），`"a = b = c"` 会产出非预期方程。
- `_do_integral`（`:193-195`）只判「两个边界都存在」；只给 `lower_bound` 不给 `upper_bound` 静默退化为不定积分，无提示。
- `_do_evaluate`（`:248-258`）检查 `free_symbols` 拒绝数值求值 —— 与沙箱无关。

### 4.2 `linalg_tool.py`（233 行）
工具名 `linear_algebra`（`:70`）。8 个操作（`:12-21`）。
- **致命**：`_to_number` 的 `sympify`（`:163`），见 §2.2。
- `:161` 对 float 调 `sympy.nsimplify` —— 实参已是 float 不涉及字符串重解析，但 `nsimplify` 是「内部裸 sympify」家族的成员，属于同类隐患。
- `_parse_matrix`（`:136-154`）**无维度上限**：只查非空与锯齿。`inv`/`eigenvals`/`rank`/`solve`（`:103-125`）无超时 → CPU DoS。
- `_parse_matrix` 接受扁平列表当列向量（`:147-148`），是设计行为。

### 4.3 `number_theory_tool.py`（164 行）
工具名 `number_theory`。7 个操作。**全部输入为 Pydantic `int`，零字符串解析，无表达式通道** —— 本文件不涉 RCE。
- `number` / `second_number` **无任何大小上限**（`:33-36` / `:37-40`），内部再 `int(number)` 强制一次（`:101`）。
- `base` 双重校验：Pydantic `ge=2, le=36`（`:41-46`）+ `_do_to_base:143-146` 复查。是双保险。
- **CPU 耗尽面**：`_do_factorize`（`:126-133`）与 `primefactors`（`:105`）把无界整数直接喂给 sympy。实测 160-bit 半素数 25 秒不返回。

### 4.4 `statistics_tool.py`（140 行）
工具名 `compute_statistics`。数字来自正则提取而非 eval（`_NUMBER_RE` `:14`，`float(token)` `:92`）—— 不涉 RCE。
- `_MAX_VALUES = 100_000`（`:15`）限制元素数；`numbers` 字段仅 `min_length=1`（`:21-27`），**无 max_length** —— 200KB 输入可接受。
- **崩溃缺陷（实测确认）**：`_NUMBER_RE` 允许 `1e999` → `float("1e999")` = `inf` → `_fmt`（`:119-122`）先跑 `int(value)` → `OverflowError: cannot convert float infinity to integer`；`_fmt(float("nan"))` → `ValueError: cannot convert float NaN to integer`。而 `compute_statistics`（`:51-84`）**无 try/except**，与同目录其余工具「错误返回字符串」的约定相悖 → 未处理异常穿透到工具层。
- `_parse_numbers`（`:87-95`）静默丢弃解析失败的 token（`except ValueError: continue`），`"1, 2, abc, 4"` → `[1.0, 2.0, 4.0]`，不报错不告警。
- `_NUMBER_RE = [-+]?\d*\.?\d+(?:[eE][-+]?\d+)?` 无嵌套量词，**无 ReDoS 风险**。

### 4.5 `currency.py`（101 行）
工具名 `convert_currency`。
- `:13` `CURRENCY_API_URL = "https://api.frankfurter.dev/v2/rate"`
- `:20-30` `from_currency` / `to_currency` 限 `min_length=3, max_length=3`，**但无字符集白名单**：`"../"` 正好 3 字符 → `:74` 的 f-string URL 拼接产生同主机路径穿越（`https://api.frankfurter.dev/v2/rate/../../../`）。无法跨主机（协议与主机是字面量），危害有限但仍是缺陷。
- `:88` `amount * float(rate)`：`amount` 有 `gt=0`（`:19`），`rate` 来自响应体，`float(rate)` 可能抛 `ValueError` —— 但外层 `try/except Exception`（`:72-79`）只包住请求部分，`:88-93` 在 try 之外 → 异常穿透。
- `:73-77` 用 `request_json(..., params={})`，即 URL 直接拼接而非 query 参数。

### 4.6 `datetime_tool.py`（236 行）
工具名 `calculate_datetime`。5 个操作。纯标准库 + `zoneinfo`，无 eval、无网络。
- `:135-152` `_parse_dt` 用 `datetime.fromisoformat`，先 `replace("Z", "+00:00")`，失败回退 `date.fromisoformat`。无异常注入面。
- **`_zone`（`:126-132`）**：`ZoneInfo(name.strip())` 对 LLM 任意字符串调用。`zoneinfo` 会读系统时区数据库；非法名字抛 `ZoneInfoNotFoundError` 已被捕获（`:131`）。但**无长度上限**，且 `zoneinfo` 可被 `TZPATH` 环境变量指到攻击者可控目录 —— 这是配置面问题，非本文件缺陷。
- **`_do_difference:178` 正确性缺陷（确认历史 F8，数值需修正）**：`f"(total {sign}{abs(delta.days)} days)"`。`delta.days` 是「天数向下取整」，不含小时部分，而 `sign` 已由 `total_seconds` 决定，`days` 来自 `secs // 86400`。三套口径并存：
  - `+26h`：输出 `1 day(s), 2 hour(s), 0 minute(s) (total 1 days)` —— 实际应为 26 小时，`total` 少算
  - `-4h`：输出 `-0 day(s), 4 hour(s), 0 minute(s) (total 1 days)` —— 同一句里 `0 day(s)` 与 `total 1 days` 自相矛盾（历史报告写的 `-0 day(s), 4 hour(s) (total -1 days)` 中，`(total` 处带了 `abs()`，不会输出负号）
- `_do_convert`（`:191-211`）：`naive` 起始时间被静默 `replace(tzinfo=from_zone)`（`:205-206`），不提示。

### 4.7 `stock.py`（144 行）
工具名 `get_stock_quote`。
- `ticker` 限 `min_length=1, max_length=20`（`:20-22`），但**无字符集约束**：`yf.Ticker(symbol)` 会把 ticker 拼进 Yahoo Finance 的请求路径，理论上可做路径穿越/注入；yfinance 内部会再次 URL 编码，实际危害取决于 yfinance 版本。
- **未捕获异常**：`:57-62` 的 `try/except` 只包住 `_default_ticker_factory` 与两个 `getattr`；`:64-106` 的 `float(price)`、`int(volume)`、`int(market_cap)`、`f"{float(price):.2f}"` 都在 try 之外。`yfinance` 的 `fast_info` 在某些状态下返回 `None` 或 `str` → `TypeError`/`ValueError` 穿透。
- `:87` 除零已防护：`(change / float(previous_close) * 100) if float(previous_close) else 0.0`。
- `:111-120` `_default_ticker_factory` 用 `yfinance`，无超时控制（yfinance 自己的 socket 超时）。

### 4.8 `weather.py`（261 行）
工具名 `get_weather`。
- `latitude`/`longitude` 有 `ge=-90, le=90` / `ge=-180, le=180`（`:47-48`），`forecast_days` 有 `ge=1, le=16`（`:49-54`）—— 本文件是 8 个工具里**约束最完整**的。
- `:138-143` `_resolve_location` 直接把城市名当 query 参数（`params={"name": query, ...}`），正确用法。
- `:229` `for index, day in enumerate(dates[:5])` —— 硬编码只显示 5 天，即使 `forecast_days=16` 也只输出 5 天。功能缺陷（不是安全缺陷）。
- `:155-156` `lat = float(first["latitude"])` / `lon = float(first["longitude"])` 在 try 块内（`:112-126`），KeyError/TypeError 被转成友好错误。

### 4.9 `__init__.py`（429 行）+ 注册链路
`__init__.py` 只是 re-export 桶，**没有任何校验层**：`:15,16,63,84,96,128,129,225` 引入本区 8 个工具，其余是文件编辑类工具。
真正的注册在 `src/backend/mcp/providers.py:148-176` 的 `BuiltinToolProvider`：

```python
factories = (
    (settings.weather_enabled, tools.build_weather_tool),
    (settings.stock_enabled, tools.build_stock_tool),
    (settings.currency_enabled, tools.build_currency_tool),
    ...
    (settings.math_enabled, tools.build_math_tool),
    (settings.statistics_enabled, tools.build_statistics_tool),
    (settings.linalg_enabled, tools.build_linalg_tool),
    (settings.number_theory_enabled, tools.build_number_theory_tool),
    (settings.datetime_enabled, tools.build_datetime_tool),
    ...
)
for enabled, factory in factories:
    if enabled:
        built.append(factory(settings))
```

每个工具经 `StructuredTool.from_function(..., args_schema=<Model>)` 构造，**Pydantic 是唯一输入校验层**。`src/backend/mcp/policy.py:250-256` 的 `PolicyTool._parse_input` 明确注释「Authorization intentionally precedes validation」——即策略层**不做任何输入内容检查**。
- `policy.py:206-207`：`allow_risk_levels` 默认 `{"read", "write", "execute"}` —— **`execute` 风险级别默认可用**。
- `policy.py:227-229`：策略只做「风险级别是否被允许 + 主体/租户是否在允许列表」，**没有任何针对 `execute` 风险的额外门禁或人工确认**。
- `providers.py:469-478` `_builtin_risk` 确实把 `solve_math`/`compute_statistics`/`linear_algebra`/`number_theory` 标为 `execute`，`date_time` 也被列入 —— **但 `date_time` 不是任何工具的真实名字**，`datetime_tool.py:79` 的工具名是 `calculate_datetime`。因此 `calculate_datetime` 被**静默降级为 `read`**。这是元数据 bug，不影响安全性（该工具确实只读），但说明风险标注本身不可靠。

### 4.10 `src/backend/tools/_http.py`（38 行，共享依赖）
`request_json`（`:11-34`）：
- `:26` `response = get(url, **kwargs)`，`timeout` 默认 10 秒（`:16`）—— 有超时，这一点做对了。
- **无 `allow_redirects=False`** → 3xx 跟随重定向，无目标校验。
- **无响应体大小限制**：`:31` `payload = response.json()` 直接解析整个 body。一个恶意响应可占满内存。
- **无 IP/域名白名单、无 SSRF 防护**（本仓库的 SSRF 防护链在 `mcp_client/security.py`，但那是给 outbound MCP 用的，`request_json` 不走那条路）。
- 被 `currency.py:73`、`stock`（间接）、`weather.py:113,145`、`map_tool` 等共享。

---

## 5. ⏱ 资源耗尽面（CPU / 内存）

### 5.1 实测量级

| 操作 | 耗时 | 结果 |
|---|---|---|
| `factorint(2**60-5)` | 0.000s | 分解成功 |
| `factorint(3**40+389)` | 0.000s | 分解成功 |
| `primefactors(3**30+119)` | 0.000s | 成功 |
| `simplify(cos^10 + sin^10)` | 0.281s | 成功 |
| `solve(x**5-5*x+12)` | **1.906s** | 返回 5 个根 |
| `integrate(exp(x**2), x)` | 0.141s | 成功 |
| `factorint(60-bit 半素数)` | 0.109s | 成功 |
| `factorint(80-bit 半素数)` | 0.391s | 成功 |
| **`factorint(160-bit 半素数)`** | **> 25.0s（子进程被杀）** | ❌ **CPU 耗尽确认** |
| `sympy.doctest('x')` | 9.94s（10.8s 带缓存） | 476 passed / 3 skipped，**全进程内执行** |
| `to_base(200 位十进制整数)` | 0.000s | 输出 876 字符 |

### 5.2 可达性

- `number_theory`：`number` 字段无上限（`number_theory_tool.py:33-36`）→ `_do_factorize`（`:126-133`）/`primefactors`（`:105`）直接吃任意大整数。LLM 只要被诱导传入一个 160-bit 半素数即可。**单请求即可挂起事件循环 25 秒以上。**
- `solve_math`：`parse_expr('factorint')` 解析出的是**真函数**（`type=builtins.function`），`parse_expr('solveset')` 同样是。即 LLM 可以直接在表达式里写 `factorint(...)`，绕过 `number_theory` 走 `solve_math` 触发同样的分解。`parse_expr('doctest')` 返回 `sympy.core.cache.LazyFunction`（**可调用**）→ `solve_math(expression="doctest('x')")` 会跑完整测试套件。
- `linear_algebra`：`_parse_matrix` 无维度上限 → `inv`/`eigenvals`/`rank`/`solve` 无超时。
- 整个仓库**无 deadline 机制**：无 `signal.alarm`、无 `asyncio.wait_for`、无独立进程隔离。

### 5.3 除零 / 溢出

- `stock.py:87` 除零已防护（见 §4.7）。
- `statistics_tool._fmt` 对 `inf`/`nan` **崩溃**（实测，见 §4.4）。这是本区唯一一处「非法输入导致未捕获异常」的可复现缺陷。
- `math_tool._do_evaluate:248-258` 的 `free_symbols` 检查只挡「含符号」，不挡「求值代价」。

---

## 6. 测试覆盖

`tests/test_agent_tools.py`：

| 范围 | 测试 | 性质 |
|---|---|---|
| `:536-605` | `solve_math` 8 个 | 全功能，零恶意输入 |
| `:208-244` | `linear_algebra` 4 个 | 全功能 |
| `:250-288` | `number_theory` 5 个 | 全功能 |
| `:191-206` | `compute_statistics` 2 个 | 全功能 |
| `:290-340` | `calculate_datetime` 多个 | 全功能 |

**全仓库 grep 确认**：`src/` 下 `__import__` 出现次数 **0**；`eval`/`exec` 在 `src/backend/tools/` 下仅出现在被沙箱化的（实际未沙箱化）位置。**没有任何测试使用 `__import__`、`os.system`、`exec`、`doctest`、`open(`、属性访问链等探针去探测解析沙箱。** `math_tool.py:152` 的错误注释之所以能存活多轮，正因为它背后没有对抗性测试。

---

## 7. 修复建议（标注实测状态）

### 7.1 `math_tool.py` —— 立即措施

**首选：Python 侧 AST 白名单预检（已验证 10/10 合法放行、11/11 攻击拒绝）**：

```python
_ALLOWED_FUNCS = {"sin","cos","tan","sec","csc","cot","sinh","cosh","tanh","asin",
                  "acos","atan","exp","log","ln","sqrt","cbrt","Abs","Max","Min",
                  "floor","ceiling","sign","conjugate","Im","Re","gamma","factorial",
                  "Rational","Integer","pi","E","oo"}

class _MathShapeOnly(_ast.NodeVisitor):
    def __init__(self): self.ok = True
    def visit_Call(self, n):
        if not (isinstance(n.func, _ast.Name) and n.func.id in _ALLOWED_FUNCS):
            self.ok = False
        self.generic_visit(n)
    def visit_Attribute(self, n): self.ok = False
    def visit_Subscript(self, n): self.ok = False
    def visit_Starred(self, n): self.ok = False
    def visit_Lambda(self, n): self.ok = False
    def visit_Constant(self, n):
        if isinstance(n.value, str): self.ok = False   # 关键：sympy 构造函数会裸 sympify 字符串

def _math_shape_ok(text):
    try: tree = _ast.parse(text, mode="eval")
    except SyntaxError: return False
    v = _MathShapeOnly(); v.visit(tree); return v.ok
```

⚠️ **注意**：`_do_solve` 在 `:206-208` 按 `=` 切分后才 `_parse` 两侧，所以预检必须**对每一侧分别做**，不能对整串做（`"2*x + 1 = 5"` 整体 `mode="eval"` 会 SyntaxError）。

**必须同时保留**：`global_dict=ns` + `ns["__builtins__"] = {}`（§3.4 已验证挡住直连 payload；虽然它挡不住 `nsimplify`/`compile`，但它降低了攻击面，且是 sympy 官方推荐的用法）。

**不要采用**：仅 `sympy.sympify(value, locals=ns)`（已验证无效）；仅名称白名单 token 转换（已验证可被 `nsimplify` 绕过）。

**超时**：`_run_math`（`:81-100`）是无 `asyncio` 的同步函数，无法用 `asyncio.wait_for`。可行方案：`concurrent.futures.ProcessPoolExecutor` + `future.result(timeout=...)`，或把解析放进受控子进程。**实测 160-bit 半素数 25 秒不返回**，说明这是必须做而不是可选的。

### 7.2 `linalg_tool.py` —— 立即措施

`_to_number`（`:157-164`）的 `isinstance(value, str)` 分支是唯一的字符串通道。三选一：

1. **推荐：直接删除该分支**，抛 `ValueError("string entries are not accepted; use numeric literals")`。数值矩阵本就不该有字符串单元格，同时修掉一个正确性问题。**已验证完全阻断**。
2. `sympy.sympify(value, strict=True)` —— 已验证阻断，但会拒绝 `"sqrt(2)/2"` 这类合法表达式。若产品需要字符串单元格则不适用。
3. 改调 `parse_expr(value, transformations=T, global_dict=ns, evaluate=True)` + AST 白名单预检。

**同时加维度上限**（`:136-154`）：例如 `if len(data) > 64 or any(len(r) > 64 for r in data)`。

### 7.3 `number_theory_tool.py`

给 `number` / `second_number` 加上限（`ge=-10**30, le=10**30` 之类，pydantic `Field(..., ge=..., le=...)`）。**注意**：`sympy.factorint` 对 160-bit（≈48 位十进制）半素数就已超过 25 秒，所以上限必须远低于此。

### 7.4 `statistics_tool.py`

1. `_fmt`（`:119-122`）先判 `math.isfinite(value)`，非有限值返回 `"inf"`/`"-inf"`/`"nan"`。
2. `compute_statistics`（`:51-84`）整体包 `try/except` 返回字符串，与同目录其余工具对齐。
3. `_parse_numbers` 丢弃 token 时应返回丢弃计数并在结果中提示。

### 7.5 `datetime_tool.py:178`

`total` 应直接用 `days` 而不是 `abs(delta.days)`；或干脆去掉 `(total N days)` 后缀。

### 7.6 `currency.py` / `_http.py`

- `currency.py:67-68` 加 `re.fullmatch(r"[A-Z]{3}", source)` 校验。
- `_http.py:26` 加 `allow_redirects=False`；`response.json()` 前检查 `Content-Length` / 流式读取设上限。

### 7.7 开关（当前状态是减分项）

- `src/config/settings.py:144-147` 与 `config/default.yaml:95-98`：**`math_enabled` / `statistics_enabled` / `linalg_enabled` / `number_theory_enabled` 默认全为 `false`** —— 这是一个真实的缓解因素。
- 但 `.env.example:217,219-221` **建议运维者设为 `true`**：`MATH_ENABLED=true`、`STATISTICS_ENABLED=true`、`LINALG_ENABLED=true`、`NUMBER_THEORY_ENABLED=true`。
- `config/production.yaml` 只有 7 行（`api_key`、`cors_allow_origins`、`web_search_enabled`、`web_search_verify_ssl`），**不覆盖这 4 个开关** → 生产环境默认关闭，但一旦有人按 `.env.example` 配置就全开。
- `src/backend/mcp/policy.py:206-207`：`allow_risk_levels` 默认含 `"execute"`，且 `:227-229` 对 execute 风险**无任何额外门禁**。所以「标成 execute」不是防护。

**建议在修复完成前，把 `.env.example` 里这 4 行改成 `false` 并加注释。**

---

## 8. 本轮未验证 / 不确定

- **AST 白名单预检只验证了 11 个攻击载荷与 10 个合法表达式**，不是穷举。特别是「sympy 构造函数裸 sympify 字符串」这一家族，我只证实了 `nsimplify` 一例；`solveset`/`Poly`/`Function`/`Piecewise` 等是否同类未逐一验证。**AST 白名单里拒绝 `str` `Constant` 是防御这一整族的关键**，但它的有效性依赖于 sympy 的重解析只发生在字符串实参上这一假设。
- `python -I` 隔离模式下确认 venv 装的是 1.14.0；**未验证 `pip install -r` 在 CI 里的实际解析结果**（是否有 lockfile、是否走 mirror）。
- `stock.py` 的 `yf.Ticker` 路径穿越实际危害**未实测**（需要联网 + yfinance 版本依赖）。
- `weather.py` 的 `_format_weather`（`:166-203`）在 `:112-126` try 块之外，若响应体结构异常会否崩溃 —— 未实测。
- `nsimplify` 绕过的**根本原因已定位**（`simplify.py:1451` 的裸 `sympify`），但**未验证 sympy 上游是否有针对该路径的修复计划**。
- `doctest('x')` 在 sympy 1.14.0 下为进程内执行；历史结论「fork 子进程」在别的版本（1.13.x?）是否成立未逐一验证。
