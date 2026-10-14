# A5 — Graph 引擎 + 应用层 + LLM 层（full-sweep-13）

> 原分区子代理死于 429 限流（仅留骨架）。本报告由**主代理自救完成**。
> ⚠️ 本报告的**并发/配额部分为完整深挖**（`chat_service.py`、`quotas.py` 全文读完）；
> **图拓扑、状态机、提示词、metrics、sanitize 部分未覆盖**，见 §6。

---

## 1. 🔒 核心发现：一次卡住的 turn 可以占死所有并发槽

### 1.1 锁与配额的取得顺序

`src/backend/application/chat_service.py:60-75`：

```python
async def complete_turn(self, request: TurnRequest) -> TurnResult:
    session = self._lifecycle.require_session(request.thread_id, request.principal)
    charge = _turn_charge(...)
    with self._quotas.acquire(request.principal, charge):      # ← ① 先拿配额租约
        await _acquire_lock(session.turn_lock)                 # ← ② 再等 turn 锁
        try:
            session = await self._lifecycle.refresh(session, request.message)
            return await self._turns.execute_locked(session, request)
        finally:
            session.turn_lock.release()
```

**配额租约在 turn 锁之前取得**，且租约在 `with` 块内（含锁等待）全程持有。

### 1.2 租约为什么拿不到就不可能松开

`quotas.py:204-232`：`acquire` 在单一 `threading.Lock` 下做全部判定，通过时 `:227-231` 递增 `_active_principals` / `_active_tenants` 并返回 `QuotaLease`。计数**只在 `_release`（`:255-263`）里递减**，即租约被释放时。所以 `await _acquire_lock(...)` 阻塞的每一毫秒，这个 principal 的并发槽都在被占用。

### 1.3 `_acquire_lock` 无超时，且取消也无法脱身

`chat_service.py:146-161`：

```python
class Lock(Protocol):
    """Minimal blocking lock contract used by chat sessions."""
    def acquire(self) -> bool: ...
    def release(self) -> None: ...


async def _acquire_lock(lock: Lock) -> None:
    worker = asyncio.create_task(asyncio.to_thread(lock.acquire))
    try:
        await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker          # ← 关键
        lock.release()
        raise
```

三个问题叠加：

1. **`Lock.acquire` 契约里没有 timeout 参数**（`:149`），只有 `acquire() -> bool`。`asyncio.to_thread(lock.acquire)` 在 worker 线程里跑一个**无限期阻塞**的调用，`to_thread` 本身也没有超时。
2. **`asyncio.shield(worker)` 使得 worker 不受外层取消影响**——设计意图是让锁获取不被中途掐断，但这同时意味着取消信号无法中断它。
3. **`except CancelledError: await worker`** —— 请求被取消时，代码**继续 `await` 那个 worker 直到它拿到锁**，然后 `release` 并 re-raise。也就是说：

   > **取消一个请求，不能放弃一次卡住的锁获取；它必须先赢到这把锁。**

### 1.4 组合后果：单点饥饿 → 全局拒绝

设 principal 并发上限 `concurrent_calls=4`（`quotas.py:91-95` 的默认 `QuotaBudget`）：

1. 会话 A 的 `turn_lock` 被某个卡住的 turn 持有（LLM 调用挂死、外部工具不返回等）。
2. 之后任何针对会话 A 的请求，都在第 2 步 `await _acquire_lock` 上阻塞，**同时占着一个配额槽**（§1.2）。
3. 三个同类请求堆积后，该 principal 的 4 个槽全部被"卡在锁上等一个永远不会结束的 turn"占满。
4. 第 4 个及之后的请求在 `quotas.py:213-217` 命中：
   ```python
   if self._active_principals.get(principal_key, 0) >= self._limits.principal.concurrent_calls:
       raise QuotaExceededError()
   ```
   **连一个都没拿到锁，就已经被配额拒绝了。**
5. 由于并发槽是按 `(tenant_id, principal_id)` 计数的（`quotas.py:202`），**同租户下其它无关会话的请求也一起饿死**。

`QuotaExceededError` 有 **5 个抛出点**（`quotas.py:210, 212, 217, 222, 250`），请求方无法区分"我速率超限了"和"我前面有人在等一把死锁"——两种情况返回同一个错误。

### 1.5 流式路径：同样的顺序，更差的失败语义

`chat_service.py:77-89`：

```python
def stream_turn(self, request: TurnRequest) -> AsyncIterator[GraphEvent]:
    """Authorize before transport start and retain quotas for the stream lifetime."""

    session = self._lifecycle.require_session(request.thread_id, request.principal)
    lease = self._quotas.acquire(...)          # ← 同步调用，立刻拿租约
    return self._stream_turn(session, request, lease)
```

注意两点：

- **`stream_turn` 是 `def` 不是 `async def`** —— 它在被调用时就同步拿租约，然后返回一个 async generator 对象。租约的获取发生在**生成器主体执行之前**。
- 租约的释放只在 `_stream_turn` 的 `finally` 里（`:106-113`：`lease.release()`）。而 `_stream_turn` 是 async generator（`:105` 有 `yield`），它的 `finally` **只在生成器被耗尽或被关闭时才运行**。

⚠️ **如果调用方创建了生成器但既不迭代也不 `aclose()`，租约会一直持有到 GC 触发 `athrow(GeneratorExit)`。** 客户端断连时若清理路径漏了 `aclose()`，就是一个租约泄漏。叠加"整个流生命周期持有并发槽"的设计（注释 `:78` 明确写了这是有意为之），一个长流响应会持续占住一个槽；一旦卡住，就变成 §1.4 的占位者。

`_stream_turn:106-113` 的清理顺序是对的（先 `aclose()` events，再 `lock.release()`，最后 `lease.release()`），但仅在正常关闭路径成立。

### 1.6 三个"零成本"操作也占并发槽

`chat_service.py:115-143`：

```python
async def history(self, thread_id, principal=None, *, purge_unsent_attachments=False):
    ...
    with self._quotas.acquire(trusted_principal):       # ← 无 charge
        return await self._lifecycle.history(...)

async def delete(self, thread_id, principal=None):
    ...
    with self._quotas.acquire(trusted_principal):       # ← 无 charge
        return await self._lifecycle.delete(...)
```

`start`（`:57`）传了 charge，`history` 和 `delete` **完全不传 charge**（`acquire` 的 `charge` 默认 `QuotaCharge()`）。后果：`concurrent_calls` 是**共享池**，`complete_turn` / `start` / `history` / `delete` 竞争同一组槽。一次批量导出历史的循环就能把 `complete_turn` 挤出并发窗口——这是与 §1.4 叠加的第二条饥饿路径。

---

## 2. 🧮 配额与成本公式

### 2.1 token 估算（`quotas.py:266-271`）

```python
def estimate_token_units(text: str | None) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
```

字符数 ÷ 4，向上取整，地板 1。空串 → 0（不是 1）。这是**确定性保守估算**，不区分中英文——对中文，`len(text)/4` 会**严重高估**（中文一字约 1-2 token，而这里按 0.25 token/字符计）。后果是中文输入的配额消耗被放大 4-8 倍，中文用户更容易触发 `tokens_per_minute` 限制（默认 principal 250,000 / tenant 1,000,000，`:91-106`）。**这是一个正确性问题，不是安全问题。**

### 2.2 成本折算（`quotas.py:274-281`）

```python
def estimated_cost_units(charge: QuotaCharge) -> int:
    return max(
        1,
        charge.cost_units,
        charge.tokens + charge.searches * 100 + charge.tool_calls * 10 + charge.retries * 5,
    )
```

线性加权求和，权重：token 1、search **100**、tool_call **10**、retry **5**。三个 `max` 分支取最大，地板 1。

⚠️ **一次 web 搜索的配额成本 = 100 个 token。** 而 `token` 估算本身是按字符/4 计的，所以一次搜索 ≈ 400 字符的中文文本配额。这个 100:1 的权重比是设计选择，但配合 §2.1 的中文高估，中文用户的搜索预算消耗得很快。

### 2.3 成本如何进入 turn charge

`chat_service.py:164-191`：`_turn_charge` 固定 `tool_calls=1`，`searches` 来自 `session_lifecycle.searches_for_turn`，`retries` 来自构造时的 `retry_reservation`（默认 0）。`_operation_charge` 用 `dataclasses.replace` 回填 `cost_units`（`:191`）。

---

## 3. ⚠️ 配额追踪字典可被唯一 principal 洪水投毒

`quotas.py:234-253`：

```python
def _window(self, windows, key, now) -> _UsageWindow:
    window = windows.get(key)
    if window is not None:
        window.prune(now)
        return window
    if len(windows) >= self._limits.max_tracked_identities:
        for candidate, existing in tuple(windows.items()):
            existing.prune(now)
            if not existing.events:
                del windows[candidate]
    if len(windows) >= self._limits.max_tracked_identities:
        raise QuotaExceededError()
    window = _UsageWindow()
    windows[key] = window
    return window
```

⚠️ **无噪声邻居（noisy neighbor）投毒**：驱逐循环只删除 **prune 后为空的窗口**（`:247-248`）。一个只要**有一个最近事件**的窗口就永远不会被驱逐。

攻击/故障模式：用大量**互不相同**的 `principal_id` 发请求，每个只发一次。字典被填满到 `max_tracked_identities`；驱逐循环发现每个窗口都非空（每个都有那一个事件），无人可删；随后 `:249-250` 抛 `QuotaExceededError`。

**而且这个状态是持久的**：只要那些一次性 principal 的窗口还没过期（`prune` 没清空它们），**后续所有请求——包括那些原本有预算的、字典里已存在的身份的请求——只要触发 `_window` 走到"新建窗口"分支，就全部被拒**。

严格地说：已存在于字典里的身份在 `:240-243` 命中缓存直接返回，不受影响。但**任何新身份都会被永久拒绝**，直到旧窗口的全部事件滑出窗口期。这比"拒绝攻击者"更糟——它把整个租户的新身份准入关掉了。

---

## 4. 配额限流判定的完整顺序

`quotas.py:200-232`，全部在单一 `threading.Lock` 下：

```
① principal 速率窗口 accepts?        → QuotaExceeded   (:209-210)
② tenant 速率窗口 accepts?           → QuotaExceeded   (:211-212)
③ principal 并发数 < 上限?           → QuotaExceeded   (:213-217)
④ tenant 并发数 < 上限?              → QuotaExceeded   (:218-222)
⑤ (窗口新建时) 追踪字典未超上限?      → QuotaExceeded   (:249-250)
⑥ 通过 → append 事件 + 递增并发计数 → 返回 QuotaLease  (:224-232)
```

**速率窗口先判，并发数后判。** 注意 `tenant_key is None` 时步骤 ②④ 整体跳过（`:207`、`:211`、`:218-220`），所以本地进程 principal（`Principal.local_process()`，见 `chat_service.py:124`、`:140`）只受 principal 级限制约束。

`_decrement`（`:289-294`）在计数归零时 `pop` 键——干净，无泄漏。

`QuotaLimits.__post_init__`（`:66-70`）用 `_bounded_positive`（`:284-286`）校验，含 `isinstance(value, bool)` 显式排除（bool 是 int 的子类）——**这是本文件做得细致的一处**。

---

## 5. 发现汇总

| # | 发现 | 位置 | 严重度 |
|---|---|---|---|
| 1 | 配额租约在 turn 锁之前取得，锁等待期间槽位被占 | `chat_service.py:69-70` + `quotas.py:227-231` | **高** |
| 2 | `Lock.acquire` 契约无 timeout，`asyncio.to_thread` 无限期阻塞 | `chat_service.py:146-155` | **高** |
| 3 | 取消时 `await worker`——必须先赢到锁才能脱身，取消无法放弃卡住的锁获取 | `chat_service.py:158-161` | **高** |
| 4 | 组合后果：1 个卡住的 turn 可占满 principal 的 4 个槽，同租户无关请求全部 `QuotaExceeded` | 综合 §1.4 | **高** |
| 5 | 流式路径同顺序；`stream_turn` 是同步 `def`，生成器未被 `aclose()` 时租约泄漏到 GC | `chat_service.py:77-89`、`:106-113` | 高 |
| 6 | `history` / `delete` 占并发槽但不传 charge，与 turn 竞争同一池 | `chat_service.py:115-143` | 中 |
| 7 | `QuotaExceededError` 5 个抛出点，调用方无法区分速率超限与锁饥饿 | `quotas.py:210,212,217,222,250` | 中（可观测性） |
| 8 | 唯一 principal 洪水可投毒追踪字典，驱逐只删空窗口 → 租户内新身份被持久拒绝 | `quotas.py:234-253` | **高** |
| 9 | `len(text)/4` token 估算对中文高估 4-8 倍，中文用户更易触限 | `quotas.py:266-271` | 中 |
| 10 | 一次搜索 = 100 token 配额，与中文高估叠加 | `quotas.py:274-281` | 低（设计） |
| 11 | `_bounded_positive` 显式排除 bool，是本文件做得细致的一处 | `quotas.py:284-286` | 正面 |

---

## 6. ⚠️ 未覆盖（本分区任务未完成，明确列出）

以下**均未读**，避免被误读为已验证：

- **图拓扑**：`graph/builder.py`、`graph/edges.py`、`graph/state.py`、`graph/executor.py`、`graph/artifacts.py`、`graph/events.py`、`graph/metrics.py` —— **节点/边/条件边/循环全部未重建**。
- **图节点**：`graph/nodes/` 下 11 个（`common`、`planning`、`decompose`、`expand`、`merge`、`condense`、`search_queries`、`web_answer`、`fallback_answer`）—— **未读**。
- **状态机字段表**：`state.py` 的字段/reducer/写入者 —— **未读**。
- **提示词普查**：`llm/prompts.py`、各节点内嵌 system/user prompt、注入守卫措辞、引用格式 `--- Source:` —— **未读**（A4 提到 `prompt_builder.py` 侧，未覆盖 LLM 侧）。
- **`llm/provider.py`**、**`llm/sanitize.py`**（回答净化规则）—— **未读**。
- **`metrics.py`** 的计时器/计数器/阈值 —— **未读**。
- **`application/turn_execution.py`**、**`session_lifecycle.py`**、**`models.py`**、**`errors.py`** —— **未读**。
- `session.turn_lock` 的**实际类型未确认**：`Lock` Protocol（`:146-151`）只声明 `acquire`/`release`。全仓库 `turn_lock` 的定义点未定位（grep 命中的是 `asyncio.Lock` 用于上传去重与 graph factory，非 turn_lock）。因此 §1.3 的"无限期阻塞"结论基于 **Protocol 无 timeout 参数 + `to_thread` 无超时** 这一组合，而非具体锁实现。**这是本报告最需要实测确认的一环。**

### 交叉验证

- 本报告 §1 的锁/配额顺序与既有记忆（quota acquired before turn lock）**一致，且更完整**：记忆未记录 `stream_turn` 是同步 `def`、未记录 5 个 `QuotaExceededError` 抛出点、未记录 `_window` 的噪声邻居投毒。
- `quotas.py:271` 的 `math.ceil(len(text)/4)` token 估算公式，与 A1 子代理独立报告的 `quotas.py:271` 一致。
