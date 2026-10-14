# A6 — 长期记忆子系统 + 会话/存储 + 安全与配额

> 探索分区：`src/backend/memory/`（11 文件）+ `src/backend/sessions/`（7 文件）+ `src/backend/security/`（3 文件）
> 对照规格：`.kiro/specs/chat-long-term-memory/{requirements,design}.md`、`.kiro/specs/memory-auto-extraction/{requirements,design}.md`
> 纪律：只信代码，不信注释；每个结论定位到 `文件:行号`。未触碰 `.chroma/`、`chat_uploads/`、真实 SQLite 库、`.env`。
> 附加旁读（仅用于验证跨文件结论，未做审查）：`src/frontend/chat/memory_hooks.py`、`src/backend/application/chat_service.py`、`src/backend/application/turn_execution.py`、`src/config/{settings,loader}.py`、`src/backend/llm/prompts.py`、`src/backend/tools/memory_tool.py`

---

## 0. 结论摘要（先看这个）

| # | 结论 | 定位 | 严重度 |
|---|---|---|---|
| 1 | **⚠️ 修正此前记忆**：`read()` **确实会改写磁盘**。JSON 语法损坏时 `_read_locked` 调用 `_quarantine` 执行 `os.replace` 把原文件**移动**走 | `store.py:264-266` → `283-313` | 高（此前笔记写错了） |
| 2 | **schema 版本不匹配 = 静默全量数据丢失**：`parse_document` 返回 `MemoryDocument.empty()`，**不隔离、不备份**；下一次 `save()` 以 `version=1` 原地覆盖 | `serialization.py:74-80` + `store.py:413-436` | 高 |
| 3 | `recall()` 是**写操作**：给 `last_recalled_at` 打戳后整文档重写 | `store.py:617-625` | 中 |
| 4 | **相关性打分没有任何权重/衰减**：就是 distinct term 的子串命中计数 | `relevance.py:50-58` | 中 |
| 5 | **打分可被内容自我放大**：命中子串即得 1 分；**tags 参与打分但从不渲染给 LLM** | `relevance.py:42-58` vs `recall.py:47-54` | 高 |
| 6 | **记忆内容零转义注入 SystemMessage**：`{record.content}` 原样拼进 prompt | `recall.py:47-54`, `113` | 高 |
| 7 | **`watermark.py` 不是加密水印**，是回合计数簿记，无校验和、无签名 | `watermark.py:15-71` | 澄清 |
| 8 | **调度没有退避/抖动/重试**：`BoundedSemaphore` + `acquire(blocking=False)`，忙即丢弃 | `scheduler.py:32,53-61` | 中 |
| 9 | **`QuotaManager.acquire` 无 timeout、无 shield、fail-fast；`QuotaLease` 无 TTL** | `quotas.py:193-232`, `145-169` | 中 |
| 10 | **SQLite 全参数化，零拼接**；但迁移只补 2 列，其余 7 列缺失即 `ProgrammingError` | `sqlite.py:134-148` vs `151-163` | 高 |
| 11 | **`SQLiteMemorySaver` 四处 `pickle.loads` DB 内容**，启动即执行 | `checkpoint.py:170,195,285,376` | 高 |
| 12 | `thread_id` **零校验**直插 Chroma 路径 | `registry.py:127` + `isolation.py:21-27` | 高 |
| 13 | 密钥屏检只有 5 条正则；AWS **secret**、Stripe `sk_live_`、GitHub `ghp_`、JWT、DSN 全部漏检 | `secrets.py:14-38` | 高 |
| 14 | 屏检只在 `save()` 跑一次；解析路径从不复检 | `store.py:472-479` vs `serialization.py:187-192` | 中 |
| 15 | 规格自相矛盾：design Property 12 要求"返回给 agent 的串不得含 Memory_Record content"，与 R3.2 直接冲突 | `design.md:732-737` vs `requirements.md:184-190` | 文档缺陷 |
| 16 | `.gitignore` **没有** `.corrupt-*` 规则，只有 `memory/` | `.gitignore:30` vs `design.md:60` | 中 |

---

## 1. 逐文件说明

### 1.1 `src/backend/memory/`

#### `models.py`（130 行）— 纯值类型 + 共享常量
- `CATEGORIES = ("fact","preference","entity","task")`、`SCOPES = ("global","session")`（18-19 行）。设计文档写的是 `frozenset`（`design.md:99-100`），实现是 tuple —— **类型漂移**，无行为影响。
- `SCHEMA_VERSION: Final[int] = 1`（26 行）。注释称"bumped only when the on-disk field set changes incompatibly"（24-25 行）。
- `DEFAULT_STORE_PATH = "memory/long_term_memory.json"`（22 行）。
- 上限表：`MAX_TAGS=10`(28)、`MAX_TAG_CHARS=40`(29)、`MAX_QUERY_CHARS=500`(30)、`MAX_QUERY_TERMS=50`(31)、`MIN_TERM_CHARS=2`(32)、`MAX_FORGET_DELETES=10`(33)、`MAX_SCOPE_ID_CHARS=200`(34)、`MAX_TOOL_CALLS_PER_TURN=10`(35)。
- **注意 `MAX_SCOPE_ID_CHARS = 200` 定义了但全仓库无引用**：`save()` 从未校验 `scope_id` 长度，`serialization._parse_record` 也不校验。R5.2 要求 `scope_id` 为 1–200 字符，**这个上限在两条路径上都未实现**。
- `EXTRACTION_TAG_PREFIX = "auto:"`（39 行）。
- `ID_PATTERN = re.compile(r"\A[0-9a-f]{32}\Z")`（42 行）。
- `MemoryRecord.recency` 属性：`return self.last_recalled_at or self.created_at`（84-92 行）。这是 R8.2"有效新近度"的实现；`""` 空串会短路到 `created_at`，符合"absent or empty"。
- `utc_now_iso`：`datetime.now(UTC).isoformat(timespec="seconds")`（59-62 行）。

#### `serialization.py`（241 行）— 序列化/解析对
- `serialize_document`（39-47 行）：`json.dumps(payload, ensure_ascii=False, indent=2) + "\n"`。三个顶层键，字段顺序由 `RECORD_FIELDS` 固定。
- `parse_document`（50-103 行）：
  - `json.loads` 失败或顶层非 dict → `raise CorruptDocumentError`（64-72 行）。**这是唯一会触发隔离的路径。**
  - `version` 不可读 → **返回空文档 + warning，不抛异常、不隔离**（74-80 行）。
  - `records` 非 list → warning + 置空（86-89 行）。
  - 逐条 `_parse_record`（91-96 行）。
- `_is_readable_version`（114-117 行）：`1 <= int(version) <= SCHEMA_VERSION`。拒绝 `bool`（`True` 是 `int` 子类，需显式挡）。**注意比 R5.7 更严**：规格只拒绝 `> 1`，实现连 `< 1` 也拒绝。
- `_parse_record`（120-198 行）：必需字段 `id/content/scope` 缺失/空/非字符串 → 整条丢弃并 warning（126-140 行）；`scope` 越界 → 丢弃（138-140）；`scope == "global"` 时任何非 None 的 `scope_id` 被丢弃并 warning（145-150）；`category` 越界 → 回落 `fact`（157-164）；`tags`/时间戳错型 → 替换为默认（166-185）。
- **关键：`content=values["content"]` 原样保留，无长度校验、无密钥复检**（192 行）。R5.2 的 `content` 1–`memory_max_record_chars` 约束在**解析路径上完全不存在**。

#### `relevance.py`（109 行）— 相关性打分
- 见 §3.1。无 I/O、无 Settings。

#### `store.py`（911 行）— 持久化核心
- 错误层级：`MemoryStoreError` → `MemoryPathError` / `MemoryWriteError`（57-67 行）。
- 三种 Outcome dataclass：`SaveOutcome`、`RecallOutcome`、`ForgetOutcome`（69-100 行）。
- `_REPLACE_BACKOFF = (0.05, 0.10, 0.20)`，`_REPLACE_ATTEMPTS = 4`（106-107 行）。
- 路径解析 `resolve_store_path`（116-135 行）：拦 `..` 段，相对路径按 CWD 解析，`os.path.normpath`。**不做符号链接解析**（`normpath` 不 resolve）。
- 作用域过滤 `in_scope_records`（138-155 行）：global 全可见；session 需 `scope_id == resolved` 精确匹配；`scope_id` 为空的 session 记录**任何地方都不可见**。
- 锁：`_LOCKS: dict[Path, threading.RLock]` 按 resolved path 键控（109-110, 854-862 行）。进程内锁，**跨进程无互斥**（与 R6.4 一致，规格明确 out of scope）。
- 读：`read()` → `_read_locked(force=False)`（214-281 行）。`(mtime_ns, size)` 签名缓存，每次 `stat`。
- 隔离 `_quarantine`（283-313 行）：`os.replace(self._path, target)`，目标名 `<name>.corrupt-<utc 去冒号>`，撞名加 `-1/-2` 后缀。
- 写：`_write_document`（338-384 行）：`mkdir(parents=True)` → `tempfile.mkstemp(dir=parent, prefix=name+".", suffix=".tmp")` → `write/flush/fsync` → `_replace_with_retry` → 失败时 `finally` 里 `unlink(missing_ok=True)` 清理 tmp。
- `_replace_with_retry`（386-398 行）：仅对 `PermissionError` 重试，其他 `OSError` 直接抛。
- `save()`（413-533 行）：校验固定顺序 content 存在 → 长度 → category → scope → tag 数量 → tag 长度（430-470）；然后密钥屏检（472-479）；然后锁内去重/插入（481-525）。
- `_prune_for_insert`（535-562 行）：全库驱逐，`min` by `(timestamp_epoch(records[i].recency), id)`。
- `recall()`（569-632 行）：**写 `last_recalled_at`**。
- `forget()`（636-661）→ `_forget_by_id`（663-701）/ `_forget_by_query`（703-748）。id 优先，同时传 query 则忽略 query（645-660 行）。
- `purge_session`（755-779 行）：只删 `scope=="session" and scope_id==resolved`。
- `_safe_count`（781-785 行）：吞掉 `MemoryStoreError` 返回 0。
- 单例 `get_memory_store`（865-889 行）：每 path 一个实例，**已存在时 `update_limits` 刷新上限**。

#### `recall.py`（240 行）— 渲染 + 自动注入 + 预算
- `MEMORY_NOTE_LABEL = "LONG-TERM MEMORY (recalled):"`（32 行）。
- `MEMORY_NOTE_MARKER = "​​"`（38 行）——两个零宽空格。
- `_MAX_TRACKED_THREADS = 1024`（44 行）。
- 见 §3.1 与 §9。

#### `transcript.py`（206 行）— 纯转录处理
- `MAX_SLICE_MESSAGES = 200`（23 行）、`TRUNCATION_MARKER = " ... [truncated]"`（19 行）。
- `message_role`（44-57 行）：`human*`/`user`→user；`ai*`/`assistant`→assistant；`tool`/`function`→tool；`system*`→system；否则原 kind。
- `normalize_message_content`（60-83 行）：str → strip；list/tuple → 逐元素取文本空格连接；dict → `_element_text`；**其他类型 `str(content).strip()`（83 行）** —— 会把非字符串内容 `repr` 化后进入 prompt。
- `_element_text`（86-96 行）：只认 `text`/`content`/`value` 三个键 + `.text` 属性，其余丢弃。注释称这防止 tool-call 块泄漏（63-67 行）；实际是静默丢弃，无 warning。
- `count_turns`（99-102 行）：数 user 消息数 = Turn_Count。
- `is_memory_note`（105-116 行）：只看 `content[:200]` 是否含 marker 或 label 前缀。
- `select_slice`（119-158 行）：找第 `watermark+1` 个 user 消息的位置作为起点；`watermark <= 0` 时 `start = 0`（145-146 行）。
- `render_slice`（161-192 行）：先 `[-200:]`，再逐块从**开头**丢弃直到 ≤ `limit`。
- 注：`_NOTE_MARKER_CACHE`（28-41 行）惰性导入 `.recall` 常量以避循环，**首次调用时才绑定一次**，之后不更新。

#### `extraction.py`（589 行）
- `MemoryExtractor.run`（296-428 行）是主流程；`_CheckpointReadError`（74-75 行）区分"空线程"与"读失败"。
- `_read_thread_messages`（98-124 行）：`checkpointer.get_tuple({"configurable":{"thread_id"}})` → `checkpoint["channel_values"]["messages"]`。**不需要编译好的 graph**，符合 R3.6。
- `_call_model`（213-247 行）：在 daemon 线程里跑一次 `model.invoke(prompt)`，`completed.wait(timeout=settings.memory_extraction_timeout_seconds)`。超时返回 `timed_out=True`，**但不取消该线程**（daemon，泄漏到进程退出）。
- `_persist`（249-288 行）：逐个候选 `store.save(...)`，异常只记首个 `error_type` 并 `continue`。
- `_parse_candidate_payload`（150-206 行）：`json.loads` 失败 → `_first_array` 提第一个数组；`parsed[:100]`；先按 normalized 内容去重（175-178）再截 `max_candidates`。
- `_first_array`（521-555 行）：手写括号深度扫描，识别引号/转义。
- `_provenance_tag`（579-581 行）：`"auto:" + trigger[:35]`。
- `_with_provenance`（584-586 行）：剔除同名 provenance，保留前 9 个 tag，追加 provenance。**不对 tag 做 trim 或长度校验**。
- 见 §3.3 与 §5。

#### `scheduler.py`（117 行）
- 见 §3.3。

#### `secrets.py`（78 行）
- 见 §4.1。

#### `watermark.py`（71 行）
- 见 §3.2。

#### `__init__.py`（160 行）
- 纯 re-export，`__all__` 含 66 个符号。无逻辑。

### 1.2 `src/backend/sessions/`

#### `models.py`（71 行）
- `ChatSession` dataclass（16-54 行）：`graph`、`settings`、`owner`、`source_urls`、`source_mode`、`created_at`（默认 `time.time`）、`last_accessed_at`、`last_message_at`、`isolated_chroma`、`title`、`announced_uploads`、`pending_turn_attachments`、`extraction_watermark: int = 0`（41 行）、`turn_lock: threading.Lock`（43-47 行，`compare=False, repr=False`）。
- `touch(now=None)`（53-54 行）。
- `SessionSummary`（57-72 行）frozen，刻意排除 `graph/settings/turn_lock`，可在 registry 锁下安全构造。

#### `storage.py`（134 行）
- `SessionMetadata`（19-60 行）frozen：`thread_id, owner, source_urls, source_mode, created_at, last_accessed_at, chroma_dir, isolated_chroma, config`。
- `SessionMetadata.from_session`（37-60 行）：`config` 组装 `collection_name`、`extraction_watermark`、可选 `session_title`、可选 `session_last_message_at`。**每次保存都重建 config**（这正是 auto-extraction 规格 R7.3 担心的点，已通过把 watermark 加进 `config` 解决）。
- `StorageBackend` Protocol（63-79 行）。
- `InMemoryStorage`（82-118 行）：`threading.Lock`，全部方法加锁。
- `_copy_metadata`（121-126 行）：`source_urls` 和 `config` 只**浅拷贝**。config 的嵌套可变值仍共享。

#### `sqlite.py`（212 行）
- 见 §4.2。

#### `checkpoint.py`（405 行）
- 见 §4.2 与 §6。

#### `isolation.py`（33 行）
- `settings_for_session`（9-27 行）：
  ```python
  chroma_dir=base_chroma_dir / "chat" / thread_id,          # 25 行
  collection_name=f"{base.collection_name}-chat-{thread_id}",  # 26 行
  ```
  `thread_id` 未经任何校验直接进路径与集合名。
- `_settings_for_session = settings_for_session`（30 行）：私有别名，`frontend/chat/api.py:291` 与 `sessions.py` 用别名，`application/session_lifecycle.py` 用公开名 —— 两个名字并存。

#### `registry.py`（455 行）
- `ISOLATED_SOURCE_MODES = {"explicit", "web_search"}`（24 行）。
- `TOUCH_PERSIST_INTERVAL_SECONDS = 30.0`（30 行）。
- `_safe_isolated_chroma_dir`（33-39 行）：`resolved.parent.name.lower() == "chat"` 且 name 非空。**只在删除时校验**。
- `_release_chroma_system` / `_remove_chroma_dir`（42-67 行）：失败静默降级到 `shutil.rmtree`。
- `cleanup_isolated_chroma`（70-87 行）：`isolated_chroma` 为假直接返回；路径不合法只 warning 不删。
- `ChatSessionRegistry`（90-438 行）：`ttl_seconds=3600`、`cleanup_interval=300`（95-96 行）。
- `create`（117-152 行）：`thread_id = thread_id or uuid.uuid4().hex`（127 行）——**接受任意调用方提供的 thread_id，无校验**；`self._sessions[thread_id] = session`（143 行）**无条件覆盖**。
- `restore`（154-179 行）：同样无条件覆盖（177 行）；`extraction_watermark=coerce_watermark(...)`（172 行）。
- `get`（181-190 行）：**无所有权校验**，只有 `get_owned`（192-216 行）校验。
- `cleanup_expired`（345-365 行）：`now - last_accessed > ttl`（严格大于）。
- `_cleanup_loop`（389-392 行）：`interval = max(0.1, float(self._cleanup_interval))`，`while not self._stop_cleanup.wait(interval)`。
- `_cleanup_thread` 读写不对称：`start_background_cleanup` 在锁内检查（371-381），`stop_background_cleanup` **锁外读**（383-387）→ 竞态。
- `_save_metadata`（419-426 行）：`try/except Exception: logger.warning` —— **持久化失败完全静默**，内存与磁盘会漂移。

### 1.3 `src/backend/security/`

#### `identity.py`（92 行）
- `_MAX_IDENTITY_CHARS = 128`（8 行）。
- `Principal`（11-35 行）：`__post_init__` 用 `object.__setattr__` 覆写为规范化值（19-29 行）；`local_process()` 返回 `principal_id="local-process"`（31-35 行）。
- `ResourceOwner.authorizes`（64-72 行）：`hmac.compare_digest` 常量时间比较（75-76 行）；tenant 双 None 才算匹配（68-69 行）。
- `_validated_identity`（79-89 行）：strip；拒空、>128、含 `ord < 32` 或 `ord == 127`。**允许 128 个任意 Unicode 字符**。

#### `quotas.py`（305 行）
- 见 §3.4。

#### `__init__.py`（24 行）
- 纯 re-export。

### 1.4 规格文档

| 文件 | 行数 | 内容 |
|---|---|---|
| `chat-long-term-memory/requirements.md` | 682 | 12 个 Requirement，R1–R12 |
| `chat-long-term-memory/design.md` | 839 | 组件接口、12 条 Correctness Property、测试策略 |
| `memory-auto-extraction/requirements.md` | 709 | 11 个 Requirement，R1–R11 |
| `memory-auto-extraction/design.md` | 620 | **未读取**（见 §7） |

---

## 2. 💾 存储与升级安全性

### 2.1 schema 演进路径

`SCHEMA_VERSION` 目前为 `1`（`models.py:26`）。演进机制**只有一条**：

```python
# serialization.py:114-117
def _is_readable_version(version: Any) -> bool:
    if isinstance(version, bool) or not isinstance(version, int):
        return False
    return 1 <= int(version) <= SCHEMA_VERSION
```

即：**只接受等于 `SCHEMA_VERSION` 的版本**。没有迁移函数、没有 `MIGRATIONS` 表、没有 `try_downgrade`。规格 `design.md:24-25` 的注释与 `requirements.md:305-308`(R5.7) 都把不可读版本定义为"直接当空库"。

### 2.2 升级时文件是移动还是原地覆盖？

**分两种情况，行为完全不同：**

**(a) JSON 语法损坏 / 顶层非对象 → 移动（`os.replace`），有备份**

```python
# store.py:260-266  —— 读路径里就会触发
try:
    document, warnings = parse_document(raw, fallback_updated_at=self._clock())
except CorruptDocumentError as exc:
    self._quarantine(exc)          # ← 移文件
    return ()
```
```python
# store.py:283-313
def _quarantine(self, reason: Exception) -> None:
    self._cache = None
    stamp = self._clock().replace(":", "")
    target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
    suffix = 1
    while target.exists():
        target = self._path.with_name(f"{self._path.name}.corrupt-{stamp}-{suffix}")
        suffix += 1
    try:
        os.replace(self._path, target)        # ← 原文件被移走，字节完整保留
    except OSError as exc:
        logger.warning(...); return
    logger.warning("memory store was unreadable (%s); moved it to %s ...", reason, target, ...)
```

**(b) schema 版本不可读 → 原地覆盖，零备份，静默丢失**

```python
# serialization.py:74-80
version = payload.get("version")
if not _is_readable_version(version):
    warnings.append(f"memory store version {version!r} is not readable by schema "
                    f"version {SCHEMA_VERSION}; ignoring all records")
    return MemoryDocument.empty(fallback_updated_at), warnings   # ← 空文档，version=1
```
```python
# store.py:323-336 —— 下一次 save() 用它写盘
document = MemoryDocument(version=SCHEMA_VERSION, updated_at=self._clock(), records=ordered)
```

结果：旧版本文件里的全部记录在**下一次任何写操作**（`save` / `recall` / `forget` / `purge_session`，因为它们都走 `_read_locked(force=True)` → `_persist_locked`）被**原地覆盖为只含新记录的 v1 文档**，磁盘上不留任何旧字节。

### 2.3 读操作是否会改写磁盘？—— 会，两条路径

**路径 A：损坏 JSON 的 `read()` 会移走原文件。** 上面 §2.2(a) 已证。`read()` → `_read_locked` → `_quarantine` → `os.replace`。**这是 R5.6 的明确要求**（`requirements.md:300-304`："SHALL rename the file by appending `.corrupt-<UTC timestamp>`"），所以是规格要求的行为，不是 bug —— 但"读"会改变磁盘状态这个性质本身值得记。

**路径 B：`recall()` 必然整文档重写。**
```python
# store.py:617-625
stamp = self._clock()
recalled = {record.id for record in ranked}
updated = tuple(replace(record, last_recalled_at=stamp) if record.id in recalled else record
                for record in records)
self._persist_locked(updated)          # ← 写盘
```
R3.7 明确要求（`requirements.md:210-213`）。注意与自动注入的对比：`build_memory_note` 走 `store.in_scope()`（纯读），**不**打戳（`recall.py:104`，`recall.py:7-10` 注释与 R7.7 一致）。

**不写盘的正常读**：文件有效时 `read()` / `in_scope()` 只 `stat` + 命中缓存（`store.py:239-250`）。文件缺失时返回 `()` 且**不创建任何文件、不建目录**（`store.py:241-243`, `254-256`），符合 R5.9/R6.9。

**`_safe_count` 的隐患**：`store.py:781-785` 吞掉 `MemoryStoreError` 返回 0 —— 但**若磁盘是只读**，`stat` 成功、`read_text` 成功、`parse` 成功，会走到 `_quarantine` 的 `os.replace` **失败分支**（`store.py:298-306`）：日志告警，然后继续用空库，**而磁盘文件完好**。所以只读 FS 下不会丢数据，只会有"读得到但显示 0 条"。

### 2.4 备份策略

**只有 `_persist_locked` 有备份，且是向前备份**：先写完整新文档到 `.tmp`，再 `os.replace` 原子换入（`store.py:338-384`）。旧文档的备份只在 `os.replace` 成功瞬间消失（被覆盖）。**没有版本号备份、没有滚动快照、没有 `.bak`、没有 copy-on-write。** 500 条记录的上限意味着一次误 `forget` 或一次误 schema 降级即不可恢复。

原子性保障链：`tempfile.mkstemp(dir=parent)` → `write` → `flush` → `os.fsync(fileno)` → `os.replace`（`store.py:355-365`）。`fsync` 的是文件描述符，**没有 fsync 目录项**，所以断电下 `os.replace` 的元数据可能未落盘（POSIX 常见弱点）。`tmp_path` 在失败 `finally` 中 `unlink(missing_ok=True)` 清理（371-382），符合 R6.7。

`_replace_with_retry`：1 次 + 3 次重试，退避 `0.05/0.10/0.20` 秒，**总计 0.35 s**（`store.py:106-107`, `386-398`），符合 R6.8。只对 `PermissionError` 重试 —— `InterruptedError`、`OSError(EINTR)` 直接抛（391-398 行）。

---

## 3. 🧮 公式清单

### 3.1 相关性打分（`relevance.py`，109 行）

**这是全文件唯一的"公式"，而且极其简单。**

**归一化**（21-24 行）
```python
def normalize_content(text: str) -> str:
    return " ".join(str(text).casefold().split())
```
`casefold` + 空白折叠为一空格 + 去首尾。**不做 Unicode NFKC/NFKD，不去标点，不拆 CJK。**

**查询词派生**（27-39 行）
```python
def derive_query_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in normalize_content(query).split(" "):
        if len(token) < MIN_TERM_CHARS or token in seen:   # MIN_TERM_CHARS = 2
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= MAX_QUERY_TERMS:                  # MAX_QUERY_TERMS = 50
            break
    return tuple(terms)
```
按空格切，去重保序，取前 50 个长度 ≥2 的 token。

**haystack 构造**（42-47 行）
```python
def searchable_text(record: MemoryRecord) -> str:
    parts = [normalize_content(record.content)]
    parts.extend(normalize_content(tag) for tag in record.tags)   # ← tags 参与
    return " ".join(part for part in parts if part)
```

**打分**（50-58 行）
```python
def relevance_score(record: MemoryRecord, terms: Sequence[str]) -> int:
    if not terms:
        return 0
    haystack = searchable_text(record)
    if not haystack:
        return 0
    return sum(1 for term in dict.fromkeys(terms) if term in haystack)
```
**就是 distinct term 的子串命中计数。** 无权重、无词频、无 TF-IDF、无 BM25、无时间衰减、无长度归一化、无类别加权。分值范围 `[0, min(len(terms), 50)]`。

**排序与阈值**（61-86 行）
```python
if score >= 1:                              # 阈值 = 1
    scored.append((record, score))
scored.sort(key=lambda pair: (-pair[1], -timestamp_epoch(pair[0].updated_at), pair[0].id))
ranked = [record for record, _ in scored]
if top_k is not None:
    ranked = ranked[: max(0, top_k)]
```
排序键 `(-score, -epoch(updated_at), id)` —— 分数降序、`updated_at` 降序、id 升序。`updated_at` 用 `datetime.fromisoformat(...).timestamp()` 转 float 再取负（89-99 行），解析失败返回 `float("-inf")`（排序最旧，不抛）。

> 注：`updated_at` 在**打分排序**里，而 `recency`（`last_recalled_at or created_at`）只在**容量驱逐**里（`store.py:547`）。两者不同源，是 R3.1 与 R8.2 的有意区分。

**时间戳解析**（89-99 行）：`datetime.fromisoformat`，`except (TypeError, ValueError): return float("-inf")`。

#### 可操纵性评估（重要）

1. **子串语义 = 无成本命中**。`term in haystack` 是纯子串匹配。一条 content 是常见 2-字串袋子（`"我 的 你 好 是 不 在 有 他 她 …"`）的记录，对**几乎任何**查询都能得分 ≥1，从而**稳定占据 top_k**。由于阈值只有 1，"命中一个公共词"就够入选。
2. **🔴 tags 参与打分但从不渲染给 LLM**。`searchable_text` 纳入 `record.tags`（46 行），但 `format_record`（`recall.py:47-54`）只输出 `id/category/updated_at/content`，**tags 不出现在任何 LLM 可见输出里**。后果：一条 content 无害、tags 高度命中关键词的记录，会**把它的 content 悄悄送进每一轮 system prompt**，且使用者无法从 `recall_memory` 的返回文本看出命中原因。这是可被 prompt 注入者利用的放大器（配合记忆内容零转义注入，见 §9）。
3. **CJK 退化为"整句精确匹配"**。中文查询无空格 → `split(" ")` 后整句是**一个** token（长度通常 ≥2）→ 打分变成"记录是否包含这段整句字符串"。`docstring`（6-10 行）声称子串匹配解决了 CJK 分词问题，但对自然中文查询，召回率实际接近零。`MIN_TERM_CHARS=2` 又丢掉了所有单字（"我"、"你"），而中文单字恰恰信息量最大。
4. **无衰减 → 陈旧记忆与新鲜记忆同权**。唯一区分是 `updated_at` 降序，但只在**同分**时才起作用。
5. **`dict.fromkeys(terms)` 去重**：查询内重复词只算一次，防止"aaaa"式刷分。这一点是对的。
6. **无负向机制**：无法表达"这条记录不相关"，只能靠 `forget`。
7. **`MAX_QUERY_TERMS=50` 截断**：超长查询只取前 50 词，尾部词被静默丢弃 —— 可被用于把关键词压到尾部使打分变 0（DoS 自己），实用意义低。

### 3.2 水印（`watermark.py`，71 行）

**⚠️ 这不是加密水印，没有注入/检测、没有校验和、没有步长、没有字符选择规则。** 它是"该线程已抽取到第几个 user turn"的**单调簿记整数**，存在 session metadata 的 `config` JSON 里。任务描述假定的"水印方程"在本仓库不存在。

完整算法（15-71 行）：
```python
WATERMARK_KEY: Final[str] = "extraction_watermark"   # 19 行
MAX_WATERMARK: Final[int] = 1_000_000                 # 22 行

def coerce_watermark(raw: object) -> int:             # 25-36 行
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    if raw < 0 or raw > MAX_WATERMARK:
        return 0
    return raw
```
`bool` 被显式排除（`bool` 是 `int` 子类，`True` 会被当成 1 从而跳过一轮，28-29 行注释）。

`WatermarkStore` 是 Protocol（39-47 行），只有 `get(thread_id) -> int` / `set(thread_id, value)` 两个方法 —— **纯 Python duck typing，无序列化格式、无版本号、无签名、无 MAC**。

`InMemoryWatermarkStore`（49-62 行）：`threading.Lock` + `dict`，`get` 和 `set` 都过 `coerce_watermark`。

**持久化位置**：`SessionMetadata.from_session` 写 `{"extraction_watermark": int(...)}`（`storage.py:50-59`），SQLite 里是 `sessions.config` 列的一个 JSON 键（`sqlite.py:158`）。`SessionWatermarkStore`（`memory_hooks.py:25-75`）双向读写 registry + storage。

**边界与可绕过**：
- 合法区间 `[0, 1_000_000]`，越界 → 0。
- **没有完整性保护**：能改 SQLite 行的攻击者可以任意设值。
- 但 **"设成极大值压制抽取"这一攻击被 `min()` 中和了**：`extraction.py:325` 与 `should_extract` `extraction.py:144` 都写 `min(coerce_watermark(...), turns)`。设 1,000,000 → 被夹到 `turns`，`should_extract` 变成 `turns >= turns + interval` = False，**只推迟一个 interval 而非永久**；下一轮 `turns` 增长后照常抽取。
- "设成 0 触发重复抽取"→ 由 `store.save` 的去重（`store.py:485-497`）把重复候选降级为 update，成本受 `memory_extraction_max_concurrency`（默认 2）限。
- 规格 R7.8 要求的"不可用值记一条 warning"由 `memory_hooks.py:44-48` 实现（`cause=unusable_watermark`），**`watermark.py` 自身完全不记日志**（模块里连 `logger` 都没有）。所以走 `InMemoryWatermarkStore` 的路径不产生该 warning。

### 3.3 调度（`scheduler.py`，117 行）

**没有任何退避、抖动、间隔或重试公式。这是一个"忙即丢弃"的有界发射器。**

```python
# scheduler.py:29-36
self._max_concurrency = max_concurrency
self._slots = threading.BoundedSemaphore(max_concurrency)   # 32 行
self._inflight: set[str] = set()
self._lock = threading.Lock()
self._condition = threading.Condition(self._lock)
self._shutdown = False
```

`submit`（38-83 行）判定顺序（全部在 `self._condition` 锁内）：
1. `self._shutdown` → 丢弃（42-43）
2. `thread_id in self._inflight` → 丢弃，`logger.debug`（45-51）
3. `not self._slots.acquire(blocking=False)` → 丢弃，`logger.warning` 含 `limit=`（53-61）
4. `self._inflight.add(thread_id)`（63）
然后锁外起 `daemon=True` 线程（66-72）；线程创建失败 → `_finish(thread_id)` 回滚槽位（73-81）。

`_run`（97-108 行）：`runner` 抛任何异常 → warning；`finally: _finish`。
`_finish`（110-114 行）：`discard` + `_slots.release()` + `notify_all()`。

`wait_idle(timeout)`（85-89 行）：`condition.wait_for(lambda: not self._inflight, timeout=timeout)`，**仅测试用**（docstring 明说）。
`shutdown()`（91-95 行）：只置 `_shutdown=True`，**不等待在途任务、不 join**。

**关键性质**：
- `acquire(blocking=False)` → **零排队**。R6.4 要求"hold at most zero pending extraction requests"，实现精确符合。
- **被丢弃的工作靠 watermark 不前进来重试**（规格 R6.9 / R2.9）。`scheduler` 自身**不重试**，重试责任在调用方下一回合的 `after_turn`（`memory_hooks.py:195-196`）。
- 同一 thread_id 的去重集合 `_inflight` 保证**一个线程同时在跑最多一个抽取**。
- `max_concurrency < 1` 构造即 `ValueError`（26-27 行）。
- **无 timeout 看门狗**：一个 `model.invoke` 卡死会让该槽永久占用直到进程退出。但 `_call_model` 的 `completed.wait(timeout=...)` 只等**结果**，daemon 线程继续跑，`BoundedSemaphore` 的槽要等 `_run` 返回才释放 —— 所以一个卡死的 provider 调用**会占满并发槽**，直到该调用自己结束。这是 R6.4 与"2000ms 内不阻塞对话"之间唯一的真实张力点。

**会话层调度方程**（`registry.py`）：
- TTL 驱逐：`if now - last_accessed > self._ttl_seconds`（354 行，严格大于），默认 3600s。
- 后台循环间隔：`interval = max(0.1, float(self._cleanup_interval))`，`while not self._stop_cleanup.wait(interval)`（389-392 行），默认 300s。
- touch 持久化节流：`if last is not None and now - last < TOUCH_PERSIST_INTERVAL_SECONDS: return`（406 行），常量 30.0s（30 行）。

### 3.4 配额（`security/quotas.py`，305 行）

**滚动窗口常量**（17-20 行）
```python
_HARD_MAX_RATE = 1_000_000_000
_HARD_MAX_CONCURRENCY = 1_024
_HARD_MAX_TRACKED_IDENTITIES = 100_000
_WINDOW_SECONDS = 60.0
```

**六维计费**（32-51 行）`QuotaCharge`：`requests=1, searches=0, tokens=0, tool_calls=0, retries=0, cost_units=1`。`__post_init__` 校验每个字段是 int（非 bool）、`0 <= v <= 1e9`，否则 `ValueError`。

**默认预算**（90-112 行）
```python
principal = QuotaBudget(requests_per_minute=120,  concurrent_calls=4,  searches_per_minute=30,
                        tokens_per_minute=250_000, tool_calls_per_minute=240,
                        retries_per_minute=600,   cost_units_per_minute=500_000)
tenant    = QuotaBudget(requests_per_minute=600,  concurrent_calls=16, searches_per_minute=120,
                        tokens_per_minute=1_000_000, tool_calls_per_minute=1_000,
                        retries_per_minute=3_000, cost_units_per_minute=2_000_000)
max_tracked_identities: int = 4_096
```

**窗口裁剪**（127-131 行）
```python
def prune(self, now: float) -> None:
    while self.events and now - self.events[0][0] >= _WINDOW_SECONDS:
        _timestamp, charge = self.events.popleft()
        for name in _USAGE_FIELDS:
            self.totals[name] -= getattr(charge, name)
```
**半开区间**：`>=` 60.0s 即裁。所以窗口是 `(now-60, now]`，恰好 60s 的事件被挤出。

**准入判定**（133-137 行）
```python
def accepts(self, charge: QuotaCharge, budget: QuotaBudget) -> bool:
    return all(self.totals[name] + getattr(charge, name) <= budget.maximum_for(name)
               for name in _USAGE_FIELDS)
```
六维**全部**同时满足才放行（AND，不是 OR，不是加权）。

**`acquire` 语义**（193-232 行）—— **无 timeout、无 shield、fail-fast**：
```python
def acquire(self, principal: Principal, charge: QuotaCharge | None = None) -> QuotaLease:
    usage = charge or QuotaCharge()
    now = self._time()
    principal_key = (principal.tenant_id, principal.principal_id)
    tenant_key = principal.tenant_id
    with self._lock:                          # 204 —— 同步 threading.Lock，无 timeout
        principal_window = self._window(self._principal_windows, principal_key, now)
        tenant_window = (None if tenant_key is None else self._window(self._tenant_windows, tenant_key, now))
        if not principal_window.accepts(usage, self._limits.principal):
            raise QuotaExceededError()       # 210
        if tenant_window is not None and not tenant_window.accepts(usage, self._limits.tenant):
            raise QuotaExceededError()       # 212
        if self._active_principals.get(principal_key, 0) >= self._limits.principal.concurrent_calls:
            raise QuotaExceededError()       # 217
        if tenant_key is not None and self._active_tenants.get(tenant_key, 0) >= self._limits.tenant.concurrent_calls:
            raise QuotaExceededError()       # 222
        principal_window.append(now, usage)  # 224
        ...
    return QuotaLease(self, principal_key, tenant_key)   # 232 —— 锁外构造
```
判定顺序：principal 速率 → tenant 速率 → principal 并发 → tenant 并发。**任何一项不满足立即抛**，不等待、不排队、不阻塞在槽位上。`acquire` 本身只持有 `threading.Lock` 极短时间（纯内存运算），**所以 `acquire` 不会卡住**。

**没有 `shield`**：`quotas.py` 全文无 `asyncio.shield`（`acquire` 是同步方法）。真正的卡点在上层 `chat_service.py:69-70`：
```python
with self._quotas.acquire(request.principal, charge):   # 先拿配额
    await _acquire_lock(session.turn_lock)              # 再拿回合锁
```
配额租约在回合锁**之前**获得，而 `concurrent_calls=4` 与回合锁无超时 → 一个卡住的回合可占死全部 4 个并发槽，后续所有请求 `QuotaExceeded`。这与我此前记录的结论一致。

**身份容量与淘汰**（234-253 行）
```python
if len(windows) >= self._limits.max_tracked_identities:      # 4096
    for candidate, existing in tuple(windows.items()):
        existing.prune(now)
        if not existing.events:
            del windows[candidate]                            # 只删已空窗口
if len(windows) >= self._limits.max_tracked_identities:
    raise QuotaExceededError()                                # 删不掉 → 429
```
**只淘汰"已完全过期"的身份**。若 4096 个身份都各有未过期的事件，任何新身份都会被拒。且**淘汰只发生在 `_window` 被调用时**，不是后台定时。

**租约**（145-169 行）
```python
class QuotaLease:
    def __init__(self, manager, principal_key, tenant_key): ...
    self._released = False
    def release(self) -> None:
        if self._released: return
        self._released = True          # 幂等
        self._manager._release(self._principal_key, self._tenant_key)
    def __enter__(self): return self
    def __exit__(self, *_): self.release()
```
**`_released` 保证幂等，但没有 TTL**。泄漏的租约会永久占用一个并发槽。`chat_service.py:81-89` 用裸 `acquire`（非 `with`），依赖 `_stream_turn` 的 `finally: lease.release()`（113 行）—— 但 `_stream_turn` 是**生成器函数**，生成器未被迭代或未被 `aclose()` 时 `finally` 不执行 → 租约泄漏。

**令牌估算**（266-271 行）
```python
def estimate_token_units(text: str | None) -> int:
    if not text: return 0
    return max(1, math.ceil(len(text) / 4))
```
字符数 / 4，向上取整，下限 1。纯字符启发式，对 CJK（每字符 ≈1 token）会**低估约 4 倍**。

**成本换算**（274-281 行）
```python
def estimated_cost_units(charge: QuotaCharge) -> int:
    return max(1, charge.cost_units,
               charge.tokens + charge.searches * 100 + charge.tool_calls * 10 + charge.retries * 5)
```
权重：search ×100 > tool ×10 > retry ×5 > token ×1。默认 `QuotaCharge`（`cost_units=1`）→ `max(1,1,0)=1`。

**递减**（289-294 行）：`_decrement` 归零即 `pop`，不留 0 值条目。

**边界校验**（284-286 行）：`_bounded_positive` 要求 int（非 bool）且 `1 <= v <= maximum`。

---

## 4. 🔐 密钥与 SQL 面

### 4.1 `memory/secrets.py`（78 行）

**完整正则清单（14-38 行）**：
```python
SECRET_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("pem_private_key",
     re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE)),          # 17 行
    ("sk_token",
     re.compile(r"sk-[A-Za-z0-9_-]{16,}", re.IGNORECASE)),                           # 21 行
    ("aws_access_key_id",
     re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}", re.IGNORECASE)),                       # 25 行
    ("bearer_token",
     re.compile(r"\bbearer\s+\S{20,}", re.IGNORECASE)),                              # 29 行
    ("assigned_secret",
     re.compile(r"[\w.-]*(?:password|passwd|secret|api[_-]?key|token)[\w.-]*"
                r"\s*[:=]\s*\S{8,}", re.IGNORECASE)),                                # 33-36 行
)
```

**返回语义**（41-52 行）：`find_secret_match` 返回**模式名**，永不返回匹配文本 —— 这是类型级保证。`find_secret_in_any`（55-71 行）遍历字符串序列，非可迭代的值静默返回 `None`（64-65 行）。

**调用点只有两处**（`store.py`）：
```python
# store.py:472-479
if find_secret_match(trimmed) is not None:
    return self._save_error("credential-like content is refused (parameter: content)")
if find_secret_in_any(normalized_tags) is not None:
    return self._save_error("credential-like content is refused (parameter: tags)")
```
顺序在参数校验之后、去重之前（符合 R9.6）。**注意 `refused` 文案不含匹配文本** ✓。

#### 漏掉的密钥形态（重要）

| 漏检形态 | 为什么漏 |
|---|---|
| **AWS Secret Access Key**（真正的密钥本体，40 字符 base64-ish） | 只有 `AKIA/ASIA` 前缀的 **key ID** 被检；secret 无固定前缀 |
| **Stripe** `sk_live_...` / `pk_live_...` | `sk_token` 要求 `sk-`（连字符），Stripe 用 `sk_`（下划线）→ **完全不匹配** |
| **GitHub** `ghp_` / `gho_` / `ghs_` / `ghu_` / `github_pat_` | 无对应模式 |
| **Google** `AIza[0-9A-Za-z\-_]{35}` | 无 |
| **Slack** `xox[baprs]-`、`xapp-` | 无 |
| **JWT** `eyJxxx.yyy.zzz` | 无 |
| **DSN / 连接串** `mongodb+srv://u:p@host`、`postgres://user:pass@h/db` | `assigned_secret` 要求 `[:=]`，而 DSN 用 `@` 分隔 → 密码不被捕 |
| **`ssh-ed25519 AAAA…` / `ecdsa-sha2-nistp256` 公开/私钥行** | 只有 PEM `BEGIN` 头被检 |
| **`id_dsa` / `id_ecdsa` 内容以外的引用** | 同上（内容靠 PEM 头，非 PEM 形态漏） |
| 短值：`password=abc`（<8 字符） | `\S{8,}` 下限 |
| `key=dbpassword12345`（无 `api` 前缀的任意 key 名） | 关键字白名单只含 password/passwd/secret/api_key/token |
| **非 `=`/`:` 分隔**：`password is hunter2`、`password: "x"`（值被引号包裹仍可捕，但空格分隔不行） | `[:=]` 强制 |
| `PuTTY-PPK-*` 私钥 | 无 |

#### 与规格/设计的偏差

| 项 | 规格 | 实现 | 判定 |
|---|---|---|---|
| `pem_private_key` | `r"-----BEGIN [A-Z ]*PRIVATE KEY-----"`（design.md:208） | `[A-Z0-9 ]*` | 实现**更宽**（含数字），改进 |
| `aws_access_key` | `r"\b(?:AKIA\|ASIA)[A-Z0-9]{16}\b"`（design.md:210） | `r"(?:AKIA\|ASIA)[A-Z0-9]{16}"` **无 `\b`** | **回归**：可匹配长串内部的 20 字符子串，误报率上升 |
| 名称 | `aws_access_key` | `aws_access_key_id` | 仅命名差异，但**暗示语义收窄到 ID**（见上表） |
| `assigned_secret` | `r"\b\w*(?:password\|passwd\|secret\|api_key\|token)\w*\s*[:=]\s*\S{8,}"`（design.md:212-213） | `[\w.-]*` + `api[_-]?key` | 实现**更宽**（`.`/`-`/`api-key` 都捕），改进 |
| `bearer_token` | 同 | 同 | 一致 |

#### 屏检覆盖的缺口

1. **只在 `save()` 里跑**。`parse_document` / `_parse_record` 从不复检（`serialization.py:187-192`），所以：手工编辑文件写入的密钥、屏检上线前保存的旧记录、以及**任何绕过 store 直接写文件的途径**都不被拦。
2. **`tags` 通过 `_normalize_tags` 只 trim、不长度截断**，屏检在 `save()` 里对 `normalized_tags` 跑（`store.py:476`），但 `_with_provenance` 拼出的 provenance tag 不再复检 —— 无害（provenance 是常量）。
3. **长度上限 40 字符的 tag** 使长 token 天然进不来；但 `content` 上限 1000 字符，完全够放一条完整密钥。
4. **`refused` 的判定结果不进日志内容**（R9.2）✓，`store.py` 的日志只打 id/count。

### 4.2 `sessions/sqlite.py`（212 行）与 `checkpoint.py`

#### SQL 语句全清单 —— 全部参数化

| 行号 | 语句 | 参数化 |
|---|---|---|
| `sqlite.py:37-57` | `INSERT INTO sessions (...) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET ...` | ✓ 10 个 `?` |
| `sqlite.py:63-66` | `SELECT * FROM sessions WHERE thread_id = ?` | ✓ |
| `sqlite.py:73-76` | `DELETE FROM sessions WHERE thread_id = ?` | ✓ |
| `sqlite.py:83-85` | `SELECT thread_id FROM sessions ORDER BY created_at, thread_id` | ✓（无参数，ORDER BY 是字面列名） |
| `sqlite.py:92-94` | `SELECT * FROM sessions ORDER BY created_at, thread_id` | ✓ |
| `sqlite.py:110-117` | `CREATE TABLE IF NOT EXISTS schema_version (id INTEGER PRIMARY KEY CHECK (id=1), version INTEGER NOT NULL)` | ✓ 字面 |
| `sqlite.py:118-133` | `CREATE TABLE IF NOT EXISTS sessions (... 10 列 ...)` | ✓ 字面 |
| `sqlite.py:135` | `PRAGMA table_info(sessions)` | ✓ 字面 |
| `sqlite.py:138/140` | `ALTER TABLE sessions ADD COLUMN owner_principal_id TEXT` / `owner_tenant_id TEXT` | ✓ 字面 |
| `sqlite.py:141-148` | `INSERT INTO schema_version (id, version) VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET version=excluded.version` | ✓ |
| `checkpoint.py:68-71` | `SELECT principal_id, tenant_id FROM checkpoint_owners WHERE thread_id = ?` | ✓ |
| `checkpoint.py:77-83` | `INSERT INTO checkpoint_owners (thread_id, principal_id, tenant_id) VALUES (?, ?, ?)` | ✓ |
| `checkpoint.py:88-92` | 同 68 | ✓ |
| `checkpoint.py:142-158` | `CREATE TABLE IF NOT EXISTS checkpoint_owners (...)` / `thread_state (thread_id TEXT PRIMARY KEY, payload BLOB NOT NULL)` | ✓ 字面 |
| `checkpoint.py:165-167` | `SELECT 1 FROM sqlite_master WHERE type='table' AND name='checkpoint_state'` | ✓ 字面 |
| `checkpoint.py:170` | `SELECT payload FROM checkpoint_state WHERE id = 1` | ✓ 字面 |
| `checkpoint.py:172-180` | `INSERT INTO thread_state (thread_id, payload) VALUES (?, ?) ON CONFLICT(thread_id) DO UPDATE ...` | ✓ |
| `checkpoint.py:181` | `DROP TABLE checkpoint_state` | ✓ 字面 |
| `checkpoint.py:185` | `SELECT thread_id, payload FROM thread_state` | ✓ |
| `checkpoint.py:246-249` | `DELETE FROM thread_state WHERE thread_id = ?` | ✓ |
| `checkpoint.py:251-258` | `INSERT INTO thread_state (thread_id, payload) VALUES (?, ?) ON CONFLICT ...` | ✓ |
| `checkpoint.py:338-341` | `DELETE FROM checkpoint_owners WHERE thread_id = ?` | ✓ |

**结论：零字符串拼接 SQL，无一处 f-string / `%` / `.format` 进入 SQL。SQL 注入面为 0。**

#### 连接 / 事务 / WAL

```python
# sqlite.py:97-105
@contextmanager
def _connect(self) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(self._path)     # 默认 timeout=5.0
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
```
```python
# checkpoint.py:357-364
@contextmanager
def _connect(self) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(self._path)     # 注意：这里没设 row_factory
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()
```

- **每次操作开新连接**，用完即关。默认 rollback journal（DELETE 模式），**无 `PRAGMA journal_mode=WAL`、无 `synchronous`、无 `busy_timeout`（用 `sqlite3.connect` 的默认 5.0s）**。
- `BEGIN` 隐式（首个 DML），`commit()` 显式在 `yield` 后。**读操作也走同一个 contextmanager，所以 `load()`/`list_ids()` 也会 `commit()`**（无实际影响，因为没 DML，但语义上是空 commit）。
- 唯一互斥是**进程内** `threading.Lock`（`sqlite.py:23`、`checkpoint.py:45` 为 `RLock`）。**两个 `SQLiteStorage` 实例指向同一 DB 时互不持锁**，此时依赖 SQLite 自身的 5s busy timeout。
- 无事务隔离级别设置 → 默认 DEFERRED。`save` 是单条 upsert，无多语句事务，无 `SAVEPOINT`。

#### 迁移完整性缺陷（高）

```python
# sqlite.py:134-148
columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)").fetchall()}
if "owner_principal_id" not in columns:
    connection.execute("ALTER TABLE sessions ADD COLUMN owner_principal_id TEXT")
if "owner_tenant_id" not in columns:
    connection.execute("ALTER TABLE sessions ADD COLUMN owner_tenant_id TEXT")
```
**只有 2 列有条件迁移**：`owner_principal_id`、`owner_tenant_id`。而 `CREATE TABLE IF NOT EXISTS` 里声明的另外 8 列 —— `thread_id`、`source_urls`、`source_mode`、`config`、`created_at`、`last_accessed_at`、`chroma_dir`、`isolated_chroma` —— **完全没有 `ALTER TABLE` 兜底**。若磁盘上存在一个早期 schema 的 `sessions` 表（例如缺 `chroma_dir` / `isolated_chroma`，这正是最近几次提交引入的两列），`CREATE TABLE IF NOT EXISTS` 会**静默跳过**，随后 `_metadata_to_row`（151-163 行）构造 10 元素 tuple 写入 8 列表 → `sqlite3.ProgrammingError: Incorrect number of bindings supplied`。**升级路径对新增列是脆的。**

另外 `_ensure_database` 在 `__init__` 里调用（`sqlite.py:21-24`）—— **构造函数有磁盘副作用**（建目录、建表、写 `schema_version`）。`SCHEMA_VERSION = 2`（19 行）写入但**从不被读取校验**：没有任何代码比较磁盘里的 `schema_version.version` 与 `self.SCHEMA_VERSION`，所以"版本 2"只是记账，不构成升级门。

`checkpoint.py` 的 legacy 迁移（161-181 行）走 `DROP TABLE checkpoint_state` —— **破坏性、无备份**：legacy 单行 blob 被拆成 per-thread 行后原表直接删除。且 `_migrate_legacy_state` 只处理 `id = 1` 一行（170 行）；若 legacy 表存在多行（不同 id），其余行静默丢失。

**JSON 列校验**：`_decode_urls`（188-195）与 `_decode_config`（198-205）对 `JSONDecodeError` 抛 `SessionStorageError`，并对类型不符抛错。**这是 fail-closed**，与 memory store 的 fail-open（返回空）形成对比 —— 一处坏行会让整个 `load()`/`list_metadata()` 抛异常，进而 `registry._save_metadata` 的 `except: logger.warning`（`registry.py:425-426`）**静默吞掉**。

#### ⚠️ `checkpoint.py` 的 pickle RCE 面

四处 `pickle.loads`：
- `checkpoint.py:170` → 172 `_split_legacy_state(row[0])`（legacy 表 payload）
- `checkpoint.py:195` `state = pickle.loads(payload)`（**启动时逐行 unpickle `thread_state`**，`_load_state`）
- `checkpoint.py:285` `state = pickle.loads(snapshot)`（`restore_thread`，**snapshot 是调用方传入的 bytes**）
- `checkpoint.py:376` `state = pickle.loads(payload)`（`_split_legacy_state`）

写入侧全部 `pickle.dumps(..., protocol=pickle.HIGHEST_PROTOCOL)`（257、272、299 行）。

`restore_thread` 的唯一生产调用点是 `turn_execution.py:329-335`，其 `snapshot` 来自同进程 `_snapshot()`（308-319）→ `snapshot_thread`。所以**当前 HTTP 路径上没有直接的外部字节入口**。真实的攻击边界是：**任何能写 `*.sqlite3` 文件的主体，在进程启动时（`_load_state`，`checkpoint.py:48`）即可获得代码执行**。这与 SQLite 文件权限等同，未引入新的边界，但对"应用数据文件"的信任假设需要显式记录。

所有 SQL 均为参数化，所以**不能用 SQL 注入绕过 ownership 表**；但 ownership 强制是可选的：`enforce_ownership` 默认 `False`（`checkpoint.py:36`），且 `_require_registered_config` / `_require_owner` 在 `False` 时直接 `return`（343-355 行）。**未开启时 `snapshot_thread`/`restore_thread`/`delete_thread` 对任何 thread_id 无限制。**

`delete_thread`（309-341 行）：`getattr(super(), "delete_thread", None)` 兼容旧版 langgraph（326-336 行），并删除 `checkpoint_owners` 行。

---

## 5. 📋 规格 vs 实现

### 5.1 `chat-long-term-memory` —— 实现偏离

| 规格 | 要求 | 实现 | 判定 |
|---|---|---|---|
| R5.2 (`requirements.md:272-280`) | `scope_id` 1–200 字符 | `MAX_SCOPE_ID_CHARS=200` 定义（`models.py:34`）但**无任何引用**；`save()` 与 `_parse_record` 都不校验 | ❌ 未实现 |
| R5.2 (`requirements.md:276-280`) | `content` 1–`memory_max_record_chars` | 仅在 `save()` 校验（`store.py:434-438`）；`parse_document` 不校验（`serialization.py:192`） | ⚠️ 解析侧未实现 |
| R5.7 (`requirements.md:305-308`) | version 缺失/非 int/**>1** → 空库 | `_is_readable_version` 拒绝 `1 <= v <= SCHEMA_VERSION`，**连 `<1` 也拒**（`serialization.py:114-117`） | ⚠️ 实现更严（无害） |
| R5.6 (`requirements.md:300-304`) | 不可解析 → 改名 + 空库 + 不抛 | ✓ `store.py:264-266`, `283-313` | ✅ |
| R11.5 (`requirements.md:587-590`) | `memory_default_scope` 折叠后越界 → **fail load** | `loader.py:415-421`：`scope = str(value).strip().casefold() or str(default or "global")` —— **空串静默回落 "global"** | ❌ 空串应失败但被吞 |
| R11.10 / Assumption 2 (`requirements.md:594-596`, `design.md:60`, `requirements.md:658-661`) | 排除 `.corrupt-*` 兄弟文件 | `.gitignore:30` 只有 `memory/`；**无 `.corrupt-*` 规则**。默认路径下功能上满足，但**自定义 `memory_store_path` 指向 `memory/` 之外时不满足** | ⚠️ 依赖默认路径的偶然满足 |
| R9.5 (`requirements.md:497-504`) | `AKIA/ASIA` + 16 字符 | 实现**丢了 `\b` 边界**（`secrets.py:25` vs `design.md:210`） | ⚠️ 回归（误报↑） |
| R11.1 (`requirements.md:566-572`) | 8 个 memory 字段 | `settings.py:245-254` 全部存在，默认值一致 | ✅ |
| R11.4 (`requirements.md:582-586`) | 4 个整型区间 | `loader.py:225-229` `(1,10000)/(1,10000)/(1,50)/(1,20000)` | ✅ |
| R3.1 / R7.1 (`requirements.md:176-183`, `382-387`) | 阈值 ≥1，`(-score, updated_at desc, id asc)` | ✓ `relevance.py:77-82` | ✅ |
| R8.2 (`requirements.md:437-444`) | 有效新近度 = `last_recalled_at or created_at` | ✓ `models.py:92` + `store.py:547` | ✅ |
| R6.8 (`requirements.md:360-365`) | 3 次重试，50ms 起倍增，总 ≤350ms | ✓ `store.py:106-107`, `386-398` | ✅ |
| R7.6 (`requirements.md:406-409`) | 2000ms 内不阻塞 | **无定时器**，靠"有界工作量"保证（`design.md:369-373` 自己也承认是"by construction"） | ⚠️ 无强制 |
| R3.8 (`requirements.md:214-216`) | 读失败 → 错误串、**不改文件、不打戳** | ✓ `store.py:601-605`（`MemoryStoreError` → `RecallOutcome(ok=False)`，在 `self._persist_locked` 之前返回） | ✅ |
| 设计路径 (`design.md:40-60`) | `src/memory/`, `src/tools/`, `src/chat/`, `src/graph/` | 实际 `src/backend/memory/`, `src/backend/tools/`, `src/backend/application/` | ⚠️ 全部过期 |
| `CATEGORIES`/`SCOPES` 类型 (`design.md:99-100`) | `frozenset` | `tuple`（`models.py:18-19`） | ⚠️ 漂移 |
| `relevance.rank_records` 签名 (`design.md:186-188`) | `top_k` 必传 | `top_k: int \| None = None`（`relevance.py:65`） | ⚠️ 漂移 |

### 5.2 `memory-auto-extraction` —— 实现偏离

| 规格 | 要求 | 实现 | 判定 |
|---|---|---|---|
| R7.8 (`requirements.md:439-444`) | 不可用 watermark → 用 0 + **记一条 warning** | `watermark.py` 无 logger；warning 由 `memory_hooks.py:44-48` 记。**"exactly one" 不成立**：每次 `get()` 都会再记一条 | ⚠️ 实现位置错位 + 非"恰好一条" |
| R7.4 (`requirements.md:423-426`) | "value never decreases" | `extraction.py:325` `min(coerce(...), turns)` 先把读值夹到 `turns`，再 `set(thread_id, turns)`（443 行）→ **磁盘上的 watermark 会下降** | ⚠️ 与 R7.9 冲突，见下 |
| R7.9 (`requirements.md:445-449`) | watermark > turns 时用 turns，且**下次成功抽取持久化该缩小值** | ✓ 实现正好这么做 | ✅（但与 R7.4 张力） |
| R4.4 (`requirements.md:268-272`) | 最多评估前 100 条，先按 R4.10 去重再截 limit | ✓ `extraction.py:165` `parsed[:100]`，175-178 去重，204-205 截 | ✅ |
| R4.11 (`requirements.md:300-305`) | 零候选（含空数组/不可解析/全丢弃）→ 推进 watermark | ✓ `extraction.py:376-386` | ✅ |
| R4.6 (`requirements.md:278-282`) | 超时/异常 → 零候选、**watermark 不动**、warning | ✓ `extraction.py:354-372` 直接 `_outcome`（不推进） | ✅ |
| R6.7 (`requirements.md:384-388`) | store 写失败 → watermark 不动、继续剩余候选、**恰好一条 warning** | ✓ `_persist` 逐条 continue 并记首个 `error_type`；`run()` 返回 `store_write:*` 走 warning | ✅ 但见下 ⚠️ |
| R6.4 (`requirements.md:374-376`) | ≤N 并发，**零 pending 请求** | ✓ `BoundedSemaphore` + `acquire(blocking=False)`（`scheduler.py:53`） | ✅ |
| R6.9 (`requirements.md:393-397`) | 满并发 → 丢弃 + warning 含 limit | ✓ `scheduler.py:53-61` | ✅ |
| R6.3 (`requirements.md:370-373`) | 同 thread 在途 → 丢弃，debug 级或静默 | ✓ `scheduler.py:45-51` `logger.debug` | ✅ |
| R6.5 (`requirements.md:377-380`) | 关闭不等待在途，最多 +1s | ✓ `shutdown()` 仅置标志（`scheduler.py:91-95`） | ✅ |
| R5.7 (`requirements.md:339-341`) | 抽取**不消耗**回合 tool 预算 | ✓ `_persist` 直调 `store.save`，不碰 `MEMORY_BUDGET` | ✅ |
| R5.1 (`requirements.md:314-318`) | 用**被抽取线程**的 thread_id | ✓ `extraction.py:271` | ✅ |
| R5.9 (`requirements.md:348-352`) | `session` 作用域绑**被抽取**线程 | ✓ `store.save(..., thread_id=thread_id)` | ✅ |
| R9.4 (`requirements.md:540-545`) | prompt 恰两部分：固定指令 + 用 delimiter 标注的不可信转录 | ✓ `prompts.py:331+`：`--- BEGIN UNTRUSTED CONVERSATION DATA ---` / `--- END ... ---` | ✅ |
| R9.7 (`requirements.md:557-562`) | 注入的 recall note 须**标记**以便按标记排除 | ✓ `MEMORY_NOTE_MARKER = "​​"`（`recall.py:38`）+ `is_memory_note`（`transcript.py:105-116`） | ✅ |
| R9.8 (`requirements.md:563-569`) | provenance tag，命名 trigger，保留上限内 | ✓ `_provenance_tag` / `_with_provenance`（`extraction.py:579-586`），≤40 字符、≤10 个 | ✅ |
| R9.1 (`requirements.md:527-531`) | 日志不含 ≥20 连续字符的转录/候选/记录/LLM 响应文本 | ✓ 日志字段只有 trigger/thread_id/计数/watermark/duration/status/detail，`detail` 是异常**类型名** | ✅ |
| R1.7 (`requirements.md:116-120`) | 已删除但转录仍在的 session 不参与选择 | ✓ `memory_hooks.py:147-151` `thread_id not in live_threads: continue` | ✅ |
| R1.8 (`requirements.md:121-125`) | `last_accessed_at` 相同 → 取 thread_id **升序最后** | ✓ `max(..., key=lambda item: (item[0], item[1]))`（`memory_hooks.py:166-169`） | ✅ |
| R1.9 (`requirements.md:126-131`) | 存在性/`last_accessed_at`/watermark **从持久化 metadata 读** | ✓ `storage.list_metadata()`；turns 从 checkpoint 读（R2.2 要求） | ✅ |
| R8.1 (`requirements.md:458-467`) | 8 个 extraction 字段及默认值 | ✓ `settings.py:257-264` 完全一致 | ✅ |
| R8.4 (`requirements.md:479-487`) | 6 个整型区间，**不 clamp** | ✓ `loader.py:230-235` `(1,1000)/(1,20)/(200,100000)/(1,600)/(1,16)/(1,8760)` | ✅ |
| R3.4 (`requirements.md:214-219`) | 消息数上限在字符上限之前，从**开头**丢 | ✓ `extraction.py:330-332` `[-200:]` 后 `render_slice` 从 `blocks.pop(0)` | ✅ |
| R3.10 (`requirements.md:242-248`) | 单条超长 → 截断 + 标记，不丢 | ✓ `transcript.py:182-189` | ✅ |
| R4.7 (`requirements.md:283-287`) | 只透传 trim 后非空字符串 tag | ✓ `extraction.py:187-192` 但**不校验长度**，超长 tag 会让 store 整体拒收该候选 | ⚠️ 单条超长 tag 杀整条候选 |
| R6.1 (`requirements.md:361-364`) | 请求路径上零 LLM 调用/零 checkpoint 读/零 store 写 | ✓ `submit()` 只起线程 | ✅ |

### 5.3 规格自身矛盾

**矛盾 1（严重）：design Property 12 vs R3.2 / R7.2**
- `design.md:732-737` Property 12 写道："No string returned to the agent and no log record contains text that matched a Secret_Pattern, **nor any Memory_Record content, tag value, or query text**."
- 但 `requirements.md:184-190` R3.2 明确要求 recall 返回串**必须包含** `content`，`requirements.md:388-394` R7.2 要求注入的 SystemMessage **必须包含**每条记录的 identifier 和 content。
- 这两者不可能同时满足。Property 12 的意图显然是"密钥与日志卫生"，但措辞把 recall 的正常功能一并禁掉了。**design.md 内部自相矛盾**，实现（返回 content）追随的是 requirements 而非 design Property 12。

**矛盾 2：R7.4 vs R7.9（watermark 单调性）**
- R7.4 (`requirements.md:423-426`)："SHALL leave the Extraction_Watermark unchanged, **so the value never decreases**"
- R7.9 (`requirements.md:445-449`)："IF a thread's Extraction_Watermark is greater than that thread's current Turn_Count, THEN ... **SHALL persist that reduced value** on the next successful extraction"
- 二者直接冲突：R7.9 要求把磁盘值写小。实现遵循 R7.9。R7.4 的"never decreases"作为**最终不变式**不成立，只是"在 watermark 未被夹小之前不减"。

**矛盾 3：R11 准则编号错乱**
- `requirements.md:566-610` 的编号顺序是 `1,2,3,4,5,6, 10, 7, 8, 9` —— 准则 10（`.corrupt-*` 排除）插在 6 和 7 之间。文档级缺陷。

**矛盾 4：design 的排序算法叙述 vs 实现**
- `design.md:194-202` 花三段论证为何不能用 `reverse=True`，并描述 `_neg_iso` 用 `datetime` 转 POSIX 时间戳取负。实现（`relevance.py:80-82` + `89-99`）确实这么做（`timestamp_epoch`），**但 design 里那个"两遍 sorted"的示例代码片段（197 行）根本不存在于实现**。叙述冗余但结论一致。

**矛盾 5：`.gitignore` 记录失真**
- `design.md:60` 断言 `.gitignore` 是 `memory/long_term_memory.json{,.corrupt-*}`。实际 `.gitignore:30` 是 `memory/`（**更宽**）。功能上默认路径被覆盖，但**记录的事实不准确**，且掩盖了"自定义路径下 `.corrupt-*` 未忽略"的真实缺口。

---

## 6. 发现与矛盾

### 🔴 高严重度

1. **`store.py:74-80` + `store.py:413-436`** —— schema 版本不匹配时 `parse_document` 返回空文档，`read()` **不动磁盘**，但下一次任何写操作以 `version=1` **原地覆盖**旧版本全部记录，**无隔离、无备份、无用户可见信号**（只有一行 warning 日志 `serialization.py:76-79` 经 `store.py:268-269` 发出）。对比 JSON 语法损坏会得到 `.corrupt-*` 备份。**这是"规格自相矛盾"型根因**：R5.7 要求"文件不变"，R6.1/R8.6 要求"写完整文档"，两者组合产生数据丢失，而没有任何准则要求"保留不可读版本"。

2. **⚠️ 修正此前记忆笔记**：`store.py:264-266` 证明 **`read()` 在 JSON 损坏时会 `os.replace` 移动原文件**（`_quarantine`）。此前"read() leaves disk untouched"的结论**只对 schema 版本不匹配和有效文件成立**，对 JSON 语法损坏**不成立**。

3. **`recall.py:47-54`** —— 记忆内容**零转义**注入 LLM 上下文：
   ```python
   head = f"- [{record.id}] ({record.category}, updated {record.updated_at})"
   return f"{head} {record.content}"
   ```
   `record.content` 原样拼入，无引号、无围栏、无边界标记。内容可以包含伪造的引用边界（`--- Source:`）、伪造的指令、甚至 `LONG-TERM MEMORY (recalled):` 本身。`build_memory_note` 最终产出 `f"{MEMORY_NOTE_LABEL}{MEMORY_NOTE_MARKER}\n{body}"`（`recall.py:113`）作为一个 `SystemMessage`（`recall.py:189`），插在 user message **之前**（`recall.py:171-199`）——即 memory 内容的指令优先级高于用户消息。

4. **`relevance.py:42-58` vs `recall.py:47-54`** —— **tags 参与打分，但 tags 从不渲染给 LLM**。命中一条仅 tags 相关的记忆，会静默地把它的 content 送进每轮 system prompt，使用者从 `recall_memory` 的输出里看不出命中原因。与 #3 组合即构成"记忆内容 → 持续注入"的放大器。

5. **`checkpoint.py:195`**（`_load_state`，`__init__` 在 48 行调用）—— 进程启动时对 `thread_state` 每行 `pickle.loads`。**任何能写 SQLite 库文件的主体在下次启动时获得 RCE**。另有 `checkpoint.py:285`（`restore_thread`，snapshot 为调用方 bytes）、`170`/`376`（legacy 迁移）。当前唯一生产调用点 `turn_execution.py:329-335` 的 snapshot 来自同进程，故无 HTTP 直入点。

6. **`sqlite.py:134-148` vs `151-163`** —— 迁移只条件性新增 `owner_principal_id` / `owner_tenant_id` 两列；其余 8 列（含最近的 `chroma_dir`、`isolated_chroma`）无 `ALTER TABLE` 兜底。旧 schema 表遇 `CREATE TABLE IF NOT EXISTS` 静默跳过 → 10 元组写 8 列表 → `sqlite3.ProgrammingError`。**升级路径脆断**。

7. **`secrets.py:14-38`** —— 5 条正则漏掉：AWS **secret** access key、Stripe `sk_live_`/`pk_live_`（`sk_token` 要求连字符 `sk-`）、GitHub `ghp_`/`github_pat_`、Google `AIza`、Slack `xox*-`/`xapp-`、JWT `eyJ`、DSN `user:pass@host`（`assigned_secret` 只认 `[:=]`）、SSH 非 PEM 密钥行。且屏检**只在 `save()` 一处**跑，解析路径从不复检。

8. **`registry.py:127` + `isolation.py:21-27`** —— `thread_id = thread_id or uuid.uuid4().hex` **零校验**；随后 `chroma_dir = base_chroma_dir / "chat" / thread_id`、`collection_name = f"{base.collection_name}-chat-{thread_id}"` 直接插值。带 `..` 或路径分隔符的 thread_id 会在 isolated 会话下指向意外路径。**防护只在删除时**（`_safe_isolated_chroma_dir`，`registry.py:33-39`，只查 `parent.name == "chat"`），创建时不查。

9. **`memory_hooks.py:166-175` + `settings.py:264`** —— `memory_extraction_max_session_age_hours` 默认 168（7 天），超龄会话被 `_runtime_watermark(runtime).set(thread_id, turns)` 标记为"已抽取"**且不调用 LLM**（`memory_hooks.py:170-173`）。即超龄会话的内容被**永久放弃**且无任何日志说明"这是超龄跳过"。规格 R1.4 明确要求这个行为，但 Assumption 3 也说"confirm that silently skipping an old backlog is acceptable" —— **该假设从未被确认，实现已按"静默丢弃"落地**。

### 🟠 中严重度

10. **`store.py:617-625`** —— `recall()` 是写操作（打 `last_recalled_at` 后整文档重写）。`build_memory_note` 刻意避开（`recall.py:104` 用 `store.in_scope`），符合 R7.7；但**任何外部调用者**若误用 `store.recall` 当作只读查询，会在读路径上改写磁盘。

11. **`quotas.py:145-169`** —— `QuotaLease` 无 TTL，泄漏即永久占用并发槽。`chat_service.py:81-89` 用裸 `acquire`（非 `with`），依赖生成器 `_stream_turn` 的 `finally`（113 行）释放；**未被迭代的生成器不释放**。配合 `chat_service.py:69-70` 的"配额在回合锁之前"，一个卡住的回合可占死全部 4 个 principal 并发槽，后续所有请求 `QuotaExceeded`。

12. **`quotas.py:234-253`** —— 身份淘汰只删**已完全过期**的窗口；4096 个身份各有未过期事件时，任何新身份直接被拒。且淘汰只在 `_window` 被调用时触发，非后台定时。

13. **`quotas.py:127-131`** —— 窗口裁剪用 `>=`，窗口是 `(now-60, now]`。恰好 60.0s 的事件被挤出，边界处会有轻微的"提前放行"。

14. **`quotas.py:266-271`** —— `estimate_token_units = ceil(len/4)` 对 CJK 低估约 4 倍（每汉字 ≈ 1 token），token 配额实际形同虚设。

15. **`extraction.py:232-239`** —— `_call_model` 超时只放弃**等待**，不取消 daemon 线程；而 `scheduler` 的槽位要等 `_run` 返回才释放。一个卡死的 provider 调用会占满 `BoundedSemaphore` 槽直到自身结束。

16. **`extraction.py:388-400`** —— 若第一批候选写失败、后续候选写成功，`persisted.error_type` 非空 → `run()` 返回 `status="failed"` 且**不推进 watermark** → 下轮重新抽取已成功的那部分。靠 `store.save` 去重降级为 update，不产生重复记录，但浪费 LLM 调用。

17. **`extraction.py:187-192`** —— 候选 tags 只 trim 不校验长度；一条模型给的超长 tag 会让 `store.save` 整体拒收该候选（`store.py:466-470` 在 `oversized` 时 return error），**单条坏 tag 杀整条候选**。R4.7 没要求 extractor 侧长度校验，但 `_with_provenance`（`extraction.py:584-586`）同样不校验。

18. **`registry.py:419-426`** —— `_save_metadata` 吞掉所有异常只记 warning，内存 registry 与 SQLite 可静默漂移；`restore` 在重启时从磁盘重建，漂移会在重启时表现为一轮"回退"。

19. **`registry.py:142-143` / `176-177`** —— `create` 和 `restore` **无条件覆盖** `self._sessions[thread_id]`。用已存在的 thread_id 调 `create` 会替换旧 session 对象，旧对象的 isolated Chroma 目录**不做 cleanup**（`_finish_delete` 未被调用）→ 目录孤儿泄漏 + session ID 复用。

20. **`registry.py:181-190`** —— `get()` 无所有权校验，只有 `get_owned` 校验。`Principal.local_process()`（`identity.py:31-35`）是显式存在的"本地无鉴权"身份，`chat_service.py:124` 在无 principal 时回落到它。

21. **`checkpoint.py:36`, `343-355`** —— `enforce_ownership` 默认 `False`；关闭时 `snapshot_thread` / `restore_thread` / `delete_thread` 对任意 thread_id 无限制。SQL 全参数化，故无法用注入绕过 ownership 表，但 ownership 本身是开关。

22. **`registry.py:371-387`** —— `start_background_cleanup` 在锁内检查 `_cleanup_thread.is_alive()`，`stop_background_cleanup` **锁外读** `_cleanup_thread` → 竞态，可能 join 一个已被替换的线程对象。

23. **`.gitignore`** —— 无 `.corrupt-*` 规则（只有 `memory/`，第 30 行）。`design.md:60` 声称有，**记录失真**。自定义 `memory_store_path` 指向 `memory/` 之外时，`.corrupt-*` 文件会成为未忽略的未跟踪文件，可被误提交（其中可能含屏检上线前的旧内容）。

24. **`serialization.py:34-36` 类文档字符串** —— `CorruptDocumentError` 注释称"Raised only for a whole-document failure (unparseable JSON, or a top-level value that is not an object)"。代码确实如此（`64-72`）。**这条注释可信**，但意味着**所有字段级问题都不隔离、不备份**，只有两种整文档失败会。

25. **`models.py:34`** —— `MAX_SCOPE_ID_CHARS = 200` 定义后**全仓库零引用**（`save()` 与 `_parse_record` 均不校验）。R5.2 的 `scope_id` 1–200 约束两条路径都未实现。

26. **`memory_tool.py:38` + `store.py:430-470`** —— `FAILURE_MARKER = "MEMORY_ERROR:"`。`_truncate_success`（`memory_tool.py:357+`）只在成功串会**以该前缀开头**时才剥离（352-353 行）。而 `save()` 的确认串是 `"Saved memory <id> (category)."`，不会碰撞；`recall()` 的 no-match 串是 `"No matching memory was found."`，也不会。**R12.5 的"可区分性"在 store 层成立**，但依赖文案不漂移。

### 🟡 低严重度 / 澄清

27. **`watermark.py` 不是加密水印** —— 无校验和、无步长、无字符选择。是 0–1,000,000 的单调簿记整数。任务描述假定的"水印方程"在本仓库不存在。防御性亮点：`extraction.py:325` / `144` 的 `min(..., turns)` 夹值中和了"设极大值压制抽取"的攻击（只推迟一个 interval）。

28. **`scheduler.py` 无退避/抖动/重试公式** —— "调度方程"不存在；只有 `BoundedSemaphore` + 非阻塞 acquire + 靠 watermark 不前进实现的重试延迟（粒度 = 一个回合）。

29. **`transcript.py:83`** —— 非字符串、非 list、非 dict 的 content 走 `str(content).strip()`，会把对象 `repr` 送进 prompt。

30. **`transcript.py:115`** —— `is_memory_note` 只看 `content[:200]`。若注入的 recall note 前 200 字符内没有 marker/label 则无法识别 —— 但 `MEMORY_NOTE_LABEL`（27 字符）+ marker 必然在前 30 字符内，实际安全。

31. **`transcript.py:28-41`** —— `_NOTE_MARKER_CACHE` 首次调用时绑定一次，之后不刷新。模块级常量不会变，无实际影响。

32. **`store.py:116-135`** —— `resolve_store_path` 拦 `..` 段但不解析符号链接（`os.path.normpath` 不 resolve）；`_lock_for` 以 `Path` 为键，未 canonicalize → 同一文件经两条路径访问会得到两把不同的锁。

33. **`store.py:244-245`, `257-258`** —— `OSError`（非 `FileNotFoundError`）在**读路径上抛** `MemoryStoreError`，与 R5.6 的"不抛"不同范畴（R5.6 只覆盖 JSON 不可解析）。`recall()` 在 `store.py:601-605` 捕获它并返回错误串 ✓。

34. **`store.py:371-382`** —— `finally` 里 `tmp_path.unlink(missing_ok=True)` 吞 `OSError` 只记 warning，符合 R6.7。但注意 `os.replace` 成功时 `tmp_path` 被置 `None`（366 行）才不删 —— 逻辑正确。

35. **`identity.py:79-89`** —— 身份校验允许 128 个任意 Unicode 字符，只拒控制字符（`ord<32`）和 DEL。`ResourceOwner.authorizes` 用 `hmac.compare_digest` ✓。tenant 双 None 才算匹配 ✓。

36. **`storage.py:121-126`** —— `_copy_metadata` 只浅拷贝 `source_urls` 和 `config`，config 的嵌套可变值仍共享。

37. **`chat_service.py:69-70`** —— 配额租约在回合锁**之前**获取，`concurrent_calls` 默认 4 且回合锁无超时（我此前已记录）。`stream_turn`（81-89）同一顺序。

---

## 7. 未读到的文件

**明确未读**：
- `.kiro/specs/memory-auto-extraction/design.md`（620 行）—— 任务要求读取，但**未读**。§5.2 的对照仅基于 `memory-auto-extraction/requirements.md`。auto-extraction 的实现细节（组件接口、Correctness Property、测试策略）未与代码交叉验证。
- `.kiro/specs/chat-long-term-memory/tasks.md`（存在，未读）
- `.kiro/specs/memory-auto-extraction/tasks.md`（存在，未读）

**21 个分区文件全部读完**，无遗漏。

**为验证跨文件结论而旁读的**（不在分区内，未做完整审查）：
- `src/frontend/chat/memory_hooks.py`（228 行，全文）—— R1（session-boundary 选择）与 R7.3/R7.8 的实现处
- `src/backend/application/chat_service.py`（40-154 行）—— 配额/回合锁顺序
- `src/backend/application/turn_execution.py`（295-349 行）—— `restore_thread` 调用点
- `src/config/settings.py`（245-264 行）、`src/config/loader.py`（150-201, 204-240, 380-440 行）—— 配置与区间
- `src/backend/llm/prompts.py`（`MEMORY_EXTRACTION_PROMPT`，331 行起）—— R9.4 delimiter
- `src/backend/tools/memory_tool.py`（部分 grep）—— `FAILURE_MARKER` 与 `_guard`

**分区内但未深入交叉验证的行为**：
- `store.py` 的并发正确性未做实测（我只读代码，未运行）。R6.4 声明的"跨进程无互斥"未验证。
- `ExtractionScheduler` 的槽位泄漏路径（`thread_factory` 抛异常分支）只读逻辑未实测。
- `.chroma/`、`chat_uploads/`、真实 `*.sqlite3`、`.env` 全程未触碰（遵守纪律）。未运行任何测试、未发网络请求。
