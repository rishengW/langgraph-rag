# A7 — Web 搜索 + RAG 检索打分面（full-sweep-13）

> 原分区子代理死于 429 限流（仅留骨架，死前最后一句是"Critical citation-boundary finding"，该发现未及写出）。
> 本报告由**主代理自救完成**，聚焦本仓库公式最密集的检索打分面。
> 根目录 `E:\langgraph-rag`。全程静态阅读，未发起任何 HTTP 请求，未读写 `.chroma/`。

---

## 0. 三条最重要的结论

1. **`reputation.py` 把"不可达"当"低质量"惩罚**——一个临时宕机的域名会被打低信誉分，惩罚理由与内容质量无关。见 §2.1。
2. **`document_quality.py` 的关键词重叠闸门对中文基本失效**——`len(token) >= 3` 把大部分中文词元过滤掉，闸门静默全放行。而同一文件 `:241-242` 存在专门的 `_cjk_character_count`，说明代码知道中文的存在却没在这里处理。见 §2.3。
3. **仓库里存在两套互不相通的 recency 模型**——`web_search/recency.py` 是加分制阶梯分，`rag/document_quality.py` 是 0-1 连续排序键，两者语义、取值域、缺失日期处理都不同。见 §2.4。

---

## 1. 阈值与常量清单

| 常量 | 值 | 位置 |
|---|---|---|
| `DEFAULT_SEMANTIC_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `semantic.py:15` |
| `DEFAULT_MIN_SIMILARITY` | `0.35` | `semantic.py:16` |
| `SEMANTIC_MAX_BONUS` | `25` | `semantic.py:19` |
| `SEMANTIC_TEXT_MAX_CHARS` | `1000` | `semantic.py:20` |
| `DEFAULT_MIN_SAMPLES` | `5` | `reputation.py:21` |
| `MAX_REPUTATION_BONUS` | `10` | `reputation.py:23` |
| `MAX_REPUTATION_PENALTY` | `10` | `reputation.py:24` |
| recency 阶梯阈值 | `90` / `365` / `730` 天，未来容忍 `-31` 天 | `recency.py:65-72` |
| 单年命中/冲突 | `+20` / `-60` | `recency.py:57` |
| 未来日期 | `-12` | `recency.py:66` |
| 老年衰减封顶 | `-min(18, 4 + years_old*3)` | `recency.py:74-78` |
| 术语最小长度 | `len(token) >= 3` | `document_quality.py:238` |

---

## 2. 🧮 打分公式大全

### 2.1 域名信誉：`reputation_delta`（`reputation.py:154-170`）

```python
if stats is None or stats.attempts < max(1, int(min_samples)):
    return 0
ratio = stats.success_ratio                      # = grounded / attempts
if ratio >= 0.5:
    return round(MAX_REPUTATION_BONUS * min(1.0, (ratio - 0.5) / 0.5))
return -round(MAX_REPUTATION_PENALTY * min(1.0, (0.5 - ratio) / 0.5))
```

**结构**：以 0.5 为中点的**分段线性映射**，两侧对称斜率，`[0.5, 1.0] → [0, +10]`，`[0.0, 0.5] → [0, -10]`。`min_samples=5` 以下返回 0（冷启动不惩罚）。

⚠️ **建模缺陷（本文件最重要）**：`success_ratio` 的定义在 `DomainStats.success_ratio`（`:42-46`）：

```python
@property
def success_ratio(self) -> float:
    if self.attempts <= 0:
        return 0.0
    return self.grounded / self.attempts
```

`attempts` 在 `:86` 是**所有三类 outcome 的累加**（`attempts = domain_reputation.attempts + 1`，而 `:63-97` 的 `record` 对 `grounded`/`rejected`/`unreachable` 一视同仁地 +1）。因此：

- 分母含 `unreachable`，**网络不可达被计入失败**，与"内容被拒"混为一谈。
- 一个**临时宕机**的域名会持续拿到 `unreachable`，`ratio` 被稀释趋近 0，最终吃满 `-10` 惩罚。

而文件头注释（`:1-5`）声称："domains whose pages keep failing readability or relevance drift down"。**`unreachable` 既不是 readability 失败也不是 relevance 失败**——它纯粹是基础设施问题。注释描述的行为与代码实现的行为不一致。

**可操纵性**：`record()` 由内部抓取结果驱动，LLM 不直接调用。因此这不是外部注入面，而是**内部指标被系统性误读**。修复方向是把 `unreachable` 从分母剔除（`ratio = grounded / (grounded + rejected)`），或对 `unreachable` 走独立的中性通道。

**顺带一处可信注释**：`:80-89` 的 SQL 用 f-string 拼 `{column}`，但 `# noqa: S608` 注释说明 column 来自 `:73-77` 的固定字面量映射（且 `outcome` 在 `:67-72` 已先做集合校验）。这是合法的"列名无法参数化"模式，注释与代码一致，是少数可信的安全注释之一。

### 2.2 语义加分：`semantic_bonus`（`semantic.py:105-118`）

```python
threshold = max(0.0, min(1.0, float(min_similarity)))
if similarity <= threshold:
    return 0
span = 1.0 - threshold
if span <= 0:
    return SEMANTIC_MAX_BONUS
return round(SEMANTIC_MAX_BONUS * min(1.0, (similarity - threshold) / span))
```

**结构**：把相似度从 `[threshold, 1.0]` 线性重标定到 `[0, 25]`，取整，地板 0。默认 `threshold=0.35`、`span=0.65`。

**设计意图（注释 `:106-110` 明确）**：只加不减——相似度不达标贡献 0，语义层**只能救召回，不能惩罚词汇匹配强的结果**。这是有意的保守设计，防止一个强语义匹配压过硬证据闸门。上限 25 也是刻意有界。

⚠️ 两处边界：
- `similarity <= threshold` 用**闭区间**，所以相似度恰好等于 0.35 时得 0 分。与"阈值以上才算"的直觉相反（差一格的实现）。
- `span <= 0` 分支返回满额 25——当 `min_similarity=1.0` 时，任何 ≤1.0 的相似度都已被前一行拦掉，此分支实际只可能在浮点越界时触及，属防御性。

**`cosine_similarity`（`:92-102`）** 是标准实现：空向量/长度不等/零向量 → 0.0。注意 `:95` 先判 `len(left) != len(right)`，所以 `:97` 的 `zip(strict=True)` 只是双保险。

⚠️ **模型缓存无淘汰**：`default_semantic_scorer`（`:71-78`）在 `model_name` 变化时重建，但 `EmbeddingSemanticScorer` 懒加载的模型（`:39-47`）**常驻进程**，无 unload 路径。换模型名会留下旧模型不释放。

### 2.3 时效性：`assess_publication_date`（`recency.py:28-78`）

**输入判定**：`recency_requested` 由三者任一成立决定（`:44-48`）——查询含 4 位年份、含 `STATUS_INTENT`、或匹配 `_RECENCY_RE`。中文词表在 `:12-14`：`最新|当前|现在|近期|最近|截至(目前|现在)?|今年|今日|今天`。

**单年份路径**（`:52-59`）：
```python
conflicts = published_on.year != required_year
score = -60 if conflicts else 20
```
**命中 +20 / 冲突 -60，不对称 3:1。** 有理由（错年份比对年份更值得惩罚），但值得记录。

**多年份路径**（`:61-62`）：直接 `return PublicationDateAssessment(applies=True)`，**score=0，完全不打分**。注释（`:36-38`）给了理由：一个对比页不可能有多个发布日期，交给正文覆盖检查处理。设计合理。

**时效阶梯**（`:64-78`）：
```python
age_days = ((today or date.today()) - published_on).days
if age_days < -31:   score = -12      # 未来日期超一个月
if age_days <= 90:   score = +18
if age_days <= 365:  score = +12
if age_days <= 730:  score = +4
years_old = max(0, (age_days - 730) // 365)
score = -min(18, 4 + years_old * 3)
```

⚠️ **730 天处有 8 分断崖**：`age_days == 730` → `+4`；`age_days == 731` → `years_old = 0` → `-min(18, 4+0) = -4`。跨过两年线瞬间掉 8 分。

⚠️ **衰减封顶**：`4 + 3*years_old >= 18` 当 `years_old >= 5`（即约 7 岁+），此后无论多老都是 `-18`。

**缺失日期 → 中性**（`:49-50`）：返回 `applies=recency_requested, score=0`，**绝不惩罚**。注释 `:33-34` 明确这是设计目标。

⚠️ **`today` 的默认值**：`:64` 是 `(today or date.today())`。`today` 作为参数传入时可以是任意值——测试友好，但生产路径若漏传就用墙钟，测试与生产行为可能分叉。

### 2.4 文档质量时效：`_recency_score`（`document_quality.py:305-309`）

```python
window = max(0, bias_days)
if window <= 0 or age_days > window:
    return 0.0
return 1.0 - (age_days / window)
```

**线性归一化到 `[0, 1]`**，`age=0` → 1.0，`age→window` → 0⁺，超过 window 硬断崖到 0.0。

⚠️ **与 §2.3 是两套互不相通的模型**：

| | `web_search/recency.py` | `rag/document_quality.py` |
|---|---|---|
| 取值域 | 整数，`[-60, +20]` | 浮点 `[0.0, 1.0]` |
| 用法 | **加法式** delta，叠加进总分 | **排序键**，`reverse=True` 直接决定顺序 |
| 缺失日期 | 中性 0（不惩罚） | 未标注 → 排序键默认 0.0（排最后） |
| 形状 | 阶梯 + 封顶线性衰减 | 纯线性 + 硬断崖 |
| 未来日期 | 显式 `-12` | `age_days` 先 `max(0, ...)` 截断（`:297`）→ 未来文档得满分 1.0 |

**最后一个格子是实质分歧**：`document_quality.py:297` 是 `age_days = max(0, (date.today() - published_date).days)`——**未来日期的文档在这套模型里拿满分 1.0**，而 `recency.py` 明确给 `-12`。一个伪造未来发布日期的页面，在 RAG 侧被最优先，在搜索侧被降权。

`_bias_recent_documents`（`:312-317`）用该分数做稳定排序。**`window <= 0`（功能关闭）时全部文档得 0.0 → 稳定排序退化为输入顺序**，不崩溃，是合理的降级。

### 2.5 关键词重叠闸门（`document_quality.py:213-238`）

```python
semantic_match = _matches_semantic_relevance(...)
if semantic_match is not None:
    return semantic_match                     # 语义优先，失败则回退

query_terms = _query_terms(config.relevance_query)
required = max(0, config.min_query_term_overlap)
if required <= 0 or not query_terms:
    return True                               # ← 这里
document_terms = set(_tokens(text))
return len(query_terms.intersection(document_terms)) >= required
```

```python
def _query_terms(query: str) -> set[str]:
    return {token for token in _tokens(query) if len(token) >= 3 and token not in STOP_WORDS}
```

⚠️ **`len(token) >= 3` 对中文基本失效。** 中文分词出的词元多为 1-2 字，经此过滤后 `query_terms` 大概率变空 → `not query_terms` 为真 → `return True`，**闸门对所有文档全放行**。

⚠️ **代码本身知道中文的存在**：同文件 `:241-242` 有专门的
```python
def _cjk_character_count(text: str) -> int:
    return sum(1 for char in text if "一" <= char <= "鿿")
```
说明作者意识到 CJK 需要特殊处理，但这个过滤用的是**按词元长度**的启发式，没有用它。**这是相关性闸门对中文内容的静默失效**，而非硬崩溃。

**语义回退路径**：`_matches_semantic_relevance`（`:200-210`）在 embedding 不可用或异常时返回 `None`（`:204`、`:209`），此时**落到关键词闸门**——也就落到上面这个对中文失效的分支。两条路径的中文支持度不一致：语义路径用多语言模型（`paraphrase-multilingual-MiniLM`），关键词路径用拉丁词元长度启发式。

### 2.6 余弦相似度三重实现（跨分区发现）

三处独立实现，判据不一致：

| 位置 | 长度判据 | `zip` | 零向量判据 |
|---|---|---|---|
| `web_search/semantic.py:92-102` | `len(left) != len(right)` → 0.0 | `strict=True` | `<= 0` |
| `rag/document_quality.py:258-266` | `len(left) != len(right)` → 0.0 | **`strict=False`** | `== 0.0` |
| `graph/nodes/common.py:509-519`（A1 子代理报告） | 未逐一核对 | 未逐一核对 | 未逐一核对 |

`strict=False` 在 `document_quality.py:261` 意味着长度不等时 `zip` 会**静默截断到较短向量**，算出一个错误的相似度而不是报错。当前有 `:259` 的长度守卫兜底，但若该守卫被移除就是静默错误。**三处应当抽公共函数**——这是本仓库重复实现漂移的典型样本。

---

## 3. 📊 检索流水线（定性）

```
query
  │
  ├─ 年份/时效意图抽取 (_YEAR_RE, _RECENCY_RE, STATUS_INTENT)
  │
  ├─ 关键词闸门 ── 语义路径优先 (多语言 embedding, ≥ min_similarity)
  │                └─ 失败 → 关键词重叠闸门 (⚠️ 中文失效)
  │
  ├─ lexical bigram 覆盖分 (硬闸门，语义层不可覆盖)
  │     └─ + semantic_bonus ∈ [0, 25]   ← 只加不减
  │
  ├─ + reputation_delta ∈ [-10, +10]   ← ⚠️ unreachable 计入失败
  │
  └─ + recency score ∈ {-60,-12,-4.., +4,+12,+18}  ← 单年 ±20/-60
```

RAG 侧（`document_quality.py`）是**独立的一套**：内容哈希去重（`:269-271` SHA-256）→ 相关性判定（语义/关键词）→ `_annotate_recency` 打 `document_quality_recency_score` 元数据 → `_bias_recent_documents` 按该键稳定降序排序。

---

## 4. 发现汇总

| # | 发现 | 位置 | 严重度 |
|---|---|---|---|
| 1 | `unreachable` 计入 `attempts` 分母，宕机域名被当低质量惩罚，与文件头注释描述的行为矛盾 | `reputation.py:43-46,63-97` vs `:1-5` | **高** |
| 2 | `len(token) >= 3` 使关键词重叠闸门对中文静默全放行；同文件有 `_cjk_character_count` 却未在此使用 | `document_quality.py:238` vs `:241-242` | **高** |
| 3 | 未来发布日期：RAG 侧给满分 1.0（`max(0, ...)` 截断），搜索侧给 -12，两套模型结论相反 | `document_quality.py:297` vs `recency.py:65-66` | 中 |
| 4 | 730 天时效断崖：730 天 +4 → 731 天 -4，8 分跳变 | `recency.py:71-77` | 中 |
| 5 | 单年匹配/冲突不对称 ±20/-60 | `recency.py:57` | 低（有意） |
| 6 | `semantic_bonus` 阈值用闭区间，恰好等于阈值得 0 分 | `semantic.py:113` | 低 |
| 7 | 余弦相似度三重实现，`strict=False` 判据不一致 | `semantic.py:92-102`、`document_quality.py:258-266`、`graph/nodes/common.py:509-519` | 中 |
| 8 | `EmbeddingSemanticScorer` 模型常驻，换模型名无 unload 路径 | `semantic.py:39-47,71-78` | 低 |
| 9 | 语义路径与关键词路径的中文支持度不一致，embedding 失败时静默落到弱路径 | `document_quality.py:200-234` | 中 |
| 10 | 两套 recency 模型互不相通（取值域、用法、缺失/未来日期处理全部不同） | `recency.py` vs `document_quality.py` | 中 |

**可信注释**：`reputation.py:89` 的 `# noqa: S608` 注释与代码一致（column 来自固定字面量映射且 outcome 已校验），是少数可以信任的安全注释。`semantic.py:17-19`、`:106-110` 的设计意图注释与实现一致。

---

## 5. 未读到 / 不确定（重要）

本报告**只覆盖了公式面**。以下分区任务**未完成**，明确列出以免被误读为已验证：

- **引用边界 / `--- Source:` 伪造链**：A7 子代理死前正要说这个"关键发现"，**该发现未写出、主代理未独立验证**。A4 分区已部分覆盖：`prompt_builder.py:97` 依赖 `content_fetcher.py:360-363` 的 `normalize_whitespace` 与 `_document_title:799-807` 的换行归一化这一**未声明的巧合不变量**，`prompt_builder` 自身零净化。这是已有结论，但 A7 那侧可能有更新的信息。
- `evidence.py`、`claim_consensus.py`（共识投票/一致率算法）、`query_constraints.py`、`query_prep.py`、`date_extractor.py`、`page_structure.py`、`content_fetcher.py`、`pdf_loader.py`、`factory.py`、`fetch_policy.py`、`prompt_builder.py`、`benchmark.py` —— **均未读**。
- `rag/chunk_context.py`、`rag/chunk_*`（分块大小/overlap/token 计数）、`rag/chroma_retriever.py`、`rag/retriever.py`、`rag/embeddings.py`、`rag/dashscope_embeddings.py`、`rag/hf_embeddings.py`、`rag/document_loader.py` —— **均未读**（分块方程与嵌入归一化未验证）。
- `web_search/` 下 `baidu.py`、`bing.py`、`duckduckgo.py`、`discovery.py`、`api_providers.py`、`playwright_loader.py` 的**出站请求面**未验证（A4 已声明其不在验证范围）。
- `graph/nodes/common.py:509-519` 的第三处余弦实现**未亲自核对**，仅据 A1 子代理报告转述。
- 未读 `.kiro/specs/` 规格文档做规格对照（那是 A6 的任务）。
