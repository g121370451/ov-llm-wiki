# Wiki

`openviking/wiki` 是可复用的 Wiki 生成核心：输入一批已经进入 OpenViking 的资源，输出一棵写在 `viking://wiki/...` 下的可浏览 Wiki。

这个目录不负责 benchmark，也不读取固定数据集。固定数据集实验放在 `benchmark/wiki`；服务侧通过独立的 `WikiService` 调用这里的通用能力。

## 当前契约

当前实现优先保证 Wiki 正文稳定生成：

- 生成 Document Card、Wiki 节点、节点 card 和节点 Markdown 正文。
- 保留节点级来源列表，写入 `source_assignments.json` 和 `nodes/<node_id>/sources/*.ref.json`。
- 正文生成阶段暂不让模型输出 claim 级 `evidence_refs` 或父层 `support_refs`。
- 不再做“生成句子必须逐字出现在正文中”的运行时校验。

这是当前设计选择。之前的 claim/ref 方案会让模型复制长 URI 或额外临时标识符，稳定性不够，而当前消费侧还没有必须依赖 claim 级 drill down 的需求。后续如果恢复 claim 级引用，应使用短临时引用交给模型复制，再由代码映射回持久 ID。

## 模块职责

```text
openviking/wiki/
├── config.py
│   └── WikiConfig 和 WikiGenerationLimits。
├── schemas.py
│   └── Card、Node、SourceRef、Document、Manifest 等 Pydantic 契约。
├── document_manifest.py
│   └── 读写资源根目录下的文档边界 manifest，并把边界展开成 WikiResourceInput。
├── service.py
│   └── 独立 Wiki 服务，负责 build_wiki、clear_wiki 和资源输入展开。
├── router.py
│   └── FastAPI 路由，提供 /api/v1/wiki/build 和 /api/v1/wiki/clear。
├── uri.py
│   └── Wiki 产物 URI 生成规则。
├── writer.py
│   └── VikingFS 写入器，负责目录、JSON、JSONL 和 Markdown 写入。
├── llm.py
│   └── Wiki LLM 调用封装，记录 prompt、schema hash 和结构化输出。
├── prompts.py
│   └── 渲染 openviking/prompts/templates/wiki/ 下的模板。
├── cards.py
│   └── 为资源文档和 Wiki 节点生成 Document Card。
├── nodes.py
│   └── 发现底层节点和父层节点。
├── assignments.py
│   └── 把节点发现结果里的来源 ID 确定性转换成 SourceRef。
├── documents.py
│   └── 生成节点 Markdown 正文，并处理结构化输出重试。
├── layer_decision.py
│   └── 判断是否继续向上生成父层节点。
├── content_loader.py
│   └── 服务侧构建时，从 VikingFS/VikingDB 加载 summary 或 raw chunk。
└── pipeline.py
    └── 总编排器，串起完整 Wiki 生成流程。
```

推荐阅读顺序：

1. `schemas.py`：先看最终数据契约。
2. `document_manifest.py`、`service.py`：看独立接口如何把资源 root 展开成文档输入。
3. `prompts.py` 和 `openviking/prompts/templates/wiki/`：看每一步给模型的输入。
4. `cards.py`、`nodes.py`、`assignments.py`、`documents.py`、`layer_decision.py`：按阶段看行为。
5. `pipeline.py`：看编排、过滤和写入时机。
6. `content_loader.py`、`writer.py`：看服务侧如何加载内容和写产物。

## 生成流程

```text
资源文档
  -> Document Cards
  -> 节点发现
  -> SourceRef 构造
  -> 节点正文
  -> 节点 card
  -> 可选上层节点发现
  -> Manifest 和运行日志
```

核心边界是 Document Card。节点发现不直接读所有长文档，而是先读压缩后的 cards。节点正文生成时，再拿该节点被选中的来源文档内容或下层节点正文内容做综合写作。`WikiNode.scope` 是节点的权威边界，长期保存在 `nodes.json`，并继续约束正文生成；node card 的 `summary` 不替代 `scope`。

## 资源输入和文档边界

独立 `build_wiki` 接口接收的是已经入库的资源 URI：

```python
client.build_wiki(resource_uris=["viking://resources/qasper_30_processed_docs"])
```

如果传入的是目录资源，Wiki 阶段不能靠递归扫描目录来猜哪些路径是一篇文档。复杂目录里同一层级可能同时包含章节、chunk、图片、表格和用户自己组织的子目录，启发式扫描不可靠。

因此，文档边界由入库解析阶段提供。`add_resource` 完成解析和落盘后，会在资源 root 下写一个隐藏 manifest：

```text
viking://resources/qasper_30_processed_docs/.wiki_documents.json
```

manifest 记录 parser 当时识别出的文档边界。目录导入 30 篇 markdown 时，这里应该有 30 条记录。`WikiService.build_wiki(...)` 收到资源 root 后会先读取这个 manifest：

- manifest 存在且有文档记录：按记录展开成多篇 `WikiResourceInput`，每条记录生成一个 Document Card；
- manifest 不存在或没有记录：保守 fallback，把传入的资源 URI 当作一篇资源文档处理。

当前 manifest 只用于把资源 root 展开到文档级输入，不用于传递正文摘要、abstract、metadata 或 benchmark 信息。

### `doc_id`、`title` 和 `relative_uri`

`ResourceDocumentDraft` 是 parser 交给 Wiki 的文档边界记录：

```python
class ResourceDocumentDraft(StrictModel):
    doc_id: NonEmptyStr
    title: NonEmptyStr
    relative_uri: str = ""
```

字段来源：

- `relative_uri`：文档相对资源 root 的路径，由 parser 在解析时确定。这是识别文档边界的核心字段。
- `doc_id`：parser 基于文档名或相对路径归一化得到，Wiki 阶段用它作为内部文档 key。
- `title`：parser 基于 markdown 标题或文件名得到，Wiki 阶段给模型和人类展示。

字段用途：

- `relative_uri` 用来拼出文档资源 URI，例如 `root_uri + relative_uri`。
- `doc_id` 贯穿 card、node discovery、source assignment 和 `sources/<doc_id>.ref.json`。
- `title` 进入 Document Card prompt、card JSON 和来源展示。

例子：

```text
本地目录 qasper_30_processed_docs/
├── paper_a.md
└── nested/
    └── paper_b.md
```

入库后 manifest 可能记录：

```json
{
  "version": 1,
  "documents": [
    {"doc_id": "paper_a", "title": "Paper A", "relative_uri": "paper_a"},
    {"doc_id": "nested_paper_b", "title": "paper_b", "relative_uri": "nested/paper_b"}
  ]
}
```

`build_wiki` 展开后会生成两篇输入：

```text
viking://resources/qasper_30_processed_docs/paper_a
viking://resources/qasper_30_processed_docs/nested/paper_b
```

每篇输入分别生成一个 Document Card。

### 节点发现与 node card

默认后端是 `facet_graph`。原始文档先生成独立的 `DocumentFacetSet`，节点发现只读取 `facet_text` 的 embedding，不读取 Card 的 `summary` 或 `candidate_topics`。近邻图同时使用 top-k 和绝对分数阈值，排除同一文档内部的 facet 边，再用 Leiden/CPM 发现候选社区。LLM 只为成熟社区生成 `title` 和 `scope`，不再决定来源归属。

Card 仍有三个用途：保存文档摘要、为大节点正文生成大纲、为 `SourceRef` 提供标题和 URI。`llm_full_context` 后端保留为小语料质量基线，只有显式选择时才会让 LLM 根据 Card 发现节点。

代码随后：

- 按不同 `doc_id` 检查社区来源数；
- 根据社区成员直接生成 source assignment；
- 在 `SourceRef` 中保存实际命中的 facet 和证据 URI；
- 为 active 节点生成正文与 node card；

`assignments.py` 不调用 LLM。它只校验来源 ID，并用 Card 的身份信息和已匹配 facet 构造 `SourceRef`。

### 上层节点

每个 node document 都会按 Markdown 标题拆成 source sections，再生成自己的 `DocumentFacetSet`。下一层继续用这些 node facets 建图和聚类，和底层沿用同一语义契约。node card 不参与 `facet_graph` 的父层发现。

上层正文 prompt 的重点是综合：输出应围绕当前层知识点组织，而不是按照 source 顺序逐个总结。

`facet_graph` 会继续尝试下一层，直到没有满足 `min_child_nodes_per_parent` 的成熟社区、候选父节点数没有严格少于当前层节点数，或达到 `max_depth`。父层中 child 集合完全相同的 facet 社区先合并为一个候选父节点并汇总支持 facets；这样既允许同一个下层节点通过不同 facets 支持多个父节点，又不会把相同的一批 children 重复包装成多个父节点。无严格节点数压缩时会在调用 LLM 命名和写入父节点前停止。`LayerDecisionRunner` 只服务 `llm_full_context` 基线。

### Bot context tree

MCP 侧的 `context_tree` tool 用于给 bot 展示某个 `viking://` URI 附近的目录上下文。它不改变 Wiki 生成结果，只读取 `nodes.json`、`source_assignments.json` 和 resource 文件树。

- 输入 resource 文档内部的文件或目录时，从该路径的直接父 resource 目录开始向下展开，不进入 Wiki DAG。
- 输入完整文档 resource root 时，从直接引用该文档的 Wiki nodes 出发，完整展开这些 nodes 的下游子图。
- 输入 Wiki node URI 时，从它的直接父 nodes 出发；没有父节点时从自身出发。
- resource 树只允许从已知完整文档 root 或其子目录展开，禁止从 `viking://`、`viking://resources` 或资源集合目录展开。
- 输出使用 `[N:<node_id>]` 和 `[D:<doc_id>]` 短引用，完整文档 URI 只在结果的 `Document URI map` 中出现一次。

## 产物结构

假设 `wiki_root_uri` 是 `viking://wiki/my_wiki/`，管线会写出：

```text
viking://wiki/my_wiki/
├── nodes.json
├── source_assignments.json
├── cards/
│   └── <doc_id>.card.json
├── facets/
│   ├── manifest.json
│   ├── <doc_id>.facets.json
│   └── nodes/
│       ├── manifest.json
│       └── <node_id>.facets.json
├── clustering/
│   └── runs/
│       ├── depth_0001.json
│       └── depth_0001.edges.jsonl
├── nodes/
│   └── <node_id>/
│       ├── card.json
│       ├── documents/
│       │   └── document.md
│       └── sources/
│           └── <ref_id>.ref.json
└── run/
    ├── config.json
    ├── prompts.jsonl
    ├── raw_outputs.jsonl
    └── logs.md
```

关键文件：

- `nodes.json`：所有发现节点，包括层级、父子关系和 rejected 节点。
- `source_assignments.json`：节点级来源引用和未分配来源。
- `cards/*.card.json`：原始文档的结构化 card。
- `facets/manifest.json` 和 `facets/*.facets.json`：原始文档的 facet cache。
- `facets/nodes/`：各层 node document 的 facet cache 和输入指纹。
- `clustering/runs/`：逐层图参数、社区成员和近邻边，供聚类质量审查。
- `nodes/<node_id>/card.json`：节点摘要，供正文生成辅助和来源引用使用。
- `nodes/<node_id>/documents/document.md`：综合生成的唯一节点正文。
- `nodes/<node_id>/sources/*.ref.json`：该节点可用的来源列表。
- `run/prompts.jsonl`：每次 LLM 调用的 prompt、schema name 和 schema hash。
- `run/raw_outputs.jsonl`：模型返回并成功解析后的结构化输出。

## LLM 行为

所有 Wiki LLM 调用都经过 `WikiLLMRunner.complete_json(...)`，再由 `StructuredVLM` 使用 JSON Schema response format 调模型。Prompt 模板放在 `openviking/prompts/templates/wiki/`。

当前会调用 LLM 的阶段：

- `document_card`
- `document_facets_batch_*`
- `facet_community`
- `node_documents`
- `node_document_outline`
- `node_documents_initial`
- `node_documents_refine`
- `node_card`

`node_discovery` 和 `next_layer_decision` 只在 `llm_full_context` 基线中调用。

文档生成阶段带结构化输出重试：模型返回的 JSON 或 Markdown 不符合契约时，会用同一个干净 prompt 最多重试 3 次。重试 prompt 不追加校验错误，避免污染模型注意力。

Prompt 边界是产品契约的一部分：

- 不能使用 benchmark question、gold answer、评测标签或目标答案。
- 不能使用外部知识。
- 每一步只能使用该阶段明确提供的资源、card 或子节点正文内容。
- 上层正文必须综合多个下层 source 文档，不要按 source 一段一段总结。

## 服务侧接入

服务入口是独立 Wiki 接口：

```text
POST /api/v1/wiki/cards/build
POST /api/v1/wiki/build
POST /api/v1/wiki/clear
```

SDK 对应方法：

```python
resource = client.add_resource(path="/path/to/docs", wait=True)
cards = client.build_wiki_cards(resource_uris=[resource["root_uri"]])
wiki = client.build_wiki(resource_uris=[resource["root_uri"]])
client.clear_wiki(preserve_cards=True)
```

`WikiService.build_wiki_cards(...)` 负责：

1. 接收已入库的 `viking://resources/...` URI。
2. 校验资源存在，并尝试读取每个 resource root 下的 `.wiki_documents.json`。
3. 如果存在文档边界 manifest，就按文档记录展开成多个 `WikiResourceInput`；否则把 resource root 当作单篇输入。
4. 按 `summary` 或 `raw_chunk` 输入生成 Document Cards。
5. 写入 card 文件、运行记录和最后提交的严格校验 manifest。

`WikiService.build_wiki(...)` 只加载并验证已有 cards，然后执行 node discovery、source
assignment、node documents、node facets、node cards 和上层聚合。默认使用 `facet_graph`，
可显式选择 `llm_full_context` 作为小规模基线。cards 缺失或指纹过期时会直接失败，
不会隐式重新生成。

`WikiService.clear_wiki(...)` 删除 `wiki_root_uri` 下的 Wiki 产物，默认是 `viking://wiki/`。底层 `VikingFS.rm(..., recursive=True)` 会联动清理这些 Wiki 文件对应的向量索引。清理接口固定幂等：目标不存在也返回成功。它不删除 `viking://resources/...` 下的原始入库文档、语义摘要、资源向量索引或 `.wiki_documents.json`。因此：

```text
add_resource -> build_wiki_cards -> build_wiki -> clear_wiki
```

清理后资源库状态应与只执行 `add_resource` 后一致。

节点调优时可调用 `clear_wiki(preserve_cards=True)`，此时只删除 nodes、source assignments
和节点阶段运行记录，同时删除 node facets 与聚类运行记录，保留原始 Document Cards 和原文档 facets。

`ResourceService.add_resource(...)` 不再接受 `build_wiki`、`wiki_card_input_mode` 或 `wiki_max_card_input_chars`。调用方必须先完成资源入库，再显式调用 Wiki 构建。

`WikiContentLoader` 支持两种 card 输入模式：

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `summary` | 读取语义摘要、overview 和 chunk abstract | 语义生成已完成且质量可用 |
| `raw_chunk` | 直接读取原始 chunk 内容 | 没有摘要，或需要绕过摘要质量问题 |

如果 `summary` 模式读不到可用摘要，构建会提前失败。此时要么先完成语义生成，要么切到 `raw_chunk`。

`max_card_input_chars` 只限制 Document Card 摘要 prompt 的输入长度，防止长文超过模型上下文。facet extraction 不使用这份截断输入，而是遍历完整 `source_sections` 后分批处理。

## 常用配置

生成规模主要由 `WikiGenerationLimits` 控制：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `max_depth` | 最多生成几层 Wiki 节点 | `6` |
| `min_refs_per_node` | 底层节点最少需要多少个文档来源 | `3` |
| `min_child_nodes_per_parent` | 父节点最少需要多少个子节点 | `3` |
| `max_concurrent_cards` | Document Card 并发生成数 | `10` |
| `max_concurrent_nodes` | 节点正文并发生成数 | `10` |
| `max_facet_batch_chars` | 单批 facet extraction 的字符上限 | `30000` |
| `facet_neighbor_limit` | 每个 facet 保留的跨文档近邻数 | `20` |
| `facet_edge_score_threshold` | 相似图绝对分数阈值 | `0.72` |
| `facet_cpm_resolution` | Leiden/CPM 聚类粒度 | `0.7` |

有些配置字段仍是扩展预留。判断真实行为时，以 `pipeline.py` 中的过滤和编排逻辑为准。

## 排查问题

如果 Wiki 没启动：

- 确认调用方在 `add_resource(wait=True)` 后先调用了 `build_wiki_cards(...)`，再调用
  `build_wiki(...)`。
- 看服务日志是否出现 Wiki pipeline 的 build 日志。
- 看返回摘要里是否有 `wiki_root_uri`、card 数和 node 数。

如果 `build_wiki` 报 Document Card cache missing or stale：

1. 查看 `viking://wiki/cards/manifest.json` 是否存在。
2. 检查源文档集合、card 输入模式、截断长度或 document-card prompt/schema 是否变化。
3. 重新执行 `build_wiki_cards(...)`，不要在 `build_wiki` 中隐式生成 cards。

如果没有生成节点正文：

1. 看 `nodes.json`，确认节点是 `active` 还是 `rejected`。
2. 看 `source_assignments.json`，确认来源数量是否足够。
3. 检查 `min_refs_per_node` 或 `min_child_nodes_per_parent` 是否过严。
4. 看 `run/prompts.jsonl`，确认该阶段真实输入。
5. 看 `run/raw_outputs.jsonl`，确认模型结构化输出。

如果是模型输出不稳定：

- 先看对应阶段是否已经有重试。
- 对 LLM 导致的格式问题，优先用同一个干净 prompt 重试。
- 不要默认把 Pydantic 错误或实现细节拼进重试 prompt，除非有明确产品理由。

核心单测：

```bash
uv run pytest tests/wiki -q
```

benchmark 全流程在包外执行：

```bash
rm -rf benchmark/wiki/wiki_storage/qasper_30 benchmark/wiki/Output/qasper_30/wiki
uv run python benchmark/wiki/run.py --config benchmark/wiki/config/qasper_30.yaml
```

重跑完整 benchmark 前先删除旧产物，避免把上一次失败或中断留下的 Wiki 文件混进判断。

## 接入新输入

服务/API 层的推荐接入方式是：

- 先用 `add_resource` 把资源写入 `viking://resources/...`。
- parser 在入库阶段写 `.wiki_documents.json`，记录文档边界。
- 调用 `build_wiki_cards(resource_uris=[root_uri])` 生成可复用 cards。
- 再调用 `build_wiki(resource_uris=[root_uri])` 构建 Wiki nodes。

如果新增 parser，希望它支持目录资源的文档级 Wiki 构建，就需要在 `ParseResult.wiki_document_drafts` 中返回文档边界。每条 draft 当前需要：

- 稳定的 `doc_id`；
- 文档相对资源 root 的 `relative_uri`；
- 非空 `title`。

这些字段只用于定位和标识文档，不应夹带摘要、abstract、metadata、benchmark gold answer、评测标签或目标答案。

如果绕过服务层，调用方应依次调用
`WikiPipeline.generate_document_cards_from_inputs(...)` 和
`WikiPipeline.run_from_stored_cards(...)`，并自己提供 `WikiResourceInput`。每个
`WikiResourceInput` 必须包含：

- 稳定的 `doc_id`；
- 真实的 `resource_uri`；
- 非空 `title`；
- loader 能读取到的 summary、abstract 或 chunk 内容；
- 不包含 benchmark gold answer、评测标签或只应评测侧可见的目标信息。

固定数据集 adapter 应放在 `benchmark/wiki`，不要放进 `openviking/wiki`。

## 当前限制和后续方向

当前限制：

- 首次建库已支持逐层 facet graph；nodes 仍只支持全量重建，不支持局部增量合并。
- HNSW 只用于当前层近邻搜索；历史 facet ANN、embedding manifest 和增量索引尚未实现。缺少 HNSW 依赖时会明确记录 `exact` fallback。
- Leiden/CPM 是 `facet_graph` 的固定社区算法；缺少 `python-igraph` 或 `leidenalg` 时构建会直接失败。
- candidate micro-cluster、node registry、稳定 node identity 对齐和 assignment threshold 尚未实现。
- 服务内构建目前固定写入 `viking://wiki/`。
- claim 级 citation 暂时关闭。
- 中途失败可能留下部分已写产物。
- 模型后端必须支持结构化 JSON response format。

后续方向：

- 引入发布/提交语义，避免失败 run 暴露半成品 Wiki。
- 在服务/API 层支持显式 `wiki_id` 或可配置 Wiki root。
- 做增量更新：复用已有 cards，只重算受影响节点。
- 如果消费侧需要，再恢复 claim 级引用，并用短临时 ref + 代码映射实现。
- 考虑 agent 化编排，让生成流程具备计划、检查和局部修复能力，同时继续以 `schemas.py` 作为最终落盘契约。
