import json

import pytest

from openviking.models.embedder.base import EmbedResult
from openviking.wiki.config import WikiConfig, WikiGenerationLimits
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.pipeline import WikiPipeline, _node_document_sections
from openviking.wiki.schemas import (
    DocumentCard,
    DocumentFacetSet,
    NodeDocument,
    PipelineArtifacts,
    ResourceDocument,
    SourceAssignmentResult,
    SourceRef,
    SourceSection,
    WikiNode,
    WikiResourceInput,
)
from openviking.wiki.writer import WikiVikingFSWriter

from .fakes import FakeClient, FakeVLM


@pytest.mark.asyncio
async def test_pipeline_generates_layer_content_before_next_layer_decision():
    docs = [_doc(index) for index in range(1, 4)]
    wiki_inputs = [_wiki_input(doc) for doc in docs]
    card_llm = WikiLLMRunner(FakeVLM([_card_content_response(index) for index in range(1, 4)]))
    node_llm = WikiLLMRunner(
        FakeVLM(
            [
                _node_discovery_response(),
                {"markdown": "# Question Answering\n\n## Methods\n\nSynthesized QA knowledge."},
                _node_card_content_response(),
                {"continue_upward": False, "reasons": ["no stable parent layer"]},
            ]
        )
    )
    client = FakeClient()
    config = WikiConfig(node_discovery_backend="llm_full_context")
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )

    await WikiPipeline(
        writer=writer,
        config=config,
        llm=card_llm,
    ).generate_document_cards_from_inputs(
        wiki_inputs,
        content_loader=FakeContentLoader(docs),
        resource_uris=["viking://resources/"],
    )
    artifacts = await WikiPipeline(
        writer=writer,
        config=config,
        llm=node_llm,
    ).run_from_stored_cards(
        wiki_inputs,
        content_loader=FakeContentLoader(docs),
        resource_uris=["viking://resources/"],
    )

    assert [record.step for record in card_llm.log.raw_outputs] == [
        "doc_card",
        "doc_card",
        "doc_card",
    ]
    assert [record.step for record in node_llm.log.raw_outputs] == [
        "node_discovery",
        "node_documents",
        "node_card",
        "next_layer_decision",
    ]
    assert "viking://wiki/nodes/question_answering/documents/document.md" in client.writes
    assert "viking://wiki/nodes/question_answering/card.json" in client.writes
    assert "viking://wiki/nodes/question_answering/node.md" not in client.writes
    assert "viking://wiki/nodes/question_answering/evidence.jsonl" not in client.writes
    assert artifacts.node_contexts[0].document.title == "Question Answering"


@pytest.mark.asyncio
async def test_pipeline_run_config_redacts_sensitive_vlm_config():
    client = FakeClient()
    config = WikiConfig(
        vlm_config={
            "provider": "volcengine",
            "api_key": "secret-key",
            "nested": {"token": "secret-token", "model": "demo"},
        }
    )
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )

    await WikiPipeline(
        writer=writer, config=config, llm=WikiLLMRunner(FakeVLM([]))
    )._write_run_records()

    run_config = json.loads(client.writes["viking://wiki/run/config.json"])
    assert run_config["model_config"]["provider"] == "volcengine"
    assert run_config["model_config"]["nested"]["model"] == "demo"
    assert run_config["model_config"]["api_key"] == "***REDACTED***"
    assert run_config["model_config"]["nested"]["token"] == "***REDACTED***"
    assert "secret-key" not in client.writes["viking://wiki/run/config.json"]
    assert "secret-token" not in client.writes["viking://wiki/run/config.json"]


@pytest.mark.asyncio
async def test_pipeline_does_not_precreate_unassigned_active_node_dirs():
    docs = [_doc(index) for index in range(1, 4)]
    wiki_inputs = [_wiki_input(doc) for doc in docs]
    card_llm = WikiLLMRunner(FakeVLM([_card_content_response(index) for index in range(1, 4)]))
    node_llm = WikiLLMRunner(
        FakeVLM(
            [
                {
                    "nodes": [
                        {
                            "title": "Question Answering",
                            "scope": "QA methods and evaluation.",
                            "supporting_source_ids": ["OARW_1", "OARW_2", "OARW_3"],
                            "merged_candidate_topics": ["question answering"],
                        },
                        {
                            "title": "Unassigned Topic",
                            "scope": "No assigned sources.",
                            "supporting_source_ids": ["OARW_1"],
                            "merged_candidate_topics": ["unknown topic"],
                        },
                    ]
                },
                {"markdown": "# Question Answering\n\n## Methods\n\nSynthesized QA knowledge."},
                _node_card_content_response(),
                {"continue_upward": False, "reasons": ["no stable parent layer"]},
            ]
        )
    )
    client = FakeClient()
    config = WikiConfig(
        node_discovery_backend="llm_full_context",
        limits=WikiGenerationLimits(min_refs_per_node=2),
    )
    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=object(),
        ctx=object(),
        config=config,
        content_writer=client,
    )

    await WikiPipeline(
        writer=writer,
        config=config,
        llm=card_llm,
    ).generate_document_cards_from_inputs(
        wiki_inputs,
        content_loader=FakeContentLoader(docs),
        resource_uris=["viking://resources/"],
    )
    artifacts = await WikiPipeline(
        writer=writer,
        config=config,
        llm=node_llm,
    ).run_from_stored_cards(
        wiki_inputs,
        content_loader=FakeContentLoader(docs),
        resource_uris=["viking://resources/"],
    )

    assert "doc_card" not in [record.step for record in node_llm.log.raw_outputs]
    assert "viking://wiki/nodes/question_answering/" in client.mkdirs
    assert "viking://wiki/nodes/unassigned_topic/" not in client.mkdirs
    assert "viking://wiki/nodes/unassigned_topic/documents/" not in client.mkdirs
    assert "viking://wiki/nodes/unassigned_topic/sources/" not in client.mkdirs
    rejected = [node for node in artifacts.nodes if node.node_id == "unassigned_topic"]
    assert rejected[0].status == "rejected"


@pytest.mark.asyncio
async def test_parent_node_always_uses_child_documents_directly(monkeypatch):
    client = FakeClient()
    pipeline = WikiPipeline(
        writer=WikiVikingFSWriter(
            viking_fs=client,
            vikingdb=object(),
            ctx=object(),
            config=WikiConfig(
                node_discovery_backend="llm_full_context",
                limits=WikiGenerationLimits(large_node_source_token_threshold=1),
            ),
            content_writer=client,
        ),
        config=WikiConfig(
            node_discovery_backend="llm_full_context",
            limits=WikiGenerationLimits(large_node_source_token_threshold=1),
        ),
        llm=WikiLLMRunner(FakeVLM([])),
    )
    child_ref = SourceRef(
        ref_id="child",
        ref_type="wiki_node",
        doc_id="child",
        resource_uri="viking://wiki/nodes/child/",
        card_uri="viking://wiki/nodes/child/card.json",
        title="Child",
        support_scope="Supports parent",
    )
    child_document = ResourceDocument(
        doc_id="child",
        resource_uri=child_ref.resource_uri,
        title="Child",
        source_sections=[
            SourceSection(
                section_uri="viking://wiki/nodes/child/documents/document.md",
                content="x" * 1000,
            )
        ],
    )
    generated = NodeDocument(title="Parent", content="# Parent\n\n## Topic\n\nBody.")
    calls = []

    async def generate_direct(node, sources):
        calls.append(sources)
        return generated

    async def fail_if_selected(*args, **kwargs):
        raise AssertionError("parent nodes must not use scope-guided selection")

    async def generate_card(*args, **kwargs):
        return DocumentCard(
            doc_id="parent",
            resource_uri="viking://wiki/nodes/parent/",
            title="Parent",
            summary="Summary",
            candidate_topics=["Topic"],
        )

    monkeypatch.setattr(pipeline.content_generator, "generate_direct", generate_direct)
    monkeypatch.setattr(pipeline.source_selector, "select", fail_if_selected)
    monkeypatch.setattr(pipeline.card_generator, "generate_node_card", generate_card)

    context = await pipeline._generate_node_context(
        WikiNode(node_id="parent", title="Parent", depth=2, scope="Parent scope"),
        SourceAssignmentResult(source_refs_by_node={"parent": [child_ref]}),
        {"child": child_document},
        {},
    )

    assert calls[0][0]["sections"] == [
        {
            "section_uri": "viking://wiki/nodes/child/documents/document.md",
            "content": "x" * 1000,
        }
    ]
    assert context.document == generated


@pytest.mark.asyncio
async def test_large_bottom_node_uses_cards_and_scope_selected_sources(monkeypatch):
    config = WikiConfig(
        node_discovery_backend="llm_full_context",
        limits=WikiGenerationLimits(large_node_source_token_threshold=1),
    )
    client = FakeClient()
    pipeline = WikiPipeline(
        writer=WikiVikingFSWriter(
            viking_fs=client,
            vikingdb=object(),
            ctx=object(),
            config=config,
            content_writer=client,
        ),
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )
    refs = [
        SourceRef(
            ref_id="source",
            doc_id="source",
            resource_uri="viking://resources/source/",
            card_uri="viking://wiki/cards/source.card.json",
            title="Source",
            support_scope="Supports node",
        )
    ]
    source_document = ResourceDocument(
        doc_id="source",
        resource_uri=refs[0].resource_uri,
        title="Source",
        source_sections=[SourceSection(section_uri="source/chunk", content="evidence")],
    )
    source_card = DocumentCard(
        doc_id="source",
        resource_uri=refs[0].resource_uri,
        title="Source",
        summary="Summary",
        candidate_topics=["Topic"],
    )
    selected = object()
    outline = "## Detail\n\n## More"
    generated = NodeDocument(title="Topic", content="# Topic\n\n## Detail\n\nBody.")
    calls = []

    async def select(node, source_refs, source_documents):
        calls.append((source_refs, source_documents))
        return selected

    async def generate_outline(node, cards):
        calls.append(cards)
        return outline

    async def generate_staged(node, generated_outline, selected_sources):
        calls.append((generated_outline, selected_sources))
        return generated

    async def fail_if_direct(*args, **kwargs):
        raise AssertionError("large bottom nodes must use staged generation")

    async def generate_card(*args, **kwargs):
        return source_card.model_copy(
            update={"doc_id": "topic", "resource_uri": "viking://wiki/nodes/topic/"}
        )

    monkeypatch.setattr(pipeline.source_selector, "select", select)
    monkeypatch.setattr(pipeline.content_generator, "generate_outline", generate_outline)
    monkeypatch.setattr(pipeline.content_generator, "generate_staged", generate_staged)
    monkeypatch.setattr(pipeline.content_generator, "generate_direct", fail_if_direct)
    monkeypatch.setattr(pipeline.card_generator, "generate_node_card", generate_card)

    await pipeline._generate_node_context(
        WikiNode(node_id="topic", title="Topic", depth=1, scope="Scope"),
        SourceAssignmentResult(source_refs_by_node={"topic": refs}),
        {"source": source_document},
        {"source": source_card},
    )

    assert calls[0] == [source_card]
    assert calls[1] == (refs, {"source": source_document})
    assert calls[2] == (outline, selected)


@pytest.mark.asyncio
async def test_facet_backend_uses_evidence_sections_and_generates_node_facets(
    monkeypatch,
):
    docs = [_doc(index) for index in range(1, 4)]
    config = WikiConfig(
        node_discovery_backend="facet_graph",
        limits=WikiGenerationLimits(max_depth=1),
    )
    client = FakeClient()

    class SameTopicEmbedder:
        def prepare_embedding_input(self, text):
            return text

        async def embed_async(self, text, is_query=False):
            return EmbedResult(dense_vector=[1.0, 0.0])

    class VikingDB:
        def get_embedder(self):
            return SameTopicEmbedder()

    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=VikingDB(),
        ctx=object(),
        config=config,
        content_writer=client,
    )
    pipeline = WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )
    cards = [DocumentCard.model_validate(_card_response(index)) for index in range(1, 4)]
    facet_sets = {
        doc.doc_id: {
            "doc_id": doc.doc_id,
            "topic_facets": [
                {
                    "facet_id": f"{doc.doc_id}:qa",
                    "facet_text": "Question answering evaluation.",
                    "source_refs": [doc.source_sections[0].section_uri],
                }
            ],
        }
        for doc in docs
    }
    pipeline.llm = WikiLLMRunner(
        FakeVLM(
            [
                {"title": "Question Answering", "scope": "QA evaluation."},
                    {
                        "markdown": (
                            "# Question Answering\n\n"
                            "## Evidence\n\nCross-document evidence."
                        )
                    },
                _node_card_content_response(),
                {
                    "topic_facets": [
                        {
                            "facet_id": "F001",
                            "facet_text": "Question answering evaluation synthesis.",
                            "source_refs": ["S0001"],
                        }
                    ],
                    "facet_relations": [],
                },
            ]
        )
    )
    pipeline.facet_generator.llm = pipeline.llm
    pipeline.content_generator.llm = pipeline.llm
    pipeline.card_generator.llm = pipeline.llm

    artifacts = await pipeline._run_from_cards(
        cards,
        PipelineArtifacts(
            facet_sets=[
                DocumentFacetSet.model_validate(payload) for payload in facet_sets.values()
            ]
        ),
        {doc.doc_id: doc for doc in docs},
        facet_sets_by_source_id={
            doc_id: DocumentFacetSet.model_validate(payload)
            for doc_id, payload in facet_sets.items()
        },
    )

    refs = artifacts.source_refs_by_node["question_answering"]
    assert all(ref.matched_facet_ids for ref in refs)
    assert all(ref.matched_source_refs == [ref.resource_uri] for ref in refs)
    assert "viking://wiki/facets/nodes/question_answering.facets.json" in client.writes
    node_facets = json.loads(
        client.writes["viking://wiki/facets/nodes/question_answering.facets.json"]
    )
    assert node_facets["doc_id"] == "question_answering"
    assert len(artifacts.facet_sets) == 4
    clustering = json.loads(
        client.writes["viking://wiki/clustering/runs/depth_0001.json"]
    )
    assert clustering["neighbor_search_backend"] == "exact"
    assert clustering["mature_community_count"] == 1
    steps = [record.step for record in pipeline.llm.log.raw_outputs]
    assert "node_discovery" not in steps
    assert "next_layer_decision" not in steps


@pytest.mark.asyncio
async def test_facet_backend_uses_node_facets_for_parent_discovery(monkeypatch):
    docs = [_doc(index) for index in range(1, 4)]
    cards = [DocumentCard.model_validate(_card_response(index)) for index in range(1, 4)]
    config = WikiConfig(
        node_discovery_backend="facet_graph",
        limits=WikiGenerationLimits(max_depth=2, min_refs_per_node=1),
    )
    client = FakeClient()

    class TextEmbedder:
        def prepare_embedding_input(self, text):
            return text

        async def embed_async(self, text, is_query=False):
            if "parent synthesis" in text:
                return EmbedResult(dense_vector=[1.0, 0.0])
            suffix = int(text.rsplit(" ", 1)[-1].rstrip("."))
            return EmbedResult(dense_vector=[1.0 if index == suffix else 0.0 for index in range(1, 4)])

    class VikingDB:
        def get_embedder(self):
            return TextEmbedder()

    writer = WikiVikingFSWriter(
        viking_fs=client,
        vikingdb=VikingDB(),
        ctx=object(),
        config=config,
        content_writer=client,
    )
    pipeline = WikiPipeline(
        writer=writer,
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )
    initial_facet_sets = {
        doc.doc_id: DocumentFacetSet(
            doc_id=doc.doc_id,
            topic_facets=[
                {
                    "facet_id": f"{doc.doc_id}:leaf",
                    "facet_text": f"leaf topic {index}.",
                    "source_refs": [doc.source_sections[0].section_uri],
                }
            ],
        )
        for index, doc in enumerate(docs, start=1)
    }

    async def generate_direct(node, sources):
        return NodeDocument(
            title=node.title,
            content=f"# {node.title}\n\nparent synthesis {node.node_id}",
        )

    async def generate_node_card(node, document, resource_uri):
        return DocumentCard(
            doc_id=node.node_id,
            resource_uri=resource_uri,
            title=node.title,
            summary=f"{node.title} summary",
            candidate_topics=[node.title],
        )

    async def generate_facets(resource_documents):
        return [
            DocumentFacetSet(
                doc_id=document.doc_id,
                topic_facets=[
                    {
                        "facet_id": f"{document.doc_id}:node_facet",
                        "facet_text": f"parent synthesis {document.doc_id}",
                        "source_refs": [document.source_sections[0].section_uri],
                    }
                ],
            )
            for document in resource_documents
        ]

    monkeypatch.setattr(pipeline.content_generator, "generate_direct", generate_direct)
    monkeypatch.setattr(pipeline.card_generator, "generate_node_card", generate_node_card)
    monkeypatch.setattr(pipeline.facet_generator, "generate", generate_facets)
    pipeline.llm = WikiLLMRunner(
        FakeVLM(
            [
                {"title": "Leaf One", "scope": "Leaf one."},
                {"title": "Leaf Two", "scope": "Leaf two."},
                {"title": "Leaf Three", "scope": "Leaf three."},
                {"title": "Parent Synthesis", "scope": "Parent over node facets."},
            ]
        )
    )

    artifacts = await pipeline._run_from_cards(
        cards,
        PipelineArtifacts(facet_sets=list(initial_facet_sets.values())),
        {doc.doc_id: doc for doc in docs},
        facet_sets_by_source_id=initial_facet_sets,
    )

    parent_refs = artifacts.source_refs_by_node["parent_synthesis"]
    assert [ref.ref_type for ref in parent_refs] == ["wiki_node", "wiki_node", "wiki_node"]
    assert all(ref.matched_facet_ids for ref in parent_refs)
    assert {facet_set.doc_id for facet_set in artifacts.facet_sets} == {
        "OARW_1",
        "OARW_2",
        "OARW_3",
        "leaf_one",
        "leaf_two",
        "leaf_three",
        "parent_synthesis",
    }
    steps = [record.step for record in pipeline.llm.log.raw_outputs]
    assert steps == ["facet_community"] * 4
    node_manifest = json.loads(
        client.writes["viking://wiki/facets/nodes/manifest.json"]
    )
    assert {entry["doc_id"] for entry in node_manifest["entries"]} == {
        "leaf_one",
        "leaf_two",
        "leaf_three",
        "parent_synthesis",
    }


@pytest.mark.asyncio
async def test_facet_backend_rejects_source_facet_mismatch():
    doc = _doc(1)
    card = DocumentCard.model_validate(_card_response(1))
    config = WikiConfig(node_discovery_backend="facet_graph")
    client = FakeClient()

    class VikingDB:
        def get_embedder(self):
            return object()

    pipeline = WikiPipeline(
        writer=WikiVikingFSWriter(
            viking_fs=client,
            vikingdb=VikingDB(),
            ctx=object(),
            config=config,
            content_writer=client,
        ),
        config=config,
        llm=WikiLLMRunner(FakeVLM([])),
    )

    with pytest.raises(RuntimeError, match=r"missing=\['OARW_1'\]"):
        await pipeline._run_from_cards(
            [card],
            PipelineArtifacts(),
            {doc.doc_id: doc},
            facet_sets_by_source_id={},
        )


def _doc(index: int) -> ResourceDocument:
    content = f"# Paper {index}\n\nContent about question answering."
    return ResourceDocument(
        doc_id=f"OARW_{index}",
        resource_uri=f"viking://resources/OARW_{index}/",
        title=f"Paper {index}",
        content_or_structure=content,
        source_sections=[
            SourceSection(
                section_uri=f"viking://resources/OARW_{index}/",
                content=content,
            )
        ],
    )


def test_node_document_sections_preserve_markdown_heading_boundaries():
    config = WikiConfig()
    context = type(
        "Context",
        (),
        {
            "node": WikiNode(node_id="topic", title="Topic", depth=1, scope="Scope"),
            "document": NodeDocument(
                title="Topic",
                content="# Topic\n\nIntro.\n\n## First\n\nOne.\n\n## Second\n\nTwo.",
            ),
        },
    )()

    sections = _node_document_sections(config, context)

    assert [section.section_uri for section in sections] == [
        "viking://wiki/nodes/topic/documents/document.md#section-0001",
        "viking://wiki/nodes/topic/documents/document.md#section-0002",
        "viking://wiki/nodes/topic/documents/document.md#section-0003",
    ]
    assert sections[1].content == "## First\n\nOne."


def _wiki_input(doc: ResourceDocument) -> WikiResourceInput:
    return WikiResourceInput(
        doc_id=doc.doc_id,
        resource_uri=doc.resource_uri,
        title=doc.title,
        document_dir_uri=doc.resource_uri,
    )


class FakeContentLoader:
    def __init__(self, docs: list[ResourceDocument]):
        self.docs_by_id = {doc.doc_id: doc for doc in docs}

    async def load_document(
        self,
        doc: WikiResourceInput,
        *,
        mode: object,
        max_card_input_chars: int,
    ) -> ResourceDocument:
        return self.docs_by_id[doc.doc_id]

    async def load_source_document(self, doc: WikiResourceInput) -> ResourceDocument:
        return self.docs_by_id[doc.doc_id]


def _card_response(index: int) -> dict:
    return {
        "doc_id": f"OARW_{index}",
        "resource_uri": f"viking://resources/OARW_{index}/",
        "title": f"Paper {index}",
        **_card_content_response(index),
    }


def _card_content_response(index: int) -> dict:
    return {
        "summary": f"Paper {index} discusses question answering.",
        "important_terms": ["question answering"],
        "candidate_topics": ["question answering"],
    }


def _node_discovery_response() -> dict:
    return {
        "nodes": [
            {
                "title": "Question Answering",
                "scope": "QA methods and evaluation.",
                "supporting_source_ids": ["OARW_1", "OARW_2", "OARW_3"],
                "merged_candidate_topics": ["question answering"],
            }
        ]
    }


def _node_card_content_response() -> dict:
    return {
        "summary": "Question answering node synthesis.",
        "important_terms": ["question answering"],
        "candidate_topics": ["question answering systems"],
    }
