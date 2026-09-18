import pytest

from openviking.models.embedder.base import EmbedResult
from openviking.wiki.facet_discovery import (
    FacetGraphEdge,
    ScalableNodeDiscoveryRunner,
    _merge_communities_with_same_sources,
    _build_similarity_graph,
    build_sparse_similarity_graph,
    discover_cpm_communities,
)
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.schemas import DocumentFacetSet, TopicFacet

from .fakes import FakeVLM


class MappingEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def prepare_embedding_input(self, text):
        return text

    async def embed_async(self, text, is_query=False):
        return EmbedResult(dense_vector=self.vectors[text])


def test_parent_communities_with_same_child_set_are_merged_with_all_facets():
    facets = {
        "a:one": ("a", TopicFacet(facet_id="a:one", facet_text="One A")),
        "b:one": ("b", TopicFacet(facet_id="b:one", facet_text="One B")),
        "a:two": ("a", TopicFacet(facet_id="a:two", facet_text="Two A")),
        "b:two": ("b", TopicFacet(facet_id="b:two", facet_text="Two B")),
        "c:other": ("c", TopicFacet(facet_id="c:other", facet_text="Other C")),
    }

    merged = _merge_communities_with_same_sources(
        [["a:one", "b:one"], ["a:two", "b:two"], ["b:one", "c:other"]],
        facets,
    )

    assert merged == [
        ["a:one", "a:two", "b:one", "b:two"],
        ["b:one", "c:other"],
    ]


def test_similarity_graph_excludes_same_document_edges_and_applies_threshold():
    facets = [
        ("a", TopicFacet(facet_id="a:1", facet_text="A1")),
        ("a", TopicFacet(facet_id="a:2", facet_text="A2")),
        ("b", TopicFacet(facet_id="b:1", facet_text="B1")),
    ]

    edges = build_sparse_similarity_graph(
        facets,
        [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1]],
        neighbor_limit=1,
        edge_score_threshold=0.8,
    )

    pairs = {(edge.left_facet_id, edge.right_facet_id) for edge in edges}
    assert ("a:1", "a:2") not in pairs
    assert pairs == {("a:1", "b:1"), ("a:2", "b:1")}


def test_cpm_keeps_unconnected_facets_separate_without_loading_leiden():
    communities = discover_cpm_communities(
        ["a", "b", "c"],
        [],
        resolution=0.5,
    )
    assert communities == [["a"], ["b"], ["c"]]


def test_large_graph_prefers_hnsw_neighbor_search(monkeypatch):
    facets = [
        (f"doc_{index}", TopicFacet(facet_id=f"facet_{index}", facet_text="topic"))
        for index in range(256)
    ]
    sentinel = [FacetGraphEdge("facet_0", "facet_1", 0.9)]

    monkeypatch.setattr(
        "openviking.wiki.facet_discovery._build_hnsw_similarity_graph",
        lambda *args, **kwargs: sentinel,
    )

    edges, backend = _build_similarity_graph(
        facets,
        [[1.0, 0.0]] * len(facets),
        neighbor_limit=20,
        edge_score_threshold=0.72,
        seed=0,
    )

    assert edges == sentinel
    assert backend == "hnsw"


def test_large_graph_falls_back_to_exact_when_hnsw_is_unavailable(monkeypatch):
    facets = [
        (f"doc_{index}", TopicFacet(facet_id=f"facet_{index}", facet_text="topic"))
        for index in range(256)
    ]

    def unavailable(*args, **kwargs):
        raise ImportError("hnswlib unavailable")

    monkeypatch.setattr(
        "openviking.wiki.facet_discovery._build_hnsw_similarity_graph",
        unavailable,
    )
    monkeypatch.setattr(
        "openviking.wiki.facet_discovery._build_exact_similarity_graph",
        lambda *args, **kwargs: [],
    )

    edges, backend = _build_similarity_graph(
        facets,
        [[1.0, 0.0]] * len(facets),
        neighbor_limit=20,
        edge_score_threshold=0.72,
        seed=0,
    )

    assert edges == []
    assert backend == "exact"


@pytest.mark.asyncio
async def test_facet_discovery_returns_existing_result_shape_and_facet_matches():
    facet_sets = {
        source_id: DocumentFacetSet(
            doc_id=source_id,
            topic_facets=[
                TopicFacet(
                    facet_id=f"{source_id}:backup",
                    facet_text=f"Backup {source_id}",
                    source_refs=[f"viking://resources/{source_id}/backup"],
                )
            ],
        )
        for source_id in ("a", "b", "c")
    }
    embedder = MappingEmbedder(
        {f"Backup {source_id}": [1.0, 0.0] for source_id in ("a", "b", "c")}
    )
    fake_vlm = FakeVLM(
        [{"title": "Database backup", "scope": "Backup and recovery."}]
    )
    runner = ScalableNodeDiscoveryRunner(
        WikiLLMRunner(fake_vlm),
        embedder,
        edge_score_threshold=0.8,
        cpm_resolution=0.5,
    )

    result = await runner.discover_layer(
        facet_sets_by_source_id=facet_sets,
        depth=1,
        min_sources_per_node=3,
    )

    assert [node.node_id for node in result.nodes] == ["database_backup"]
    assignment = result.source_assignments.assignments[0]
    assert assignment.source_ids == ["a", "b", "c"]
    assert assignment.facet_matches_by_source_id["a"][0].facet_id == "a:backup"


@pytest.mark.asyncio
async def test_parent_layer_stops_before_llm_when_candidates_do_not_reduce_count(
    monkeypatch,
):
    facet_sets = {
        source_id: DocumentFacetSet(
            doc_id=source_id,
            topic_facets=[
                TopicFacet(facet_id=f"{source_id}:{index}", facet_text=f"{source_id} {index}")
                for index in range(3)
            ],
        )
        for source_id in ("a", "b", "c")
    }
    all_facets = [
        facet
        for facet_set in facet_sets.values()
        for facet in facet_set.topic_facets
    ]
    vectors = {facet.facet_text: [1.0, 0.0] for facet in all_facets}
    monkeypatch.setattr(
        "openviking.wiki.facet_discovery.discover_cpm_communities",
        lambda *args, **kwargs: [
            ["a:0", "b:0"],
            ["a:1", "c:0"],
            ["b:1", "c:1"],
        ],
    )
    fake_vlm = FakeVLM([])
    runner = ScalableNodeDiscoveryRunner(WikiLLMRunner(fake_vlm), MappingEmbedder(vectors))

    result = await runner.discover_layer(
        facet_sets_by_source_id=facet_sets,
        depth=2,
        min_sources_per_node=2,
    )

    assert result.nodes == []
    assert fake_vlm.calls == []
    assert (
        runner.last_run_artifact["parent_layer_stop_reason"]
        == "no_strict_node_count_reduction"
    )
