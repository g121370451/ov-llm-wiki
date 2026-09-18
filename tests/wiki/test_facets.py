import pytest

from openviking.models.embedder.base import EmbedResult
from openviking.wiki.facets import DocumentFacetGenerator, legacy_facet_set_from_card
from openviking.wiki.llm import WikiLLMRunner
from openviking.wiki.schemas import DocumentCard, ResourceDocument, SourceSection

from .fakes import FakeVLM


class TextEmbedder:
    async def embed_async(self, text, is_query=False):
        vector = [1.0, 0.0] if "backup" in text.lower() else [0.0, 1.0]
        return EmbedResult(dense_vector=vector)

    def prepare_embedding_input(self, text):
        return text


@pytest.mark.asyncio
async def test_facet_generator_batches_all_sections_and_merges_duplicates():
    fake_vlm = FakeVLM(
        [
            {
                "topic_facets": [
                    {
                        "facet_id": "F1",
                        "facet_text": "Database backup and recovery.",
                        "source_refs": ["S0001"],
                    }
                ],
                "facet_relations": [],
            },
            {
                "topic_facets": [
                    {
                        "facet_id": "F1",
                        "facet_text": "Database backup and recovery.",
                        "source_refs": ["S0001"],
                    },
                    {
                        "facet_id": "F2",
                        "facet_text": "Replication and failover.",
                        "source_refs": ["S0001"],
                    },
                ],
                "facet_relations": [
                    {
                        "source_facet_id": "F2",
                        "target_facet_id": "F1",
                        "relation_text": "Replication does not replace backup.",
                        "source_refs": ["S0001"],
                    }
                ],
            },
        ]
    )
    generator = DocumentFacetGenerator(
        WikiLLMRunner(fake_vlm),
        max_batch_chars=8,
        embedder=TextEmbedder(),
    )
    document = ResourceDocument(
        doc_id="doc",
        resource_uri="viking://resources/doc",
        title="Document",
        source_sections=[
            SourceSection(section_uri="viking://resources/doc/one", content="12345678"),
            SourceSection(section_uri="viking://resources/doc/two", content="abcdefgh"),
        ],
    )

    facet_set = await generator.generate_one(document)

    assert len(fake_vlm.calls) == 2
    assert "S0001" in fake_vlm.calls[0]
    assert "viking://resources/doc/one" not in fake_vlm.calls[0]
    assert len(facet_set.topic_facets) == 2
    backup = next(facet for facet in facet_set.topic_facets if "backup" in facet.facet_text.lower())
    assert backup.source_refs == [
        "viking://resources/doc/one",
        "viking://resources/doc/two",
    ]
    assert facet_set.facet_relations[0].source_facet_id != facet_set.facet_relations[0].target_facet_id


@pytest.mark.asyncio
async def test_facet_generator_rejects_unknown_source_refs_then_retries():
    valid = {
        "topic_facets": [
            {
                "facet_id": "F1",
                "facet_text": "Database backup.",
                "source_refs": ["S0001"],
            }
        ],
        "facet_relations": [],
    }
    fake_vlm = FakeVLM(
        [
            {
                "topic_facets": [
                    {
                        "facet_id": "F1",
                        "facet_text": "Database backup.",
                        "source_refs": ["viking://resources/unknown"],
                    }
                ],
                "facet_relations": [],
            },
            valid,
        ]
    )
    generator = DocumentFacetGenerator(WikiLLMRunner(fake_vlm))
    document = ResourceDocument(
        doc_id="doc",
        resource_uri="viking://resources/doc",
        title="Document",
        source_sections=[
            SourceSection(section_uri="viking://resources/doc/one", content="content")
        ],
    )

    facet_set = await generator.generate_one(document)

    assert len(fake_vlm.calls) == 2
    assert "Allowed source_refs for this batch: ['S0001']" in fake_vlm.calls[1]
    assert facet_set.topic_facets[0].source_refs == ["viking://resources/doc/one"]


def test_legacy_card_adapter_preserves_topics_without_claiming_evidence():
    facet_set = legacy_facet_set_from_card(
        DocumentCard(
            doc_id="doc",
            resource_uri="viking://resources/doc",
            title="Document",
            summary="Summary",
            candidate_topics=["Backup", "Replication"],
        )
    )

    assert [facet.facet_text for facet in facet_set.topic_facets] == [
        "Backup",
        "Replication",
    ]
    assert all(not facet.source_refs for facet in facet_set.topic_facets)
