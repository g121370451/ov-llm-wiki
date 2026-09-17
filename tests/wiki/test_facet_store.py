import pytest

from openviking.wiki.config import WikiConfig, WikiGenerationLimits
from openviking.wiki.facet_store import DocumentFacetStore
from openviking.wiki.schemas import DocumentFacetSet, ResourceDocument, SourceSection, TopicFacet
from openviking.wiki.uri import facet_json_uri, facet_manifest_uri
from openviking.wiki.writer import WikiVikingFSWriter
from openviking_cli.exceptions import FailedPreconditionError

from .fakes import FakeClient


@pytest.mark.asyncio
async def test_facet_store_round_trip_and_source_validation():
    client = FakeClient()
    config = WikiConfig(limits=WikiGenerationLimits(max_facet_batch_chars=1234))
    writer = WikiVikingFSWriter(
        viking_fs=client, vikingdb=object(), ctx=object(), config=config, content_writer=client
    )
    store = DocumentFacetStore(
        viking_fs=client, writer=writer, config=config, ctx=writer.ctx
    )
    document = _document("content")
    facet_set = DocumentFacetSet(
        doc_id="doc",
        topic_facets=[
            TopicFacet(
                facet_id="doc:1",
                facet_text="Backup.",
                source_refs=["viking://resources/doc/one"],
            )
        ],
    )

    await store.replace(
        facet_sets=[facet_set],
        resource_documents=[document],
        max_batch_chars=1234,
    )

    assert await store.load_validated([document]) == [facet_set]
    assert client.write_order[-1] == facet_manifest_uri(config)
    assert facet_json_uri(config, "doc") in client.writes

    with pytest.raises(FailedPreconditionError):
        await store.load_validated([_document("changed")])


def _document(content: str) -> ResourceDocument:
    return ResourceDocument(
        doc_id="doc",
        resource_uri="viking://resources/doc",
        title="Document",
        source_sections=[
            SourceSection(section_uri="viking://resources/doc/one", content=content)
        ],
    )
