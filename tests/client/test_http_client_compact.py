"""Unit tests for request-body compaction on find/search/add_resource.

Older OpenViking instances use ``model_config = ConfigDict(extra="forbid")`` and
reject any field they do not yet define. The SDK must not attach optional fields
as ``null``/``{}`` when the caller never set them, otherwise the whole request
fails with ``body.<field>: Extra inputs are not permitted`` (see PR #2799 for the
equivalent CLI fix).
"""

from typing import Any, Dict, List

from openviking_cli.client.http import AsyncHTTPClient


class _FakeHTTPClient:
    """Records the last request and returns a canned response."""

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.next_response: Any = object()

    async def post(self, path, *, json=None, headers=None):
        self.calls.append({"method": "POST", "path": path, "json": json, "headers": headers})
        return self.next_response


def _client_with_fake() -> tuple[AsyncHTTPClient, _FakeHTTPClient]:
    client = AsyncHTTPClient(url="http://localhost:1933")
    fake = _FakeHTTPClient()
    client._http = fake
    client._handle_response_data = lambda response: {"result": {}}
    return client, fake


def test_compact_request_body_drops_null_and_empty_args():
    body = {
        "query": "hi",
        "score_threshold": None,
        "tags": None,
        "args": {},
        "wait": False,
        "create_parent": True,
        "filter": {"k": "v"},
    }
    compacted = AsyncHTTPClient._compact_request_body(body)
    # Non-null values are kept, including `False` and non-empty objects.
    assert compacted["query"] == "hi"
    assert compacted["wait"] is False
    assert compacted["create_parent"] is True
    assert compacted["filter"] == {"k": "v"}
    # Null fields and an empty `args` are dropped so pre-field servers accept it.
    assert "score_threshold" not in compacted
    assert "tags" not in compacted
    assert "args" not in compacted


def test_compact_request_body_keeps_non_empty_args():
    body = {"path": "x", "args": {"feishu_access_token": "u-x"}}
    compacted = AsyncHTTPClient._compact_request_body(body)
    assert compacted["args"] == {"feishu_access_token": "u-x"}


async def test_find_omits_unset_optional_fields():
    client, fake = _client_with_fake()

    await client.find("hello")

    payload = fake.calls[-1]["json"]
    assert payload["query"] == "hello"
    assert payload["limit"] == 10
    # `tags` is the field that breaks `find` against pre-#2706 strict instances.
    for dropped in ("score_threshold", "filter", "context_type", "tags"):
        assert dropped not in payload


async def test_find_keeps_explicit_tags():
    client, fake = _client_with_fake()

    await client.find("hello", tags=["a", "b"])

    payload = fake.calls[-1]["json"]
    assert payload["tags"] == ["a", "b"]


async def test_wiki_http_requests_keep_card_and_discovery_inputs_separate():
    client, fake = _client_with_fake()

    await client.build_wiki_cards(
        ["viking://resources/demo"],
        card_input_mode="raw_chunk",
        max_card_input_chars=1234,
    )
    card_call = fake.calls[-1]
    assert card_call["path"] == "/api/v1/wiki/cards/build"
    assert card_call["json"]["card_input_mode"] == "raw_chunk"
    assert card_call["json"]["max_card_input_chars"] == 1234

    await client.build_wiki(
        ["viking://resources/demo"],
        node_discovery_backend="facet_graph",
    )
    wiki_call = fake.calls[-1]
    assert wiki_call["path"] == "/api/v1/wiki/build"
    assert wiki_call["json"]["node_discovery_backend"] == "facet_graph"
    assert "card_input_mode" not in wiki_call["json"]
    assert "max_card_input_chars" not in wiki_call["json"]

    await client.clear_wiki(preserve_cards=True)
    clear_call = fake.calls[-1]
    assert clear_call["path"] == "/api/v1/wiki/clear"
    assert clear_call["json"]["preserve_cards"] is True

async def test_search_omits_unset_optional_fields():
    client, fake = _client_with_fake()

    await client.search("hello")

    payload = fake.calls[-1]["json"]
    assert payload["query"] == "hello"
    for dropped in ("session_id", "score_threshold", "filter", "context_type", "tags"):
        assert dropped not in payload


async def test_add_resource_omits_empty_args_and_null_fields():
    client, fake = _client_with_fake()

    # A non-existent path is sent as a plain `path` body (no temp-file upload).
    await client.add_resource("https://example.com/doc")

    payload = fake.calls[-1]["json"]
    assert payload["path"] == "https://example.com/doc"
    # `args` is the field that breaks `add-resource` against pre-#2549 instances.
    assert "args" not in payload
    for dropped in ("to", "parent", "timeout", "ignore_dirs", "include", "exclude"):
        assert dropped not in payload


async def test_add_resource_keeps_explicit_args():
    client, fake = _client_with_fake()

    await client.add_resource("https://example.com/doc", args={"feishu_access_token": "u-x"})

    payload = fake.calls[-1]["json"]
    assert payload["args"] == {"feishu_access_token": "u-x"}
