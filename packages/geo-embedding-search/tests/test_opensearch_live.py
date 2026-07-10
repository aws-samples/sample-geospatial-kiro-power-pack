"""Gated live check against a real managed OpenSearch cluster.

Skipped unless ``RUN_LIVE_OPENSEARCH=1`` and ``OPENSEARCH_URL`` are set (with
``OPENSEARCH_USERNAME``/``OPENSEARCH_PASSWORD`` if the cluster requires auth).
This never runs a local engine - it connects to a hosted service - so it stays
out of the default hermetic run.

    RUN_LIVE_OPENSEARCH=1 OPENSEARCH_URL=https://... \
        pytest packages/geo-embedding-search/tests/test_opensearch_live.py
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_OPENSEARCH") != "1" or not os.environ.get("OPENSEARCH_URL"),
    reason="set RUN_LIVE_OPENSEARCH=1 and OPENSEARCH_URL to run the live OpenSearch check",
)


def test_live_round_trip() -> None:
    from geo_embedding_search.opensearch_store import OpenSearchVectorStore
    from geo_embedding_search.store import search_embeddings, store_embedding

    index = "geo-embeddings-livetest-%s" % uuid.uuid4().hex[:8]
    store = OpenSearchVectorStore(
        url=os.environ["OPENSEARCH_URL"],
        username=os.environ.get("OPENSEARCH_USERNAME"),
        password=os.environ.get("OPENSEARCH_PASSWORD"),
        index=index,
    )
    rid = uuid.uuid4().hex
    conf = store_embedding([1.0, 0.0, 0.0], {"label": "east"}, store=store, record_id=rid)
    assert conf.retrievable
    hits = search_embeddings([1.0, 0.0, 0.0], k=1, store=store)
    assert hits and hits[0].id == rid
    store.delete(rid)
