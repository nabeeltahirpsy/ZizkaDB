import datetime
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi import HTTPException

from services.event_write import (
    _pgvector_literal,
    validate_event_input,
    write_event,
)


def test_pgvector_literal_formats_as_bracketed_csv():
    """asyncpg has no codec for pgvector's `vector` type — a raw list param
    fails with 'expected str, got list'. This must produce pgvector's text
    input format so `$1::vector` can parse it."""
    assert _pgvector_literal([0.1, -0.2, 3.0]) == "[0.1,-0.2,3.0]"
    assert _pgvector_literal([]) == "[]"


def _async_cm(return_value=None):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=return_value)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.fixture
def mock_pool(monkeypatch):
    """Pool whose acquire() yields a mocked connection with a mocked
    transaction context manager. Returns (pool, conn)."""
    pool = AsyncMock()
    conn = AsyncMock()

    row = {
        "event_id": "12345678-1234-1234-1234-123456789abc",
        "timestamp": datetime.datetime(
            2026,
            1,
            1,
            tzinfo=datetime.timezone.utc,
        ),
        "sequence_no": 7,
    }
    conn.fetchrow.return_value = row

    txn = _async_cm()
    conn.transaction = MagicMock(return_value=txn)
    pool.acquire = MagicMock(return_value=_async_cm(conn))

    monkeypatch.setattr(
        "services.event_write.get_pool",
        lambda: pool,
    )

    return pool, conn


@pytest.fixture
def mock_qdrant(monkeypatch):
    qdrant = AsyncMock()

    monkeypatch.setattr(
        "services.event_write.get_qdrant",
        lambda: qdrant,
    )

    return qdrant


@pytest.mark.asyncio
async def test_write_event_success(
    monkeypatch,
    mock_pool,
    mock_qdrant,
):
    pool, conn = mock_pool
    monkeypatch.setattr("services.event_write.embeddings_enabled", lambda: True)
    monkeypatch.setenv("EMBED_SYNC", "true")
    monkeypatch.setattr(
        "services.event_write.event_to_text",
        lambda event, data: "hello world",
    )

    monkeypatch.setattr(
        "services.event_write.generate_embedding",
        AsyncMock(return_value=[0.1] * 1536),
    )

    result = await write_event(
        tenant_id="tenant1",
        agent="agent1",
        event="message",
        data={"text": "hello"},
    )

    assert result["event_id"] == "12345678-1234-1234-1234-123456789abc"
    assert result["sequence_no"] == 7
    assert result["indexed"] is True
    assert result["index_status"] == "indexed"

    # Agent upsert runs inside the transaction on the acquired connection;
    # embedding update + usage meter stay outside on the pool.
    assert conn.execute.await_count == 1
    mock_qdrant.upsert.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_upsert_counts_events_inside_transaction(mock_pool, mock_qdrant):
    """The agent upsert maintains event_count itself (first event counts as
    1 via the INSERT value, later events via ON CONFLICT DO UPDATE, #235) —
    and it runs inside the same transaction as the event INSERT (#230)."""
    pool, conn = mock_pool

    await write_event(
        tenant_id="tenant1",
        agent="agent1",
        event="message",
        data={"text": "hello"},
    )

    sql_calls = [call.args[0] for call in conn.execute.await_args_list]
    assert len(sql_calls) == 1
    assert "INSERT INTO agents" in sql_calls[0]
    assert "event_count" in sql_calls[0]
    assert "ON CONFLICT (agent_id, tenant_id)" in sql_calls[0]
    assert "event_count = agents.event_count + 1" in sql_calls[0]
    # The event INSERT ran on the same connection, inside the transaction.
    conn.fetchrow.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_insert_rolls_back_agent_upsert(mock_pool, mock_qdrant):
    """Regression test for #230: if the event INSERT fails, the agent upsert
    must roll back with it — no phantom agents, no counter drift."""
    pool, conn = mock_pool
    conn.fetchrow.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await write_event(
            tenant_id="tenant1",
            agent="agent1",
            event="message",
            data={"text": "hello"},
        )

    # The upsert ran inside the transaction...
    assert conn.execute.await_count == 1
    txn_cm = conn.transaction.return_value
    # ...and __aexit__ received the exception -> transaction rolled back.
    exit_call = txn_cm.__aexit__.await_args
    assert exit_call.args[0] is RuntimeError


@pytest.mark.asyncio
async def test_write_event_pending_when_async_embed(
    monkeypatch,
    mock_pool,
    mock_qdrant,
):
    pool, conn = mock_pool
    monkeypatch.setattr("services.event_write.embeddings_enabled", lambda: True)
    monkeypatch.delenv("EMBED_SYNC", raising=False)
    monkeypatch.setattr(
        "services.event_write.event_to_text",
        lambda event, data: "hello world",
    )

    result = await write_event(
        tenant_id="tenant1",
        agent="agent1",
        event="message",
        data={"text": "hello"},
    )

    assert result["indexed"] is False
    assert result["index_status"] == "pending"
    mock_qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_write_event_without_embedding(
    monkeypatch,
    mock_pool,
    mock_qdrant,
):
    monkeypatch.setattr(
        "services.event_write.event_to_text",
        lambda event, data: "hello",
    )

    monkeypatch.setattr(
        "services.event_write.generate_embedding",
        AsyncMock(return_value=None),
    )

    result = await write_event(
        tenant_id="tenant",
        agent="agent",
        event="message",
        data={},
    )

    mock_qdrant.upsert.assert_not_called()
    assert result["indexed"] is False


@pytest.mark.asyncio
async def test_embedding_failure_does_not_fail_request(
    monkeypatch,
    mock_pool,
    mock_qdrant,
):
    pool, conn = mock_pool
    monkeypatch.setattr("services.event_write.embeddings_enabled", lambda: True)
    monkeypatch.setenv("EMBED_SYNC", "true")
    monkeypatch.setattr(
        "services.event_write.event_to_text",
        lambda event, data: "hello",
    )

    async def fail(*args, **kwargs):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(
        "services.event_write.generate_embedding",
        fail,
    )

    result = await write_event(
        tenant_id="tenant",
        agent="agent",
        event="message",
        data={},
    )

    assert result["event_id"] == "12345678-1234-1234-1234-123456789abc"
    assert result["indexed"] is False
    assert result["index_status"] == "failed"
    mock_qdrant.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_usage_meter_failure_does_not_fail_request(
    monkeypatch,
    mock_pool,
):
    pool, conn = mock_pool
    monkeypatch.setattr(
        "services.event_write.get_qdrant",
        lambda: AsyncMock(),
    )

    async def execute(*args, **kwargs):
        if "usage_daily" in args[0]:
            raise RuntimeError("usage table failed")
        return "OK"

    pool.execute.side_effect = execute

    result = await write_event(
        tenant_id="tenant",
        agent="agent",
        event="message",
        data={},
    )

    assert result["event_id"] == "12345678-1234-1234-1234-123456789abc"


# ── #230 input validation: malformed input must fail with a 400
#    before anything is written ───────────────────────────────────


def _assert_400(excinfo):
    assert isinstance(excinfo.value, HTTPException)
    assert excinfo.value.status_code == 400


def test_validate_accepts_normal_input():
    validate_event_input(
        event="tool_call",
        data={"text": "hello", "n": 1.5, "nested": {"list": [1, "a", None, True]}},
        parent_id="12345678-1234-1234-1234-123456789abc",
        session_id="s1",
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {"x": "a\x00b"},  # NUL in a string value
        {"x": ["ok", {"deep": "a\x00b"}]},  # NUL nested
        {"\x00key": "v"},  # NUL in a key
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": [1, float("-inf")]},
    ],
)
async def test_malformed_data_rejected_before_any_write(mock_pool, data):
    pool, conn = mock_pool
    with pytest.raises(HTTPException) as excinfo:
        await write_event(
            tenant_id="tenant1", agent="t2", event="e", data=data,
        )
    _assert_400(excinfo)
    # Nothing was written: no connection was even acquired.
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_nul_in_event_name_rejected(mock_pool):
    pool, _ = mock_pool
    with pytest.raises(HTTPException) as excinfo:
        await write_event(
            tenant_id="tenant1", agent="t2", event="e\x00vil", data={},
        )
    _assert_400(excinfo)
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_unpaired_surrogate_rejected(mock_pool):
    pool, _ = mock_pool
    with pytest.raises(HTTPException) as excinfo:
        await write_event(
            tenant_id="tenant1", agent="t2", event="ok", data={"x": "a\ud800b"},
        )
    _assert_400(excinfo)
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_non_uuid_parent_id_rejected(mock_pool):
    pool, _ = mock_pool
    with pytest.raises(HTTPException) as excinfo:
        await write_event(
            tenant_id="tenant1",
            agent="t2",
            event="e",
            data={},
            parent_id="not-a-uuid",
        )
    _assert_400(excinfo)
    # The parent lookup (which would 500 on the bad UUID) never ran.
    pool.fetchval.assert_not_called()
    pool.acquire.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_metadata_rejected(mock_pool):
    pool, _ = mock_pool
    with pytest.raises(HTTPException) as excinfo:
        await write_event(
            tenant_id="tenant1",
            agent="t2",
            event="e",
            data={},
            metadata={"x": float("nan")},
        )
    _assert_400(excinfo)
    pool.acquire.assert_not_called()
