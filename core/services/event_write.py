"""Shared event write path for API and dashboard test pings."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import uuid

from db.connection import get_pool, get_qdrant
from qdrant_client.models import PointStruct
from services.embeddings import generate_embedding, event_to_text
from services.entitlements import embeddings_enabled
from services.exceptions import bad_request

logger = logging.getLogger(__name__)


def _pgvector_literal(embedding: list[float]) -> str:
    """asyncpg has no built-in codec for pgvector's `vector` type, so a raw
    list param fails with 'expected str, got list'. pgvector accepts its
    text input format (e.g. "[0.1,0.2]") cast via `::vector` instead."""
    return "[" + ",".join(repr(x) for x in embedding) + "]"


def _check_text_safety(text: str, where: str) -> None:
    """Reject strings Postgres text/jsonb columns cannot store.

    Postgres rejects NUL bytes in any text value and lone surrogates in
    UTF-8 input; without an early check those reach the INSERT and blow
    up with a 500 (issue #230).
    """
    if "\x00" in text:
        raise bad_request(f"{where} must not contain NUL characters")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise bad_request(f"{where} contains invalid Unicode (e.g. unpaired surrogates)")


def _check_json_value(value: object, path: str = "$") -> None:
    """Reject payloads that cannot round-trip through Postgres jsonb.

    `json.dumps` happily serializes NaN/Infinity (as the bare tokens `NaN`,
    `Infinity`) and passes NUL bytes through, but Postgres jsonb rejects
    all of them — historically surfacing as a 500 after partial writes
    (issue #230). Fail here with a 400 instead, before anything is written.
    """
    if isinstance(value, str):
        _check_text_safety(value, f"data{path}")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise bad_request(
                f"data{path} is not a finite number: "
                "NaN/Infinity are not valid JSON and are rejected by Postgres"
            )
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                _check_text_safety(key, f"data{path} key")
            _check_json_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_json_value(item, f"{path}[{index}]")


def validate_event_input(
    *,
    event: str,
    data: dict,
    parent_id: str | None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Fail fast with a 400 on input Postgres cannot store.

    Runs before any write, so malformed input is rejected with a clear
    message and nothing is persisted — no phantom agent row, no counter
    drift (fixes #230).
    """
    if not isinstance(event, str) or not event:
        raise bad_request("event must be a non-empty string")
    _check_text_safety(event, "event")
    if not isinstance(data, dict):
        raise bad_request("data must be a JSON object")
    _check_json_value(data)
    if session_id is not None:
        if not isinstance(session_id, str):
            raise bad_request("session_id must be a string")
        _check_text_safety(session_id, "session_id")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise bad_request("metadata must be a JSON object")
        _check_json_value(metadata, path="$.metadata")
    if parent_id is not None:
        try:
            uuid.UUID(str(parent_id))
        except (ValueError, AttributeError, TypeError):
            raise bad_request(f"parent_id '{parent_id}' is not a valid UUID")


async def write_event(
    *,
    tenant_id: str,
    agent: str,
    event: str,
    data: dict,
    parent_id: str | None = None,
    session_id: str | None = None,
    metadata: dict | None = None,
) -> dict:
    validate_event_input(
        event=event,
        data=data,
        parent_id=parent_id,
        session_id=session_id,
        metadata=metadata,
    )

    pool = get_pool()

    if parent_id:
        parent_tenant_id = await pool.fetchval(
            "SELECT tenant_id FROM events WHERE event_id = $1",
            parent_id,
        )
        if parent_tenant_id is None or str(parent_tenant_id) != str(tenant_id):
            raise bad_request(
                f"parent_id '{parent_id}' does not exist or belongs to a different tenant"
            )

    content = json.dumps({"event": event, "data": data}, sort_keys=True)
    checksum = hashlib.sha256(content.encode()).hexdigest()

    embed_on = embeddings_enabled()
    initial_status = "pending" if embed_on else "skipped"

    # The agent upsert (which also maintains event_count, #235) and the
    # event INSERT run inside one transaction: if the INSERT fails, the
    # agent row and its counter increment roll back with it — no phantom
    # agents, no counter drift (fixes #230).
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO agents (agent_id, tenant_id, event_count)
                VALUES ($1, $2, 1)
                ON CONFLICT (agent_id, tenant_id)
                DO UPDATE SET last_seen = NOW(), event_count = agents.event_count + 1
                """,
                agent,
                tenant_id,
            )

            row = await conn.fetchrow(
                """
                INSERT INTO events (
                    tenant_id, agent_id, event_type, data,
                    parent_event_id, session_id, checksum, metadata, index_status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING event_id, timestamp, sequence_no
                """,
                tenant_id,
                agent,
                event,
                json.dumps(data),
                parent_id,
                session_id,
                checksum,
                json.dumps(metadata) if metadata else None,
                initial_status,
            )

    event_id = str(row["event_id"])

    indexed = False
    index_status = initial_status
    if embed_on and os.getenv("EMBED_SYNC", "false").lower() in ("1", "true", "yes"):
        try:
            text = event_to_text(event, data)
            embedding = await generate_embedding(text, tenant_id)
            if embedding:
                await pool.execute(
                    "UPDATE events SET embedding = $1::vector, index_status = 'indexed' WHERE event_id = $2",
                    _pgvector_literal(embedding),
                    row["event_id"],
                )
                qdrant = get_qdrant()
                await qdrant.upsert(
                    collection_name="agent_events",
                    points=[
                        PointStruct(
                            id=event_id,
                            vector=embedding,
                            payload={
                                "tenant_id": tenant_id,
                                "agent_id": agent,
                                "event_type": event,
                                "timestamp": row["timestamp"].isoformat(),
                            },
                        )
                    ],
                )
                indexed = True
                index_status = "indexed"
        except Exception as e:
            logger.warning("Embedding/index skipped for event %s: %s", event_id, e)
            await pool.execute(
                "UPDATE events SET index_status = 'failed' WHERE event_id = $1",
                row["event_id"],
            )
            index_status = "failed"

    try:
        await pool.execute(
            """
            INSERT INTO usage_daily (tenant_id, date, events_written)
            VALUES ($1, CURRENT_DATE, 1)
            ON CONFLICT (tenant_id, date)
            DO UPDATE SET events_written = usage_daily.events_written + 1
            """,
            tenant_id,
        )
    except Exception as e:
        logger.warning("usage_daily meter skipped for tenant %s: %s", tenant_id, e)

    return {
        "event_id": event_id,
        "timestamp": row["timestamp"].isoformat(),
        "sequence_no": row["sequence_no"],
        "checksum": checksum,
        "indexed": indexed,
        "index_status": index_status,
    }
