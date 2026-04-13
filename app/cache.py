"""
Redis cache layer for the Clinical Documentation AI.

Two cache namespaces:
  soap:{md5}        — full SOAP note results,      TTL = 1 hour
  guidelines:{md5}  — FAISS retrieval results,     TTL = 24 hours

Design: all Redis errors are silently swallowed so the app degrades
gracefully when Redis is unavailable (cache miss → run pipeline normally).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379")
SOAP_TTL     = int(os.getenv("CACHE_SOAP_TTL",       3600))   # 1 hour
GUIDELINES_TTL = int(os.getenv("CACHE_GUIDELINES_TTL", 86400)) # 24 hours

_client: Optional[aioredis.Redis] = None


def get_client() -> aioredis.Redis:
    """Return a shared async Redis client (lazy-initialised)."""
    global _client
    if _client is None:
        _client = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _client


async def close_client() -> None:
    """Close the Redis connection — call during app shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── SOAP note cache ───────────────────────────────────────────────────────────

async def get_soap(transcript: str) -> Optional[dict]:
    """Return cached SOAP note or None on miss / Redis error."""
    try:
        key    = f"soap:{_md5(transcript)}"
        cached = await get_client().get(key)
        if cached:
            logger.info("Cache HIT  %s", key)
            return json.loads(cached)
        logger.info("Cache MISS %s", key)
    except Exception as exc:
        logger.warning("Redis get_soap error (degrading gracefully): %s", exc)
    return None


async def set_soap(transcript: str, soap: dict) -> None:
    """Store SOAP note in Redis with TTL."""
    try:
        key = f"soap:{_md5(transcript)}"
        await get_client().setex(key, SOAP_TTL, json.dumps(soap))
        logger.info("Cache SET  %s  (TTL=%ds)", key, SOAP_TTL)
    except Exception as exc:
        logger.warning("Redis set_soap error (degrading gracefully): %s", exc)


# ── Guidelines retrieval cache ────────────────────────────────────────────────

async def get_guidelines(query: str) -> Optional[list[dict]]:
    """Return cached guideline chunks or None on miss / Redis error."""
    try:
        key    = f"guidelines:{_md5(query)}"
        cached = await get_client().get(key)
        if cached:
            logger.info("Cache HIT  %s", key)
            return json.loads(cached)
        logger.info("Cache MISS %s", key)
    except Exception as exc:
        logger.warning("Redis get_guidelines error (degrading gracefully): %s", exc)
    return None


async def set_guidelines(query: str, chunks: list[dict]) -> None:
    """Store guideline chunks in Redis with TTL."""
    try:
        key = f"guidelines:{_md5(query)}"
        await get_client().setex(key, GUIDELINES_TTL, json.dumps(chunks))
        logger.info("Cache SET  %s  (TTL=%ds)", key, GUIDELINES_TTL)
    except Exception as exc:
        logger.warning("Redis set_guidelines error (degrading gracefully): %s", exc)
