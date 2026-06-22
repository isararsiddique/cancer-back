"""
Verifiable Audit Chain — immutable, hash-chained audit log.

Each audit entry includes a SHA-256 hash of (previous_hash + entry_data),
creating a tamper-evident chain similar to a blockchain. If any entry is
modified or deleted, the chain verification will fail.

This addresses 35 USC 101 by demonstrating a concrete technical mechanism
(cryptographic hash chain) that cannot be characterized as merely "organizing
human activity" or an "abstract idea."
"""
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any

GENESIS_HASH = "0" * 64  # Genesis block


def compute_entry_hash(
    previous_hash: str,
    timestamp: str,
    user_id: str,
    action: str,
    resource: str,
    detail: str = "",
) -> str:
    """Compute the SHA-256 hash for an audit entry, chaining from the previous entry."""
    payload = json.dumps({
        "prev": previous_hash,
        "ts": timestamp,
        "user": user_id,
        "action": action,
        "resource": resource,
        "detail": detail,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_chain(entries: list) -> Dict[str, Any]:
    """
    Verify the integrity of an audit chain.
    Returns {"valid": True/False, "broken_at": index or None, "total": count}.
    """
    prev = GENESIS_HASH
    for i, entry in enumerate(entries):
        expected = compute_entry_hash(
            previous_hash=prev,
            timestamp=entry.get("timestamp", ""),
            user_id=entry.get("user_id", ""),
            action=entry.get("action", ""),
            resource=entry.get("resource", ""),
            detail=entry.get("detail", ""),
        )
        if entry.get("chain_hash") != expected:
            return {"valid": False, "broken_at": i, "total": len(entries)}
        prev = expected
    return {"valid": True, "broken_at": None, "total": len(entries)}
