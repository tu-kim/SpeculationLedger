"""core — ledger 자료구조·수학. torch/cuda import 금지 (불변식 I5).

sim과 online(vllm_plugin)이 이 패키지를 공유한다 (CLAUDE.md §1 설계 대원칙).
"""

from core.store import LedgerStore
from core.types import (
    InvalidationEvent,
    LedgerStats,
    Posterior,
    PosteriorCand,
    Scope,
    ScopeKind,
    Segment,
    VerifyOutcome,
)

__all__ = [
    "LedgerStore",
    "Scope",
    "ScopeKind",
    "Segment",
    "VerifyOutcome",
    "Posterior",
    "PosteriorCand",
    "LedgerStats",
    "InvalidationEvent",
]
