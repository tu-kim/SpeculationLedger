"""(a′) dense-key positive-only proposer — key/value 효과 분리 통제군 (CLAUDE.md §7).

- key: 최근 window 토큰의 recency-decay feature-hash 64-d 사영 → int8 격자 양자화.
  SENSE(arXiv 2606.00021) 공개 PCA 가중치는 오프라인 환경에서 미확보 —
  결정적 feature-hash 사영을 프록시로 쓴다(docs/DECISIONS.md A-6; 확보 시 교체 지점은
  `_embed` 하나다).
- index: 기본 faiss IndexFlatL2(정확 탐색 = dense key의 상한 성능, (a) 대비 보수적 비교).
  ivfpq=True면 IVF-PQ(nlist 64, m 8)로 전환해 압축 검색 현실치를 잰다.
- value: positive-only — 매치된 과거 위치의 realized 연속열을 그대로 제안.
- 격리: repo_id 단위로 index/corpus 분리 (tenant가 repo_id에 접혀 있음, I3).
"""

from __future__ import annotations

import numpy as np

from core.signature import fmix64
from core.types import DraftTree, Scope, VerifyOutcome
from sim.proposers import BaseProposer, ProposeCtx

_DECAY = 0.8
_TRAIN_MIN = 4096


class _RepoIndex:
    def __init__(self, dims: int, use_ivfpq: bool):
        import faiss

        faiss.omp_set_num_threads(1)  # 결정성 (I4)
        self._faiss = faiss
        self.dims = dims
        self.use_ivfpq = use_ivfpq
        self.index = faiss.IndexFlatL2(dims)
        self.trained_ivf = False
        self.stream: list[int] = []  # repo 단위 realized corpus (세션 경계 넘어 연결)
        self.offsets: list[int] = []  # faiss id → stream 끝 위치
        self._pending: list[np.ndarray] = []

    def _maybe_upgrade(self) -> None:
        if not self.use_ivfpq or self.trained_ivf:
            return
        if self.index.ntotal < _TRAIN_MIN:
            return
        faiss = self._faiss
        flat = self.index
        xb = faiss.rev_swig_ptr(flat.get_xb(), flat.ntotal * self.dims)
        xb = np.array(xb, dtype=np.float32).reshape(flat.ntotal, self.dims)
        quant = faiss.IndexFlatL2(self.dims)
        ivf = faiss.IndexIVFPQ(quant, self.dims, 64, 8, 8)
        ivf.train(xb)
        ivf.add(xb)
        ivf.nprobe = 8
        self.index = ivf
        self.trained_ivf = True

    def add(self, vec: np.ndarray, end_off: int) -> None:
        self.index.add(vec.reshape(1, -1))
        self.offsets.append(end_off)
        self._maybe_upgrade()

    def search(self, vec: np.ndarray) -> tuple[float, int]:
        if self.index.ntotal == 0:
            return float("inf"), -1
        d, i = self.index.search(vec.reshape(1, -1), 1)
        return float(d[0][0]), int(i[0][0])


class DenseKeyProposer(BaseProposer):
    def __init__(
        self,
        dims: int = 64,
        window: int = 16,
        max_dist: float = 0.35,
        use_ivfpq: bool = False,
        name: str = "dense",
    ):
        self.dims = dims
        self.window = window
        self.max_dist = max_dist  # 정규화 벡터 간 squared-L2 임계
        self.use_ivfpq = use_ivfpq
        self.name = name
        self._by_repo: dict[int, _RepoIndex] = {}
        self._repo_id = 0
        self._scope: Scope | None = None
        self.lookups = 0
        self.hits = 0

    # ------------------------------------------------------------ embedding
    def _embed(self, recent: tuple[int, ...]) -> np.ndarray:
        """SENSE-PCA 교체 지점. 현재: 위치 민감 feature-hash + recency decay → int8 격자."""
        v = np.zeros(self.dims, dtype=np.float64)
        w = list(recent[-self.window :])
        for i, tok in enumerate(reversed(w)):  # i=0 최신
            h = fmix64((tok + 1) * 0x9E3779B97F4A7C15 ^ fmix64(i + 1))
            j = h % self.dims
            sign = 1.0 if (h >> 8) & 1 else -1.0
            v[j] += sign * (_DECAY**i)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        q = np.clip(np.round(v * 127.0), -127, 127)  # int8 사영 (§10 저장 규약과 정합)
        out = (q / 127.0).astype(np.float32)
        return out

    def _idx(self, repo_id: int) -> _RepoIndex:
        idx = self._by_repo.get(repo_id)
        if idx is None:
            idx = _RepoIndex(self.dims, self.use_ivfpq)
            self._by_repo[repo_id] = idx
        return idx

    # -------------------------------------------------------------- protocol
    def begin_request(self, scope: Scope) -> None:
        self._scope = scope
        self._repo_id = scope.repo_id()

    def propose(self, ctx: ProposeCtx) -> DraftTree:
        tree = DraftTree()
        if len(ctx.recent) < 2:
            return tree
        idx = self._idx(self._repo_id)
        self.lookups += 1
        dist, fid = idx.search(self._embed(ctx.recent))
        if fid < 0 or dist > self.max_dist:
            return tree
        self.hits += 1
        end = idx.offsets[fid]
        cont = idx.stream[end + 1 : end + 1 + ctx.budget]
        parent = -1
        for tok in cont:
            parent = tree.add(tok, parent)
        return tree

    def harvest(self, ev: VerifyOutcome) -> None:
        idx = self._idx(self._repo_id)
        window: list[int] = list(ev.ctx_tail[-self.window :])
        for tok in ev.realized():
            idx.stream.append(tok)
            window.append(tok)
            if len(window) > self.window:
                window.pop(0)
            if len(window) >= 2:
                idx.add(self._embed(tuple(window)), len(idx.stream) - 1)

    def stats(self) -> dict:
        keys = sum(i.index.ntotal for i in self._by_repo.values())
        toks = sum(len(i.stream) for i in self._by_repo.values())
        return {
            "keys": keys,
            "bytes": keys * self.dims + toks * 4,  # int8 사영 기준
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hits / self.lookups, 6) if self.lookups else 0.0,
        }
