"""SWE-bench Lite-50 고정 목록 생성 (결정적: instance_id 사전순 상위 50).

실행(1회, 네트워크 필요 — datasets 의존성은 uvx로 일회 사용):
    uvx --with datasets python bench/swebench/make_lite50.py
생성된 lite50.txt는 커밋한다 — 이 파일이 Phase 2의 고정 계약이다 (CLAUDE.md §2).
"""

import pathlib

OUT = pathlib.Path(__file__).parent / "lite50.txt"


def main() -> int:
    from datasets import load_dataset

    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    ids = sorted(r["instance_id"] for r in ds)
    sel = ids[:50]
    OUT.write_text("\n".join(sel) + "\n", encoding="utf-8")
    print(f"{len(sel)} instances → {OUT} (total {len(ids)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
