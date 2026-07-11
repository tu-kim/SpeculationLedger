"""golden gates.json의 휘발 필드(git_hash) 정규화 — I4 golden은 '계산'의 결정성을
고정하는 장치이고, git hash는 커밋마다 정당하게 변한다 (§8의 기록 의무는 results/가 짐).
"""

import pathlib
import re
import sys


def normalize(b: bytes) -> bytes:
    return re.sub(rb'"git_hash":"[^"]*"', b'"git_hash":"GOLDEN"', b)


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    p.write_bytes(normalize(p.read_bytes()))
    print(f"normalized: {p}")
