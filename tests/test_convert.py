"""sim.convert: schema v1 검증기·token_log 변환기."""

import json

import pytest

from sim.convert import (
    convert_token_log,
    normalize_record,
    read_traces,
    validate_record,
)


def _valid_record():
    return {
        "schema_version": 1,
        "request_id": "r1",
        "ts": 0,
        "scope": {"tenant": "t", "repo": "t/r", "session": "t/r/s", "instance_id": "i"},
        "model": "m",
        "tokenizer_hash": "abc",
        "steps": [
            {
                "pos": 0,
                "proposed": [5, 6],
                "accepted_len": 1,
                "bonus": 7,
                "topk_ids": [[5, 9], [7, 9]],
                "topk_logp_q8": [[2, 30], [1, 25]],
                "seg": [1, 1],
                "t_us": 10,
            }
        ],
        "final_text_sha": "x",
    }


def test_valid_record_passes():
    assert validate_record(_valid_record()) == []


def test_missing_key_caught():
    rec = _valid_record()
    del rec["tokenizer_hash"]
    assert any("tokenizer_hash" in e for e in validate_record(rec))


def test_accepted_len_bounds_checked():
    rec = _valid_record()
    rec["steps"][0]["accepted_len"] = 5
    assert any("accepted_len" in e for e in validate_record(rec))


def test_topk_alignment_checked():
    rec = _valid_record()
    rec["steps"][0]["topk_ids"] = [[5]]  # realized 2인데 1개
    assert any("topk_ids" in e for e in validate_record(rec))


def test_q8_range_checked():
    rec = _valid_record()
    rec["steps"][0]["topk_logp_q8"] = [[2, 300], [1, 25]]
    assert any("q8" in e for e in validate_record(rec))


def test_normalize_reconstructs_realized_stream():
    req = normalize_record(_valid_record())
    assert req.tokens == [5, 7]  # accepted prefix + bonus
    assert req.seg == [1, 1]
    assert req.topk_ids[0] == (5, 9)


def test_token_log_converter_roundtrip(tmp_path):
    src = {
        "request_id": "r9",
        "ts": 3,
        "scope": {"tenant": "t", "repo": "t/r", "session": "t/r/s"},
        "tokenizer_hash": "h",
        "tokens": [1, 2, 3],
        "seg": [0, 1, 2],
        "events": [{"pos": 2, "type": "file_edit", "file": "a.py"}],
    }
    rec = convert_token_log(src)
    assert validate_record(rec) == []
    req = normalize_record(rec)
    assert req.tokens == [1, 2, 3]
    assert req.events[0].file == "a.py"

    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    reqs = read_traces([str(p)])
    assert len(reqs) == 1


def test_read_traces_strict_raises(tmp_path):
    p = tmp_path / "bad.jsonl"
    rec = _valid_record()
    del rec["steps"][0]["bonus"]
    p.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError):
        read_traces([str(p)])


def test_read_traces_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_traces(["/nonexistent/*.jsonl"])
