"""Ranking behaviour: Pareto correctness, valid JSON, ordering, and failure handling."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ranking.pareto_front import compute_pareto_fronts
from ranking.pipeline_ranker import generate_final_ranking, parse_target_scores


def _docking_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(code, scores, bbb=0.9, qed=0.8):
    return {"candidate_code": code, "source_id": code, "smiles": "C",
            "bbb_probability": bbb, "qed": qed, "target_scores": json.dumps(scores)}


def test_pareto_matches_brute_force():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 6, size=(80, 4)).astype(float)
    df = pd.DataFrame(X, columns=list("abcd"))
    ranks = compute_pareto_fronts(df.copy(), ["a", "b"], ["c", "d"])["pareto_rank"].to_numpy()

    C = np.column_stack([X[:, 0], X[:, 1], -X[:, 2], -X[:, 3]])
    dom = lambda i, j: (C[i] <= C[j]).all() and (C[i] < C[j]).any()
    expected, remaining, k = np.zeros(len(C), int), set(range(len(C))), 0
    while remaining:
        k += 1
        front = [j for j in remaining if not any(dom(i, j) for i in remaining if i != j)]
        for j in front:
            expected[j] = k
        remaining -= set(front)
    assert (ranks == expected).all()


def test_stop_after_ranks_enough_rows():
    df = pd.DataFrame({"a": range(10), "b": range(10)})  # a strict chain: 10 fronts of 1
    out = compute_pareto_fronts(df, ["a", "b"], [], stop_after=3)
    assert list(out["pareto_rank"][:3]) == [1, 2, 3]
    assert out["pareto_rank"][3:].nunique() == 1


def test_output_is_strict_json_and_ordered(tmp_path: Path):
    rng = np.random.default_rng(0)
    rows = [_row(f"M{i}", {t: -float(rng.uniform(5, 12)) for t in ("A", "B", "C")},
                 bbb=float(rng.uniform()), qed=float(rng.uniform())) for i in range(200)]
    _docking_csv(tmp_path / "in.csv", rows)
    out = tmp_path / "out.json"
    ranked = generate_final_ranking(tmp_path / "in.csv", out, top_k=10)

    text = out.read_text()
    assert "Infinity" not in text and "NaN" not in text
    json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict constant {c}"))

    assert len(ranked) == 10
    keys = [(r["pareto_rank"], -r["crowding_distance"], r["mean_vina"]) for r in ranked]
    assert keys == sorted(keys)
    assert len({r["composite_score"] for r in ranked}) > 1
    assert all(0.0 <= r["composite_score"] <= 1.0 for r in ranked)
    assert all(r["vina_score"] == pytest.approx(r["mean_vina"]) for r in ranked)


def test_failed_docking_candidates_are_dropped(tmp_path: Path):
    _docking_csv(tmp_path / "in.csv", [
        _row("OK", {"A": -8.0, "B": -7.0}),
        _row("BAD", {"A": -12.0, "B": 0.0}),   # 0.0 == failed docking
    ])
    ranked = generate_final_ranking(tmp_path / "in.csv", tmp_path / "out.json")
    assert [r["candidate_code"] for r in ranked] == ["OK"]


def test_everything_failed_gives_empty_list(tmp_path: Path):
    _docking_csv(tmp_path / "in.csv", [_row("X", {"A": -8.0, "B": 0.0})])
    assert generate_final_ranking(tmp_path / "in.csv", tmp_path / "out.json") == []


def test_unrankable_input_raises(tmp_path: Path):
    pd.DataFrame([{"candidate_code": "a", "bbb_probability": 0.5}]).to_csv(tmp_path / "in.csv", index=False)
    with pytest.raises(ValueError):
        generate_final_ranking(tmp_path / "in.csv", tmp_path / "out.json")


def test_parse_target_scores_variants():
    assert parse_target_scores('{"A": -1.0}') == {"A": -1.0}
    assert parse_target_scores("{'A': -1.0}") == {"A": -1.0}
    assert parse_target_scores(float("nan")) == {}
    assert parse_target_scores("garbage") == {}
