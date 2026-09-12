#!/usr/bin/env python3
"""Fast, solver-free consistency checks for the frozen paper artifacts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "reproduction/expected_metrics.json").read_text())
PRIMARY_ROLES = {"surrogate_best", "medoid", "interior_1", "interior_2", "interior_3", "interior_4"}
METADATA = {"role", "draw", "predicted_loss", "predicted_loss_lo95_lat", "predicted_loss_hi95_lat", "sigma_latent_std", "sigma_predictive_std", "in_S_eps"}


def load_json(path: Path):
    assert path.is_file(), f"missing {path.relative_to(ROOT)}"
    return json.loads(path.read_text())


def rows(path: Path):
    assert path.is_file(), f"missing {path.relative_to(ROOT)}"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, *, atol=1e-12, rtol=1e-10, label="value"):
    assert math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol), (
        f"{label}: expected {expected}, found {actual}"
    )


def verify_year(year: str):
    base = ROOT / "results" / year
    expected = EXPECTED["years"][year]

    params = rows(base / "parameters.csv")
    perf = rows(base / "final_performance_summary.csv")
    assert len(params) == EXPECTED["design_size"] == len(perf)
    param_ids = {int(float(r["run_id"])) for r in params}
    perf_ids = {int(float(r["run_id"])) for r in perf}
    assert param_ids == perf_ids == set(range(1, 101)), f"{year}: design IDs are not 1..100"

    for row in perf:
        components = [float(row[f"nrmse_{name}"]) for name in ("coal", "wind", "solar", "mlt")]
        reconstructed = sum(x * x for x in components) / 4.0
        close(reconstructed, row["total_loss"], atol=2e-13, label=f"{year} run {row['run_id']} loss")

    split = load_json(base / "master_data_split.json")
    train = set(map(int, split["train_indices"]))
    test = set(map(int, split["test_indices"]))
    assert len(train) == EXPECTED["split"]["train"]
    assert len(test) == EXPECTED["split"]["test"]
    assert not train & test and train | test == set(range(1, 101))

    validation = load_json(base / "validation_summary.json")
    close(validation["test_r2"], expected["test_r2"], label=f"{year} test R2")
    close(validation["cv_r2"], expected["cv_r2"], label=f"{year} CV R2")
    active = load_json(base / "active_parameters.json")
    assert active["active_parameters"] == expected["active_parameters"]
    assert validation["final_dimensions"] == len(expected["active_parameters"])

    robustness = load_json(base / "active_selection_robustness/robustness_summary.json")
    assert robustness["baseline_reproduced"] is True
    assert robustness["overall_status"] == expected["robustness_status"]

    search = load_json(base / "surrogate_search_summary.json")
    assert search["n_samples"] == expected["search_points"]
    close(search["L_min"], expected["loss_min"], label=f"{year} search minimum")
    assert search["epsilon_sweep"]["0.1"]["n_members"] == expected["eps10_members"]
    dense = rows(base / "dense_surrogate_search.csv")
    assert len(dense) == expected["search_points"]
    close(min(float(r["predicted_loss"]) for r in dense), expected["loss_min"], atol=1e-12,
          label=f"{year} dense-search minimum")

    candidates = {r["role"]: r for r in rows(base / "revalidation_candidates_eps10.csv") if r["role"] in PRIMARY_ROLES}
    assert set(candidates) == PRIMARY_ROLES
    archives = list((base / "back_check_archive").glob("*/backcheck_result.json"))
    assert len(archives) == 6, f"{year}: expected six revalidation records"
    seen = set()
    for path in archives:
        record = load_json(path)
        role = record["role"]
        assert role in PRIMARY_ROLES and role not in seen
        seen.add(role)
        result = record["result"]
        components = [float(result[f"raw_{name}_error"]) for name in ("coal", "wind", "solar", "mlt")]
        reconstructed = sum(x * x for x in components) / 4.0
        close(reconstructed, result["performance_score"], atol=2e-13, label=f"{year} {role} loss")
        close(result["performance_score"], expected["backcheck_loss"][role], label=f"{year} {role} archived loss")
        candidate = candidates[role]
        for name, value in record["params"].items():
            assert name in candidate, f"{year} {role}: candidate lacks {name}"
            close(value, candidate[name], atol=5.1e-4, rtol=0, label=f"{year} {role} {name}")
    assert seen == PRIMARY_ROLES

    print(f"PASS {year}: 100 designs, {len(expected['active_parameters'])} retained parameters, "
          f"{expected['eps10_members']} S10% members, 6 UCED revalidations")


def verify_reporting_metrics():
    report = {r["year"]: r for r in rows(ROOT / "paper/supplementary/data/surrogate_validation_summary.csv")}
    for year, expected in EXPECTED["years"].items():
        close(report[year]["test_r2"], expected["test_r2"], label=f"{year} reported test R2")
        assert int(report[year]["n_test"]) == EXPECTED["split"]["test"]


def verify_no_machine_paths():
    for base in (ROOT / "results",):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
                text = path.read_text(errors="ignore")
                assert "/Users/" not in text and "/home/" not in text, f"machine-specific path in {path}"


if __name__ == "__main__":
    for year in ("2016", "2021"):
        verify_year(year)
    verify_reporting_metrics()
    verify_no_machine_paths()
    print("PASS: all archived 2016 and 2021 results are internally consistent.")
