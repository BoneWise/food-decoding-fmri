from __future__ import annotations

import argparse
import json
from pathlib import Path
from analyses._paths import DATA_ROOT, RUNS_ROOT, OUTPUT_ROOT, FSL_ROOT, ATLAS_ROOT
from typing import Any

import pandas as pd


DEFAULT_GLM_ROOT = DATA_ROOT / "derived" / "sessionwise"
DEFAULT_BETA_INDEX = DEFAULT_GLM_ROOT / "beta_index.csv"
DEFAULT_OUT_DIR = OUTPUT_ROOT / "searchlight" / "qc"
DEFAULT_REQUIRED_SESSIONS = ("ses-01", "ses-02", "ses-03")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate run-wise rating1/rating4 beta_index.csv for the primary "
            "session-wise 3-fold searchlight MVPA."
        )
    )
    parser.add_argument("--beta-index-csv", default=str(DEFAULT_BETA_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--required-sessions", default=",".join(DEFAULT_REQUIRED_SESSIONS))
    parser.add_argument("--skip-file-check", action="store_true")
    return parser


def _truthy(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _required_sessions(value: str) -> list[str]:
    sessions = [item.strip() for item in value.split(",") if item.strip()]
    if len(sessions) < 2:
        raise ValueError("--required-sessions must contain at least two session ids.")
    return sessions


def _file_exists(path_text: Any) -> bool:
    text = "" if path_text is None else str(path_text).strip()
    return bool(text) and Path(text).expanduser().exists()


def build_session_counts(df: pd.DataFrame, required_sessions: list[str]) -> pd.DataFrame:
    subjects = sorted(df["sub"].dropna().astype(str).unique().tolist())
    ratings = [1, 4]
    index = pd.MultiIndex.from_product(
        [subjects, required_sessions, ratings],
        names=["sub", "session", "rating_value"],
    )
    counts = (
        df.groupby(["sub", "session", "rating_value"], dropna=False)
        .agg(
            n_beta_samples=("beta_file", "size"),
            n_total_trials=("n_trials", "sum"),
        )
        .reindex(index, fill_value=0)
        .reset_index()
    )
    wide = counts.pivot(index=["sub", "session"], columns="rating_value", values="n_beta_samples").fillna(0)
    wide.columns = [f"n_rating{int(col)}_betas" for col in wide.columns]
    wide = wide.reset_index()
    trials = counts.pivot(index=["sub", "session"], columns="rating_value", values="n_total_trials").fillna(0)
    trials.columns = [f"n_rating{int(col)}_trials" for col in trials.columns]
    trials = trials.reset_index()
    out = wide.merge(trials, on=["sub", "session"], how="left")
    for col in ("n_rating1_betas", "n_rating4_betas", "n_rating1_trials", "n_rating4_trials"):
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].astype(int)
    out["has_both_classes"] = (out["n_rating1_betas"] > 0) & (out["n_rating4_betas"] > 0)
    out["min_class_betas"] = out[["n_rating1_betas", "n_rating4_betas"]].min(axis=1)
    return out.sort_values(["sub", "session"]).reset_index(drop=True)


def build_fold_counts(session_counts: pd.DataFrame, required_sessions: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, sub_df in session_counts.groupby("sub", sort=True):
        sub_df = sub_df.set_index("session")
        for test_session in required_sessions:
            test = sub_df.loc[test_session]
            train = sub_df.loc[[session for session in required_sessions if session != test_session]]
            train_r1 = int(train["n_rating1_betas"].sum())
            train_r4 = int(train["n_rating4_betas"].sum())
            test_r1 = int(test["n_rating1_betas"])
            test_r4 = int(test["n_rating4_betas"])
            rows.append(
                {
                    "sub": subject_id,
                    "test_session": test_session,
                    "train_rating1_betas": train_r1,
                    "train_rating4_betas": train_r4,
                    "test_rating1_betas": test_r1,
                    "test_rating4_betas": test_r4,
                    "train_has_both_classes": train_r1 > 0 and train_r4 > 0,
                    "test_has_both_classes": test_r1 > 0 and test_r4 > 0,
                    "fold_usable": train_r1 > 0 and train_r4 > 0 and test_r1 > 0 and test_r4 > 0,
                }
            )
    return pd.DataFrame(rows)


def build_subject_eligibility(session_counts: pd.DataFrame, fold_counts: pd.DataFrame) -> pd.DataFrame:
    session_summary = (
        session_counts.groupby("sub", as_index=False)
        .agg(
            n_sessions=("session", "nunique"),
            n_sessions_with_both=("has_both_classes", "sum"),
            min_session_class_betas=("min_class_betas", "min"),
            total_rating1_betas=("n_rating1_betas", "sum"),
            total_rating4_betas=("n_rating4_betas", "sum"),
        )
    )
    fold_summary = (
        fold_counts.groupby("sub", as_index=False)
        .agg(
            n_usable_folds=("fold_usable", "sum"),
            min_train_rating1_betas=("train_rating1_betas", "min"),
            min_train_rating4_betas=("train_rating4_betas", "min"),
            min_test_rating1_betas=("test_rating1_betas", "min"),
            min_test_rating4_betas=("test_rating4_betas", "min"),
        )
    )
    out = session_summary.merge(fold_summary, on="sub", how="left")
    out["eligible_session3fold"] = (out["n_sessions_with_both"] == out["n_sessions"]) & (
        out["n_usable_folds"] == out["n_sessions"]
    )
    return out.sort_values(["eligible_session3fold", "sub"], ascending=[False, True]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    beta_index_csv = Path(args.beta_index_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    required_sessions = _required_sessions(args.required_sessions)

    df = pd.read_csv(beta_index_csv)
    rows_total = int(len(df))
    df = df[df["rating_value"].isin([1, 4])].copy()
    if "estimated_successfully" in df.columns:
        df = df[_truthy(df["estimated_successfully"])].copy()
    df["sub"] = df["sub"].astype(str)
    df["session"] = df["session"].astype(str)
    df["rating_value"] = df["rating_value"].astype(int)
    if not args.skip_file_check:
        df["beta_file_exists"] = df["beta_file"].map(_file_exists)
        missing_files = int((~df["beta_file_exists"]).sum())
        df = df[df["beta_file_exists"]].copy()
    else:
        missing_files = None

    session_counts = build_session_counts(df, required_sessions)
    fold_counts = build_fold_counts(session_counts, required_sessions)
    eligibility = build_subject_eligibility(session_counts, fold_counts)

    session_counts_path = out_dir / "sessionwise_beta_counts.csv"
    fold_counts_path = out_dir / "sessionwise_cv_fold_counts.csv"
    eligibility_path = out_dir / "sessionwise_cv_eligibility.csv"
    summary_path = out_dir / "sessionwise_cv_summary.json"
    session_counts.to_csv(session_counts_path, index=False)
    fold_counts.to_csv(fold_counts_path, index=False)
    eligibility.to_csv(eligibility_path, index=False)

    summary = {
        "beta_index_csv": str(beta_index_csv),
        "rows_total": rows_total,
        "usable_rating14_beta_rows": int(len(df)),
        "missing_beta_files": missing_files,
        "required_sessions": required_sessions,
        "num_subjects": int(eligibility["sub"].nunique()),
        "num_eligible_subjects": int(eligibility["eligible_session3fold"].sum()),
        "eligible_subjects": eligibility.loc[eligibility["eligible_session3fold"], "sub"].tolist(),
        "min_session_class_betas": int(eligibility["min_session_class_betas"].min()),
        "session_counts_csv": str(session_counts_path),
        "fold_counts_csv": str(fold_counts_path),
        "eligibility_csv": str(eligibility_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
