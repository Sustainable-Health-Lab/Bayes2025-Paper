import ast
import json
from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path


def compute_observations(df: pd.DataFrame) -> Dict[str, float]:
    observations = {}

    # Basic counts (correct)
    observations["num_patients"] = df["n_w"].sum() + df["n_p"].sum()
    observations["num_hospitals"] = len(df)

    # Patients per hospital statistics
    for worp in ['w', 'p']:
        key = f"n_{worp}"
        observations[f'mean_{worp}p'] = df[key].mean()
        observations[f'median_{worp}p'] = df[key].median()
        observations[f'min_{worp}p'] = df[key].min()
        observations[f'fq_{worp}p'] = df[key].quantile(0.25)
        observations[f'tq_{worp}p'] = df[key].quantile(0.75)
        observations[f'max_{worp}p'] = df[key].max()
    

    # Race demographics (correct)
    observations["num_poc"] = df["n_p"].sum()
    observations["num_white"] = df["n_w"].sum()
    observations["percent_poc"] = (
        observations["num_poc"] / observations["num_patients"]
    ) * 100
    observations["percent_white"] = (
        observations["num_white"] / observations["num_patients"]
    ) * 100

    # Statistical tests results (correct)
    observations["num_hospitals_fep"] = sum(df["qmp"] < 0.05)
    observations["percent_hospitals_fep"] = (
        observations["num_hospitals_fep"] / observations["num_hospitals"]
    ) * 100
    observations["num_hospitals_mwp"] = sum(df["mwp"] < 0.05)
    observations["percent_hospitals_mwp"] = (
        observations["num_hospitals_mwp"] / observations["num_hospitals"]
    ) * 100

    # Initialize day distributions
    dist_pmpcd = [0] * 200
    dist_wmpcd = [0] * 200
    dist_dbdfe = [0] * 200

    # Process significant days and build distributions (correct)
    total_days = observations["num_hospitals"] * 200
    num_pmpcd_days = 0
    num_wmpcd_days = 0
    num_pcd_days = 0
    num_dbdfe_days = 0

    def parse_day_list(day_string: str) -> np.ndarray:
        # Remove brackets and convert to numpy array
        return np.fromstring(day_string.strip("[]"), sep=" ", dtype=int)

    for _, row in df.iterrows():
        # Parse day lists
        pmpcd_days = parse_day_list(row["pmpcd_sig_days"])
        wmpcd_days = parse_day_list(row["wmpcd_sig_days"])
        dbdfe_days = parse_day_list(row["dbdfe_sig_days"])

        # assert that the intersection of pmpcd and wmpcd is empty
        assert len(set(pmpcd_days) & set(wmpcd_days)) == 0

        # Update counts
        num_pmpcd_days += len(pmpcd_days)
        num_wmpcd_days += len(wmpcd_days)
        num_dbdfe_days += len(dbdfe_days)
        num_pcd_days += len(set(pmpcd_days.tolist() + wmpcd_days.tolist()))

        # Update distributions
        for day in pmpcd_days:
            dist_pmpcd[day-1] += 1
        for day in wmpcd_days:
            dist_wmpcd[day-1] += 1
        for day in dbdfe_days:
            dist_dbdfe[day-1] += 1

    # Store day counts and percentages (correct)
    observations["num_days_pmpcdp"] = num_pmpcd_days
    observations["num_days_wmpcdp"] = num_wmpcd_days
    observations["num_days_pcdp"] = num_pcd_days
    observations["num_days_dbdfep"] = num_dbdfe_days

    observations["percent_days_pmpcdp"] = (num_pmpcd_days / total_days) * 100
    observations["percent_days_wmpcdp"] = (num_wmpcd_days / total_days) * 100
    observations["percent_days_pcdp"] = (num_pcd_days / total_days) * 100
    observations["percent_days_dbdfep"] = (num_dbdfe_days / total_days) * 100

    # Compute hospital counts with at least 1 or 10 positive days
    for metric in ["pcd", "pmpcd", "wmpcd", "dbdfe"]:
        for threshold in [1, 5, 10, 15, 20, 25, 30, 35]:
            if metric == "pcd":
                # this is correct because it is not possible for a single day to be both pmpcd and wmpcd
                # we check that this is true with an assert statement above
                count = sum(
                    (df["count_pmpcd_sig"] + df["count_wmpcd_sig"]) >= threshold
                )
            else:
                col = f"count_{metric}_sig"
                count = sum(df[col] >= threshold)

            suffix = "day" if threshold == 1 else "days"
            observations[f"num_hospitals_with_{threshold}_{metric}p_{suffix}"] = count
            observations[f"percent_hospitals_with_{threshold}_{metric}p_{suffix}"] = (
                count / observations["num_hospitals"]
            ) * 100

    # Check for hospitals with both PMPCD+ and WMPCD+ days
    observations["is_hospital_with_pmpcdp_and_wmpcdp"] = any(
        (df["count_pmpcd_sig"] > 0) & (df["count_wmpcd_sig"] > 0)
    )

    # Store distributions
    observations["distribution_of_pmpcdp_days"] = dist_pmpcd
    observations["distribution_of_wmpcdp_days"] = dist_wmpcd
    observations["distribution_of_dbdfep_days"] = dist_dbdfe

    return observations


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def generate_observations(path_to_summaries, path_to_output) -> Dict:
    df = pd.read_csv(path_to_summaries)
    observations = compute_observations(df)
    assert validate_observations(observations) == []
    with open(path_to_output, "w") as f:
        json.dump(observations, f, cls=NumpyEncoder)
    print(f"Saved observations to {path_to_output}")
    return observations


def validate_observations(observations: Dict) -> List[str]:
    """Validate that all required variables are present in the observations dictionary."""
    required_variables = [
        "num_patients",
        "num_hospitals",
        "num_poc",
        "percent_poc",
        "num_white",
        "percent_white",
        "num_hospitals_fep",
        "percent_hospitals_fep",
        "num_hospitals_mwp",
        "percent_hospitals_mwp",
        "num_days_pmpcdp",
        "percent_days_pmpcdp",
        "num_days_wmpcdp",
        "percent_days_wmpcdp",
        "num_days_pcdp",
        "percent_days_pcdp",
        "num_days_dbdfep",
        "percent_days_dbdfep",
        "num_hospitals_with_1_pcdp_day",
        "num_hospitals_with_10_pcdp_days",
        "percent_hospitals_with_1_pcdp_day",
        "percent_hospitals_with_10_pcdp_days",
        "num_hospitals_with_1_pmpcdp_day",
        "num_hospitals_with_10_pmpcdp_days",
        "percent_hospitals_with_1_pmpcdp_day",
        "percent_hospitals_with_10_pmpcdp_days",
        "num_hospitals_with_1_wmpcdp_day",
        "num_hospitals_with_10_wmpcdp_days",
        "percent_hospitals_with_1_wmpcdp_day",
        "percent_hospitals_with_10_wmpcdp_days",
        "num_hospitals_with_1_dbdfep_day",
        "num_hospitals_with_10_dbdfep_days",
        "percent_hospitals_with_1_dbdfep_day",
        "percent_hospitals_with_10_dbdfep_days",
        "is_hospital_with_pmpcdp_and_wmpcdp",
        "distribution_of_pmpcdp_days",
        "distribution_of_wmpcdp_days",
        "distribution_of_dbdfep_days",
    ]

    missing_variables = [var for var in required_variables if var not in observations]
    return missing_variables



def main():
    save_path = Path(
        "/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper"
    )
    summaries_path = save_path / "summaries.csv"
    observations_path = save_path / "Export/observations.json"
    observations = generate_observations(summaries_path, observations_path)
    missing_vars = validate_observations(observations)
    if missing_vars:
        print(f"Missing variables: {missing_vars}")
    assert not missing_vars

if __name__ == '__main__':
    main()
