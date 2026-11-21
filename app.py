"""Streamlit dashboard for the Informative Dropout Profiler.

Three tabs:
1. Regimes - one paired fleet masked three ways: per-sensor missing
   rates, the profiler's statistics, and its diagnosis vs the truth.
2. Illusion gap - held-out RUL RMSE by regime and imputer for one
   predictor, with the complete-data reference.
3. Evaluation - the aggregated multi-seed headline: illusion gaps,
   imputation-ranking reversal, diagnosis and driver accuracy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from dropout_profiler.evaluate import run_evaluation, run_single_seed
from dropout_profiler.models import IMPUTERS, PREDICTORS, REGIMES, SENSOR_NAMES


@st.cache_data(show_spinner=False)
def load_single_seed(seed):
    return run_single_seed(seed, n_train=20, n_test=12)


@st.cache_data(show_spinner=False)
def load_evaluation():
    return run_evaluation(seeds=(201, 202), n_train=20, n_test=12)


def tab_regimes():
    st.subheader("Three dropout regimes over one paired fleet")
    seed = st.selectbox("Fleet seed", [7, 21, 42], index=0)
    result = load_single_seed(int(seed))
    rates = dict(
        (
            regime,
            [
                result["regimes"][regime]["profile"]["per_sensor_rate"][name]
                for name in SENSOR_NAMES
            ],
        )
        for regime in REGIMES
    )
    st.write(
        "Per-sensor missing rate by regime (sensor order: %s). The "
        "three regimes hide the same share of the panel - what "
        "differs is *which* cells go missing."
        % ", ".join(SENSOR_NAMES)
    )
    st.bar_chart(rates)
    rows = []
    for regime in REGIMES:
        profile = result["regimes"][regime]["profile"]
        rows.append(
            {
                "regime": regime,
                "truth driver": result["regimes"][regime]["truth"]["driver"],
                "diagnosis": profile["diagnosis"],
                "same-group lift": round(profile["same_group_lift"], 2),
                "cross-group lift": round(profile["cross_group_lift"], 2),
                "proxy gradient": round(profile["proxy_gradient"], 3),
                "run length": round(profile["mean_run_length"], 2),
                "worst group": profile["worst_group"],
            }
        )
    st.dataframe(rows)


def tab_gap():
    st.subheader("Robustness illusion gap")
    predictor = st.selectbox("Predictor", list(PREDICTORS), index=0)
    result = load_single_seed(7)
    st.write(
        "Held-out RUL RMSE (days) for %s on the seed-7 fleets. "
        "Complete-data reference: %.2f days. The iid column is what a "
        "standard robustness benchmark reports; the informative "
        "column is what deployment actually delivers."
        % (predictor, result["complete_rmse"][predictor])
    )
    rows = []
    for imputer in IMPUTERS:
        row = {"imputer": imputer}
        for regime in REGIMES:
            row[regime] = round(
                result["regimes"][regime]["rmse"][imputer][predictor], 2
            )
        row["gap (informative - iid)"] = round(
            row["informative"] - row["iid"], 2
        )
        rows.append(row)
    st.dataframe(rows)
    chart = dict(
        (
            regime,
            [
                result["regimes"][regime]["rmse"][imputer][predictor]
                for imputer in IMPUTERS
            ],
        )
        for regime in REGIMES
    )
    st.write("RMSE by imputer (order: %s)." % ", ".join(IMPUTERS))
    st.bar_chart(chart)


def tab_evaluation():
    st.subheader("Headline evaluation (2 seeds, reduced fleets)")
    evaluation = load_evaluation()
    rows = [
        {"metric": "missing rate (iid)", "value": round(evaluation["missing_rate"]["iid"], 3)},
        {"metric": "missing rate (grouped)", "value": round(evaluation["missing_rate"]["grouped"], 3)},
        {"metric": "missing rate (informative)", "value": round(evaluation["missing_rate"]["informative"], 3)},
        {"metric": "illusion gap, chosen config, grouped (days)", "value": round(evaluation["gap_grouped"], 2)},
        {"metric": "illusion gap, chosen config, informative (days)", "value": round(evaluation["gap_informative"], 2)},
        {"metric": "illusion gap, mean-fill cells, informative (days)", "value": round(evaluation["gap_informative_mean_fill"], 2)},
        {"metric": "worst cell gap, informative (days)", "value": round(evaluation["worst_gap_informative"], 2)},
        {"metric": "worst cell", "value": evaluation["worst_gap_cell"]},
        {"metric": "imputation reversal rate (iid -> informative)", "value": round(evaluation["reversal_rate_informative"], 3)},
        {"metric": "diagnosis accuracy (profiler)", "value": round(evaluation["diagnosis_accuracy"], 3)},
        {"metric": "diagnosis accuracy (MAR screen)", "value": round(evaluation["mar_screen_accuracy"], 3)},
        {"metric": "diagnosis accuracy (always iid)", "value": round(evaluation["always_iid_accuracy"], 3)},
        {"metric": "driver identification accuracy", "value": round(evaluation["driver_accuracy"], 3)},
    ]
    st.dataframe(rows)
    st.write(
        "Full 6-seed numbers with standard fleets are in results/ "
        "(python -m dropout_profiler evaluate)."
    )


def main():
    st.title("Informative Dropout Profiler")
    st.write(
        "PHM benchmarks simulate iid sensor dropout. Real sensors "
        "drop out in physical groups and failure-proximately - the "
        "failing machine kills its own sensors. This dashboard shows "
        "what that mismatch costs, and reads the mask to name the "
        "mechanism."
    )
    tab1, tab2, tab3 = st.tabs(["Regimes", "Illusion gap", "Evaluation"])
    with tab1:
        tab_regimes()
    with tab2:
        tab_gap()
    with tab3:
        tab_evaluation()


if __name__ == "__main__":
    main()
