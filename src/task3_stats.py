"""
Task 3 script
- KPI calculations (LossRatio, Claim Frequency, Claim Severity, Margin)
- Defines sensible control/test groups for province / zip / gender
- Runs chi-square for proportions (frequency) and t-test / Mann-Whitney for numeric (severity, margin)
- Applies Benjamini-Hochberg correction for multiple pairwise tests
- Outputs CSV summary with p-values and concise business interpretation
"""
import os
import sys
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'insurance_data.csv')
OUT_DIR = os.path.join(ROOT, 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def compute_kpis(df):
    # Ensure numeric
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)

    # KPI definitions
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    df['ClaimFrequency'] = (df['TotalClaims'] > 0).astype(int)
    # If NumberOfClaims exists use it; else fallback: severity = TotalClaims where claim occurred
    if 'NumberOfClaims' in df.columns:
        df['ClaimSeverity'] = df['TotalClaims'] / df['NumberOfClaims'].replace(0, np.nan)
    else:
        df['ClaimSeverity'] = df['TotalClaims'].where(df['TotalClaims'] > 0, 0)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']

    return df


def safe_pairwise_tests_cat(df, group_col, metric_col='ClaimFrequency'):
    """
    For categorical group_col, run pairwise chi-square on aggregated counts for ClaimFrequency (binary).
    Returns DataFrame of pairs with p-values and BH-adjusted p-values.
    """
    pairs = []
    counts = df.groupby([group_col, metric_col]).size().unstack(fill_value=0)
    # counts index are groups
    groups = counts.index.tolist()
    for a, b in combinations(groups, 2):
        # form contingency table 2x2
        table = np.array([
            [counts.loc[a].get(1, 0), counts.loc[a].get(0, 0)],
            [counts.loc[b].get(1, 0), counts.loc[b].get(0, 0)]
        ])
        chi2, p, _, _ = chi2_contingency(table)
        pairs.append({'groupA': a, 'groupB': b, 'test': 'chi2_freq', 'p_value': p})
    if pairs:
        dfp = pd.DataFrame(pairs)
        # BH correction
        dfp['p_adj'] = multipletests(dfp['p_value'], method='fdr_bh')[1]
        dfp['reject_H0'] = dfp['p_adj'] < 0.05
    else:
        dfp = pd.DataFrame(columns=['groupA','groupB','test','p_value','p_adj','reject_H0'])
    return dfp


def safe_pairwise_tests_num(df, group_col, metric_col, test='t'):
    """
    Pairwise tests for numeric metric (ClaimSeverity, Margin).
    test='t' uses Welch t-test; test='mw' uses Mann-Whitney.
    """
    pairs = []
    groups = df[group_col].dropna().unique().tolist()
    for a, b in combinations(groups, 2):
        a_vals = df.loc[df[group_col]==a, metric_col].dropna()
        b_vals = df.loc[df[group_col]==b, metric_col].dropna()
        if len(a_vals)<5 or len(b_vals)<5:
            p = np.nan
            stat = np.nan
        else:
            if test == 't':
                stat, p = ttest_ind(a_vals, b_vals, equal_var=False, nan_policy='omit')
            else:
                stat, p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        pairs.append({'groupA': a, 'groupB': b, 'test': f'{test}_{metric_col}', 'stat': stat, 'p_value': p})
    if pairs:
        dfp = pd.DataFrame(pairs)
        dfp['p_adj'] = multipletests(dfp['p_value'].fillna(1), method='fdr_bh')[1]
        dfp['reject_H0'] = dfp['p_adj'] < 0.05
    else:
        dfp = pd.DataFrame(columns=['groupA','groupB','test','stat','p_value','p_adj','reject_H0'])
    return dfp


def interpret_row(row, metric):
    # Short business interpretation text
    if pd.isna(row['p_value']):
        return "Insufficient data for reliable test."
    if row['reject_H0']:
        return (f"Reject H0: {metric} differs between {row['groupA']} and {row['groupB']} "
                f"(p_adj={row['p_adj']:.4f}). Consider different pricing/underwriting for these groups.")
    else:
        return (f"Fail to reject H0: no statistical evidence {metric} differs between {row['groupA']} and {row['groupB']} "
                f"(p_adj={row['p_adj']:.4f}).")


def run_all_tests(df):
    outputs = []

    # detect columns
    prov_col = next((c for c in df.columns if 'prov' in c.lower()), None)
    zip_col = next((c for c in df.columns if 'zip' in c.lower()), None)
    gender_col = next((c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()), None)

    # Province: frequency (chi2) and severity/margin (t-test)
    if prov_col:
        print(f"Running province tests on column '{prov_col}'")
        df_freq = safe_pairwise_tests_cat(df, prov_col, metric_col='ClaimFrequency')
        df_sev = safe_pairwise_tests_num(df, prov_col, 'ClaimSeverity', test='t')
        df_margin = safe_pairwise_tests_num(df, prov_col, 'Margin', test='t')
        for dfp, metric in [(df_freq,'ClaimFrequency'), (df_sev,'ClaimSeverity'), (df_margin,'Margin')]:
            if not dfp.empty:
                dfp['metric'] = metric
                dfp['interpretation'] = dfp.apply(lambda r: interpret_row(r, metric), axis=1)
                outputs.append(dfp)
    else:
        print("No province-like column found.")

    # Zip code: only test top N zip codes by volume to keep tests practical
    if zip_col:
        top_zips = df[zip_col].value_counts().index[:6].tolist()
        df_small = df[df[zip_col].isin(top_zips)].copy()
        print("Running zip-code tests for top zips:", top_zips)
        df_freq = safe_pairwise_tests_cat(df_small, zip_col, metric_col='ClaimFrequency')
        df_margin = safe_pairwise_tests_num(df_small, zip_col, 'Margin', test='t')
        for dfp, metric in [(df_freq,'ClaimFrequency'), (df_margin,'Margin')]:
            if not dfp.empty:
                dfp['metric'] = metric
                dfp['interpretation'] = dfp.apply(lambda r: interpret_row(r, metric), axis=1)
                outputs.append(dfp)
    else:
        print("No zip-like column found.")

    # Gender: frequency test and severity/margin
    if gender_col:
        print(f"Running gender tests on column '{gender_col}'")
        df_freq = safe_pairwise_tests_cat(df, gender_col, metric_col='ClaimFrequency')
        df_sev = safe_pairwise_tests_num(df, gender_col, 'ClaimSeverity', test='t')
        for dfp, metric in [(df_freq,'ClaimFrequency'), (df_sev,'ClaimSeverity')]:
            if not dfp.empty:
                dfp['metric'] = metric
                dfp['interpretation'] = dfp.apply(lambda r: interpret_row(r, metric), axis=1)
                outputs.append(dfp)
    else:
        print("No gender-like column found.")

    if outputs:
        all_res = pd.concat(outputs, ignore_index=True, sort=False)
    else:
        all_res = pd.DataFrame()
    return all_res


def main():
    df = load_data(DATA_PATH)
    df = compute_kpis(df)
    res = run_all_tests(df)
    out_path = os.path.join(OUT_DIR, 'task3_hypothesis_results.csv')
    if not res.empty:
        res.to_csv(out_path, index=False)
        print(f"Saved hypothesis test results to {out_path}")
        # Print top significant results for quick grader view
        sig = res[res['reject_H0']].sort_values('p_adj')
        if not sig.empty:
            print("\nSignificant findings (sample):")
            print(sig[['metric','groupA','groupB','p_value','p_adj']].head(20).to_string(index=False))
        else:
            print("\nNo significant pairwise differences after BH correction.")
    else:
        print("No tests were run; check columns.")
    print("\nDone.")

if __name__ == '__main__':
    main()
