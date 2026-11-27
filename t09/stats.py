"""
Stats: Statistical analysis module for the T09 UnimNeuron project.
Performs paired statistical tests to compare B0, S1, S2, and S3,
evaluating differences in predictive performance and rule count.

- Usage: python -m t09.stats --summary evaluation/summary_runs.csv --out evaluation
- Dependencies: pandas, numpy, scipy.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare


def holm_adjust(pvalues):
    """
    Apply the Holm correction to multiple p-values.
    Adjusts p-values to reduce the risk of false positives
    when performing several statistical comparisons.
    """
    order=np.argsort(pvalues); m=len(pvalues); adjusted=np.zeros(m); running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.,float(pvalues[idx])*(m-rank)))
        adjusted[idx]=running
    return adjusted


def paired_analysis(df,metric):
    """
    Perform paired statistical comparisons between model variants.
    Compares each pruning strategy (S1, S2, S3) against B0 using
    the Wilcoxon signed-rank test, and compares all four models
    using the Friedman test when sufficient matched runs exist.
    Only valid experiments are included, and comparisons are
    matched by stream and random seed. Multiple configurations
    of the same variant must be resolved before analysis.
    """
    df=df[df.valid.astype(str).str.lower().isin(['true','1'])].copy()

    # Do not combine parameter configurations: pick configurations BEFORE running these comparisons.
    if df.duplicated(['stream','seed','variant']).any():
        raise ValueError('Several hyperparameter configurations per stream/seed/variant: select one fixed configuration first')

    p=df.pivot_table(index=['stream','seed'],columns='variant',values=metric,aggfunc='first')
    tests=[]

    for variant in ['S1','S2','S3']:
        if variant not in p.columns or 'B0' not in p.columns:continue
        z=p[['B0',variant]].dropna(); n=len(z)
        if n<5:continue

        diff=z[variant]-z['B0']

        if np.allclose(diff,0):pv=1.
        else:pv=float(wilcoxon(diff,zero_method='wilcox').pvalue)

        tests.append({
            'metric':metric,
            'contrast':f'{variant} - B0',
            'pairs':n,
            'mean_difference':float(diff.mean()),
            'p_raw':pv
        })

    if tests:
        adjusted=holm_adjust([r['p_raw'] for r in tests])
        for r,pa in zip(tests,adjusted):r['p_holm']=pa

    if set(['B0','S1','S2','S3']).issubset(p.columns):
        z=p[['B0','S1','S2','S3']].dropna()

        # At least five matched units; repeated seeds within a single stream are NOT independent datasets.
        if len(z)>=5:
            stat,pv=friedmanchisquare(
                *(z[c] for c in ['B0','S1','S2','S3'])
            )
            tests.append({
                'metric':metric,
                'contrast':'Friedman omnibus',
                'pairs':len(z),
                'statistic':float(stat),
                'p_raw':float(pv),
                'p_holm':np.nan
            })

    return pd.DataFrame(tests)


def main():
    """
    Run the statistical analysis of the experimental results.
    """
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary',default='evaluation/summary_runs.csv')
    ap.add_argument('--out',default='evaluation')

    a=ap.parse_args()
    df=pd.read_csv(a.summary)
    out=Path(a.out)
    out.mkdir(exist_ok=True,parents=True)

    all_tests=[]

    for metric in ['accuracy','rules_mean']:
        ts=paired_analysis(df,metric)
        if not ts.empty:all_tests.append(ts)

    if all_tests:
        res=pd.concat(all_tests,ignore_index=True)
        res.to_csv(out/'paired_tests.csv',index=False)
        print(res.to_string(index=False))
    else:
        print('Insufficient matched runs for inferential statistics; report descriptive results only.')

    print('Reminder: non-significance is not proof of equivalence; check 1 p.p. accuracy margin separately.')


if __name__=='__main__':
    main()