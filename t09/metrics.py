"""Metrics: Evaluation module for the T09 UnimNeuron project. 
Calculates predictive, structural, and computational metrics
for B0, S1, S2, and S3, and generates performance and rule-count plots.

- Usage: python -m t09.metrics --results results --out evaluation --warmup 50.
- Dependencies: pandas, numpy, matplotlib.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REQUIRED = ['t','y_true','y_pred','n_rules','pruned_now','created_now','recreated_now','capacity_deletions','time_s']


def macro_f1(true, pred):
    """
    Calculate the macro-averaged F1-score for classification predictions.
    This function calculates the F1-score separately for each class
    and then computes their average, giving equal importance to
    every class regardless of how many observations it contains.
    """
    classes=np.union1d(true, pred)
    scores=[]
    for c in classes:
        tp=np.sum((true==c)&(pred==c)); fp=np.sum((true!=c)&(pred==c)); fn=np.sum((true==c)&(pred!=c))
        denom=2*tp+fp+fn
        scores.append(float(2*tp/denom) if denom else 0.0)
    return float(np.mean(scores)) if scores else float('nan')


def evaluate_csv(path, warmup=50):
    """
    Evaluate a single experimental run using its CSV results.
    This function reads the observations recorded during a model
    execution and calculates predictive, structural, and resource
    metrics (prequential accuracy, Macro-F1 score, mean, maximum, 
    and final number of rules, number of pruning, creation, and 
    recreation events, total evaluation runtime and average processing
    time per observation)

    The first 'warmup' observations are excluded because the model
    is still in its initial learning phase.
    """
    path=Path(path)
    df=pd.read_csv(path)
    missing=set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if len(df)<=warmup: raise ValueError(f'{path}: {len(df)} observations <= warmup {warmup}')
    if df['t'].duplicated().any() or not df['t'].is_monotonic_increasing: raise ValueError(f'{path}: t must be monotonic and unique')
    config_path=path.with_suffix('.json')
    conf=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    ev=df.iloc[warmup:].copy()
    y=ev.y_true.to_numpy(); p=ev.y_pred.to_numpy()
    counts=ev.n_rules.to_numpy(dtype=float)
    run={
      'file':str(path),'stream':conf.get('stream',path.parent.name),
      'variant':conf.get('variant',re.match(r'^(B0|S1|S2|S3)',path.stem).group(1) if re.match(r'^(B0|S1|S2|S3)',path.stem) else 'unknown'),
      'seed':conf.get('seed',int(re.search(r'_seed(\d+)',path.stem).group(1)) if re.search(r'_seed(\d+)',path.stem) else -1),
      'params':json.dumps(conf.get('params',{}),sort_keys=True),
      'n_evaluated':len(ev),'warmup':warmup,
      'accuracy':float(np.mean(y==p)),'macro_f1':macro_f1(y,p),
      'rules_mean':float(np.mean(counts)),'rules_max':int(max(counts)), 'rules_final':int(counts[-1]),
      'pruned':int(ev.pruned_now.sum()),'created':int(ev.created_now.sum()),'recreated':int(ev.recreated_now.sum()),
      'capacity_deletions':int(df.capacity_deletions.sum()),
      'runtime_eval_s':float(ev.time_s.sum()),'ms_per_obs':float(ev.time_s.mean()*1000),
      'valid':bool(int(df.capacity_deletions.sum())==0)
    }
    return run,ev


def add_compression(summary):
    """
    Calculate structural compression relative to the baseline model B0.
    Each pruning variant is compared with the baseline using the same
    stream, random seed, and number of evaluated observations.
    The compression ratio measures the proportion of rules retained
    by a variant compared with B0.
    The rule reduction percentage expresses how many fewer rules
    the variant maintains on average.
    If there is no unique valid matching baseline, the compression
    values remain NaN.
    """
    result=summary.copy(); result['compression_ratio']=np.nan; result['rules_reduction_pct']=np.nan
    for idx,row in result.iterrows():
        b=result[(result.stream==row['stream'])&(result.seed==row['seed'])&(result.n_evaluated==row['n_evaluated'])&(result.variant=='B0')&(result.valid)]
        if len(b)==1 and b.iloc[0].rules_mean>0:
            ratio=row.rules_mean/b.iloc[0].rules_mean
            result.loc[idx,'compression_ratio']=ratio
            result.loc[idx,'rules_reduction_pct']=100*(1-ratio)
    return result


def checkpoint_csv(ev,window=500,stride=100):
    """
    Calculate predictive accuracy at regular checkpoints using a
    moving window of recent observations.
    Instead of measuring accuracy over the entire stream, this
    function measures performance over the last 'window'
    observations at each checkpoint, which helps identify temporary
    performance drops caused by concept drift and observe how quickly
    the model adapts.
    If there are fewer observations than the window size, an
    empty DataFrame is returned.
    """
    correct=(ev.y_true.to_numpy()==ev.y_pred.to_numpy()).astype(float)
    if len(ev)<window: return pd.DataFrame(columns=['t','window_accuracy','n_rules'])
    rows=[]
    for stop in range(window,len(ev)+1,stride):
        part=ev.iloc[stop-window:stop]
        rows.append({'t':int(part.t.iloc[-1]),'window_accuracy':float(correct[stop-window:stop].mean()),'n_rules':float(part.n_rules.iloc[-1])})
    return pd.DataFrame(rows)


def plot_runs(group, out, window=500, stride=100, drift_points=None):
    """
    Generate plots showing model complexity and predictive performance
    throughout a data stream.
    1. n_rules.png - Shows how the number of active rules changes over
    time for each model variant.
    2. window_accuracy.png - Shows moving-window accuracy over time, allowing
    the comparison of predictive stability between variants.
    These plots are especially important for analysing whether
    pruning reduces structural complexity while maintaining
    predictive stability under concept drift.
    """
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    for metric in ['n_rules','window_accuracy']:
        fig,ax=plt.subplots(figsize=(8,4))
        for record,ev in group:
            if metric=='n_rules':
                xs=ev.t.to_numpy(); ys=ev.n_rules.to_numpy()
            else:
                cp=checkpoint_csv(ev,window,stride)
                xs=cp.t.to_numpy();ys=cp.window_accuracy.to_numpy()
            if len(xs): ax.plot(xs,ys,label=record['variant'])
        for drift in drift_points or []:ax.axvline(drift,linestyle='--',color='gray',alpha=.6)
        ax.set(xlabel='Observation t',ylabel='Number of rules' if metric=='n_rules' else f'Accuracy (last {window})',title=metric.replace('_',' ').title())
        ax.legend();fig.tight_layout();fig.savefig(out/f'{metric}.png',dpi=160);plt.close(fig)


def main():
    """
    Execute the complete evaluation pipeline for the experimental
    results generated by B0, S1, S2, and S3.
    This function coordinates all the previous functions.
    This function provides the main entry point for evaluating
    the preliminary and final experiments of the project.
    """
    ap=argparse.ArgumentParser(); ap.add_argument('--results',default='results'); ap.add_argument('--out',default='evaluation')
    ap.add_argument('--warmup',type=int,default=50);ap.add_argument('--window',type=int,default=500);ap.add_argument('--stride',type=int,default=100)
    args=ap.parse_args();output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    runs=[];all_evs={}; configs={}
    for f in sorted(Path(args.results).rglob('*.csv')):
        if not re.match(r'^(B0|S1|S2|S3)',f.stem):continue
        r,ev=evaluate_csv(f,args.warmup)
        runs.append(r); all_evs[r['file']]=ev
        jf=f.with_suffix('.json')
        configs[r['file']]=json.loads(jf.read_text()) if jf.exists() else {}
    if not runs:raise SystemExit(f'No model CSVs found under {args.results}. Execute experiments first.')
    summary=add_compression(pd.DataFrame(runs));summary.to_csv(output/'summary_runs.csv',index=False)
    for (stream,seed,n),grp in summary.groupby(['stream','seed','n_evaluated']):
        grp=grp.sort_values('file').drop_duplicates('variant')
        members=[(r.to_dict(),all_evs[r['file']]) for _,r in grp.iterrows() if r['valid']]
        points=[]
        for _,r in grp.iterrows():
            c=configs[r['file']]; points=c.get('stream_info',{}).get('drift_points',[])
            if points:break
        plot_runs(members,output/'figures'/str(stream)/f'seed{seed}_n{n}',args.window,args.stride,points)
    print(summary[['stream','variant','seed','accuracy','macro_f1','rules_mean','rules_reduction_pct','pruned','ms_per_obs','valid']].to_string(index=False))
    print('Saved',output/'summary_runs.csv')


if __name__=='__main__':main()

