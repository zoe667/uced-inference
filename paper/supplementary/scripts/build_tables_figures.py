"""Reporting only: regenerate supplementary tables and figures from frozen results.

Run extract_gp_validation.jl first. No GP fitting, screening changes, or UCED runs.
Python dependencies: numpy, scipy, matplotlib.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
YEARS = {2016: ROOT/'results/2016', 2021: ROOT/'results/2021'}
for d in ['tables', 'figures', 'data']:
    (OUT/d).mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
COLORS = {2016:'#29628c', 2021:'#b75b32'}

def readj(p): return json.loads(p.read_text())
def readcsv(p):
    with p.open() as f: return list(csv.DictReader(f))
def canon(s): return s.replace('CHP_Retrofitted_', 'Coal_Retrofitted_')
def label(s):
    s=canon(s)
    if s=='MLT_Band': return 'MLT band'
    group='R' if s.startswith('Coal_Retrofitted') else 'C'
    kind='min' if '_Min_Power_' in s else 'time'
    return f'{group}-{kind} {s.split("_")[-1]}'
def tex(s): return str(s).replace('_', r'\_')
def table(name, headers, rows, align=None):
    align=align or ('l'+'r'*(len(headers)-1))
    t=[r'\begin{tabular}{@{}'+align+r'@{}}',r'\toprule',
       ' & '.join(headers)+r' \\',r'\midrule']
    t+=[' & '.join(map(str,row))+r' \\' for row in rows]
    t += [r'\bottomrule',r'\end{tabular}']
    (OUT/'tables'/f'{name}.tex').write_text('\n'.join(t)+'\n')
def csvout(name, rows):
    with (OUT/'data'/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def figout(name):
    plt.savefig(OUT/'figures'/f'{name}.pdf')
    plt.savefig(OUT/'figures'/f'{name}.png',dpi=180)
    plt.close()
def spearman_matrix(a):
    ranks=np.empty_like(a,dtype=float)
    for j in range(a.shape[1]):
        order=np.argsort(a[:,j],kind='mergesort')
        ranks[order,j]=np.arange(1,len(a)+1,dtype=float)
    return np.corrcoef(ranks,rowvar=False)

summary={int(r['year']):r for r in readcsv(OUT/'data/surrogate_validation_summary.csv')}
domains={y:{canon(p['name']):p for p in readj(r/'lhs_design_domain.json')['parameters']} for y,r in YEARS.items()}
param_order=list(domains[2016])
table('parameter_domain', ['ID','Parameter','2016 bounds','2021 bounds','Years'],[
    [f'P{i+1}',label(n),f"[{domains[2016][n]['lower']:.2f}, {domains[2016][n]['upper']:.2f}]",
     f"[{domains[2021][n]['lower']:.2f}, {domains[2021][n]['upper']:.2f}]",'Both']
    for i,n in enumerate(param_order)],'llrrl')
csvout('parameter_domain',[dict(id=f'P{i+1}',parameter=n,lower_2016=domains[2016][n]['lower'],upper_2016=domains[2016][n]['upper'],lower_2021=domains[2021][n]['lower'],upper_2021=domains[2021][n]['upper']) for i,n in enumerate(param_order)])

rows=[]
for y,r in YEARS.items():
    s=summary[y]
    for mode in ['full','reduced']:
        cv=readj(r/f'cv_predictions_{mode}.json')
        rows.append([y,mode.capitalize(),f"{float(s[mode+'_cv_r2']):.4f}",f"{cv['overall_rmse']:.4f}",f"{float(s[mode+'_test_r2']):.4f}",f"{float(s[mode+'_test_rmse'])*1e4:.3f}",s['n_test']])
table('gp_performance',['Year','GP',r'CV $R^2$',r'CV RMSE$^{a}$',r'Test $R^2$',r'Test RMSE$^{b}$',r'$n_{test}$'],rows,'llrrrrr')

ardrows=[]; arddata=[]; boundrows=[]
robustness_available = all(
    (r/'active_selection_robustness/ard_bound_sensitivity.csv').exists()
    and (r/'active_selection_robustness/robustness_summary.json').exists()
    for r in YEARS.values()
)
for y,r in YEARS.items():
    a=readj(r/'master_lengthscales.json')['official_full_dimension_screening']
    logs=np.asarray(a['fold_log_lengthscales'])
    assert logs.shape==(5,13)
    votes=(logs >= a['log_ls_hi']-0.1).sum(axis=0)
    assert np.array_equal(votes,a['ceiling_votes'])
    for j,n in enumerate(a['feature_columns']):
        ardrows.append([y,label(n),*[f'{v:.2f}' for v in logs[:,j]],f'{votes[j]}/5','Drop' if votes[j]>=4 else 'Retain'])
        arddata.append(dict(year=y,parameter=canon(n),**{f'log_ls_fold_{k+1}':logs[k,j] for k in range(5)},votes=int(votes[j]),retained=bool(votes[j]<4)))
    if robustness_available:
        b=readcsv(r/'active_selection_robustness/ard_bound_sensitivity.csv')
        if y==2016: b16={canon(x['Parameter']):x for x in b}
        else: b21={canon(x['Parameter']):x for x in b}
table('ard_full',['Year','Parameter','F1','F2','F3','F4','F5','Hits','Rule'],ardrows,'llrrrrrrl')
csvout('ard_full',arddata)
if robustness_available:
    for n in param_order:
        boundrows.append([label(n),*[f"{d[n][k]}/5" for d in [b16,b21] for k in ['votes_0p5x','votes_1p0x','votes_2p0x']]])
    table('ard_bounds',['Parameter',r'2016: $0.5\times$',r'$1\times$',r'$2\times$',r'2021: $0.5\times$',r'$1\times$',r'$2\times$'],boundrows,'lrrrrrr')
    cfrows=[]
    for y,r in YEARS.items():
        a=readj(r/'active_selection_robustness/robustness_summary.json')['cross_fitted_ablation']
        cfrows.append([y,f"{a['full_r2']:.4f}",f"{a['reduced_r2']:.4f}",f"{a['full_rmse']:.4f}",f"{a['reduced_rmse']:.4f}",f"{100*(a['reduced_rmse']/a['full_rmse']-1):+.2f}\\%"])
    table('crossfit',['Year',r'Full $R^2$',r'Reduced $R^2$','Full RMSE','Reduced RMSE',r'$\Delta$ RMSE'],cfrows)
else:
    note='% Not generated: rerun 04b against runs_2016_100 and runs_2021_100.\n'
    (OUT/'tables/ard_bounds.tex').write_text(note)
    (OUT/'tables/crossfit.tex').write_text(note)

pred=readcsv(OUT/'data/surrogate_holdout_predictions.csv')
fig,axs=plt.subplots(1,2,figsize=(7.0,2.45))
for ax,(y,r) in zip(axs,YEARS.items()):
    d=[x for x in pred if int(x['year'])==y]
    x=np.array([float(v['uced_loss']) for v in d]);z=np.array([float(v['gp_predicted_loss']) for v in d])
    lo=min(x.min(),z.min());hi=max(x.max(),z.max());pad=(hi-lo)*.08
    ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#777',lw=.9,ls='--')
    ax.scatter(x,z,color=COLORS[y],s=24,zorder=3)
    ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel='Original UCED loss',ylabel='Reduced-GP prediction',title=f"{y}  |  test $R^2$ = {float(summary[y]['test_r2']):.3f}")
    ax.ticklabel_format(axis='both',style='plain',useOffset=False)
    ax.tick_params(labelsize=8);ax.grid(alpha=.13)
fig.tight_layout();figout('holdout_validation')

epsvals=[.05,.1,.2,.3]; allstats=[]; sets={}; matrices={}; geometryrows=[]; counts=[]
for y,r in YEARS.items():
    saved=readj(r/'surrogate_search_summary.json')
    d=readcsv(OUT/'data'/f'dense_predictions_{y}.csv')
    names=[canon(x['name']) for x in saved['search_domain']]
    arr=np.array([[float(row[n if n in row else n.replace('Coal_Retrofitted_','CHP_Retrofitted_')]) for n in names] for row in d])
    losses=np.array([float(row['predicted_loss']) for row in d])
    assert len(d)==20000
    for e in epsvals:
        mask=losses <= (1+e)*losses.min()
        ss=saved['epsilon_sweep'][str(e)]
        assert np.array_equal(mask,np.array([row['in_S_'+str(e).replace('.','p')].lower()=='true' for row in d]))
        counts.append(dict(year=y,epsilon=e,n_members=int(mask.sum()),fraction=float(mask.mean()),loss_min=float(losses.min()),cutoff=float((1+e)*losses.min())))
        for j,n in enumerate(names):
            p=domains[y][n];v=arr[mask,j];q=np.quantile(v,[.25,.5,.75]);niqr=(q[2]-q[0])/(p['upper']-p['lower'])
            match=next(t for t in ss['parameters'] if canon(t['parameter'])==n)
            if y==2021: assert abs(niqr-match['normalized_iqr'])<1e-10
            allstats.append(dict(year=y,epsilon=e,parameter=n,n_members=int(mask.sum()),minimum=float(v.min()),q25=float(q[0]),median=float(q[1]),q75=float(q[2]),maximum=float(v.max()),normalized_iqr=float(niqr)))
        if e==.1:
            sets[y]=(names,arr[mask]);corr=spearman_matrix(arr[mask]);matrices[y]=corr
            if y==2021: assert np.max(np.abs(corr-np.asarray(saved['supplement']['geometry']['spearman'])))<1e-10
            for j,n in enumerate(names):
                for k in range(j+1,len(names)): geometryrows.append(dict(year=y,parameter_1=n,parameter_2=names[k],spearman_rho=float(corr[j,k])))
csvout('region_quantiles',allstats);csvout('region_sizes',counts);csvout('pairwise_spearman',geometryrows)
primary=[x for x in allstats if x['epsilon']==.1]
macros=[]
for y,suffix in [(2016,'Sixteen'),(2021,'TwentyOne')]:
    c=next(x for x in counts if x['year']==y and x['epsilon']==.1)
    p=next(x for x in primary if x['year']==y and x['parameter']=='MLT_Band')
    macros.extend([r'\newcommand{\PoolCount'+suffix+'}{'+str(c['n_members'])+'}',
                   r'\newcommand{\PoolShare'+suffix+'}{'+f"{100*c['fraction']:.2f}"+r'\%}',
                   r'\newcommand{\MinLoss'+suffix+'}{'+f"{c['loss_min']:.7f}"+'}',
                   r'\newcommand{\MLTIQR'+suffix+'}{'+f"{p['normalized_iqr']:.3f}"+'}'])
(OUT/'tables/result_macros.tex').write_text('\n'.join(macros)+'\n')
table('region_sizes_wide',['Year',r'$5\%$',r'$10\%$',r'$20\%$',r'$30\%$'],[
    [y,*[str(next(x['n_members'] for x in counts if x['year']==y and x['epsilon']==e)) for e in epsvals]] for y in YEARS],'lrrrr')
table('primary_region',['Year','Parameter','Min','Q25','Median','Q75','Max','nIQR'],[
    [x['year'],label(x['parameter']),*[f"{x[k]:.3f}" for k in ['minimum','q25','median','q75','maximum','normalized_iqr']]] for x in primary],'llrrrrrr')
table('epsilon_iqr',['Year','Parameter',r'$5\%$',r'$10\%$',r'$20\%$',r'$30\%$'],[
    [x['year'],label(x['parameter']),*[f"{next(t['normalized_iqr'] for t in allstats if t['year']==x['year'] and t['parameter']==x['parameter'] and t['epsilon']==e):.3f}" for e in epsvals]] for x in primary],'llrrrr')
table('region_sizes',['Year',r'$\epsilon$',r'$|S_\epsilon|$','Pool share'],[
    [x['year'],f"{100*x['epsilon']:.0f}\\%",x['n_members'],f"{100*x['fraction']:.2f}\\%"] for x in counts],'lrrr')

fig,axs=plt.subplots(1,2,figsize=(7.0,2.55),gridspec_kw={'width_ratios':[1,1.3]})
for ax,(y,r) in zip(axs,YEARS.items()):
    d=[x for x in primary if x['year']==y]
    ax.barh([label(x['parameter']) for x in d],[x['normalized_iqr'] for x in d],color=COLORS[y],height=.6)
    ax.axvline(.5,color='#888',ls=':',lw=1)
    ax.set_xlim(0,.56);ax.invert_yaxis();ax.set_xlabel('IQR / design-domain width');ax.set_title(str(y));ax.grid(axis='x',alpha=.15)
fig.tight_layout();figout('normalized_iqr')

fig,axs=plt.subplots(1,2,figsize=(7.0,3.0))
for ax,(y,r) in zip(axs,YEARS.items()):
    names,_=sets[y];m=matrices[y]
    im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1)
    labs=[label(n).replace(' ','\n') for n in names]
    ax.set_xticks(range(len(names)),labs,fontsize=7);ax.set_yticks(range(len(names)),labs,fontsize=7)
    ax.set_title(str(y))
    for j in range(len(names)):
        for k in range(len(names)):ax.text(k,j,f'{m[j,k]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(m[j,k])>.7 else 'black')
fig.tight_layout();figout('pairwise_geometry')

fig,axs=plt.subplots(2,2,figsize=(7,4.0))
for col,(y,r) in enumerate(YEARS.items()):
    names,a=sets[y];b=a[:,names.index('MLT_Band')]
    mlt_domain=domains[y]['MLT_Band']
    for row,n in enumerate(['CHP_NonRetrofitted_Min_Power_300-660','CHP_NonRetrofitted_Min_Power_0-300']):
        ax=axs[row,col];ax.scatter(b,a[:,names.index(n)],s=4,alpha=.24,color=COLORS[y],rasterized=True)
        ax.set_xlim(mlt_domain['lower'],mlt_domain['upper']);ax.set_ylim(.3,.75);ax.set_ylabel(label(n));ax.set_xlabel('MLT band');ax.grid(alpha=.12)
        if row==0:ax.set_title(str(y))
fig.tight_layout();figout('joint_region')

sources=[]
for y,r in YEARS.items():
    names=['parameters.csv','lhs_design_domain.json','active_parameters.json','master_lengthscales.json','master_data_split.json','master_fold_assignments.json','plotting_data_reduced.json','cv_predictions_full.json','cv_predictions_reduced.json','trained_metamodel_full.jld2','trained_metamodel_reduced.jld2','surrogate_search_summary.json','dense_surrogate_search.csv']
    if robustness_available:
        names.append('active_selection_robustness/robustness_summary.json')
    for n in names:
        p=r/n;sources.append(dict(year=y,path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
csvout('source_manifest',sources)
print('Checked refreshed 20,000-point pools and region membership for both years; archived quantiles/correlations match the current summaries.')
if not robustness_available:
    print('Skipped 04b robustness tables because the final 100-run folders do not contain current robustness outputs.')
