import json,csv,hashlib
from pathlib import Path
O=Path(__file__).resolve().parents[1]; R=O.parents[1]
a=json.loads((O/'data/revalidation_audit.json').read_text())
roles=['medoid','interior_1','interior_2','interior_3','interior_4','surrogate_best']
rows=sorted([x for x in a if x['year']=='2021'],key=lambda x:roles.index(x['role']))
assert len(rows)==6
candidates=list(csv.DictReader((R/'results/2021/revalidation_candidates_eps10.csv').open()))
flat=[]
for x in a:
 c=x['components'];assert abs(x['actual']-sum(c['raw_'+k+'_error']**2 for k in ['coal','wind','solar','mlt'])/4)<1e-12
 flat.append(dict(year=x['year'],role=x['role'],gp_loss=x['prediction'],uced_loss=x['actual'],residual=x['actual']-x['prediction'],**{k:c['raw_'+k+'_error'] for k in ['coal','wind','solar','mlt']},**x['params']))
for x in rows:
 c=next(c for c in candidates if c['role']==x['role'])
 for k,v in x['params'].items():assert abs(float(c[k.replace('Coal_Retrofitted_','CHP_Retrofitted_')])-v)<1e-12
 assert x['prediction']<=1.1*.00710584661017041
with (O/'data/revalidation.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=flat[0].keys());w.writeheader();w.writerows(flat)
with (O/'data/revalidation_source_manifest.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['path','sha256'])
 for p in [R/x['source'] for x in a]+[R/'results/2021/revalidation_candidates_eps10.csv',R/'results/2016/revalidation_candidates_eps10.csv']:
  w.writerow([str(p.relative_to(R)),hashlib.sha256(p.read_bytes()).hexdigest()])
def name(x):return {'medoid':'Center representative','surrogate_best':'Surrogate best'}.get(x['role'],x['role'].replace('interior_','Interior '))
s=['\\begin{tabular}{@{}lrrr@{}}\\toprule','Point & GP loss & UCED loss & UCED $-$ GP \\\\ \\midrule']
for x in rows:s.append(f"{name(x)} & {x['prediction']:.6f} & {x['actual']:.6f} & {x['actual']-x['prediction']:+.6f} \\\\")
s+=['\\bottomrule\\end{tabular}'];(O/'tables/revalidation_2021.tex').write_text('\n'.join(s)+'\n')
s=['\\begin{tabular}{@{}lrrrr@{}}\\toprule','Point & Coal & Wind & Solar & Exchange \\\\ \\midrule']
for x in rows:s.append(name(x)+' & '+' & '.join(f"{x['components']['raw_'+k+'_error']:.4f}" for k in ['coal','wind','solar','mlt'])+r' \\')
s+=['\\bottomrule\\end{tabular}'];(O/'tables/revalidation_components_2021.tex').write_text('\n'.join(s)+'\n')
print('Validated six candidate parameter vectors, current region membership and component identities.')
print('2021 mean absolute residual:',sum(abs(x['actual']-x['prediction']) for x in rows)/6)
# Both years are reported; 2016 roles retain their original selection provenance.
for year,run in [('2016','2016_100'),('2021','2021_100')]:
 rr=sorted([x for x in a if x['year']==year],key=lambda x:roles.index(x['role']))
 assert len(rr)==6
 cc=list(csv.DictReader((R/f'results/{run[:4]}/revalidation_candidates_eps10.csv').open()))
 for x in rr:
  c=next(c for c in cc if c['role']==x['role'])
  for k,v in x['params'].items():assert abs(float(c[k.replace('Coal_Retrofitted_','CHP_Retrofitted_')])-v)<1e-12
 for component in [False,True]:
  s=[r'\begin{tabular}{@{}l'+('rrrr' if component else 'rrr')+r'@{}}\toprule',('Point & Coal & Wind & Solar & Exchange' if component else 'Point & GP loss & UCED loss & UCED $-$ GP')+r' \\ \midrule']
  for x in rr:
   vals=[x['components']['raw_'+k+'_error'] for k in ['coal','wind','solar','mlt']] if component else [x['prediction'],x['actual'],x['actual']-x['prediction']]
   s.append(name(x)+' & '+' & '.join(f'{v:.4f}' if component else f'{v:.6f}' for v in vals)+r' \\')
  s.append(r'\bottomrule\end{tabular}')
  (O/f'tables/revalidation_{"components_" if component else ""}{year}.tex').write_text('\n'.join(s)+'\n')
print('Both years: twelve parameter vectors match archived candidate records.')
