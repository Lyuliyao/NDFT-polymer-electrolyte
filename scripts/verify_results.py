from pathlib import Path
import ast, json, math, statistics

ROOT = Path(__file__).resolve().parents[1]

def profile_error(row):
    return 100 * statistics.mean(v['rel_l2'] for v in row['profile'].values())

def main():
    reference=json.loads((ROOT/'results/tables/paper-results.json').read_text())
    checked=0
    for label, recorded in reference['models'].items():
        model=ROOT/'data/models'/recorded['system']/recorded['model']
        metrics=json.loads((model/'metrics.json').read_text())
        name='predict_c05' if recorded['system']=='lj1' else 'predict_c005'
        path=model/name/'metrics.json'
        transfer=json.loads(path.read_text()).get('transfer_rows',[]) if path.exists() else []
        transfer=[r for r in transfer if r['tag'] in {f'p{i:02d}' for i in range(22)}]
        heldout=metrics['heldout_rows']
        def mean(rows):
            use=rows if recorded['final_iterates'] else [r for r in rows if r.get('el',{}).get('converged',True) and profile_error(r)<=50]
            return statistics.mean(profile_error(r) for r in use) if use else None
        for key,rows in [('ho',heldout),('c5',transfer),('all_test',heldout+transfer)]:
            a,b=mean(rows),recorded[key]
            assert a==b or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)),(label,key,a,b)
            checked+=1
        train=set(metrics.get('train_runs',[])); val=set(metrics.get('val_runs',[])); test=set(metrics.get('heldout_runs',[]))
        assert not(train&val or train&test or val&test),label
        checked+=1
    for path in list((ROOT/'code').rglob('*.py'))+list((ROOT/'scripts').rglob('*.py')):
        ast.parse(path.read_text(),filename=str(path))
    print(f'{len(reference["models"])} selected fits; {checked} independent result/split checks passed. No training or MD performed.')

if __name__=='__main__':
    main()
