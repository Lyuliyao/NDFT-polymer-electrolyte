from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
from prepare_workspace import prepare

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description='Reproduce the paper from archived metrics, profiles and selected configurations.')
    p.add_argument('action',choices=['verify','figure','train','evaluate'])
    p.add_argument('--system',choices=['eps75','eps2','lj1'],default='eps75')
    p.add_argument('--form',choices=['ours','pair'],default='ours')
    p.add_argument('--figure',choices=['1','2','3','4','5'],default='3')
    a=p.parse_args()
    roots=prepare()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),JAX_PLATFORMS='cpu',SIP_ROOT=str(roots[a.system]),NDFT_EPS75_ROOT=str(roots['eps75']),NDFT_EPS2_ROOT=str(roots['eps2']))
    if a.action=='verify':
        command=[sys.executable,str(ROOT/'scripts/verify_results.py')]
    elif a.action=='figure':
        filename={'1':'make_fig1.py','2':'make_fig2.py','3':'make_generalization.py','4':'make_fig3.py','5':'make_fig4.py'}[a.figure]
        command=[sys.executable,str(ROOT/'code/figures'/filename)]
    else:
        data=json.loads((ROOT/'results/tables/paper-results.json').read_text())
        key=a.form+' '+{'eps75':'7.5','eps2':'2','lj1':'lj'}[a.system]
        model=data['models'][key]
        if a.action=='evaluate':
            dest=ROOT/'work/evaluation'/a.system/model['model']
            dest.mkdir(parents=True,exist_ok=True)
            source=ROOT/'data/models'/a.system/model['model']
            for name in ('config.json','params.pkl','metrics.json'):
                shutil.copy2(source/name,dest/name)
            command=[sys.executable,'-m','learn.protocol','eval','--model',str(dest)]
        else:
            seed=int(model['model'].rsplit('_s',1)[1])
            conc=['0.1','0.2','0.4','0.6','0.7'] if a.system=='lj1' else ['0.01','0.02','0.04','0.06','0.08']
            command=[sys.executable,'-m','learn.protocol','joint','--variants','v2' if a.form=='ours' else 'v1','--joint-conc',*conc,'--M','6','--s-max','1','--depth','1','--C','4','--hidden','64','--readout-hidden','128','128','--kernel-net','32','32','--activation','softplus','--n-ref','0.4' if a.system=='lj1' else '0.02','--nval','3','--steps','4000','--lr','0.001','--weight-decay','0.00001','--stab-weight','300','--eval-every','20','--patience','50','--kboost','2','--kboost-mode','k','--loss','force','--select','best','--seed',str(seed),'--name-suffix','_reproduced']
    subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':
    main()
