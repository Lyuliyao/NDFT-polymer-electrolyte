from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
from prepare_workspace import prepare

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {'1': ['make_fig1.py'], '2': ['make_fig2.py'], '3': ['make_generalization.py'], '4': ['kernel_k1.py', 'make_fig3.py'], '5': ['make_fig6.py']}
EPS = ('7.5', '5', '4', '3', '2')

def main():
    p=argparse.ArgumentParser(description='Reproduce the paper from archived metrics, profiles and selected configurations.')
    p.add_argument('action',choices=['verify','figure','train','evaluate'])
    p.add_argument('--system',choices=['eps75','eps2','lj1','epsx'],default='eps75')
    p.add_argument('--form',choices=['transferable','ours','pair'],default='transferable')
    p.add_argument('--figure',choices=sorted(FIGURES),default='3')
    p.add_argument('--leave-out',choices=EPS,default=None,help='transferable form: dielectric constant left out of training')
    p.add_argument('--seed',type=int,default=None)
    a=p.parse_args()
    roots=prepare()
    env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),JAX_PLATFORMS='cpu',SIP_ROOT=str(roots[a.system]),NDFT_EPS75_ROOT=str(roots['eps75']),NDFT_EPS2_ROOT=str(roots['eps2']),NDFT_EPSX_ROOT=str(roots['epsx']))
    for e in EPS:
        env[f"NDFT_EPS{e.replace('.','')}_ROOT"]=str(roots['eps'+e.replace('.','')])
    if a.action=='verify':
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_results.py')],cwd=ROOT,env=env,check=True)
        return
    if a.action=='figure':
        for filename in FIGURES[a.figure]:
            subprocess.run([sys.executable,str(ROOT/'code/figures'/filename)],cwd=ROOT,env=env,check=True)
        return
    data=json.loads((ROOT/'results/tables/paper-results.json').read_text())
    if a.form=='transferable':
        system=a.system if a.system in ('eps75','eps2') else 'eps75'
        model=data['models']['transferable '+{'eps75':'7.5','eps2':'2'}[system]]
    else:
        key=a.form+' '+{'eps75':'7.5','eps2':'2','lj1':'lj','epsx':'7.5'}[a.system]
        model=data['models'][key]; system=model['system']
    if a.action=='evaluate':
        dest=ROOT/'work/evaluation'/system/model['model']
        dest.mkdir(parents=True,exist_ok=True)
        source=ROOT/'data/models'/system/model['model']
        for name in ('config.json','params.pkl','metrics.json'):
            shutil.copy2(source/name,dest/name)
        env['SIP_ROOT']=str(roots[system])
        command=[sys.executable,'-m','learn.protocol','eval','--model',str(dest)]
    else:
        common=['--nval','3','--depth','1','--C','4','--readout-hidden','128','128','--kernel-net','32','32','--M','6','--activation','softplus',
                '--basis-learn','--s-max','4','--basis-lr-scale','30','--steps','12000','--ckpt-every','2000','--patience','100000',
                '--kboost','2','--kboost-mode','k']
        if a.form=='transferable':
            seed=0 if a.seed is None else a.seed
            env['SIP_ROOT']=str(roots['epsx'])
            env['SIP_EPS_ROOTS']=':'.join(f"{e}={roots['eps'+e.replace('.','')]}" for e in EPS)
            command=[sys.executable,'-m','learn.protocol','joint','--variants','v2',*common,'--joint-conc','0.01','0.02','0.04','0.06','0.08',
                     '--lb-mode','poly','--seed',str(seed),'--name-suffix',f'_kn1softplus_kbK2_bw_reproduced_s{seed}']
            if a.leave_out:
                command+=['--eps-test',a.leave_out]
        else:
            seed=int(model['model'].rsplit('_s',1)[1]) if a.seed is None else a.seed
            lj=system=='lj1'
            conc=['0.1','0.2','0.4','0.6','0.7'] if lj else ['0.01','0.02','0.04','0.06','0.08']
            command=[sys.executable,'-m','learn.protocol','joint','--variants','v2' if a.form=='ours' else 'v1',*common,'--joint-conc',*conc,
                     '--n-ref','0.4' if lj else '0.02','--seed',str(seed),'--name-suffix',f"_kn1softplus_kbK2_bw{'_lj1' if lj else ''}_reproduced_s{seed}"]
    subprocess.run(command,cwd=ROOT/'code',env=env,check=True)

if __name__=='__main__':
    main()
