"""Prediction roots for the size test: one learn/ root per big box.

learn/ assumes a cubic box (A = L^2, nbar = n_pairs / L^3).  A box n times
longer along z is entered as an equivalent cube of edge L_z = n L with
n_pairs = nbar L_z^3 = 480 n^3, which gives the right grid, the right nbar and
the right EL normalisation (A only multiplies F and cancels in mu); this is the
convention models.fine_geometry already uses.  Sk.npz (l_B, MD S(k)) is the
small box's.  Field runs are symlinks to the MD data on scratch.
"""
import json, os, shutil, glob
import numpy as np
S = '/mnt/gs21/scratch/lyuliyao/salt_in_polymer'
R = '/mnt/research/MultiscaleML_group/Liyao'
OUT = f'{R}/salt_in_polymer_bigbox'
T = 'c001_c002_c004_c006_c008'
SYS = {
  '75': dict(L=24.502285, sk=f'{S}/runs/pilot_T1.0_c0.04_one_both/zerofield/Sk.npz',
             models={'V1': f'{S}/runs/learn/v1_{T}_long',
                     **{f'V2_s{s}': f'{R}/salt_in_polymer_eps75/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_long_s{s}' for s in (0, 1, 2)}}),
  '2':  dict(L=24.416564, sk=f'{S}/eps2/runs/prod_T1.0_c0.04/zero_field/Sk.npz',
             models={'V1': f'{R}/salt_in_polymer_eps2/runs/learn/v1_{T}_full',
                     **{f'V2_s{i}': f'{R}/salt_in_polymer_eps2/runs/learn/joint_v2_L1_C4_R128x128_H64_{T}_val3_full{s}' for i, s in enumerate(('', '_s1', '_s2'))}}),
}
def build(name, e, n, field_dirs):
    root = f'{OUT}/{name}'; sysd = SYS[e]; L = sysd['L']
    st = f'{root}/runs/{name}'; os.makedirs(f'{st}/zero_field', exist_ok=True)
    os.makedirs(f'{root}/runs/field_analysis', exist_ok=True); os.makedirs(f'{root}/runs/learn', exist_ok=True)
    os.makedirs(f'{root}/splits', exist_ok=True)
    if not os.path.islink(f'{root}/code'): os.symlink(f'{R}/salt_in_polymer_eps2/code', f'{root}/code')
    json.dump(dict(conc=0.04, lz=n * L, n_pairs=480 * n ** 3, volume=(n * L) ** 3,
                   note=f'equivalent cube for a box {n}x longer along z (nbar = 480/L^3)'), open(f'{st}/zero_field/D.json', 'w'), indent=1)
    shutil.copy(sysd['sk'], f'{st}/zero_field/Sk.npz')
    for d in field_dirs:
        link = f'{st}/{os.path.basename(d)}'
        if not os.path.lexists(link): os.symlink(d, link)
    for lab, m in sysd['models'].items():
        dst = f'{root}/runs/learn/{lab}'; os.makedirs(dst, exist_ok=True)
        for f in ('config.json', 'params.pkl', 'metrics.json'): shutil.copy(f'{m}/{f}', dst)
        open(f'{dst}/SOURCE.txt', 'w').write(m + '\n')
    for csv in (f'{S}/bigbox/field_analysis/acceptance_summary.csv',):
        shutil.copy(csv, f'{root}/runs/field_analysis/acceptance_summary.csv')
    print('built', root, len(field_dirs), 'field dirs')
tags = ('p01', 'p09', 'p13', 'p15', 'p30', 'p31')
for e in ('75', '2'):
    for n in (1, 2, 4, 8):
        src = f'{S}/bigbox/eps{e}_c0.04_x{n}'
        if not os.path.isdir(src): continue
        build(f'eps{e}_c0.04_x{n}', e, n, [f'{src}/field_{t}' for t in tags if os.path.isdir(f'{src}/field_{t}')])
# eps_r = 2 at 1x: the production runs (TRAINING runs for these tags) and, separately, the C7 control of p13 (never trained on)
build('eps2_c0.04_x1_prod', '2', 1, [f'{S}/eps2/runs/prod_T1.0_c0.04/field_{t}' for t in ('p01', 'p09', 'p13', 'p15')])
build('eps2_c0.04_x1_control', '2', 1, [f'{S}/bigbox/control_eps2_c0.04_C7/field_p13'])
