"""Run on amd20; emit the compact MD/neural map cache of the aperiodic 2D potential p55 to stdout
(2026-10-06, replaces the egg carton p50).

Read-only: no simulations, model fitting, or source-file changes.
"""
import io,sys,json
import numpy as np
out={};sources={}
SEED={"eps75":0,"eps2":1}   # the reported model per dielectric constant (2026-10-06)
for system in ("eps75","eps2"):
    a=f"/mnt/gs21/scratch/lyuliyao/salt_in_polymer/field2d/{system}_c0.04/field_p55/profiles2d.npz"
    b=f"/mnt/research/MultiscaleML_group/Liyao/salt_in_polymer_field2d/{system}/p55_pred.npz"
    md=np.load(a,allow_pickle=True);pred=np.load(b,allow_pickle=True)
    assert np.allclose(md["lo"],pred["lo"])
    for k in ("lo","lz","cation_n","cation_n_err"):
        out[system+"_"+k]=md[k]
    out[system+"_pred"]=pred[f"n_NF s{SEED[system]}"][0]
    sources[system]={"md":a,"prediction":b,"prediction_key":f"n_NF s{SEED[system]}"}
out["sources"]=json.dumps(sources)
f=io.BytesIO();np.savez_compressed(f,**out);sys.stdout.buffer.write(f.getvalue())
