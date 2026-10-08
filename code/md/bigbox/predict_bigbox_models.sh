#!/bin/bash
# Copy trained models into the long-box prediction roots and predict their c = 0.04 field runs there.
#   bash bigbox/predict_bigbox_models.sh LABEL "<model dir with %s for the seed>" [roots...]
# e.g. bash bigbox/predict_bigbox_models.sh KB4 "$R/salt_in_polymer_eps75/runs/learn/joint_..._kn1softplus_kb4_long_s%s" eps75_c0.04_x2 eps75_c0.04_x4
set -e
R=/mnt/research/MultiscaleML_group/Liyao; B=$R/salt_in_polymer_bigbox; PY=/mnt/home/lyuliyao/.conda/envs/heat/bin/python
LAB=$1; PAT=$2; shift 2
F='^Jax plugin|jax_plugins|cuInit|versions_helpers|_check_cuda|plugin_module.initialize|RuntimeWarning|profile_uncertainties|^  sn, sf'
for root in "$@"; do
  for s in 0 1 2; do
    m=$(printf "$PAT" $s); dst=$B/$root/runs/learn/${LAB}_s$s
    [ -f $m/params.pkl ] || { echo "missing $m"; continue; }
    mkdir -p $dst; cp $m/config.json $m/params.pkl $m/metrics.json $dst/; echo "$m" > $dst/SOURCE.txt
    cd $B/$root/code && SIP_ROOT=$B/$root JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      $PY -m learn.protocol predict --model $dst --conc 0.04 --log $B/$root/runs/learn/protocol.log 2>&1 | grep -vE "$F" | grep "^  ->" | cut -c1-200
  done
done
