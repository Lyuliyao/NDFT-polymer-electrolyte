#!/bin/bash
# Validate the implemented force field and external potential against the model.
set -e
cd "$(dirname "$0")"
source ../env.sh
fail=0

echo "### 1. pair potentials vs. the analytic model"
rm -f pw_*.dat            # pair_write APPENDS; a stale file corrupts the check
$LMP -in in.pairwrite -var ff ../templates/ff.lmp -var tabfile ../tools/solvation.table \
     -log pw.log > /dev/null
G=$(grep -m1 "G vector" pw.log | awk '{print $5}')
cd .. && $SIP_PY tests/check_pairwrite.py $G || fail=1
cd tests

echo
echo "### 2. dielectric must scale the PPPM k-space term too"
Q=7.725213; QS=$($SIP_PY -c "print(repr($Q/7.5**0.5))")
A=$($LMP -in in.dielectric -var qp $Q  -var qm -$Q  -var eps 7.5 -log /dev/null | grep RESULT)
B=$($LMP -in in.dielectric -var qp $QS -var qm -$QS -var eps 1.0 -log /dev/null | grep RESULT)
$SIP_PY - "$A" "$B" <<'PY' || fail=1
import sys
a = [float(x) for x in sys.argv[1].split()[2::2]]
b = [float(x) for x in sys.argv[2].split()[2::2]]
d = max(abs(x-y)/max(abs(x),1e-30) for x, y in zip(a, b))
print(f"  max rel difference between the two parameterisations: {d:.2e}")
print("  DIELECTRIC OK" if d < 1e-9 else "  *** DIELECTRIC NOT APPLIED TO KSPACE ***")
sys.exit(0 if d < 1e-9 else 1)
PY

echo
echo "### 3. fix addforce must apply exactly -dV_alpha/dz"
$SIP_PY ../tools/make_potential.py --lz 24.0 --seed 4242 --kind both \
        --pp-neutral 1.7 --pp-charged 2.2 --tag extchk \
        --out-json extchk.json --out-lmp extchk.lmp > /dev/null
rm -f extcheck.dump
$LMP -in in.extforce -var lz 24.0 -var pot extchk.lmp -log /dev/null > /dev/null
$SIP_PY check_extforce.py extchk.json || fail=1

echo
[ $fail -eq 0 ] && echo "ALL VALIDATION TESTS PASSED" || echo "*** SOME TESTS FAILED ***"
exit $fail
