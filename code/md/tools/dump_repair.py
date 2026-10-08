#!/usr/bin/env python3
"""Repair a LAMMPS text dump that lost a stretch of writes.

On 2026-09-27 a gs21 I/O fault dropped a few kB of buffered output from runs
that kept going: a frame stops mid-line and the next surviving frame's
`ITEM: TIMESTEP` is glued to the cut line, e.g.
    12130 2 22.7306 14.7062 0.4027 -41.296 -70.578ITEM: TIMESTEP
dumpio then fails with "cannot reshape".  The simulation itself was not
affected, only what reached the file, so the frames after the gap are good.

This tool splits glued headers back onto their own line and keeps only frames
whose atom block is complete (NUMBER OF ATOMS lines of the header's column
count, all numeric) and whose step increases.  It reports every dropped frame
and every gap in the step sequence.

    dump_repair.py cat.dump [ani.dump ...]          repair; original kept as .orig
    dump_repair.py --check cat.dump [...]           report only, write nothing
"""
import argparse
import os
import re
import sys

GLUE = re.compile(r"(?<!^)ITEM: ")


def frames(fh):
    """Yield (header_lines, atom_lines) with glued headers split off; a cut
    line is yielded as None inside the atom block, which marks it damaged."""

    def lines():
        for raw in fh:
            s = raw.rstrip("\n")
            m = GLUE.search(s)
            if m:
                yield ("CUT", s[:m.start()])
                yield ("LINE", s[m.start():])
            else:
                yield ("LINE", s)

    hdr, atoms, in_atoms = [], [], False
    for kind, s in lines():
        if kind == "CUT":
            atoms.append(None)
            continue
        if s.startswith("ITEM: TIMESTEP"):
            if hdr:
                yield hdr, atoms
            hdr, atoms, in_atoms = [s], [], False
        elif s.startswith("ITEM:"):
            hdr.append(s)
            in_atoms = s.startswith("ITEM: ATOMS")
        elif in_atoms:
            atoms.append(s)
        else:
            hdr.append(s)
    if hdr:
        yield hdr, atoms


def check_frame(hdr, atoms):
    """-> (step, ok, reason)"""
    try:
        step = int(hdr[1])
        nat = int(hdr[hdr.index("ITEM: NUMBER OF ATOMS") + 1])
        ncol = len(hdr[-1].split()) - 2
    except (ValueError, IndexError):
        return None, False, "bad header"
    if len(atoms) != nat:
        return step, False, f"{len(atoms)} atom lines, want {nat}"
    for a in atoms:
        if a is None:
            return step, False, "cut line"
        p = a.split()
        if len(p) != ncol:
            return step, False, f"line with {len(p)} fields"
        try:
            [float(x) for x in p]
        except ValueError:
            return step, False, "non-numeric field"
    return step, True, ""


def process(fn, write):
    out = open(fn + ".tmp", "w") if write else None
    nkeep = 0
    last, dstep, gaps, drops = None, None, [], []
    with open(fn) as fh:
        for hdr, atoms in frames(fh):
            step, ok, why = check_frame(hdr, atoms)
            if ok and last is not None and step <= last:
                ok, why = False, f"step {step} not after {last}"
            if not ok:
                drops.append((step, why))
                continue
            if last is not None:
                d = step - last
                if dstep is None:
                    dstep = d
                elif d != dstep:
                    gaps.append((last, step))
            last = step
            nkeep += 1
            if out:
                out.write("\n".join(hdr) + "\n")
                out.write("\n".join(atoms) + "\n")
    if out:
        out.close()
    if write:
        if not drops:
            # nothing dropped: keep the original, the rewrite has the same content
            os.remove(fn + ".tmp")
        else:
            os.replace(fn, fn + ".orig")
            os.replace(fn + ".tmp", fn)
    return dict(file=fn, kept=nkeep, dropped=drops, gaps=gaps, dstep=dstep, last=last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--check", action="store_true", help="report only")
    a = ap.parse_args()
    for fn in a.dumps:
        try:
            r = process(fn, write=not a.check)
        except OSError:
            # e.g. ENOSPC from the storage pool: leave the original untouched
            if os.path.exists(fn + ".tmp"):
                os.remove(fn + ".tmp")
            raise
        miss = sum((b - e) // r["dstep"] - 1 for e, b in r["gaps"]) if r["dstep"] else 0
        tag = "OK" if not r["dropped"] and not r["gaps"] else "DAMAGED"
        print(f"{tag} {fn}: kept {r['kept']} frames (dstep {r['dstep']}, last step {r['last']}), "
              f"dropped {len(r['dropped'])}, gaps {len(r['gaps'])} ({miss} frames missing)", flush=True)
        for s, why in r["dropped"]:
            print(f"    dropped step {s}: {why}")
        for e, b in r["gaps"]:
            print(f"    gap {e} -> {b}")


if __name__ == "__main__":
    main()
