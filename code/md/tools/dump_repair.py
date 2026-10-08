#!/usr/bin/env python3

import argparse
import os
import re
import sys

GLUE = re.compile(r"(?<!^)ITEM: ")


def frames(fh):


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
