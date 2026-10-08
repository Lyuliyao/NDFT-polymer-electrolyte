#!/usr/bin/env python3

import argparse
import os


def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def repair_chunk(lines):

    head = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    out, kept, dropped = list(head), 0, []
    ncol = None
    i = 0
    while i < len(body):
        p = body[i].split()
        is_hdr = len(p) == 3 and "." not in p[0] and "." not in p[1] and all(is_num(x) for x in p)
        if not is_hdr:
            i += 1
            continue
        step, nch = int(p[0]), int(p[1])
        rows = body[i + 1:i + 1 + nch]
        if ncol is None and rows:
            ncol = len(rows[0].split())
        ok = len(rows) == nch and all(len(r.split()) == ncol and all(is_num(x) for x in r.split())
                                       for r in rows)
        if ok:
            out.append(body[i]); out.extend(rows); kept += 1
            i += 1 + nch
        else:
            dropped.append(step)

            i += 1
            while i < len(body):
                q = body[i].split()
                if len(q) == 3 and "." not in q[0] and "." not in q[1] and all(is_num(x) for x in q):
                    break
                i += 1
    return out, kept, dropped


def repair_time(lines):
    head = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    ncol = len(body[0].split()) if body else 0
    good = [ln for ln in body if len(ln.split()) == ncol and all(is_num(x) for x in ln.split())]
    bad = [ln.split()[0] if ln.split() else "?" for ln in body if ln not in good]
    return head + good, len(good), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for fn in a.files:
        lines = open(fn).read().splitlines()

        chunk = os.path.basename(fn).startswith("prof_") or any(ln.startswith("# Chunk") for ln in lines[:4])
        out, kept, dropped = (repair_chunk if chunk else repair_time)(lines)
        tag = "OK" if not dropped and (kept or not lines) else "DAMAGED"
        print(f"{tag} {fn}: {'blocks' if chunk else 'rows'} kept {kept}, dropped {len(dropped)}"
              + (f" (steps {dropped[:6]}{'...' if len(dropped) > 6 else ''})" if dropped else ""), flush=True)
        if not a.check and kept == 0 and lines:

            os.replace(fn, fn + ".orig")
            print(f"    {fn}: no complete block, moved to .orig")
            continue
        if dropped and not a.check:
            try:
                with open(fn + ".tmp", "w") as f:
                    f.write("\n".join(out) + "\n")
            except OSError:
                if os.path.exists(fn + ".tmp"):
                    os.remove(fn + ".tmp")
                raise
            os.replace(fn, fn + ".orig")
            os.replace(fn + ".tmp", fn)


if __name__ == "__main__":
    main()
