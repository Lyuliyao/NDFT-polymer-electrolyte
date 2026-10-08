#!/usr/bin/env python3

import os
import sys


def truncate(path, step):
    kept = dropped = 0
    cut = None
    with open(path, "rb") as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            if not line.startswith(b"ITEM: TIMESTEP"):
                continue
            tline = fh.readline()
            try:
                t = int(tline)
            except ValueError:
                cut = pos if cut is None else cut
                break
            if t >= step:
                if cut is None:
                    cut = pos
                dropped += 1
            else:
                kept += 1
    if cut is not None:
        os.truncate(path, cut)
    return kept, dropped, cut


def main():
    if len(sys.argv) != 3:
        raise SystemExit('Usage: dump_truncate.py <dump> <step>; retain frames with timestep below step.')
    path, step = sys.argv[1], int(sys.argv[2])
    if not os.path.exists(path):
        print(f"{path}: not there, nothing to truncate")
        return
    kept, dropped, cut = truncate(path, step)
    where = "unchanged" if cut is None else f"truncated at byte {cut}"
    print(f"{os.path.basename(path)}: kept {kept} frames below step {step}, "
          f"dropped {dropped}, {where}")


if __name__ == "__main__":
    main()
