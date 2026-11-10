"""Trim Turbospectrum .bsyn line lists to a wavelength range.

bsyn reads and parses every line of every list for each synthesis, so lines far outside the synthesis
window cost time without changing the spectrum. Each species block (header "'<species> ' <ion> <nlines>",
a name line, then <nlines> lines starting with the wavelength in A) keeps only lines with
lo <= lambda <= hi; the header count is rewritten and empty blocks are dropped. Keep a margin of a few
tens of A around the synthesis window so line wings at the edges are unaffected. Species with no line in
the range get no output file.

    python scripts/trim_bsyn_linelist.py SRC_DIR DST_DIR --lo 3760 --hi 6930
"""
import argparse
import glob
import json
import os
import re

HEADER = re.compile(r"^'\s*[\d.]+\s*'\s+\d+\s+(\d+)\s*$")


def trim_file(src, dst, lo, hi):
    kept_total = read_total = 0
    tmp = dst + ".tmp"
    with open(src) as fin, open(tmp, "w") as fout:
        line = fin.readline()
        while line:
            m = HEADER.match(line.rstrip("\n"))
            if not m:
                raise ValueError(f"{src}: expected a species header, got {line[:80]!r}")
            name = fin.readline()
            n = int(m.group(1))
            kept = []
            for _ in range(n):
                row = fin.readline()
                if lo <= float(row.split(None, 1)[0]) <= hi:
                    kept.append(row)
            read_total += n
            if kept:
                head = line[: m.start(1)] + str(len(kept)) + line[m.end(1):]
                fout.write(head if head.endswith("\n") else head + "\n")
                fout.write(name)
                fout.writelines(kept)
                kept_total += len(kept)
            line = fin.readline()
    if kept_total:
        os.replace(tmp, dst)
    else:  # an empty list file is not a valid bsyn input; leave the species out
        os.remove(tmp)
    return read_total, kept_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("dst_dir")
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    a = ap.parse_args()
    os.makedirs(a.dst_dir, exist_ok=True)
    summary = {"src": os.path.abspath(a.src_dir), "lo": a.lo, "hi": a.hi, "files": {}}
    for src in sorted(glob.glob(os.path.join(a.src_dir, "*.bsyn"))):
        dst = os.path.join(a.dst_dir, os.path.basename(src))
        n_in, n_out = trim_file(src, dst, a.lo, a.hi)
        summary["files"][os.path.basename(src)] = {"lines_in": n_in, "lines_out": n_out}
        print(f"{os.path.basename(src)}: {n_in} -> {n_out}", flush=True)
    with open(os.path.join(a.dst_dir, "TRIM_PROVENANCE.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
