#!/usr/bin/env python3
"""Extend the Gaia-ESO (GES v5) Turbospectrum molecular line lists blueward of 4200 A from Kurucz-format files.

The molecules-420-920nm lists start at 4200 A, so a 3780-6910 A synthesis has no molecular lines below 4200 A.
The same molecular data are on Kurucz's site in RMOLECASC format: CN violet (Brooke et al. 2014,
molecules/cn/cnbxbrookek.asc) and CH (Masseron et al. 2014, molecules/ch/chmasseron.asc). This script

  1. matches the Kurucz lines to the GES list in an overlap window (default 4200-4300 A) on wavelength and
     lower-level energy, and derives the log gf offset between the two conventions (CN: -0.290 dex, CH: 0.000);
  2. keeps one isotopologue (--iso, the Kurucz isotope code) and calibrates it against its own GES file;
  3. writes the Kurucz lines in [lo, 4200) A in the GES file's own Turbospectrum conventions (column 5 = lower J
     for CN, upper 2J+1 for CH; the GES radiative-damping placeholder), log gf shifted onto the GES scale;
  4. converts the overlap window the same way and checks it reproduces the GES entries (wavelength, energy, log gf).

    python kurucz_molecules_to_bsyn.py cn KURUCZ.asc GES.bsyn OUT.bsyn [--lo 3760]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

CM_PER_EV = 8065.544
CONVENTION = {  # GES column 5 and the radiative-damping value used for the new lines
    "cn": dict(col5="j_low", raddmp=1.0),
    "ch": dict(col5="2j_up+1", raddmp=1.0),
}


def read_kurucz(path):
    rows = []
    for line in open(path):
        try:
            wl = float(line[0:10]) * 10.0; gf = float(line[10:17])
            j1 = float(line[17:22]); e1 = abs(float(line[22:32])); j2 = float(line[32:37]); e2 = abs(float(line[37:48]))
        except ValueError:
            continue
        lo_first = e1 <= e2
        fields = line[48:].split()  # e.g. "607X02f2 B02f2 12 718 K": lower label, upper label, isotope code, ...
        rows.append((wl, gf, (e1 if lo_first else e2) / CM_PER_EV, j1 if lo_first else j2, j2 if lo_first else j1,
                     fields[2] if len(fields) >= 3 else "", line[48:].rstrip()))
    return rows


def read_ges(path):
    lines = open(path).readlines()
    header = lines[:2]
    data = []
    for l in lines[2:]:
        p = l.split()
        if p and p[0][0].isdigit():
            data.append([float(x) for x in p[:6]])
    return header, np.asarray(data)


def match(K, G, lo, hi):
    Kw = np.array([k[0] for k in K]); Ke = np.array([k[2] for k in K])
    g = G[(G[:, 0] >= lo) & (G[:, 0] < hi) & (G[:, 2] > -2.0)]
    pairs = []
    for row in g:
        m = np.nonzero((np.abs(Kw - row[0]) < 0.006) & (np.abs(Ke - row[1]) < 0.01))[0]
        if m.size:
            pairs.append((row, int(m[np.argmin(np.abs(Kw[m] - row[0]))])))
    return g, pairs


def fmt(k, mol, dgf, conv):
    wl, gf, elo, jlo, jup, _tag, label = k
    col5 = jlo if conv["col5"] == "j_low" else 2 * jup + 1
    lab = "".join(ch for ch in label.split()[0] if ch.isalnum())[:12] if label.split() else "kurucz"
    return f"{wl:.3f} {elo:.3f} {gf + dgf:.3f} .00 {col5:.1f} {conv['raddmp']:.3E} 'X' 'X'   0.0    1.0 '{lab}'\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("molecule", choices=sorted(CONVENTION)); ap.add_argument("kurucz"); ap.add_argument("ges"); ap.add_argument("out")
    ap.add_argument("--lo", type=float, default=3760.0); ap.add_argument("--edge", type=float, default=4200.0)
    ap.add_argument("--overlap", type=float, nargs=2, default=(4200.0, 4300.0))
    ap.add_argument("--iso", required=True, help="Kurucz isotope code to keep (CN: 12 = 12C14N, 13 = 13C14N, 15 = 12C15N; CH: 12, 13)")
    a = ap.parse_args()
    conv = CONVENTION[a.molecule]
    K = [k for k in read_kurucz(a.kurucz) if k[5] == a.iso]; header, G = read_ges(a.ges)
    g, pairs = match(K, G, *a.overlap)
    dgf = float(np.median([row[2] - K[j][1] for row, j in pairs]))
    tags = Counter(K[j][5] for _, j in pairs); tag = a.iso
    # 4. round trip on the overlap window, in the GES conventions
    rt = []
    for row, j in pairs:
        p = fmt(K[j], a.molecule, dgf, conv).split()
        rt.append((float(p[0]) - row[0], float(p[1]) - row[1], float(p[2]) - row[2], float(p[4]) - row[4]))
    rt = np.asarray(rt)
    keep = [k for k in K if a.lo <= k[0] < a.edge and k[5] == tag]
    keep.sort(key=lambda k: k[0])
    species = header[0].split("'")[1]
    with open(a.out, "w") as fh:
        fh.write(f"'{species}' 1 {len(keep)}\n")
        fh.write(f"'Kurucz {a.kurucz.split('/')[-1]} -> GES conventions, loggf {dgf:+.3f}, {a.lo:.0f}-{a.edge:.0f} A'\n")
        for k in keep:
            fh.write(fmt(k, a.molecule, dgf, conv))
    rep = dict(molecule=a.molecule, kurucz=a.kurucz, ges=a.ges, out=a.out, overlap=a.overlap,
               ges_strong_lines_in_overlap=int(len(g)), matched=len(pairs), isotopologue_tag=tag, tags_in_matches=dict(tags),
               loggf_offset_added=dgf, roundtrip_max_abs=dict(wavelength=float(np.abs(rt[:, 0]).max()), energy=float(np.abs(rt[:, 1]).max()),
                                                              loggf=float(np.abs(rt[:, 2]).max()), col5=float(np.abs(rt[:, 3]).max())),
               roundtrip_frac_exact_loggf=float(np.mean(np.abs(rt[:, 2]) <= 0.0015)),
               lines_written=len(keep), range=[a.lo, a.edge], strong_lines_written=int(sum(k[1] + dgf > -2 for k in keep)))
    json.dump(rep, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
