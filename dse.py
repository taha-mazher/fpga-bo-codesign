"""Design-space evaluator for the UCI cascade: one design point -> accuracy, sensor time, cycles, LUTs, FFs, DSPs,
WNS and power, appended as one row of out/dse_results.csv (and read back from there if already evaluated).

Knobs: H1 stage-1 hidden neurons, H2 stage-2 hidden neurons, WW weight bits, TAU gate margin (stage-1 logits).
Splits: 'val' is 3-fold cross-validation over the 21 official training subjects, accuracy pooled over folds (what
the optimisers see); hardware is built and checked from the last fold. 'test' trains on all 21 and scores the
official test subjects (final report only).
Per point: train both stages (cached per split, H1, H2), quantise and export (cascade_hw.run), iverilog bit-exact
check of the RTL in WSL, then Vivado synthesis and implementation at 50 MHz with vectorless power.
Usage: python dse.py H1 H2 WW TAU [val|test]      (6 8 16 1 test is the hand-tuned baseline cascade)
"""
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from cascade_hw import SS, run
from gyro_policies import UCI, build as build_uci, fit_models

HERE = os.path.dirname(os.path.abspath(__file__))
VIVADO = os.environ.get('VIVADO', r'C:\Xilinx\Vivado\2024.2\bin\vivado.bat')
CSV = os.path.join(HERE, 'out', 'dse_results.csv')
PERIOD_NS = 20
_data, _models = {}, {}


def data():
    if not _data:
        uci = build_uci()
        subj = np.array([e['subject'] for e in uci])
        test_subj = list(set(np.loadtxt(os.path.join(UCI, 'Test', 'subject_id_test.txt'), dtype=int)))
        tr, te = np.flatnonzero(~np.isin(subj, test_subj)), np.flatnonzero(np.isin(subj, test_subj))
        _data.update(uci=uci, test=[(tr, te)], val=[(tr[a], tr[b]) for a, b in GroupKFold(3).split(tr, groups=subj[tr])])
    return _data


def models(split, fold, h1, h2):
    if (split, fold, h1, h2) not in _models:
        with torch.enable_grad():
            m1, sc1, m2 = fit_models(data()['uci'], data()[split][fold][0], h1=h1, h2=h2)
        _models[split, fold, h1, h2] = (m1, sc1, m2.m, m2.sc)
    return _models[split, fold, h1, h2]


def grab(path, pat):
    return float(re.search(pat, open(path).read()).group(1))


def cached(split, h1, h2, ww, tau):
    if os.path.exists(CSV):
        for r in csv.DictReader(open(CSV)):
            if (r['split'], int(r['H1']), int(r['H2']), int(r['WW']), float(r['TAU'])) == (split, h1, h2, ww, tau):
                return {k: v if k == 'split' else float(v) for k, v in r.items()}


def evaluate(h1, h2, ww, tau, split='val'):
    if (row := cached(split, h1, h2, ww, tau)) is not None:
        return row
    name = f'cascade_dse_{split[0]}_{h1}_{h2}_{ww}_{tau:g}'
    t0 = time.time()
    torch.set_grad_enabled(False)
    res = [run(name[8:], data()['uci'], te, *models(split, k, h1, h2), dict(M=SS, M2=SS, tau=tau, tau_keep=0.0, keep_all=True), ww)
           for k, (_, te) in enumerate(data()[split])]  # each fold overwrites the export; the last one is built
    acc = 100 * sum(r['int'][0] for r in res) / sum(r['int'][1] for r in res)
    sensor = np.mean([r['acc']['int'][1] for r in res])
    t1 = time.time()

    sh = '/mnt/c' + os.path.join(HERE, 'rtl', 'run_cascade_sim.sh')[2:].replace('\\', '/')
    sim = subprocess.run(['wsl', '-e', 'bash', sh, f'mem/{name}/'], capture_output=True, text=True).stdout
    assert 'RESULT: PASS' in sim, f'RTL is not bit-exact with Python:\n{sim}'
    win, s2, c1, c2 = map(int, re.search(r'windows=(\d+).*stage2=(\d+) sensor_on.*stage1=(\d+) stage2=(\d+)', sim).groups())
    cycles = c1 + s2 / win * (c2 - c1)
    t2 = time.time()

    vdir = os.path.join(HERE, 'vivado')
    subprocess.run([VIVADO, '-mode', 'batch', '-nojournal', '-log', f'build_{name}.log', '-source', 'build_cascade.tcl',
                    '-tclargs', name], cwd=vdir, check=True, capture_output=True)
    u, t, p = (os.path.join(vdir, f'out_{name}', f) for f in ('utilization_routed.rpt', 'timing_routed.rpt', 'power_vectorless.rpt'))
    dyn = grab(p, r'\| Dynamic \(mW\)\s*\|\s*([\d.]+)')
    row = dict(split=split, H1=h1, H2=h2, WW=ww, TAU=tau, acc=round(acc, 3), sensor_on=round(sensor, 2), cycles=round(cycles, 1),
               LUTs=grab(u, r'\| Slice LUTs\s*\|\s*(\d+)'), FFs=grab(u, r'\| Slice Registers\s*\|\s*(\d+)'),
               DSP=grab(u, r'\| DSPs\s*\|\s*(\d+)'), WNS=grab(t, r'WNS\(ns\).*\n.*\n\s*(-?[\d.]+)'),
               power_mW=grab(p, r'Total On-Chip Power \(mW\)\s*\|\s*([\d.]+)'), dyn_mW=dyn,
               dyn_nJ=round(dyn * cycles * PERIOD_NS * 1e-3, 3),  # mW * ns = pJ
               t_python=round(t1 - t0, 1), t_sim=round(t2 - t1, 1), t_vivado=round(time.time() - t2, 1))
    new = not os.path.exists(CSV)
    with open(CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, row)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)
    return row


if __name__ == '__main__':
    h1, h2, ww = map(int, sys.argv[1:4])
    evaluate(h1, h2, ww, float(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else 'val')
