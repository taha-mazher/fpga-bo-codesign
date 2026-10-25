"""Training-seed noise of chosen designs: integer-model accuracy and sensor on-time over 5 training seeds, on the
validation CV (what the optimisers saw) and the official test split. Python only, no Vivado (hardware does not
depend on the seed beyond weight values). Usage: python noise_check.py"""
import os
import shutil

import numpy as np
import torch

from cascade_hw import SS, run
from dse import HERE, data
from gyro_policies import fit_models

DESIGNS = {'P baseline': (6, 8, 16, 1.0), 'A most accurate': (7, 2, 12, 0.5), 'B no gyro': (12, 2, 6, 0.0),
           'C smallest': (3, 2, 9, 0.25)}
SEEDS = range(5)


def score(split, h1, h2, ww, tau, seed):
    res = []
    for tr, te in data()[split]:
        with torch.enable_grad():
            m1, sc1, m2 = fit_models(data()['uci'], tr, seed=seed, h1=h1, h2=h2)
        res.append(run('noise_tmp', data()['uci'], te, m1, sc1, m2.m, m2.sc,
                       dict(M=SS, M2=SS, tau=tau, tau_keep=0.0, keep_all=True), ww))
    return 100 * sum(r['int'][0] for r in res) / sum(r['int'][1] for r in res), np.mean([r['acc']['int'][1] for r in res])


torch.set_grad_enabled(False)
rows = []
for label, d in DESIGNS.items():
    for split in ('val', 'test'):
        s, failed = [], 0
        for seed in SEEDS:
            try:
                s.append(score(split, *d, seed))
            except AssertionError:  # this seed's weights do not fit WW bits
                failed += 1
        s = np.array(s)
        rows.append(f'{label:16s} {split:4s} acc {s[:, 0].mean():6.2f} +- {s[:, 0].std(ddof=1):4.2f}  '
                    f'(min {s[:, 0].min():6.2f})  gyro {s[:, 1].mean():5.2f} +- {s[:, 1].std(ddof=1):4.2f}%  '
                    f'seeds not fitting {d[2]} bits: {failed}/{len(SEEDS)}')
        print(rows[-1], flush=True)
shutil.rmtree(os.path.join(HERE, 'rtl', 'mem', 'cascade_noise_tmp'), ignore_errors=True)
os.remove(os.path.join(HERE, 'out', 'cascade_noise_tmp.mem'))
print('done')
