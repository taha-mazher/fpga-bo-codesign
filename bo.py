"""Multi-objective search of the UCI cascade design space: Gaussian-process BO against random search and NSGA-II at
equal budgets of completed designs, scored on the validation subjects. Objectives: accuracy (max), LUTs (min),
gyro on-time (min). Studies live in out/bo.db and resume; any design already built is read from out/dse_results.csv.
Usage: python bo.py run METHOD SEED N      METHOD = gp | random | nsga2
       python bo.py report                 hypervolume vs completed designs, and each study's Pareto front
"""
import os
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # ponytail: anaconda MKL and pip torch each ship libiomp5md; one conda env would remove this
import numpy as np
import optuna
from optuna._hypervolume import compute_hypervolume
from optuna.trial import TrialState

from dse import HERE, evaluate

DB = 'sqlite:///' + os.path.join(HERE, 'out', 'bo.db')
IDEAL = np.array([-100.0, 0.0, 0.0])   # (-accuracy %, LUTs, gyro on %)
REF = np.array([-80.0, 500.0, 50.0])   # worst corner; hypervolume is the fraction of the IDEAL-REF box dominated
SAMPLERS = dict(gp=lambda s: optuna.samplers.GPSampler(seed=s, n_startup_trials=8),
                random=lambda s: optuna.samplers.RandomSampler(seed=s),
                nsga2=lambda s: optuna.samplers.NSGAIISampler(seed=s, population_size=8))


def objective(t):
    r = evaluate(t.suggest_int('H1', 2, 16), t.suggest_int('H2', 2, 24), t.suggest_int('WW', 6, 16),
                 t.suggest_float('TAU', 0, 4, step=0.25))
    for k in ('cycles', 'dyn_nJ', 'WNS', 'FFs'):
        t.set_user_attr(k, r[k])
    return r['acc'], r['LUTs'], r['sensor_on']


def done(study):
    return [t for t in study.trials if t.state == TrialState.COMPLETE]


def run(method, seed, n):
    study = optuna.create_study(study_name=f'{method}_{seed}', storage=DB, load_if_exists=True,
                                directions=['maximize', 'minimize', 'minimize'], sampler=SAMPLERS[method](seed))
    while len(done(study)) < n and len(study.trials) < 2 * n:  # failed designs (too few weight bits) don't use budget
        study.optimize(objective, n_trials=1, catch=(Exception,))


def hv(trials):
    L = np.array([[-t.values[0], t.values[1], t.values[2]] for t in trials])
    return compute_hypervolume((np.minimum(L, REF) - IDEAL) / (REF - IDEAL), np.ones(3))


def report():
    for s in optuna.get_all_study_names(DB):
        tr = done(optuna.load_study(study_name=s, storage=DB))
        print(f'\n{s}: {len(tr)} designs, hypervolume after ' +
              '  '.join(f'{i}: {hv(tr[:i]):.4f}' for i in range(5, len(tr) + 1, 5)))
        front = optuna.load_study(study_name=s, storage=DB).best_trials
        for t in sorted(front, key=lambda t: t.values[1]):
            p = t.params
            print(f"  H1 {p['H1']:2d} H2 {p['H2']:2d} WW {p['WW']:2d} TAU {p['TAU']:4.2f} | acc {t.values[0]:6.2f}%  "
                  f"LUTs {t.values[1]:4.0f}  gyro {t.values[2]:5.2f}%  cycles {t.user_attrs['cycles']:6.1f}")


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sys.argv[1] == 'run':
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        report()
