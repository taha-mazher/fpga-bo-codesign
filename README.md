# Bayesian optimisation with Vivado in the loop

Hardware and algorithm co-design of a small FPGA activity classifier. A Gaussian-process optimiser picks the next design to build; every candidate is trained, quantised, checked bit-exact against a Python integer model, then synthesised, placed and routed in Vivado before it is scored.

The design is a two-stage, sensor-gated classifier for human activity recognition (UCI HAPT data). A small network on the accelerometer answers every window and switches on a gyroscope only when it is unsure between sitting and standing; a second network then uses both sensors. The hand-tuned version of this cascade is the baseline here.

## Design space and objectives

| Knob | Range | What it changes |
|---|---|---|
| `H1` | 2 to 16 | stage-1 hidden neurons |
| `H2` | 2 to 24 | stage-2 hidden neurons |
| `WW` | 6 to 16 bits | weight precision |
| `TAU` | 0 to 4, step 0.25 | gate margin: how unsure stage 1 must be before the gyroscope turns on |

Three objectives: accuracy (maximise), post-route LUTs (minimise), gyroscope on-time (minimise). Accuracy for the optimisers is 3-fold cross-validation over the 21 training subjects; the official test subjects are only used for the final check.

Target: Xilinx Artix-7 (xc7a35t), 50 MHz, Vivado 2024.2. One design takes about 75 s end to end.

## Results

### Optimiser comparison

Three seeds per method, 40 designs each. Hypervolume is the fraction of the accuracy, LUT and gyroscope box (80 to 100%, 0 to 500 LUTs, 0 to 50%) covered by the method's best trade-offs; higher is better.

| Method | after 10 designs | after 20 | after 40 (worst to best seed) |
|---|---|---|---|
| Gaussian-process BO | **0.253** | **0.278** | **0.287** (0.284 to 0.291) |
| Random search | 0.213 | 0.247 | 0.255 (0.252 to 0.259) |
| NSGA-II | 0.213 | 0.230 | 0.252 (0.241 to 0.260) |

Every GP seed beat every random and NSGA-II seed. GP after 10 designs matches random search after 40, so it needs about 4x fewer Vivado builds for the same quality of trade-offs. Full log: `out/bo_report.txt`.

### Chosen designs on the test subjects

Trained with 5 seeds each; accuracy is mean and standard deviation over seeds. Hardware figures are from the Vivado build of seed 0.

| Design | H1-H2, bits, TAU | Test accuracy | LUTs | Gyroscope on | Dynamic energy per window |
|---|---|---|---|---|---|
| Hand-tuned baseline | 6-8, 16, 1.0 | 94.27 ± 0.50% | 358 | 14.8% | 7.1 nJ |
| A | 7-2, 12, 0.5 | 93.98 ± 0.59% | 315 (-12%) | 9.2% (-38%) | 6.4 nJ |
| C | 3-2, 9, 0.25 | 92.80 ± 1.07% | 306 (-15%) | 6.0% (-60%) | 3.2 nJ |

Design A matches the baseline's accuracy within seed noise with 12% fewer LUTs and 38% less gyroscope time. Design C halves the cycles and the dynamic energy for about 1.5 points of accuracy.

### Surrogate accuracy

10-fold cross-validation over all 294 built designs (`surrogate_check.py`):

| Model | Mean error in LUTs | Within 5% | R² |
|---|---|---|---|
| Gaussian process | 4.0 LUTs (1.1%) | 99% | 0.97 |
| Linear regression | 9.7 LUTs (2.7%) | 86% | 0.86 |

## What did not hold up

- **The optimiser's apparent accuracy gain was noise.** On a single training seed the GP found designs 1.5 points above the baseline. Retrained with five seeds, the baseline scores 93.4 ± 0.5% in cross-validation rather than the 92.8% of that one seed, and the gain disappears. The real improvement is in LUTs and gyroscope time at the same accuracy.
- **Dropping the gyroscope does not work.** A wide stage 1 with 6-bit weights and no gyroscope looked best on cross-validation but scored 92.8% on the test subjects, and for one seed in five its weights do not fit in 6 bits at all.
- **The hypervolume comparison uses one training seed per design.** Seed noise is about 0.5 points; scoring each design over several seeds would make the comparison cleaner.

## Reproducing

Requirements: Python 3.11+ with PyTorch, scikit-learn, SciPy, pandas and Optuna 5; Vivado 2024.2 (the free Standard edition covers Artix-7); Icarus Verilog in WSL for the bit-exact check.

1. Download the UCI [Smartphone-Based Recognition of Human Activities and Postural Transitions](https://archive.ics.uci.edu/dataset/341) dataset and point `HAPT_DIR` at the folder that contains `RawData/` and `Test/` (default: `data/HAPT`).
2. One design: `python dse.py H1 H2 WW TAU [val|test]`, for example `python dse.py 6 8 16 1 test` for the baseline. Each run appends a row to `out/dse_results.csv` and reuses it if the design was built before.
3. The full search: `.\run_bo.ps1` (PowerShell; 9 studies, about 4 hours), then `python bo.py report`.
4. Checks: `python noise_check.py`, `python surrogate_check.py`.

On Windows with Anaconda, set `KMP_DUPLICATE_LIB_OK=TRUE` before running the optimiser (Anaconda's MKL and PyTorch each ship an OpenMP runtime). Set `VIVADO` if Vivado is not at `C:\Xilinx\Vivado\2024.2\bin\vivado.bat`.

## Files

| File | Purpose |
|---|---|
| `rtl/cascade.v` | the classifier: one multiply-accumulate engine for both stages, all knobs as parameters |
| `rtl/tb_cascade.v`, `rtl/run_cascade_sim.sh` | bit-exact testbench against the Python integer model |
| `vivado/build_cascade.tcl` | synthesis, implementation and reports for one design |
| `cascade_hw.py` | training output to fixed-point weights, test vectors and RTL parameters |
| `dse.py` | evaluates one design end to end |
| `bo.py`, `run_bo.ps1` | the optimiser studies and the overnight run |
| `noise_check.py`, `surrogate_check.py` | the checks above |
| `out/` | every built design, the optimiser database and the check outputs |
