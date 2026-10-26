# Quantum Reservoir Computing for Water-Level Forecasting

Two self-contained reference implementations of quantum reservoir computing (QRC)
for multi-step time-series forecasting. Each script is standalone — no shared
modules, no local imports.

| Script | Model | Reservoir |
|---|---|---|
| `qrc/QRC.py` | **QRC** — pure quantum | quantum reservoir only |
| `qrc/OCQRC.py` | **OCQRC** — coupled | classical ESN reservoir **+** quantum reservoir |

Both scripts share identical algorithms, hyper-parameter search, evaluation and
output format. The only difference is whether the classical ESN reservoir is
enabled (`use_classical`).

The quantum reservoir follows the onion-style classical–quantum design described
in [qBraid/Classical-Quantum-Reservoir-Computing](https://github.com/qBraid/Classical-Quantum-Reservoir-Computing).

---

## Method summary

- **Quantum reservoir** — a fixed (untrained) `EfficientSU2` circuit: `ry` rotations
  with full entanglement. The circuit is never optimised, so there is no barren
  plateau.
- **Non-unitary feedback** — the measurement of the previous step is fed back as a
  rotation angle via `arccos(<Z>)`, which produces the memory and nonlinearity the
  reservoir needs.
- **Observables** — all single-qubit `<Z_i>` plus all two-qubit `<Z_i Z_j>`
  correlators, giving `n(n+1)/2` features from `n` qubits.
- **Onion layout** — `n_qrc_layers` quantum reservoirs with different feedback
  strengths run in parallel, covering both short- and long-range memory.
- **Readout** — Ridge regression. Only the linear readout is trained.
- **Multi-step forecasting** — autoregressive: each prediction is fed back as the
  next input.

## Prediction protocol

| Data | Input steps | Predicted steps | Window file |
|---|---:|---:|---|
| Water level (`water`) | 120 | 24 | 144 columns |
| Benchmark (`benchmark`) | 18 | 18 | 36 columns |

Each CSV row is one sample: `[0:input_len]` is the input history and
`[input_len:input_len+pred_len]` is the target. Files use a sliding window with
stride 1.

- Train/test split: chronological, `train_ratio = 0.7` (first 70 % train, last 30 % test).
- Normalisation: min–max on the training split, mapped to `[0, 1]`.
- All reported metrics are computed after de-normalisation.

## Installation

```bash
pip install -r requirements.txt
```

Tested with `numpy 2.4.4`, `pandas 3.0.3`, `scikit-learn 1.8.0`,
`qiskit 2.5.1`, `qiskit-aer 0.17.2` on Python 3.13.

## Usage

```bash
cd qrc

# inspect the execution plan without running anything
python QRC.py --dry-run

# hyper-parameter search + final full run (default mode)
python QRC.py --datasets <name> --only water

# no search: run directly with the given hyper-parameters
python QRC.py --mode final --num-qubits 4 --ridge-param 1e-2

# the coupled model
python OCQRC.py --datasets <name> --only water
```

### Common options

| Option | Description |
|---|---|
| `--mode {tune,final}` | `tune` = staged search then final run; `final` = no search |
| `--only {all,water,benchmark}` | restrict to one protocol group |
| `--datasets a,b` | comma-separated subset |
| `--grid "num_qubits=3,4;ridge_param=1e-4,1e-2"` | custom search space |
| `--stages ridge,b` | restrict which search stages run |
| `--metric {Avg_MSE,Avg_SMAPE}` | selection metric for the search |
| `--dry-run` | print the plan only |
| `--num-qubits`, `--n-qrc-layers`, `--crc-size`, `--b`, `--f-b`, `--ridge-param` | fixed hyper-parameters |
| `--sweep-max-train`, `--sweep-max-test`, `--final-max-train`, `--final-max-test` | sampling budgets |

Hyper-parameter search is a staged coordinate descent over
`ridge_param → b → f_b → crc_size → num_qubits → n_qrc_layers`. Each stage selects
its best value and writes it back before the next stage starts.

### Datasets

Dataset names are **not** hard-coded in the scripts. Pass them with `--datasets`
and each name is resolved automatically from the filesystem:

| Name given | Resolved file | Protocol |
|---|---|---|
| `<name>` | `data/<name>.csv` | `water` (120 → 24) |
| `<name>` | `data/benchmark_18_18/<name>.csv` | `benchmark` (18 → 18) |

```bash
python QRC.py --datasets my_data --only water
python QRC.py --datasets series_a,series_b --only benchmark
```

If the same name exists under both locations, use `--only water` /
`--only benchmark` to disambiguate. Names may also be listed in
`WATER_STATIONS` / `BENCHMARKS` at the top of each script, but that is optional
and only affects runs that do not pass `--datasets`.

## Output

Each *model × dataset* pair is written to `outputs/<MODEL>/<dataset>/`:

| File | Content |
|---|---|
| `metrics.csv` | per-step MSE and SMAPE (`step, MSE, SMAPE`) |
| `metrics_full.json` | overall RMSE, NRMSE, R², MAE |
| `artifacts.npz` | ground truth and predictions |
| `meta.json` | model, protocol, hyper-parameters, normalisation |
| `timing.json` | wall-clock timings |
| `best_params.json`, `grid_search_results.csv` | search results (in `tune` mode) |

Summary tables, rebuilt by scanning all per-dataset `metrics.csv`:

| Model | Summary file |
|---|---|
| `QRC.py` | `outputs/analysis_results.csv` |
| `OCQRC.py` | `outputs/analysis_results_OCQRC.csv` |

## Known limitations

- **Recursive divergence.** Autoregressive feedback can blow up for a few starting
  windows, producing very large errors for those samples. This is a general
  reservoir-computing failure mode and affects both models. Clipping each feedback
  prediction to the training range `[min, max]` usually fixes it.
- **`crc_size` is inactive for the pure quantum model.** `QRC.py` does not use the
  ESN reservoir, so every `crc_size` trial gives the same result. The `crc_size`
  search stage can be removed from `STAGES` to save roughly 4× search time.
