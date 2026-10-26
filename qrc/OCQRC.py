"""OCQRC - coupled classical-quantum reservoir computing for multi-step water-level forecasting.

Protocol
  water      120 inputs -> 24 predictions  (144-column windows)
  benchmark   18 inputs -> 18 predictions  ( 36-column windows)

Modes
  tune    staged hyper-parameter search, then final full run (default)
  final   no search, use the configured hyper-parameters

Usage
  python OCQRC.py --dry-run
  python OCQRC.py --datasets <name> --only water
  python OCQRC.py --mode final --num-qubits 4 --ridge-param 1e-2
"""


from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from copy import deepcopy
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.parameter import Parameter
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.backends import AerSimulator

# ---------------------------------------------------------------- metrics & output


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / (denom + 1e-12)) * 100.0)


def evaluate_steps(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-step MSE and SMAPE."""
    rows = []
    for step in range(y_true.shape[1]):
        yt, yp = y_true[:, step], y_pred[:, step]
        rows.append({
            "step": f"t+{step + 1}",
            "MSE": mean_squared_error(yt, yp),
            "SMAPE": smape(yt, yp),
        })
    return pd.DataFrame(rows)


def save_results(output_dir: Path, model_name: str, y_true: np.ndarray,
                 y_pred: np.ndarray, meta: dict | None = None) -> pd.DataFrame:
    """Write metrics.csv, artifacts.npz and meta.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_steps(y_true, y_pred)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    payload = {"model_name": model_name, "pred_len": int(y_true.shape[1])}
    if meta:
        payload.update(meta)

    np.savez_compressed(output_dir / "artifacts.npz", y_true=y_true, y_pred=y_pred)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{model_name} evaluation:")
    print(metrics.to_string(index=False, formatters={"MSE": "{:.6f}".format,
                                                     "SMAPE": "{:.4f}".format}))
    return metrics


warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUTPUTS = os.path.join(ROOT, "outputs")
MODEL = "OCQRC"
SWEEP_ROOT = os.path.join(OUTPUTS, "sweep_OCQRC")

# ============================================================================
# ============================================================================
RUN_WATER = True
RUN_BENCHMARK = True

TUNE_WATER = None
TUNE_BENCH = None

SWEEP_MAX_TRAIN_WATER, SWEEP_MAX_TEST_WATER = 1000, 200
SWEEP_MAX_TRAIN_BENCH, SWEEP_MAX_TEST_BENCH = 1000, 200
FINAL_MAX_TRAIN_WATER, FINAL_MAX_TEST_WATER = 3000, 800
FINAL_MAX_TRAIN_BENCH, FINAL_MAX_TEST_BENCH = 3000, 800

METRIC = "Avg_MSE"
TRAIN_RATIO = 0.7
SCALE = 1.0
# ============================================================================

# ---------------------------------------------------------------- dataset registry
#
# Dataset names are not hard-coded here. Pass them on the command line:
#
#   python OCQRC.py --datasets my_dataset --only water
#
# A name given via --datasets is resolved automatically:
#   <name> -> data/<name>.csv                  protocol "water"      (120 -> 24)
#   <name> -> data/benchmark_18_18/<name>.csv   protocol "benchmark"  ( 18 -> 18)
#
# Names may also be listed explicitly below, e.g.
#   WATER_STATIONS = ["station_A", "station_B"]
#   BENCHMARKS     = ["benchmark_A"]
# but this is optional and only affects runs that do not pass --datasets.
WATER_STATIONS: list[str] = []
BENCHMARKS: list[str] = []

DATA = os.path.join(ROOT, "data")
BENCH = os.path.join(DATA, "benchmark_18_18")
DATASETS = (
    [(n, os.path.join(DATA, f"{n}.csv"), "water") for n in WATER_STATIONS]
    + [(n, os.path.join(BENCH, f"{n}.csv"), "benchmark") for n in BENCHMARKS]
)
PROTOCOLS = {"water": dict(ctx=120, pred=24), "benchmark": dict(ctx=18, pred=18)}


BASE_CFG = {
    "num_qubits": 4, "n_qrc_layers": 1, "crc_size": 850,
    "b": -0.33, "f_b": 0.11, "ridge_param": 3e-4,
}
INT_PARAMS = {"num_qubits", "n_qrc_layers", "crc_size"}
PARAM_ORDER = ["num_qubits", "n_qrc_layers", "crc_size", "b", "f_b", "ridge_param"]

STAGES = [
    ("ridge", "ridge_param", [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1]),
    ("b", "b", [-0.60, -0.45, -0.33, -0.22, -0.12]),
    ("f_b", "f_b", [0.05, 0.08, 0.11, 0.16, 0.25]),
    ("crc_size", "crc_size", [200, 400, 850, 1200]),
    ("num_qubits", "num_qubits", [3, 4, 5, 6]),
    ("n_qrc_layers", "n_qrc_layers", [1, 2, 3]),
]

RESULT_COLUMNS = [
    "run_id", "timestamp", "dataset", "protocol", "stage",
    "num_qubits", "n_qrc_layers", "crc_size", "b", "f_b", "ridge_param",
    "max_train", "max_test", "n_train_used", "n_test_used",
    "Avg_MSE", "Avg_SMAPE", "RMSE", "NRMSE", "R2", "MAE",
    "train_reservoir_sec", "train_fit_sec", "inference_sec", "wall_sec",
]


def _fmt_hms(seconds):
    seconds = int(round(seconds))
    seconds = max(seconds, 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class _ProgressBar:
    def __init__(self, total, desc="progress", width=24):
        self.total = int(total)
        self.desc = desc
        self.width = width
        self.start = time.perf_counter()
        self.n = 0
        self._last_print = 0.0
        self._finished = False

    def update(self, k=1):
        if self._finished or self.total <= 0:
            return
        self.n = min(self.n + k, self.total)
        now = time.perf_counter()
        if self.n < self.total and (now - self._last_print) < 0.1:
            return
        self._last_print = now
        self._render(now)

    def _render(self, now):
        elapsed = now - self.start
        frac = self.n / self.total
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        if frac > 0:
            est_total = elapsed / frac
            eta = est_total - elapsed
        else:
            est_total = 0.0
            eta = 0.0
        end_ts = datetime.fromtimestamp(now + eta).strftime("%H:%M:%S")
        line = (f"\r{self.desc} [{bar}] {self.n}/{self.total} "
                f"{frac * 100:5.1f}%  elapsed {_fmt_hms(elapsed)}  "
                f"ETA {_fmt_hms(eta)}  est {_fmt_hms(est_total)}  end {end_ts}")
        print(line, end="", flush=True)
        if self.n >= self.total:
            self._finished = True
            print()

    def close(self):
        if not self._finished and self.total > 0:
            self.n = self.total
            self._render(time.perf_counter())
        elif self.total <= 0:
            self._finished = True


def qrc_circuit(num_qubits: int) -> QuantumCircuit:
    phicirc = EfficientSU2(
        num_qubits=num_qubits, su2_gates=["ry"], skip_final_rotation_layer=True,
        entanglement="full", reps=1, parameter_prefix="phi",
    )
    vcirc1 = EfficientSU2(
        num_qubits=num_qubits, su2_gates=["ry"], skip_final_rotation_layer=False,
        entanglement="full", reps=1, parameter_prefix="v1",
    )
    vcirc2 = EfficientSU2(
        num_qubits=num_qubits, su2_gates=["ry"], skip_final_rotation_layer=False,
        entanglement="full", reps=1, parameter_prefix="v2",
    )
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc = qc.compose(phicirc)
    qc = qc.compose(vcirc1)
    qc = qc.compose(vcirc2)

    f_b = Parameter("f_b b x")
    zs = [Parameter(f"b y_{i}") for i in range(num_qubits)]
    vars_ = [f_b] + zs[:-1] + zs * 4
    qrc = qc.assign_parameters(vars_)
    return qrc


class OCQRC_Model:
    def __init__(
        self,
        num_qubits: int = 4,
        crc_size: int = 850,
        use_classical: bool = True,
        use_quantum: bool = True,
        backends=None,
        b: float = -0.33,
        f_bs: list = None,
        ridge_param: float = 3e-4,
        ctx_len: int = 120,
        pred_len: int = 24,
    ):
        self.num_qubits = num_qubits
        self.crc_size = crc_size
        self.observables = self.observables_qrc_circuit()
        self.circuit = self.full_circuit()

        self.b = b
        self.f_bs = f_bs if f_bs is not None else [0.1]

        self.context_len = ctx_len
        self.pred_len = pred_len
        self.ridge_param = ridge_param
        self.ridge = None
        self.train_X = None
        self.train_Y = None

        self.train_results = []
        self.train_outputs = []
        self.train_reservoir_sec = 0.0
        self.train_fit_sec = 0.0

        self.backends = backends
        self.pms = []
        self.pmc = []
        for backend in self.backends:
            self.pms.append(generate_preset_pass_manager(1, backend=backend))
            self.pmc.append(self.pms[-1].run(self.circuit))
        self.last_output = [[0] for _ in range(len(self.backends))]

        self.use_classical = use_classical
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.init_qrc()
        else:
            self.initial_exp_values = None
        if self.use_classical:
            self.init_crc()
        else:
            self.init_xinit = None

    def init_crc(self):
        self.inSize = 1
        self.outSize = 1
        self.resSize = self.crc_size
        np.random.seed(2)
        self.Win = (np.random.rand(self.resSize, 1 + self.inSize) - 0.5) * 1
        self.W = np.random.rand(self.resSize, self.resSize) - 0.5
        rhoW = max(abs(np.linalg.eig(self.W)[0]))
        self.W *= 1.00 / rhoW
        self.init_xinit = np.tanh(np.ones((self.resSize, 1)) / np.sqrt(self.resSize))
        self.x = self.init_xinit.copy()

    def evolve_crc(self, t0):
        if self.use_classical:
            u = t0 / 3.33
            self.x = np.tanh(np.dot(self.Win, np.vstack((1, u))) + np.dot(self.W, self.x))
            return list(np.vstack((1, u, self.x)).flatten())
        return []

    def observables_qrc_circuit(self):
        observables = []
        for i in range(0, self.num_qubits):
            op = ["I"] * self.num_qubits
            op[self.num_qubits - 1 - i] = "Z"
            observables.append(SparsePauliOp(Pauli("".join(op))))
        for i, j in combinations(range(self.num_qubits), r=2):
            op = ["I"] * self.num_qubits
            op[self.num_qubits - 1 - i] = "Z"
            op[self.num_qubits - 1 - j] = "Z"
            observables.append(SparsePauliOp(Pauli("".join(op))))

        self._int_observables = np.zeros(len(observables), dtype=int)
        for o, obs in enumerate(observables):
            pstring = obs.paulis[0].to_instruction().params[0]
            birep = int("".join(["0" if p != "Z" else "1" for p in pstring]), base=2)
            self._int_observables[o] = birep
        return observables

    def full_circuit(self):
        return qrc_circuit(self.num_qubits)

    def calc_observables(self, samples):
        exps = np.zeros(len(self.observables), dtype=float)
        for i, r in enumerate(samples):
            for o, obs in enumerate(self._int_observables):
                exps[o] += (-1) ** bin(obs & i).count("1") * r
        return exps

    def init_qrc(self):
        self.last_output = [list(np.ones(self.num_qubits) * 0.5) for _ in range(len(self.backends))]
        self.initial_exp_values = deepcopy(self.last_output)

    def evolve_qrc(self, t0):
        obsvs = []
        if self.use_quantum:
            for c in range(len(self.backends)):
                dt, dtu = self.b, self.f_bs[c]
                f_b = t0 * dt * dtu
                zs = self.last_output[c]
                params = [list(np.array(np.arccos(zs[:])) * dt) + [f_b]]
                rcirc = self.pmc[c].assign_parameters(params[0])
                rcirc.save_statevector()
                state = self.backends[c].run([rcirc]).result().get_statevector().data
                probs = (np.conj(state) * state).real
                obsvs_new = self.calc_observables(probs)
                self.last_output[c] = obsvs_new[: self.num_qubits]
                obsvs += list(obsvs_new)
        return list(np.array(obsvs))

    def combined(self, t0):
        return self.evolve_qrc(t0) + self.evolve_crc(t0)

    def _evolve_seq(self, seq):
        """Reset the reservoir, feed seq scalar by scalar, return final state.

        warmup: seq[0] is fed 3 times, the remaining values once."""
        seq = list(seq)
        self.last_output = deepcopy(self.initial_exp_values)
        self.x = deepcopy(self.init_xinit)
        if len(seq) == 0:
            return np.array(self.combined(0.0), dtype=float)
        _ = self.combined(seq[0])
        _ = self.combined(seq[0])
        state = self.combined(seq[0])
        for v in seq[1:]:
            state = self.combined(v)
        return np.array(state, dtype=float)

    def train(self, x):
        ctx_len, pred_len = self.context_len, self.pred_len
        batch_size = x.shape[0]
        X_all, Y_all = [], []
        t_res = time.perf_counter()
        pbar = _ProgressBar(batch_size, desc="[train]")
        for b in range(batch_size):
            context = x[b, :ctx_len]
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            state = self._evolve_seq(context)
            X_all.append(np.array(state, dtype=float))
            Y_all.append(x[b, ctx_len])
            pbar.update(1)
        pbar.close()
        self.train_X = np.array(X_all)
        self.train_Y = np.array(Y_all).reshape(-1, 1)
        self.train_reservoir_sec = time.perf_counter() - t_res
        t_fit = time.perf_counter()
        self.fit()
        self.train_fit_sec = time.perf_counter() - t_fit

    def fit(self, ridge_param=None):
        """Fit the Ridge readout (state -> next value)."""
        rp = self.ridge_param if ridge_param is None else ridge_param
        self.ridge = Ridge(rp)
        self.ridge.fit(self.train_X, self.train_Y)

    def forward(self, x, num_predict):
        pred_len = self.pred_len
        batch_size = x.shape[0]
        all_outputs = np.zeros((batch_size, pred_len))
        pbar = _ProgressBar(batch_size, desc="[test ]")
        for b in range(batch_size):
            context = x[b, :]
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            state = self._evolve_seq(context)
            p = float(self.ridge.predict(np.array(state, dtype=float).reshape(1, -1))[0])
            all_outputs[b, 0] = p
            for k in range(1, pred_len):
                state = self.combined(p)
                p = float(self.ridge.predict(np.array(state, dtype=float).reshape(1, -1))[0])
                all_outputs[b, k] = p
            pbar.update(1)
        pbar.close()
        return all_outputs


def load_csv_windows(path: str) -> np.ndarray:
    p = path.replace("/", os.sep)
    data = np.loadtxt(p, delimiter=",", ndmin=2)
    return data.astype(np.float64)


def split_indices(n: int, train_ratio: float):
    n_train = max(1, int(round(n * train_ratio)))
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:]


def prepare_io(path: str, ctx_len: int, pred_len: int,
               train_ratio: float = 0.7, stride: int = 1, scale: float = 1.0):
    raw = load_csv_windows(path)
    tr_idx, te_idx = split_indices(len(raw), train_ratio)
    mn = float(raw[tr_idx].min())
    mx = float(raw[tr_idx].max())
    train_raw = raw[tr_idx]
    test_raw = raw[te_idx]
    if stride > 1:
        train_raw = train_raw[::stride]
        test_raw = test_raw[::stride]

    def norm(arr):
        return (arr - mn) / (mx - mn + 1e-12) * scale

    return {
        "train_windows": norm(train_raw),
        "test_context": norm(test_raw)[:, :ctx_len],
        "test_target": norm(test_raw)[:, ctx_len: ctx_len + pred_len],
        "stats": (mn, mx, scale),
        "n_train_used": len(train_raw),
        "n_test": len(test_raw),
    }


def denormalize(arr: np.ndarray, mn: float, mx: float, scale: float = 1.0) -> np.ndarray:
    return arr / (scale + 1e-12) * (mx - mn) + mn


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(pred).reshape(target.shape)
    mse = float(np.mean((pred - target) ** 2))
    rmse = float(np.sqrt(mse))
    std = float(target.std())
    nrmse = rmse / (std + 1e-12)
    r2 = float(r2_score(target.flatten(), pred.flatten()))
    mae = float(np.mean(np.abs(pred - target)))
    return {"RMSE": rmse, "NRMSE": nrmse, "R2": r2, "MAE": mae}


def _score(preds, target) -> dict:
    step_df = evaluate_steps(target, preds)
    out = {
        "Avg_MSE": float(step_df["MSE"].mean()),
        "Avg_SMAPE": float(step_df["SMAPE"].mean()),
    }
    out.update({k: float(v) for k, v in compute_metrics(preds, target).items()})
    return out


def build_model(params: dict, proto: str) -> OCQRC_Model:
    n_q = max(1, int(params["n_qrc_layers"]))
    backends = [AerSimulator(method="statevector", device="CPU") for _ in range(n_q)]
    f_bs = [params["f_b"] * (1.0 + 0.25 * i) for i in range(n_q)]
    proto_cfg = PROTOCOLS[proto]
    return OCQRC_Model(
        num_qubits=int(params["num_qubits"]),
        crc_size=int(params["crc_size"]),
        use_classical=True,
        use_quantum=True,
        backends=backends,
        b=params["b"],
        f_bs=f_bs,
        ridge_param=params["ridge_param"],
        ctx_len=proto_cfg["ctx"],
        pred_len=proto_cfg["pred"],
    )


def run_model(data: dict, params: dict) -> dict:
    """Train and evaluate one dataset."""
    ctx, pred = data["ctx"], data["pred"]
    model = build_model(params, data["proto"])
    train_windows = data["train_windows"]
    n_train_used = len(train_windows)

    t_train = time.perf_counter()
    model.train(train_windows)
    train_wall = time.perf_counter() - t_train

    test_context, test_target = data["test_context"], data["test_target"]
    n_test_used = len(test_context)
    t_inf = time.perf_counter()
    preds_norm = model.forward(test_context, pred)
    inference_sec = time.perf_counter() - t_inf

    mn, mx, scale = data["stats"]
    preds = denormalize(preds_norm, mn, mx, scale)
    target = denormalize(test_target, mn, mx, scale)

    train_total = model.train_reservoir_sec + model.train_fit_sec
    timing = {
        "training_total_sec": round(train_total, 4),
        "train_reservoir_sec": round(model.train_reservoir_sec, 4),
        "train_readout_fit_sec": round(model.train_fit_sec, 4),
        "train_wall_sec": round(train_wall, 4),
        "inference_sec": round(inference_sec, 4),
        "n_train_windows": n_train_used,
        "n_test_windows": n_test_used,
        "train_windows_per_sec": round(n_train_used / train_total, 3) if train_total > 0 else None,
    }
    return {"preds": preds, "target": target, "timing": timing,
            "n_train_used": n_train_used, "n_test_used": n_test_used}


def _capped(data: dict, budget: dict) -> dict:
    """Data view truncated by the sampling budget."""
    d = dict(data)
    if budget.get("max_train"):
        d["train_windows"] = d["train_windows"][: budget["max_train"]]
    if budget.get("max_test"):
        d["test_context"] = d["test_context"][: budget["max_test"]]
        d["test_target"] = d["test_target"][: budget["max_test"]]
    return d


def eval_full(params: dict, budget: dict, data: dict) -> dict:
    res = run_model(_capped(data, budget), params)
    row = dict(params)
    row.update(_score(res["preds"], res["target"]))
    row.update({
        "n_train_used": res["n_train_used"],
        "n_test_used": res["n_test_used"],
        "train_reservoir_sec": res["timing"]["train_reservoir_sec"],
        "train_fit_sec": res["timing"]["train_readout_fit_sec"],
        "inference_sec": res["timing"]["inference_sec"],
        "wall_sec": res["timing"]["training_total_sec"] + res["timing"]["inference_sec"],
    })
    return row


def eval_ridge_group(params: dict, budget: dict, data: dict, values: list):
    """The reservoir is evolved once; only the readout is refit per ridge value."""
    d = _capped(data, budget)
    model = build_model(params, d["proto"])
    train_windows = d["train_windows"]
    t0 = time.perf_counter()
    model.train(train_windows)
    shared_train_sec = time.perf_counter() - t0
    res_sec = model.train_reservoir_sec

    test_ctx, test_tgt = d["test_context"], d["test_target"]
    mn, mx, scale = d["stats"]
    pred_len = d["pred"]

    for i, alpha in enumerate(values):
        w0 = time.perf_counter()
        tf = time.perf_counter()
        model.fit(alpha)
        fit_sec = time.perf_counter() - tf
        ti = time.perf_counter()
        preds_norm = model.forward(test_ctx, pred_len)
        inf_sec = time.perf_counter() - ti

        preds = denormalize(preds_norm, mn, mx, scale)
        target = denormalize(test_tgt, mn, mx, scale)

        row = dict(params)
        row["ridge_param"] = alpha
        row.update(_score(preds, target))
        row.update({
            "n_train_used": len(train_windows),
            "n_test_used": len(test_ctx),
            "train_reservoir_sec": round(res_sec if i == 0 else 0.0, 4),
            "train_fit_sec": round(fit_sec, 4),
            "inference_sec": round(inf_sec, 4),
            "wall_sec": round((shared_train_sec if i == 0 else 0.0)
                              + time.perf_counter() - w0, 3),
        })
        yield row


class ResultStore:
    """Append search results to disk (supports resume)."""

    def __init__(self, path: str, dataset: str, protocol: str, metric: str):
        self.path = path
        self.dataset = dataset
        self.protocol = protocol
        self.metric = metric
        self.rows: list[dict] = []
        self.done: set[str] = set()
        self._n = 0
        if os.path.isfile(path):
            try:
                old = pd.read_csv(path)
                for _, r in old.iterrows():
                    d = r.to_dict()
                    self.rows.append(d)
                    cfg = {k: d[k] for k in BASE_CFG}
                    for k in INT_PARAMS:
                        cfg[k] = int(cfg[k])
                    bud = {"max_train": int(d["max_train"]), "max_test": int(d["max_test"])}
                    self.done.add(_key(cfg, bud))
                self._n = len(self.rows)
                print(f"[OCQRC] resume: {self._n} previous result(s) from {path}")
            except Exception as e:
                print(f"[OCQRC] WARNING: could not read {path} ({e}); starting fresh")

    def has(self, cfg, budget):
        return _key(cfg, budget) in self.done

    def add(self, row: dict, budget: dict, stage: str):
        self._n += 1
        full = {c: None for c in RESULT_COLUMNS}
        full.update(row)
        full.update({
            "run_id": self._n,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": self.dataset,
            "protocol": self.protocol,
            "stage": stage,
            "max_train": budget["max_train"],
            "max_test": budget["max_test"],
        })
        self.rows.append(full)
        cfg = {k: row[k] for k in BASE_CFG}
        self.done.add(_key(cfg, budget))
        self.flush()
        return full

    def flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        pd.DataFrame(self.rows, columns=RESULT_COLUMNS).to_csv(self.path, index=False)

    def best(self):
        if not self.rows:
            return None
        return min(self.rows, key=lambda r: float(r[self.metric]))

    def lookup(self, cfg, budget):
        key = _key(cfg, budget)
        for r in self.rows:
            c = {k: r[k] for k in BASE_CFG}
            for k in INT_PARAMS:
                c[k] = int(c[k])
            b = {"max_train": int(r["max_train"]), "max_test": int(r["max_test"])}
            if _key(c, b) == key:
                return r
        return None


def _key(cfg: dict, budget: dict) -> str:
    parts = [f"{k}={cfg[k]!r}" for k in sorted(cfg)]
    parts += [f"{k}={budget[k]!r}" for k in sorted(budget)]
    return "|".join(parts)


def _parse_grid(spec: str) -> dict:
    out = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, vals = chunk.partition("=")
        name = name.strip()
        if name not in BASE_CFG:
            raise SystemExit(f"[OCQRC] unknown parameter in grid: {name}. "
                             f"Valid: {sorted(BASE_CFG)}")
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            if not v:
                continue
            parsed.append(int(float(v)) if name in INT_PARAMS else float(v))
        if not parsed:
            raise SystemExit(f"[OCQRC] no values for {name}")
        out[name] = parsed
    if not out:
        raise SystemExit("[OCQRC] empty grid")
    return out


def run_search(mode: str, grid_spec: str, stages_arg: str, no_fast_ridge: bool,
               metric: str, data: dict, budget: dict, store: ResultStore,
               cfg_base: dict | None = None) -> tuple:
    """Run the search; return (winning params, store)."""
    cfg = dict(cfg_base or BASE_CFG)
    t_sweep = time.perf_counter()
    counter = {"done": 0}

    def _report(row, stage, total_hint=None):
        counter["done"] += 1
        best = store.best()
        elapsed = time.perf_counter() - t_sweep
        print(f"[OCQRC] #{counter['done']}" + (f"/{total_hint}" if total_hint else "")
              + f" stage={stage} " + " ".join(f"{k}={row[k]}" for k in PARAM_ORDER)
              + f" -> MSE={row['Avg_MSE']:.6f} SMAPE={row['Avg_SMAPE']:.4f}"
              + f"  [{_fmt_hms(elapsed)} elapsed]")
        if best is not None:
            print(f"      best so far: {metric}={float(best[metric]):.6f}  "
                  + " ".join(f"{k}={best[k]}" for k in PARAM_ORDER))

    if mode == "grid":
        space = _parse_grid(grid_spec)
        names = list(space)
        plan = []
        for combo in product(*(space[n] for n in names)):
            trial = dict(cfg)
            trial.update(dict(zip(names, combo)))
            plan.append(trial)
        total = len(plan)
        for trial in plan:
            if store.has(trial, budget):
                print("[OCQRC] skip (already done): " + " ".join(f"{k}={trial[k]}" for k in PARAM_ORDER))
                continue
            full = store.add(eval_full(trial, budget, data), budget, "grid")
            _report(full, "grid", total)
        best = store.best()
        win = {k: float(best[k]) for k in PARAM_ORDER}
        for k in INT_PARAMS:
            win[k] = int(win[k])
        return win, store

    if stages_arg:
        wanted = {s.strip() for s in stages_arg.split(",") if s.strip()}
        unknown = wanted - {s[0] for s in STAGES}
        if unknown:
            raise SystemExit(f"[OCQRC] unknown stage(s): {sorted(unknown)}. Valid: {[s[0] for s in STAGES]}")
        stage_list = [s for s in STAGES if s[0] in wanted]
    else:
        stage_list = list(STAGES)

    for name, param, values in stage_list:
        print(f"\n=== stage '{name}': tuning {param} over {values} ===")
        use_fast = (param == "ridge_param") and not no_fast_ridge
        pending = [v for v in values
                   if not (store.has({**cfg, param: v}, budget))]
        if use_fast and pending:
            for row in eval_ridge_group(cfg, budget, data, pending):
                full = store.add(row, budget, name)
                _report(full, name)
        else:
            for v in pending:
                trial = dict(cfg)
                trial[param] = v
                full = store.add(eval_full(trial, budget, data), budget, name)
                _report(full, name)

        best_v, best_score = cfg[param], float("inf")
        for v in values:
            rec = store.lookup({**cfg, param: v}, budget)
            if rec is None:
                continue
            s = float(rec[metric])
            if s < best_score:
                best_score, best_v = s, v
        if best_v != cfg[param]:
            print(f"[OCQRC] stage '{name}': {param} {cfg[param]} -> {best_v} ({metric}={best_score:.6f})")
        else:
            print(f"[OCQRC] stage '{name}': keep {param}={cfg[param]} ({metric}={best_score:.6f})")
        cfg[param] = best_v
    return cfg, store


def write_tuning_artefacts(name: str, proto: str, best_cfg: dict, best_scores: dict) -> None:
    ds_dir = os.path.join(OUTPUTS, MODEL, name)
    os.makedirs(ds_dir, exist_ok=True)
    pred_len = PROTOCOLS[proto]["pred"]
    best_meta = {
        "model": MODEL,
        "protocol": proto,
        "input_len": PROTOCOLS[proto]["ctx"],
        "pred_len": pred_len,
        "selection_metric": f"avg_{pred_len}step_mse",
        "selection_score": float(best_scores.get("Avg_MSE", "nan")),
        "best_avg_mse": float(best_scores.get("Avg_MSE", "nan")),
        "best_avg_smape": float(best_scores.get("Avg_SMAPE", "nan")),
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
        **{k: best_cfg[k] for k in PARAM_ORDER},
    }
    with open(os.path.join(ds_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_meta, f, ensure_ascii=False, indent=2)
    sr = os.path.join(SWEEP_ROOT, name, "sweep_results.csv")
    if os.path.isfile(sr):
        import shutil
        shutil.copyfile(sr, os.path.join(ds_dir, "grid_search_results.csv"))
    print(f"[OCQRC]    best_params.json + grid_search_results.csv -> {ds_dir}")


def finalize_outputs(name: str, proto: str, params: dict, res: dict,
                     pred_len: int) -> None:
    """Save all outputs for one dataset."""
    ds_dir = os.path.join(OUTPUTS, MODEL, name)
    os.makedirs(ds_dir, exist_ok=True)
    meta = {
        "model": MODEL,
        "mode": "coupled (classical ESN + quantum reservoir)",
        "train_ratio": TRAIN_RATIO,
        "protocol": proto,
        "input_len": PROTOCOLS[proto]["ctx"],
        "pred_len": pred_len,
        **params,
        "normalize": f"min-max by train to [0,{SCALE}] (min-max on training split)",
        "timing": res["timing"],
        "best_params": params,
        "selection_metric": f"avg_{pred_len}step_mse",
    }
    save_results(Path(ds_dir), MODEL, res["target"], res["preds"], meta=meta)
    full = compute_metrics(res["preds"], res["target"])
    with open(os.path.join(ds_dir, "metrics_full.json"), "w") as f:
        json.dump(full, f, indent=2)
    import platform as _platform
    import qiskit as _qiskit
    timing_report = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": _platform.python_version(),
            "qiskit": _qiskit.__version__,
            "backend": "AerSimulator(statevector, CPU)",
            "platform": _platform.platform(),
        },
        "timing": res["timing"],
    }
    with open(os.path.join(ds_dir, "timing.json"), "w") as f:
        json.dump(timing_report, f, indent=2)
    print("  full metrics:", {k: round(float(v), 5) for k, v in full.items()})
    print(f"  training: total={res['timing']['training_total_sec']}s "
          f"(reservoir={res['timing']['train_reservoir_sec']}s, "
          f"readout_fit={res['timing']['train_readout_fit_sec']}s), "
          f"inference={res['timing']['inference_sec']}s")


def rebuild_summary() -> pd.DataFrame | None:
    """Rebuild the summary CSV from all per-dataset metrics.csv."""
    rows = []
    model_root = os.path.join(OUTPUTS, MODEL)
    if not os.path.isdir(model_root):
        return None
    for name in sorted(os.listdir(model_root)):
        mpath = os.path.join(model_root, name, "metrics.csv")
        if not os.path.isfile(mpath):
            continue
        df = pd.read_csv(mpath)
        rows.append({
            "Dataset": name,
            "Model": MODEL,
            "Avg_MSE": float(df["MSE"].mean()),
            "Avg_SMAPE": float(df["SMAPE"].mean()),
        })
    if not rows:
        return None
    summ = pd.DataFrame(rows)
    summ["MSE_Rank"] = summ.groupby("Dataset")["Avg_MSE"].rank(method="min").astype(int)
    summ["SMAPE_Rank"] = summ.groupby("Dataset")["Avg_SMAPE"].rank(method="min").astype(int)
    os.makedirs(OUTPUTS, exist_ok=True)
    summ.to_csv(os.path.join(OUTPUTS, "analysis_results_OCQRC.csv"), index=False)
    return summ


def _lookup_dataset(name: str, prefer: str | None = None):
    """Map a bare name to (name, path, protocol) by locating its CSV file."""
    candidates = [
        (os.path.join(DATA, f"{name}.csv"), "water"),
        (os.path.join(BENCH, f"{name}.csv"), "benchmark"),
    ]
    if prefer == "benchmark":
        candidates.reverse()
    for path, proto in candidates:
        if os.path.isfile(path):
            return (name, path, proto)
    return None


def resolve_run_list(cli_datasets: str | None, cli_only: str | None) -> list:
    if cli_datasets:
        known = {d[0]: d for d in DATASETS}
        prefer = cli_only if cli_only in ("water", "benchmark") else None
        selected, missing = [], []
        for name in (s.strip() for s in cli_datasets.split(",") if s.strip()):
            hit = known.get(name) or _lookup_dataset(name, prefer)
            if hit is None:
                missing.append(name)
            else:
                selected.append(hit)
        if missing:
            print(f"[OCQRC] not found under data/ or data/benchmark_18_18/: {sorted(missing)}")
        return selected
    if cli_only and cli_only != "all":
        return [d for d in DATASETS if d[2] == cli_only]

    def matches(cfg, name):
        return cfg is True or (isinstance(cfg, (list, tuple, set)) and name in cfg)

    return [d for d in DATASETS
            if matches(RUN_WATER if d[2] == "water" else RUN_BENCHMARK, d[0])]


def _tune_spec(proto: str) -> str | None:
    return TUNE_WATER if proto == "water" else TUNE_BENCH


def _sweep_budget(proto: str, cli_train=None, cli_test=None) -> dict:
    if proto == "water":
        mt, ms = SWEEP_MAX_TRAIN_WATER, SWEEP_MAX_TEST_WATER
    else:
        mt, ms = SWEEP_MAX_TRAIN_BENCH, SWEEP_MAX_TEST_BENCH
    return {"max_train": cli_train if cli_train is not None else mt,
            "max_test": cli_test if cli_test is not None else ms}


def _final_budget(proto: str, cli_train=None, cli_test=None) -> dict:
    if proto == "water":
        mt, ms = FINAL_MAX_TRAIN_WATER, FINAL_MAX_TEST_WATER
    else:
        mt, ms = FINAL_MAX_TRAIN_BENCH, FINAL_MAX_TEST_BENCH
    return {"max_train": cli_train if cli_train is not None else mt,
            "max_test": cli_test if cli_test is not None else ms}


def main():
    ap = argparse.ArgumentParser(description="OCQRC: coupled classical-quantum reservoir for multi-step forecasting")
    ap.add_argument("--mode", choices=["tune", "final"], default="tune")
    ap.add_argument("--only", choices=["all", "water", "benchmark"], default="all")
    ap.add_argument("--datasets", default=None, help="comma-separated subset (overrides code config)")
    ap.add_argument("--grid", default=None, help="custom search space (overrides TUNE_WATER/BENCH)")
    ap.add_argument("--stages", default=None, help="restrict staged search, e.g. ridge,b")
    ap.add_argument("--no-fast-ridge", action="store_true")
    ap.add_argument("--metric", choices=["Avg_MSE", "Avg_SMAPE"], default=METRIC)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    for name in PARAM_ORDER:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float,
                        default=BASE_CFG[name])
    ap.add_argument("--sweep-max-train", type=int, default=None)
    ap.add_argument("--sweep-max-test", type=int, default=None)
    ap.add_argument("--final-max-train", type=int, default=None)
    ap.add_argument("--final-max-test", type=int, default=None)
    args = ap.parse_args()

    fixed = {name: getattr(args, name) for name in PARAM_ORDER}
    for k in INT_PARAMS:
        fixed[k] = int(fixed[k])

    plan, skipped = [], []
    for name, path, proto in resolve_run_list(args.datasets, args.only):
        if not os.path.isfile(path):
            print(f"[OCQRC] MISSING FILE  {name} -> {path}")
            continue
        plan.append((name, path, proto))

    print("=" * 78)
    print(f"[OCQRC] model={MODEL}  mode={args.mode}  metric={args.metric}  train_ratio={TRAIN_RATIO}")
    if args.mode == "tune":
        print(f"[OCQRC] search: " + (f"--grid {args.grid}" if args.grid else
                                   "default staged (TUNE_WATER/TUNE_BENCH)"))
        print(f"[OCQRC] sweep budget water={SWEEP_MAX_TRAIN_WATER}/{SWEEP_MAX_TEST_WATER}  "
              f"bench={SWEEP_MAX_TRAIN_BENCH}/{SWEEP_MAX_TEST_BENCH}")
        print(f"[OCQRC] final budget water={FINAL_MAX_TRAIN_WATER}/{FINAL_MAX_TEST_WATER}  "
              f"bench={FINAL_MAX_TRAIN_BENCH}/{FINAL_MAX_TEST_BENCH}")
    elif args.mode == "final":
        print(f"[OCQRC] fixed params: " + " ".join(f"{k}={v}" for k, v in fixed.items()))
    sel = resolve_run_list(args.datasets, args.only)
    print(f"[OCQRC] selected: {len(sel)} ({sum(1 for d in sel if d[2] == 'water')} water, "
          f"{sum(1 for d in sel if d[2] == 'benchmark')} benchmark)")
    for i, (name, path, proto) in enumerate(plan, 1):
        tuned = "tuned" if os.path.isfile(os.path.join(OUTPUTS, MODEL, name, "best_params.json")) \
            else "need-tune"
        print(f"    {i:>2}. [{proto:<9}] {name:<28} ({tuned})")
    print("=" * 78)

    if args.dry_run:
        print("[OCQRC] dry-run: nothing executed.")
        return

    if not plan:
        print("[OCQRC] nothing to run.")
        rebuild_summary()
        return

    batch_start = time.perf_counter()
    ok, failed = [], []
    for i, (name, path, proto) in enumerate(plan, 1):
        print(f"\n{'-' * 78}\n[OCQRC] ({i}/{len(plan)}) {name}  [{proto}]  "
              f"(elapsed {_fmt_hms(time.perf_counter() - batch_start)})\n{'-' * 78}")
        ctx, pred = PROTOCOLS[proto]["ctx"], PROTOCOLS[proto]["pred"]
        data = prepare_io(path, ctx, pred, TRAIN_RATIO, 1, SCALE)
        data["ctx"], data["pred"], data["proto"] = ctx, pred, proto

        if args.mode == "final":
            params = dict(fixed)
            print(f"[OCQRC]    fixed params: {params}")
        else:
            s_budget = _sweep_budget(proto, args.sweep_max_train, args.sweep_max_test)
            grid_spec = args.grid or _tune_spec(proto)
            store = ResultStore(os.path.join(SWEEP_ROOT, name, "sweep_results.csv"),
                                name, proto, args.metric)
            t0 = time.perf_counter()
            params, store = run_search(
                mode="grid" if grid_spec else "stage",
                grid_spec=grid_spec, stages_arg=args.stages,
                no_fast_ridge=args.no_fast_ridge, metric=args.metric,
                data=data, budget=s_budget, store=store)
            print(f"[OCQRC]    search done in {_fmt_hms(time.perf_counter() - t0)}; best: "
                  + " ".join(f"{k}={params[k]}" for k in PARAM_ORDER))
            best_scores = {k: float(store.best()[k]) for k in
                           ["Avg_MSE", "Avg_SMAPE", "RMSE", "NRMSE", "R2", "MAE"]}
            write_tuning_artefacts(name, proto, params, best_scores)

        f_budget = _final_budget(proto, args.final_max_train, args.final_max_test)
        data2 = prepare_io(path, ctx, pred, TRAIN_RATIO, 1, SCALE)
        data2["ctx"], data2["pred"], data2["proto"] = ctx, pred, proto
        tr = data2["train_windows"][: f_budget["max_train"]] if f_budget["max_train"] else data2["train_windows"]
        te_ctx = data2["test_context"][: f_budget["max_test"]] if f_budget["max_test"] else data2["test_context"]
        te_tgt = data2["test_target"][: f_budget["max_test"]] if f_budget["max_test"] else data2["test_target"]
        data2["train_windows"], data2["test_context"], data2["test_target"] = tr, te_ctx, te_tgt

        t0 = time.perf_counter()
        res = run_model(data2, params)
        finalize_outputs(name, proto, params, res, pred)
        print(f"[OCQRC]    final done in {_fmt_hms(time.perf_counter() - t0)}")
        ok.append(name)

    total = time.perf_counter() - batch_start
    print("\n" + "=" * 78)
    print(f"[OCQRC] finished {len(ok)}/{len(plan)} in {_fmt_hms(total)}")
    if failed:
        print(f"[OCQRC] FAILED: {failed}")

    summ = rebuild_summary()
    if summ is not None:
        print("\n[OCQRC] analysis_results.csv")
        print(summ.to_string(index=False))

    with open(os.path.join(OUTPUTS, "ocqrc_experiment_report.json"), "w") as f:
        json.dump({
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "mode": args.mode, "metric": args.metric,
            "succeeded": ok,
            "failed": [{"dataset": n, "stage": c} for n, c in failed],
            "skipped_already_done": skipped,
            "batch_wall_sec": round(total, 2),
        }, f, indent=2)
    print(f"[OCQRC] report -> {os.path.join(OUTPUTS, 'qrc_experiment_report.json')}")


if __name__ == "__main__":
    main()
