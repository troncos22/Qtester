# Qiskit Circuit Hardware Estimator

An analysis tool that checks whether a Qiskit quantum circuit can be simulated locally (CPU/GPU) and compares its execution time, estimated fidelity and, optionally, cost on real IBM Quantum processors (QPUs).

## Features

* **Automatic simulation method:** picks the cheapest local method that fits in memory: `statevector` (any circuit, up to ~30 qubits), `stabilizer` (Clifford-only circuits, thousands of qubits) or `matrix_product_state` (low-entanglement circuits, many qubits). Each method can also be forced.
* **Memory estimation (RAM/VRAM):** computes the memory each method needs. Statevector supports `single` precision (complex64, 8 bytes per amplitude) or `double` precision (complex128, 16 bytes per amplitude).
* **Automatic hardware detection:** reads available RAM with `psutil` and free GPU VRAM via `PyTorch` or `nvidia-smi`. In `auto` mode, statevector runs on the GPU if `qiskit-aer-gpu` is installed and there is enough VRAM; otherwise it runs on the CPU. Stabilizer and MPS always run on the CPU.
* **Local simulation:** runs the circuit with `AerSimulator` (or `StatevectorSampler` if Aer is not installed) and returns the shot distribution.
* **Per-QPU transpilation:** transpiles the circuit to each QPU's real topology and native gates before estimating, so the SWAPs added by routing and gate decompositions are counted.
* **Real-hardware projection:** for each QPU, estimates execution time from the number of two-qubit layers and the QPU speed (MCPS), and fidelity (ESP) from the 1Q, 2Q and readout errors.
* **Cost estimate (optional):** turns the estimated QPU time into USD using IBM Quantum plan prices, with a cost-per-QPU chart.
* **QPU data updates:** reads live calibration data from the IBM Quantum API with your account, with web scraping as a fallback.

## Project files

| File | Description |
|---|---|
| `qiskit_circuit_estimator.py` | Main module: `analyze_circuit_hardware` and `load_runtime_backends`. |
| `actualizacion_data_QPU.py` | Regenerates `data.csv` from the IBM Quantum API or by scraping [quantum.cloud.ibm.com/computers](https://quantum.cloud.ibm.com/computers). |
| `data.csv` | Local QPU database: qubits, processor type, error rates, T1/T2, MCPS, queue and status. |
| `testing_qc.py` | Usage example with a 4-qubit GHZ circuit. |
| `requirements.txt` | Project dependencies. |

## Installation

Requires Python 3.9 or later.

```bash
git clone <repository-url>
cd Qtester_mf
python -m venv .venv
source .venv/bin/activate
pip install qiskit qiskit-aer qiskit-ibm-runtime pandas numpy matplotlib psutil
```

`qiskit-ibm-runtime` provides the QPU topologies used for transpilation and access to the IBM Quantum API. Without it, the estimator falls back to counting gates on the untranspiled circuit.

Optional dependencies:

```bash
# To update data.csv by scraping (fallback when the API is not available)
pip install playwright
playwright install chromium

# For GPU detection and simulation (requires an NVIDIA GPU with CUDA)
pip install torch qiskit-aer-gpu
```

> **Note:** `qiskit-aer` and `qiskit-aer-gpu` provide the same `qiskit_aer` package. Install only one of them: if you want GPU support, uninstall `qiskit-aer` first. `requirements.txt` lists `torch` as well, so on a machine without CUDA it is best to install the dependencies manually as shown above.

### IBM Quantum account (optional)

Live QPU data and exact per-QPU transpilation need a saved IBM Quantum account. Save it once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<your API key>")
```

Without an account, everything still works: the estimator uses `data.csv` and representative fake backends from `qiskit-ibm-runtime`.

## Quick start

```python
from qiskit import QuantumCircuit
from qiskit_circuit_estimator import analyze_circuit_hardware

qc = QuantumCircuit(4)
qc.h(0)
for i in range(3):
    qc.cx(i, i + 1)
qc.measure_all()

results = analyze_circuit_hardware(
    circuit=qc,
    csv_path="data.csv",
    shots=1024,
    precision="single",
    target_device="auto",
    show_plots=True,
    show_cost=True,      # optional cost-per-QPU column and chart
    cost_plan="payg",
)

print(results["local_feasible"], results["sim_method"], results["device_used"])
print(results["qpu_summary"][["QPU", "Type", "Transpiled_2Q", "QPU_Time_s",
                              "Estimated_Fidelity", "Estimated_Cost_USD"]])
print(results["counts"])
```

Or run the included example directly:

```bash
python testing_qc.py
```

### Exact layouts with your IBM Quantum account

By default, each QPU is transpiled with a fake backend that has the same topology. To use the exact current layout of the QPUs your account can access:

```python
from qiskit_circuit_estimator import analyze_circuit_hardware, load_runtime_backends

results = analyze_circuit_hardware(qc, backends=load_runtime_backends())
```

### Large circuits

A 120-qubit GHZ circuit cannot be simulated with a statevector, but it is Clifford, so `auto` picks the stabilizer method:

```python
qc = QuantumCircuit(120)
qc.h(0)
for i in range(119):
    qc.cx(i, i + 1)

results = analyze_circuit_hardware(qc, show_plots=False)
print(results["sim_method"])        # 'stabilizer'
print(results["mem_by_method_gb"])  # memory each method would need
```

## `analyze_circuit_hardware` reference

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `circuit` | `QuantumCircuit` | — | Circuit to analyze. If it has no measurements, `measure_all()` is added. |
| `csv_path` | `str` | `'data.csv'` | Path to the QPU database. |
| `shots` | `int` | `1024` | Number of shots for the simulation and the QPU estimates. |
| `seed_sim` | `int` | `None` | Simulator seed, for reproducible results. |
| `precision` | `str` | `'double'` | `'single'` or `'double'` (statevector only). `'single'` halves the required memory. |
| `target_device` | `str` | `'auto'` | `'auto'`, `'GPU'` or `'CPU'`. |
| `sim_method` | `str` | `'auto'` | `'auto'`, `'statevector'`, `'stabilizer'` or `'matrix_product_state'`. |
| `only_online` | `bool` | `True` | If `True`, only QPUs with *Online* status are compared. |
| `show_plots` | `bool` | `True` | Shows the shot histogram and comparison charts. |
| `max_parallel_threads` | `int` | `None` | Passed to `AerSimulator`. |
| `max_parallel_experiments` | `int` | `None` | Passed to `AerSimulator`. |
| `max_parallel_shots` | `int` | `None` | Passed to `AerSimulator`. |
| `backends` | `dict` | `None` | `{qpu_name: backend}` used to transpile each QPU, e.g. from `load_runtime_backends()`. |
| `optimization_level` | `int` | `2` | Transpiler optimization level (0–3) for the per-QPU transpilation. |
| `seed_transpiler` | `int` | `1234` | Transpiler seed, for reproducible gate counts. |
| `show_cost` | `bool` | `False` | Adds the `Estimated_Cost_USD` column and a cost-per-QPU chart. |
| `cost_plan` | `str` | `'payg'` | IBM Quantum plan: `'open'` ($0), `'payg'` ($96/min), `'flex'` ($72/min) or `'premium'` ($48/min). |
| `price_per_minute` | `float` | `None` | Custom price in USD per QPU minute; overrides `cost_plan`. |

### Return value

A dictionary with the following keys:

| Key | Description |
|---|---|
| `local_feasible` | `True` if some simulation method fits on the available device. |
| `cpu_feasible` / `gpu_feasible` | Whether a statevector simulation fits on each device. |
| `device_used` | `'CPU'`, `'GPU'`, or `None` if it could not be simulated. |
| `sim_method` | Simulation method used, or `None`. |
| `is_clifford` | `True` if the circuit contains only Clifford gates. |
| `mem_by_method_gb` | Memory each method would need (`inf` if the method cannot run the circuit). |
| `precision` | Precision used (`'single'` or `'double'`). |
| `gpu_detected` | GPU name, or `None`. |
| `ram_required_gb` / `vram_required_gb` | Memory needed by the selected method (same value in both keys). |
| `ram_available_gb` / `vram_available_gb` | Detected free memory. |
| `logical_metrics` | Gate counts, depth and 2Q layers of the untranspiled circuit. |
| `qpu_summary` | `DataFrame` with one row per QPU, sorted by time (see below). |
| `price_per_minute` | Price used for the cost estimate, or `None` if `show_cost=False`. |
| `counts` | Count dictionary `{bitstring: n}`; empty if simulation is not feasible. |
| `fig_hardware` | Matplotlib figure: memory, per-QPU execution time and, with `show_cost`, per-QPU cost. |
| `fig_shots` | Shot histogram, or `None`. |

`qpu_summary` columns:

| Column | Description |
|---|---|
| `QPU`, `Type`, `Chip_Qubits` | QPU name, processor family and number of qubits. |
| `Compatible` | `True` if the circuit fits on the chip. |
| `Topology_Source` | Backend used for transpilation: `provided`, a fake backend name (e.g. `fake_fez`) or `logical` (not transpiled). |
| `Transpiled_2Q`, `Transpiled_Depth`, `Layers_2Q` | Two-qubit gates, depth and two-qubit layers after transpilation. |
| `MCPS`, `QPU_Time_s` | QPU speed and estimated execution time. |
| `Estimated_Fidelity` | ESP estimate (see below). |
| `Estimated_Cost_USD` | Only with `show_cost=True`. |
| `Pending_Jobs`, `Status` | Queue length (when the CSV has it) and QPU status. |

## Estimation model

* **Simulation method (`auto`):** tries `statevector` first (GPU, then CPU), then `stabilizer` if the circuit is Clifford, then `matrix_product_state`, and uses the first one that fits in free memory.
* **Memory:**
  * Statevector: `2^n · bytes_per_amplitude · 1.2`. Each extra qubit doubles the memory: at `double` precision, 30 qubits need about 19 GB.
  * Stabilizer: a tableau of `2n · (2n + 1)` bits, so a few kilobytes even for hundreds of qubits.
  * Matrix product state: an upper bound in which the bond dimension across each cut between neighboring qubits at most doubles with every CX crossing it, capped at `2^min(k, n − k)`. The real memory use is often lower.
* **Transpilation:** each QPU uses, in order of preference, the backend passed in `backends`, a fake backend with the same name (`ibm_fez` → `fake_fez`), or a fake backend of the same family (Heron → `fake_fez`, Nighthawk → `fake_nighthawk`). QPUs that share a topology are transpiled only once.
* **QPU time:** `shots · max(1, L₂Q) / MCPS`, where `L₂Q` is the number of two-qubit layers in the transpiled circuit. Queue time and network latency are not included.
* **Estimated fidelity (ESP):** `(1 − ε₂Q)^N₂Q · (1 − ε₁Q)^N₁Q · (1 − ε_RO)^N_meas`, where `ε₂Q` is the layered two-qubit error, `ε₁Q` the median single-qubit error (when available in the CSV), `ε_RO` the readout error, and the `N` are gate counts from the transpiled circuit.
* **Cost:** `QPU_Time_s / 60 · price_per_minute`. Prices are IBM Quantum list prices (checked October 2026, [ibm.com/quantum/pricing](https://www.ibm.com/quantum/pricing)). IBM bills the actual quantum time, which includes overhead this model does not include, so real bills are usually somewhat higher.

These figures are a rough guide: they do not model decoherence during idle time, crosstalk, or error mitigation.

## Updating the QPU database

```bash
python actualizacion_data_QPU.py                 # IBM Quantum API (default)
python actualizacion_data_QPU.py --source scrape # web scraping only
python actualizacion_data_QPU.py --source both   # API + scraped QPUs your account cannot see
python actualizacion_data_QPU.py --out other.csv # write to a different file
```

* **`api`** reads live calibration data for the QPUs your saved account can access. The layered 2Q error is computed from IBM's 100-qubit layer fidelity (`EPLG = 1 − LF^(1/99)`), the same value the website shows.
* **`scrape`** opens IBM Quantum's public computers page in headless Chromium and extracts the table. It covers every public QPU, but if IBM changes the page layout, the column indices in the script may need adjusting.
* If the API fails (no saved account or no network), the script falls back to scraping automatically.

> **Note:** an Open plan account only sees a few QPUs, so `--source api` may produce a shorter list than the website. Use `--source both` to keep comparing against every public QPU.

Expected `data.csv` format (columns missing from scraped rows are left empty):

```csv
QPU,Qubits,Type,2Q Gate,2Q Error (Best),2Q Error (Layered),1Q Error,Readout Error,T1 (us),T2 (us),MCPS,Pending Jobs,Status
ibm_fez,156,Heron r2,cz,0.001317,0.004859,0.000289,0.010254,149.42,101.94,3864 Hz,1,Online
```

Older `data.csv` files without the new columns still work.

## Roadmap

* Turn the estimator into an API.

## License

Distributed under the MIT License. See [`license.md`](license.md).
