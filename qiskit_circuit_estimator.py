import math
import re
import shutil
import subprocess
import warnings
from functools import lru_cache
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram

# Representative fake backends (topology + native gates) per QPU family,
# used when neither a real backend nor a same-name fake backend is available.
_FAMILY_TEMPLATES = {
    'heron': 'fake_fez',
    'nighthawk': 'fake_nighthawk',
}

# IBM Quantum list prices in USD per minute of QPU time (ibm.com/quantum/pricing, checked 2026-10).
# 'open' is free but limited to 10 minutes per month.
_PLAN_PRICE_PER_MIN = {'open': 0.0, 'payg': 96.0, 'flex': 72.0, 'premium': 48.0}

# Local simulation methods, in the order 'auto' tries them
_SIM_METHODS = ['statevector', 'stabilizer', 'matrix_product_state']
_CLIFFORD_BASIS = ['id', 'x', 'y', 'z', 'h', 's', 'sdg', 'sx', 'sxdg', 'cx', 'cy', 'cz', 'swap',
                   'measure', 'reset', 'delay']  # barriers are directives and always pass through

def _parse_mcps_to_hz(val: str) -> float:
    """Converts frequencies ('109.7 kHz', '250 Hz') to numeric Hz."""
    s = str(val).strip().lower()
    match = re.search(r"([0-9.]+)\s*([a-z]*)", s)
    if not match:
        return np.nan
    num, unit = float(match.group(1)), match.group(2)
    multipliers = {'mhz': 1e6, 'khz': 1e3, 'hz': 1.0}
    return num * multipliers.get(unit, 1.0)

def _error_value(row, *columns) -> float:
    """Returns the first non-empty error rate among the given columns (NaN if none)."""
    for col in columns:
        val = row.get(col, np.nan)
        if pd.notna(val):
            return float(val)
    return np.nan

@lru_cache(maxsize=None)
def _fake_backend(name: str):
    """Loads a fake backend from qiskit-ibm-runtime by name, or None if unavailable."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # fake_nighthawk warns about its error values; only topology is used
            return FakeProviderForBackendV2().backend(name)
    except Exception:
        return None

def load_runtime_backends(service=None) -> dict:
    """
    Returns {qpu_name: backend} for every QPU visible to the saved IBM Quantum account,
    to pass as `backends=` so each QPU is transpiled to its exact current layout.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = service or QiskitRuntimeService()
    return {b.name: b for b in service.backends(simulator=False)}

def _resolve_backend(qpu_name: str, qpu_type: str, backends: dict = None):
    """
    Picks the backend whose topology and native gates represent this QPU.
    Priority: user-provided backend > same-name fake backend > family template.
    Returns (backend, source_label) or (None, 'logical') if nothing is available.
    """
    if backends and qpu_name in backends:
        return backends[qpu_name], 'provided'

    fake_name = 'fake_' + qpu_name.removeprefix('ibm_')
    backend = _fake_backend(fake_name)
    if backend is not None:
        return backend, fake_name

    family = str(qpu_type).strip().split()[0].lower() if qpu_type else ''
    template = _FAMILY_TEMPLATES.get(family)
    if template:
        backend = _fake_backend(template)
        if backend is not None:
            return backend, template

    return None, 'logical'

def _circuit_metrics(circ: QuantumCircuit) -> dict:
    """Counts the quantities that drive QPU time and fidelity on an (ideally transpiled) circuit."""
    is_2q = lambda inst: inst.operation.num_qubits == 2 and inst.operation.name != 'barrier'
    n_2q = sum(1 for inst in circ.data if is_2q(inst))
    n_1q = sum(1 for inst in circ.data
               if inst.operation.num_qubits == 1 and inst.operation.name not in ('measure', 'barrier', 'reset', 'delay'))
    return {
        'n_2q': n_2q,
        'n_1q': n_1q,
        'n_measures': circ.count_ops().get('measure', 0),
        'depth': circ.depth(),
        'layers_2q': circ.depth(is_2q) if n_2q else 0,
    }

def _as_clifford_circuit(circ: QuantumCircuit):
    """Rewrites the circuit with Clifford gates only (for the stabilizer method), or None if it is not Clifford."""
    try:
        return transpile(circ, basis_gates=_CLIFFORD_BASIS, optimization_level=0)
    except Exception:
        return None  # e.g. T or arbitrary rotations cannot be expressed in the Clifford basis

def _statevector_mem_gb(n_qubits: int, bytes_per_amplitude: int) -> float:
    """2^n amplitudes plus a 20% margin for buffers."""
    if n_qubits > 60:
        return math.inf
    return (2 ** n_qubits) * bytes_per_amplitude / (1024 ** 3) * 1.2

def _stabilizer_mem_gb(n_qubits: int) -> float:
    """Stabilizer tableau: 2n generators of (2n + 1) bits each, plus a 20% margin."""
    return (2 * n_qubits) * (2 * n_qubits + 1) / 8 / (1024 ** 3) * 1.2

def _as_u_cx_circuit(circ: QuantumCircuit):
    """
    Rewrites the circuit with u + cx gates only, or None if that fails.
    Used for the MPS memory bound and to run MPS on Aer, whose own target rejects more than 63 qubits.
    """
    try:
        return transpile(circ, basis_gates=['u', 'cx', 'measure', 'reset', 'delay'], optimization_level=0)
    except Exception:
        return None

def _mps_mem_gb(flat: QuantumCircuit, bytes_per_amplitude: int = 16) -> float:
    """
    Upper bound on matrix product state memory (Aer MPS stores complex128) for a u + cx circuit.
    The bond dimension across the cut between qubits k and k+1 at most doubles with every CX
    that crosses it, and can never exceed 2^min(k+1, n-k-1).
    """
    if flat is None:
        return math.inf
    n = flat.num_qubits
    if n < 2:
        return bytes_per_amplitude * 2 / (1024 ** 3)

    # Difference array: a CX between qubits a < b crosses cuts a .. b-1
    diff = [0] * n
    for inst in flat.data:
        if inst.operation.name == 'cx':
            a, b = sorted(flat.find_bit(q).index for q in inst.qubits)
            diff[a] += 1
            diff[b] -= 1
    crossings = np.cumsum(diff)[:-1]

    log2_chi = [0] + [min(int(c), k + 1, n - k - 1) for k, c in enumerate(crossings)] + [0]
    # Site tensor i has shape (chi_{i-1}, 2, chi_i)
    log2_bytes = [log2_chi[i] + log2_chi[i + 1] + 1 + math.log2(bytes_per_amplitude) for i in range(n)]
    if max(log2_bytes) > 80:
        return math.inf
    return sum(2.0 ** x for x in log2_bytes) / (1024 ** 3) * 1.2

def _get_gpu_memory_info() -> dict:
    """Queries free and total VRAM (in GB) via PyTorch or nvidia-smi."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return {
                "detected": True,
                "name": torch.cuda.get_device_name(0),
                "vram_free_gb": free_bytes / (1024 ** 3),
                "vram_total_gb": total_bytes / (1024 ** 3)
            }
    except ImportError:
        pass

    if shutil.which("nvidia-smi"):
        try:
            cmd = "nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,nounits,noheader"
            output = subprocess.check_output(cmd, shell=True, text=True).strip().split('\n')[0]
            name, free_mb, total_mb = [x.strip() for x in output.split(',')]
            return {
                "detected": True,
                "name": name,
                "vram_free_gb": float(free_mb) / 1024.0,
                "vram_total_gb": float(total_mb) / 1024.0
            }
        except Exception:
            pass

    return {
        "detected": False,
        "name": None,
        "vram_free_gb": 0.0,
        "vram_total_gb": 0.0
    }

def analyze_circuit_hardware(
    circuit: QuantumCircuit,
    csv_path: str = 'data.csv',
    shots: int = 1024,
    seed_sim: int = None,
    precision: str = 'double',  # 'single' (complex64: 8 bytes) or 'double' (complex128: 16 bytes)
    target_device: str = 'auto',  # 'auto', 'GPU' or 'CPU'
    sim_method: str = 'auto',  # 'auto', 'statevector', 'stabilizer' or 'matrix_product_state'
    only_online: bool = True,
    show_plots: bool = True,
    max_parallel_threads: int = None,
    max_parallel_experiments: int = None,
    max_parallel_shots: int = None,
    backends: dict = None,  # optional {qpu_name: BackendV2}, e.g. from QiskitRuntimeService
    optimization_level: int = 2,
    seed_transpiler: int = 1234,
    show_cost: bool = False,  # adds an Estimated_Cost_USD column and a cost-per-QPU plot
    cost_plan: str = 'payg',  # 'open', 'payg', 'flex' or 'premium'
    price_per_minute: float = None,  # overrides the plan price (USD per QPU minute)
) -> dict:
    """
    Evaluates local feasibility (RAM/VRAM in single/double precision) and compares against QPUs.
    With sim_method='auto', the local simulation uses statevector if it fits in memory, otherwise
    stabilizer (Clifford-only circuits) or matrix product state (low-entanglement circuits).
    The circuit is transpiled to each QPU's topology and native gates before estimating
    execution time and fidelity.
    """
    # 1. Normalize and validate numeric precision
    prec = str(precision).strip().lower()
    if prec not in ['single', 'double']:
        raise ValueError("The 'precision' parameter must be 'single' or 'double'.")

    # complex64 = 8 bytes per amplitude | complex128 = 16 bytes per amplitude
    bytes_per_amplitude = 8 if prec == 'single' else 16

    # 2. Load and prepare the QPU database
    df = pd.read_csv(csv_path)
    df['MCPS_Hz'] = df['MCPS'].apply(_parse_mcps_to_hz)
    df['Online_Bool'] = df['Status'].str.strip().str.lower().str.startswith('online')
    df_qpu = df[df['Online_Bool']].copy() if only_online else df.copy()

    method = str(sim_method).strip().lower()
    if method != 'auto' and method not in _SIM_METHODS:
        raise ValueError(f"The 'sim_method' parameter must be 'auto' or one of {_SIM_METHODS}.")
    device_pref = str(target_device).strip().upper()

    # 3. Memory requirements per simulation method (RAM / VRAM)
    n_qubits = circuit.num_qubits
    circ_sim = circuit.copy()
    if not any(inst.operation.name == 'measure' for inst in circ_sim.data):
        circ_sim.measure_all()

    clifford_circ = _as_clifford_circuit(circ_sim)
    u_cx_circ = _as_u_cx_circuit(circ_sim)
    mem_by_method = {
        'statevector': _statevector_mem_gb(n_qubits, bytes_per_amplitude),
        'stabilizer': _stabilizer_mem_gb(n_qubits) if clifford_circ is not None else math.inf,
        'matrix_product_state': _mps_mem_gb(u_cx_circ),
    }

    # CPU diagnostics
    ram_local = psutil.virtual_memory()
    ram_avail_gb = ram_local.available / (1024 ** 3)
    cpu_feasible = mem_by_method['statevector'] <= ram_avail_gb

    # GPU diagnostics
    gpu_info = _get_gpu_memory_info()
    vram_avail_gb = gpu_info["vram_free_gb"]
    gpu_hw_feasible = gpu_info["detected"] and (mem_by_method['statevector'] <= vram_avail_gb)

    # 4. Simulation method and compute device selection
    aer_gpu_supported = False
    try:
        from qiskit_aer import AerSimulator
        aer_gpu_supported = 'GPU' in AerSimulator().available_devices()
    except ImportError:
        pass

    gpu_feasible = gpu_hw_feasible and aer_gpu_supported

    selected_method = None
    selected_device = None
    for m in (_SIM_METHODS if method == 'auto' else [method]):
        # Aer only runs statevector on GPU; stabilizer and MPS are CPU-only
        if m == 'statevector' and device_pref in ('AUTO', 'GPU') and gpu_feasible:
            selected_method, selected_device = m, 'GPU'
        elif device_pref != 'GPU' and mem_by_method[m] <= ram_avail_gb:
            selected_method, selected_device = m, 'CPU'
        if selected_method:
            break

    local_feasible = selected_method is not None
    mem_req_gb = mem_by_method[selected_method] if local_feasible else mem_by_method['statevector']

    # 5. Circuit metrics for quantum hardware (transpiled per QPU topology)
    circ_hw = circ_sim
    logical_metrics = _circuit_metrics(circ_hw)

    transpile_cache = {}  # source label -> metrics; QPUs sharing a topology are transpiled once
    qpu_results = []
    for _, row in df_qpu.iterrows():
        backend, source = _resolve_backend(row['QPU'], row['Type'], backends)
        compatible = n_qubits <= row['Qubits'] and (backend is None or n_qubits <= backend.num_qubits)

        metrics = None
        if compatible:
            if source not in transpile_cache:
                if backend is None:
                    transpile_cache[source] = logical_metrics
                else:
                    try:
                        circ_t = transpile(circ_hw, backend=backend,
                                           optimization_level=optimization_level,
                                           seed_transpiler=seed_transpiler)
                        transpile_cache[source] = _circuit_metrics(circ_t)
                    except Exception as exc:
                        warnings.warn(f"Transpilation for {row['QPU']} ({source}) failed: {exc}. "
                                      "Using logical circuit counts.")
                        transpile_cache[source] = logical_metrics
            metrics = transpile_cache[source]

        est_time_s = np.nan
        esp_fidelity = np.nan
        if metrics is not None:
            # MCPS = layers of 2Q gates per second; every shot needs at least one layer
            layers = max(1, metrics['layers_2q'])
            if row['MCPS_Hz'] > 0:
                est_time_s = shots * layers / row['MCPS_Hz']

            err_2q = _error_value(row, '2Q Error (Layered)', '2Q Error (Best)')
            err_ro = _error_value(row, 'Readout Error')
            err_1q = _error_value(row, '1Q Error')  # absent in older/scraped CSVs -> factor ignored
            if not (np.isnan(err_2q) or np.isnan(err_ro)):
                esp_fidelity = ((1.0 - err_2q) ** metrics['n_2q']) * ((1.0 - err_ro) ** metrics['n_measures'])
                if not np.isnan(err_1q):
                    esp_fidelity *= (1.0 - err_1q) ** metrics['n_1q']

        qpu_results.append({
            'QPU': row['QPU'],
            'Type': row['Type'],
            'Chip_Qubits': row['Qubits'],
            'Compatible': compatible,
            'Topology_Source': source,
            'Transpiled_2Q': metrics['n_2q'] if metrics else None,
            'Transpiled_Depth': metrics['depth'] if metrics else None,
            'Layers_2Q': metrics['layers_2q'] if metrics else None,
            'MCPS': row['MCPS'],
            'QPU_Time_s': round(est_time_s, 6) if not np.isnan(est_time_s) else None,
            'Estimated_Fidelity': round(esp_fidelity, 4) if not np.isnan(esp_fidelity) else None,
            'Pending_Jobs': row.get('Pending Jobs'),
            'Status': row['Status'].replace('\n', ' ')
        })

    df_result = pd.DataFrame(qpu_results).sort_values(by='QPU_Time_s', ascending=True)

    if show_cost:
        if price_per_minute is None:
            plan = str(cost_plan).strip().lower()
            if plan not in _PLAN_PRICE_PER_MIN:
                raise ValueError(f"The 'cost_plan' parameter must be one of {list(_PLAN_PRICE_PER_MIN)}.")
            price_per_minute = _PLAN_PRICE_PER_MIN[plan]
        df_result['Estimated_Cost_USD'] = (df_result['QPU_Time_s'].astype(float) / 60.0 * price_per_minute).round(4)

    # 6. Simulation and shot distribution
    counts = {}
    fig_shots = None

    if local_feasible:
        try:
            from qiskit_aer import AerSimulator
        except ImportError:
            AerSimulator = None

        if AerSimulator is not None:
            sim_options = {
                'method': selected_method,
                'device': selected_device,
            }
            if selected_method == 'statevector':
                sim_options['precision'] = prec
            if max_parallel_threads is not None:
                sim_options['max_parallel_threads'] = max_parallel_threads
            if max_parallel_experiments is not None:
                sim_options['max_parallel_experiments'] = max_parallel_experiments
            if max_parallel_shots is not None:
                sim_options['max_parallel_shots'] = max_parallel_shots

            try:
                sim = AerSimulator(**sim_options)
                to_run = {'stabilizer': clifford_circ,
                          'matrix_product_state': u_cx_circ}.get(selected_method) or transpile(circ_sim, sim)
                job = sim.run(to_run, shots=shots, seed_simulator=seed_sim)
                counts = job.result().get_counts()
            except Exception as exc:
                warnings.warn(f"Aer simulation ({selected_method}, {selected_device}) failed: {exc}")
        elif selected_method == 'statevector':
            from qiskit.primitives import StatevectorSampler
            job = StatevectorSampler(seed=seed_sim).run([circ_sim], shots=shots)
            counts = job.result()[0].join_data().get_counts()
        else:
            warnings.warn(f"The '{selected_method}' method requires qiskit-aer.")

        if show_plots and counts:
            fig_shots = plot_histogram(
                counts,
                title=f"Distribution of {shots} Shots ({selected_method}, {selected_device} - {prec})"
            )
    else:
        req = " | ".join(f"{m}: {v:.2f} GB" if math.isfinite(v) else f"{m}: n/a"
                         for m, v in mem_by_method.items())
        print(f"Local simulation NOT feasible ({prec}). Required -> {req} | "
              f"Free RAM: {ram_avail_gb:.2f} GB | Free VRAM: {vram_avail_gb:.2f} GB")

    # 7. Comparison plots
    if show_cost:
        fig_hw, (ax_hw, ax_qpu, ax_cost) = plt.subplots(1, 3, figsize=(21, 5))
    else:
        fig_hw, (ax_hw, ax_qpu) = plt.subplots(1, 2, figsize=(15, 5))

    method_label = selected_method or 'statevector'
    labels = [f'Required\n({method_label}, {prec})', 'Avail. RAM\n(CPU)']
    # An infinite requirement cannot be drawn; cap the bar just above the largest available memory
    plot_req = mem_req_gb if math.isfinite(mem_req_gb) else 1.5 * max(ram_avail_gb, vram_avail_gb)
    values = [plot_req, ram_avail_gb]
    colors = ['#e67e22', '#3498db']

    if gpu_info["detected"]:
        labels.append(f"Avail. VRAM\n({gpu_info['name'][:10]}...)")
        values.append(vram_avail_gb)
        colors.append('#2ecc71' if gpu_feasible else '#95a5a6')

    bars = ax_hw.bar(labels, values, color=colors, width=0.45)
    ax_hw.set_ylabel("Gigabytes (GB)")
    status_txt = f"FEASIBLE ({selected_method}, {selected_device})" if local_feasible else "INSUFFICIENT"
    ax_hw.set_title(f"Feasibility ({n_qubits} Qs, {prec}): {status_txt}")
    for b in bars:
        h = b.get_height()
        ax_hw.annotate(f"{h:.3f} GB" if h < 1 else f"{h:.2f} GB",
                       xy=(b.get_x() + b.get_width() / 2, h),
                       xytext=(0, 4), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')

    df_valid = df_result[df_result['QPU_Time_s'].notnull()]
    if not df_valid.empty:
        qpu_bars = ax_qpu.barh(df_valid['QPU'], df_valid['QPU_Time_s'], color='#9b59b6')
        ax_qpu.set_xlabel("Estimated execution time (seconds)")
        ax_qpu.set_title(f"Execution Time per QPU ({shots} shots)")
        for b in qpu_bars:
            w = b.get_width()
            ax_qpu.annotate(f"{w:.4f} s",
                            xy=(w, b.get_y() + b.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=9)
    else:
        ax_qpu.text(0.5, 0.5, "No compatible online QPU", ha='center', va='center')

    if show_cost:
        if not df_valid.empty:
            cost_bars = ax_cost.barh(df_valid['QPU'], df_valid['Estimated_Cost_USD'], color='#16a085')
            ax_cost.set_xlabel("Estimated cost (USD)")
            ax_cost.set_title(f"Cost per QPU (${price_per_minute:g}/min)")
            ax_cost.margins(x=0.18)  # room for the value labels
            for b in cost_bars:
                w = b.get_width()
                ax_cost.annotate(f"${w:.2f}",
                                 xy=(w, b.get_y() + b.get_height() / 2),
                                 xytext=(5, 0), textcoords="offset points",
                                 ha='left', va='center', fontsize=9)
        else:
            ax_cost.text(0.5, 0.5, "No compatible online QPU", ha='center', va='center')

    plt.tight_layout()
    if show_plots:
        plt.show()

    # Both RAM and VRAM keys are returned explicitly for full compatibility with scripts
    return {
        "local_feasible": local_feasible,
        "cpu_feasible": cpu_feasible,
        "gpu_feasible": gpu_feasible,
        "device_used": selected_device,
        "sim_method": selected_method,
        "is_clifford": clifford_circ is not None,
        "mem_by_method_gb": {m: round(v, 6) for m, v in mem_by_method.items()},
        "precision": prec,
        "gpu_detected": gpu_info["name"] if gpu_info["detected"] else None,
        "ram_required_gb": round(mem_req_gb, 4),
        "vram_required_gb": round(mem_req_gb, 4),
        "ram_available_gb": round(ram_avail_gb, 4),
        "vram_available_gb": round(vram_avail_gb, 4),
        "logical_metrics": logical_metrics,
        "qpu_summary": df_result,
        "price_per_minute": price_per_minute if show_cost else None,
        "counts": counts,
        "fig_hardware": fig_hw,
        "fig_shots": fig_shots
    }
