"""
Qiskit Circuit Resource, Runtime & Cost Estimator (IBM Quantum Edition)
======================================================================

A comprehensive evaluation module for Quantum Circuits designed for contribution
to the Qiskit ecosystem (Qiskit SDK / Qiskit Community / Qiskit Addons).

Key Capabilities:
1. Circuit Analysis:
   - Evaluates total and active qubits, circuit depth, 2-qubit gate density.
   - Categorizes Clifford vs. Non-Clifford (T-gate) operations.
   - Analyzes heavy-hex topology routing overhead and SWAP insertion penalties.
2. IBM Quantum Modality Runtime & Cost Modeling:
   - Evaluates Superconducting Transmon architectures open on IBM Quantum:
     * Tunable-Coupler Transmon (IBM Heron r2, CZ gates ~60 ns).
     * Fixed-Frequency Transmon (IBM Eagle r3, ECR gates ~350 ns, Falcon r5, Osprey).
   - Configurable target processor selection (default: IBM Heron).
   - Computes single-shot duration, readout time, reset/repolarization, and batch overhead.
   - Calculates estimated execution costs: IBM Quantum Runtime pay-as-you-go ($1.60/sec)
     and IBM Quantum Open Plan (10 minutes free allocation monthly).
3. Local Classical Hardware Inspection & Simulation Viability:
   - Discovers local CPU cores, RAM (total and available), and GPU acceleration status.
   - Evaluates feasibility across simulation methods:
     * Statevector (Schrödinger direct simulation, $O(2^n)$ memory).
     * Clifford / Stabilizer (Aaronson-Gottesman theorem, polynomial memory).
     * Matrix Product State (MPS / Tensor Networks for low-entanglement).
     * Extended Stabilizer (Clifford + small T-depth).
   - Recommends the optimal simulation method or warns of Out-Of-Memory (OOM).

Author: jorge Moreno Farías
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import platform
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union


# =====================================================================
# 1. Quantum Architecture & Modality Enums & Data Structures
# =====================================================================

class QPUModality(str, Enum):
    """
    Quantum Processor modalities available on IBM Quantum Platform.
    IBM Quantum hardware systems operate exclusively on superconducting transmon modalities.
    """
    SUPERCONDUCTING = "Superconducting Transmon"
    SUPERCONDUCTING_TUNABLE_COUPLER = "Superconducting Tunable-Coupler Transmon (IBM Heron)"
    SUPERCONDUCTING_FIXED_FREQUENCY = "Superconducting Fixed-Frequency Transmon (IBM Eagle / Falcon)"


class TopologyType(str, Enum):
    """QPU coupling maps and coupling topology graphs."""
    HEAVY_HEX = "Heavy-Hexagon (IBM Quantum Standard)"
    SQUARE_GRID = "Square Lattice"
    ALL_TO_ALL = "All-to-All"


class PricingModelType(str, Enum):
    """Billing and allocation models on IBM Quantum."""
    RUNTIME_PAY_AS_YOU_GO = "IBM Quantum Runtime ($1.60/sec pay-as-you-go)"
    OPEN_PLAN_FREE = "IBM Quantum Open Plan (Free 10 min/month allocation)"
    PREMIUM_RESERVED = "IBM Quantum Premium (Capacity Allocation)"


@dataclass
class QPUProfile:
    """Hardware profile and timing parameters for an IBM Quantum Processor."""
    name: str
    vendor: str
    modality: QPUModality
    total_qubits: int
    topology: TopologyType
    
    # Gate durations (in nanoseconds: 1 ns = 1e-9 s)
    t_1q_ns: float             # Single-qubit gate duration (microwave pulse)
    t_2q_ns: float             # Two-qubit entangling gate duration (CZ or ECR)
    t_readout_ns: float        # Measurement / readout duration
    t_reset_ns: float          # Active reset / thermal relaxation time
    
    # Architecture routing overhead estimate (multiplier for non-local 2Q gates)
    avg_swap_overhead_factor: float
    
    # Pricing configuration
    pricing_model: PricingModelType
    price_per_runtime_sec: float = 1.60     # Standard IBM Quantum Runtime rate ($1.60/s)
    
    # Typical system latency & container overhead
    job_init_overhead_sec: float = 5.0      # Qiskit Runtime session startup & queue packing
    max_shots_per_job: int = 100000


# Standard IBM Quantum Hardware Registry
STANDARD_QPUS: Dict[str, QPUProfile] = {
    "ibm_heron": QPUProfile(
        name="IBM Heron (r2)",
        vendor="IBM Quantum",
        modality=QPUModality.SUPERCONDUCTING_TUNABLE_COUPLER,
        total_qubits=156,
        topology=TopologyType.HEAVY_HEX,
        t_1q_ns=20.0,             # ~20 ns single-qubit microwave pulses
        t_2q_ns=60.0,             # Tunable coupler CZ gates (~60 ns)
        t_readout_ns=700.0,       # Fast multiplexed dispersive readout
        t_reset_ns=250_000.0,     # ~250 µs repetition delay / active reset
        avg_swap_overhead_factor=2.2, # Heavy-hex routing overhead
        pricing_model=PricingModelType.RUNTIME_PAY_AS_YOU_GO,
        price_per_runtime_sec=1.60, # Standard IBM Quantum Runtime pay-as-you-go rate
        job_init_overhead_sec=5.0
    ),
    "ibm_eagle": QPUProfile(
        name="IBM Eagle (r3)",
        vendor="IBM Quantum",
        modality=QPUModality.SUPERCONDUCTING_FIXED_FREQUENCY,
        total_qubits=127,
        topology=TopologyType.HEAVY_HEX,
        t_1q_ns=35.0,
        t_2q_ns=350.0,            # Cross-resonance ECR gates (~300-400 ns)
        t_readout_ns=900.0,
        t_reset_ns=300_000.0,     # ~300 µs
        avg_swap_overhead_factor=2.8,
        pricing_model=PricingModelType.RUNTIME_PAY_AS_YOU_GO,
        price_per_runtime_sec=1.60,
        job_init_overhead_sec=6.0
    ),
    "ibm_eagle_open": QPUProfile(
        name="IBM Eagle (Open Plan)",
        vendor="IBM Quantum",
        modality=QPUModality.SUPERCONDUCTING_FIXED_FREQUENCY,
        total_qubits=127,
        topology=TopologyType.HEAVY_HEX,
        t_1q_ns=35.0,
        t_2q_ns=350.0,
        t_readout_ns=900.0,
        t_reset_ns=300_000.0,
        avg_swap_overhead_factor=2.8,
        pricing_model=PricingModelType.OPEN_PLAN_FREE,
        price_per_runtime_sec=0.00, # 10 min/month free allocation
        job_init_overhead_sec=6.0
    ),
    "ibm_falcon": QPUProfile(
        name="IBM Falcon (r5)",
        vendor="IBM Quantum",
        modality=QPUModality.SUPERCONDUCTING_FIXED_FREQUENCY,
        total_qubits=27,
        topology=TopologyType.HEAVY_HEX,
        t_1q_ns=35.0,
        t_2q_ns=300.0,
        t_readout_ns=750.0,
        t_reset_ns=250_000.0,
        avg_swap_overhead_factor=2.0,
        pricing_model=PricingModelType.RUNTIME_PAY_AS_YOU_GO,
        price_per_runtime_sec=1.60,
        job_init_overhead_sec=4.0
    ),
    "ibm_osprey": QPUProfile(
        name="IBM Osprey",
        vendor="IBM Quantum",
        modality=QPUModality.SUPERCONDUCTING_FIXED_FREQUENCY,
        total_qubits=433,
        topology=TopologyType.HEAVY_HEX,
        t_1q_ns=40.0,
        t_2q_ns=400.0,
        t_readout_ns=1000.0,
        t_reset_ns=300_000.0,
        avg_swap_overhead_factor=3.2,
        pricing_model=PricingModelType.RUNTIME_PAY_AS_YOU_GO,
        price_per_runtime_sec=1.60,
        job_init_overhead_sec=8.0
    )
}

# Alias mapping for user convenience and spelling variations
QPU_ALIASES: Dict[str, str] = {
    "heron": "ibm_heron",
    "heleron": "ibm_heron", # Common phonetic typo
    "ibm_heron": "ibm_heron",
    "torino": "ibm_heron",  # ibm_torino is Heron r2
    "marrakesh": "ibm_heron",
    "fez": "ibm_heron",
    "eagle": "ibm_eagle",
    "ibm_eagle": "ibm_eagle",
    "brisbane": "ibm_eagle", # ibm_brisbane is Eagle r3
    "kyoto": "ibm_eagle",    # ibm_kyoto is Eagle r3
    "osaka": "ibm_eagle",
    "sherbrooke": "ibm_eagle",
    "open": "ibm_eagle_open",
    "open_plan": "ibm_eagle_open",
    "falcon": "ibm_falcon",
    "ibm_falcon": "ibm_falcon",
    "osprey": "ibm_osprey",
    "ibm_osprey": "ibm_osprey"
}


# =====================================================================
# 2. Circuit Resource Analysis Data Structures
# =====================================================================

CLIFFORD_GATES = {
    "h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "id", "i",
    "cx", "cz", "swap", "cnot", "cy", "barrier", "measure", "reset"
}

NON_CLIFFORD_GATES = {
    "t", "tdg", "rx", "ry", "rz", "u", "u1", "u2", "u3", "p",
    "ccx", "cswap", "rxx", "ryy", "rzz", "cp", "crx", "cry", "crz"
}


@dataclass
class CircuitMetrics:
    """Metrics extracted from a quantum circuit."""
    num_qubits: int
    active_qubits: int
    num_clbits: int
    depth: int
    depth_2q: int
    total_gates: int
    gates_1q: int
    gates_2q: int
    gates_multi_q: int
    num_measurements: int
    num_resets: int
    num_barriers: int
    ops_distribution: Dict[str, int]
    is_clifford_only: bool
    t_count: int               # Count of T / Tdg gates
    non_clifford_count: int    # Total count of non-Clifford rotations and gates
    entangling_ratio: float    # 2Q gates / total gates
    interaction_graph_edges: int


# =====================================================================
# 3. Circuit Analyzer (Native Qiskit & Fallback Compatibility)
# =====================================================================

class CircuitAnalyzer:
    """Analyzes QuantumCircuit structures to extract detailed resource metrics."""
    
    @staticmethod
    def analyze(circuit: Any) -> CircuitMetrics:
        """
        Analyze a QuantumCircuit object.
        Compatible with Qiskit's `QuantumCircuit` or any object implementing
        the standard `num_qubits`, `count_ops()`, and circuit inspection interface.
        """
        num_qubits = getattr(circuit, "num_qubits", 0)
        num_clbits = getattr(circuit, "num_clbits", 0)
        
        if hasattr(circuit, "count_ops") and callable(circuit.count_ops):
            ops = dict(circuit.count_ops())
        else:
            ops = {}

        active_qubit_indices = set()
        gates_1q = 0
        gates_2q = 0
        gates_multi_q = 0
        num_measurements = 0
        num_resets = 0
        num_barriers = 0
        non_clifford_count = 0
        t_count = 0
        interaction_edges = set()
        
        data = getattr(circuit, "data", [])
        
        for item in data:
            if hasattr(item, "operation"):
                op = item.operation
                qargs = item.qubits
                name = op.name.lower()
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                op = item[0]
                qargs = item[1]
                name = getattr(op, "name", str(op)).lower()
            else:
                continue
            
            q_indices = []
            for q in qargs:
                if hasattr(circuit, "find_bit"):
                    q_indices.append(circuit.find_bit(q).index)
                elif hasattr(q, "_index"):
                    q_indices.append(q._index)
                elif isinstance(q, int):
                    q_indices.append(q)
                else:
                    try:
                        q_indices.append(list(circuit.qubits).index(q))
                    except (ValueError, AttributeError):
                        pass
            
            for qi in q_indices:
                active_qubit_indices.add(qi)
            
            q_len = len(q_indices)
            
            if name == "measure":
                num_measurements += 1
            elif name == "reset":
                num_resets += 1
            elif name == "barrier":
                num_barriers += 1
            else:
                if q_len == 1:
                    gates_1q += 1
                elif q_len == 2:
                    gates_2q += 1
                    if len(q_indices) == 2:
                        edge = tuple(sorted((q_indices[0], q_indices[1])))
                        interaction_edges.add(edge)
                elif q_len > 2:
                    gates_multi_q += 1
            
            if name in ("t", "tdg"):
                t_count += 1
                non_clifford_count += 1
            elif name in NON_CLIFFORD_GATES:
                non_clifford_count += 1
            elif name not in CLIFFORD_GATES:
                non_clifford_count += 1

        if not data and ops:
            for op_name, count in ops.items():
                op_l = op_name.lower()
                if op_l == "measure":
                    num_measurements += count
                elif op_l == "barrier":
                    num_barriers += count
                elif op_l == "reset":
                    num_resets += count
                elif op_l in ("cx", "cz", "swap", "cnot", "cy", "rxx", "ryy", "rzz", "cp"):
                    gates_2q += count
                elif op_l in ("ccx", "cswap"):
                    gates_multi_q += count
                else:
                    gates_1q += count
                
                if op_l in ("t", "tdg"):
                    t_count += count
                    non_clifford_count += count
                elif op_l in NON_CLIFFORD_GATES:
                    non_clifford_count += count
                elif op_l not in CLIFFORD_GATES:
                    non_clifford_count += count
            active_qubit_indices = set(range(num_qubits))

        depth = getattr(circuit, "depth", lambda: 0)
        depth_val = depth() if callable(depth) else depth
        
        depth_2q = gates_2q // max(1, (num_qubits // 2)) if gates_2q > 0 else 0
        total_gates = gates_1q + gates_2q + gates_multi_q
        active_count = len(active_qubit_indices) if active_qubit_indices else num_qubits
        is_clifford_only = (non_clifford_count == 0)
        entangling_ratio = (gates_2q + gates_multi_q) / max(1, total_gates)

        return CircuitMetrics(
            num_qubits=num_qubits,
            active_qubits=active_count,
            num_clbits=num_clbits,
            depth=depth_val,
            depth_2q=depth_2q,
            total_gates=total_gates,
            gates_1q=gates_1q,
            gates_2q=gates_2q,
            gates_multi_q=gates_multi_q,
            num_measurements=num_measurements,
            num_resets=num_resets,
            num_barriers=num_barriers,
            ops_distribution=ops,
            is_clifford_only=is_clifford_only,
            t_count=t_count,
            non_clifford_count=non_clifford_count,
            entangling_ratio=entangling_ratio,
            interaction_graph_edges=len(interaction_edges)
        )


# =====================================================================
# 4. Local Classical Hardware Inspector & Simulator Evaluator
# =====================================================================

class SimFeasibility(str, Enum):
    FEASIBLE = "FEASIBLE"                       # Comfortably within RAM & CPU limits
    WARNING_HIGH_RAM = "WARNING: HIGH RAM"       # Uses >75% of available RAM
    INFEASIBLE_OOM = "INFEASIBLE (OOM)"         # Exceeds available RAM (Crash)
    HIGHLY_EFFICIENT = "HIGHLY EFFICIENT"       # Polynomial or exact tensor contraction


@dataclass
class LocalHardwareSpecs:
    """Discovered specifications of the local host system."""
    cpu_model: str
    logical_cores: int
    physical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    platform_os: str = field(default_factory=platform.system)


@dataclass
class SimulationMethodAssessment:
    """Feasibility evaluation for a specific classical simulation method."""
    method_name: str
    aer_method_flag: str
    feasibility: SimFeasibility
    required_memory_gb: float
    memory_percentage_of_available: float
    estimated_sim_time_sec: float
    explanation: str
    is_recommended: bool


@dataclass
class LocalSimulationReport:
    """Complete diagnostic report for local simulation viability."""
    hardware: LocalHardwareSpecs
    assessments: Dict[str, SimulationMethodAssessment]
    can_simulate_locally: bool
    recommended_method: Optional[str]
    summary_verdict: str
    guidance_notes: List[str]


class LocalHardwareInspector:
    """Inspects host machine resources and evaluates classical simulation viability."""

    @staticmethod
    def get_system_specs() -> LocalHardwareSpecs:
        """Query CPU, RAM, and GPU capabilities with multi-platform fallbacks."""
        logical_cores = os.cpu_count() or 1
        physical_cores = max(1, logical_cores // 2)
        total_ram_gb = 8.0
        available_ram_gb = 6.0
        cpu_model = platform.processor() or "Generic x86_64/ARM CPU"
        
        # Linux /proc/meminfo parsing
        if sys.platform.startswith("linux"):
            try:
                meminfo = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            meminfo[parts[0].strip()] = parts[1].strip()
                
                if "MemTotal" in meminfo:
                    kb = float(meminfo["MemTotal"].split()[0])
                    total_ram_gb = kb / (1024.0 * 1024.0)
                if "MemAvailable" in meminfo:
                    kb = float(meminfo["MemAvailable"].split()[0])
                    available_ram_gb = kb / (1024.0 * 1024.0)
                elif "MemFree" in meminfo:
                    kb = float(meminfo["MemFree"].split()[0])
                    available_ram_gb = kb / (1024.0 * 1024.0)
            except Exception:
                pass
                
        # macOS sysctl fallback
        elif sys.platform == "darwin":
            try:
                import subprocess
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
                total_ram_gb = float(out) / (1024.0 ** 3)
                available_ram_gb = total_ram_gb * 0.7
            except Exception:
                pass
                
        # Windows fallback via ctypes
        elif sys.platform == "win32":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram_gb = stat.ullTotalPhys / (1024.0 ** 3)
                available_ram_gb = stat.ullAvailPhys / (1024.0 ** 3)
            except Exception:
                pass

        gpu_available = False
        gpu_name = None
        gpu_vram = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
        except ImportError:
            try:
                import subprocess
                smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).decode()
                parts = smi.strip().split(",")
                if len(parts) >= 2:
                    gpu_available = True
                    gpu_name = parts[0].strip()
                    gpu_vram = float(parts[1].strip()) / 1024.0
            except Exception:
                pass

        return LocalHardwareSpecs(
            cpu_model=cpu_model,
            logical_cores=logical_cores,
            physical_cores=physical_cores,
            total_ram_gb=round(total_ram_gb, 2),
            available_ram_gb=round(available_ram_gb, 2),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=round(gpu_vram, 2) if gpu_vram else None
        )

    @classmethod
    def evaluate_simulation_viability(
        cls,
        metrics: CircuitMetrics,
        hardware: Optional[LocalHardwareSpecs] = None,
        shots: int = 1000
    ) -> LocalSimulationReport:
        """
        Evaluate feasibility of executing local simulation across various Qiskit Aer methods.
        """
        if hardware is None:
            hardware = cls.get_system_specs()

        n = metrics.active_qubits
        assessments: Dict[str, SimulationMethodAssessment] = {}
        guidance: List[str] = []

        # 1. Statevector Simulation: Memory = 2^n * 16 bytes (complex128)
        if n <= 60:
            sv_bytes = (2 ** n) * 16.0
            sv_gb = sv_bytes / (1024.0 ** 3)
        else:
            sv_gb = 1e15
            
        sv_pct = (sv_gb / max(0.1, hardware.available_ram_gb)) * 100.0
        sv_time_est = max(0.01, (2 ** min(n, 28)) * metrics.total_gates * 1.5e-9 / hardware.logical_cores)
        
        if sv_gb <= hardware.available_ram_gb * 0.70:
            sv_feas = SimFeasibility.FEASIBLE
            sv_expl = f"Statevector requires {sv_gb:.2f} GB RAM ({sv_pct:.1f}% of available). Smooth local simulation."
        elif sv_gb <= hardware.available_ram_gb:
            sv_feas = SimFeasibility.WARNING_HIGH_RAM
            sv_expl = f"Statevector requires {sv_gb:.2f} GB RAM ({sv_pct:.1f}% of available). May trigger system swap."
        else:
            sv_feas = SimFeasibility.INFEASIBLE_OOM
            sv_expl = f"Statevector requires {sv_gb:.2f} GB RAM, exceeding available {hardware.available_ram_gb:.2f} GB. Will crash with OOM."

        assessments["statevector"] = SimulationMethodAssessment(
            method_name="Statevector (Exact Schrödinger)",
            aer_method_flag="statevector",
            feasibility=sv_feas,
            required_memory_gb=round(sv_gb, 4),
            memory_percentage_of_available=round(sv_pct, 1),
            estimated_sim_time_sec=round(sv_time_est, 2),
            explanation=sv_expl,
            is_recommended=False
        )

        # 2. Stabilizer / Clifford Simulation (Gottesman-Knill Theorem)
        stab_bytes = ((2 * n + 1) ** 2) / 8.0
        stab_gb = stab_bytes / (1024.0 ** 3)
        stab_time_est = max(0.001, (metrics.total_gates * (n ** 2) * 1e-8) + (shots * 1e-4))
        
        if metrics.is_clifford_only:
            stab_feas = SimFeasibility.HIGHLY_EFFICIENT
            stab_expl = (
                f"Circuit is 100% Clifford! Tableau requires only {stab_bytes:.1f} bytes. "
                f"Simulable in polynomial time for thousands of qubits."
            )
        else:
            stab_feas = SimFeasibility.INFEASIBLE_OOM
            stab_expl = f"Circuit contains {metrics.non_clifford_count} non-Clifford gates (T, Rz, U3, etc.). Stabilizer simulation is exact only for Clifford circuits."

        assessments["stabilizer"] = SimulationMethodAssessment(
            method_name="Clifford Stabilizer (Aaronson-Gottesman)",
            aer_method_flag="stabilizer",
            feasibility=stab_feas,
            required_memory_gb=round(stab_gb, 6),
            memory_percentage_of_available=round((stab_gb / hardware.available_ram_gb) * 100.0, 4),
            estimated_sim_time_sec=round(stab_time_est, 3),
            explanation=stab_expl,
            is_recommended=False
        )

        # 3. Extended Stabilizer (Bravyi-Smith-Smolin)
        t = metrics.t_count
        if t <= 40 and not metrics.is_clifford_only and metrics.non_clifford_count == metrics.t_count:
            ext_overhead = min(1e9, 2 ** (t * 0.47))
            ext_ram_gb = stab_gb + (ext_overhead * 1e-6)
            ext_time = max(0.01, ext_overhead * 1e-5)
            ext_feas = SimFeasibility.FEASIBLE if ext_ram_gb < hardware.available_ram_gb else SimFeasibility.WARNING_HIGH_RAM
            ext_expl = f"T-count is {t} (<= 40). Extended stabilizer can efficiently approximate output state."
        else:
            ext_ram_gb = 0.0
            ext_time = 0.0
            ext_feas = SimFeasibility.INFEASIBLE_OOM
            ext_expl = f"T-count / Non-Clifford count is {metrics.non_clifford_count} (>40 or arbitrary continuous rotations). Extended stabilizer exponential scaling is prohibitive."

        assessments["extended_stabilizer"] = SimulationMethodAssessment(
            method_name="Extended Stabilizer (Bravyi decomposition)",
            aer_method_flag="extended_stabilizer",
            feasibility=ext_feas,
            required_memory_gb=round(ext_ram_gb, 4),
            memory_percentage_of_available=round((ext_ram_gb / hardware.available_ram_gb) * 100.0, 2),
            estimated_sim_time_sec=round(ext_time, 2),
            explanation=ext_expl,
            is_recommended=False
        )

        # 4. Matrix Product State (MPS / Tensor Networks)
        bond_dim = min(256, max(8, 2 ** min(8, int(metrics.depth_2q * metrics.entangling_ratio * 4))))
        mps_bytes = n * 2 * (bond_dim ** 2) * 16.0
        mps_gb = mps_bytes / (1024.0 ** 3)
        mps_time = max(0.01, metrics.total_gates * (bond_dim ** 3) * 5e-8)

        if mps_gb < hardware.available_ram_gb * 0.6:
            if metrics.entangling_ratio < 0.35 or metrics.depth <= 30:
                mps_feas = SimFeasibility.FEASIBLE
                mps_expl = f"Moderate entanglement (bond dimension ~{bond_dim}). MPS memory is only {mps_gb:.3f} GB."
            else:
                mps_feas = SimFeasibility.WARNING_HIGH_RAM
                mps_expl = f"Deep entangling circuit. Truncation errors may occur if bond dimension ({bond_dim}) is capped."
        else:
            mps_feas = SimFeasibility.INFEASIBLE_OOM
            mps_expl = f"Estimated MPS bond dimension requires {mps_gb:.2f} GB RAM, exceeding local system resources."

        assessments["matrix_product_state"] = SimulationMethodAssessment(
            method_name="Matrix Product State (MPS / Tensor Network)",
            aer_method_flag="matrix_product_state",
            feasibility=mps_feas,
            required_memory_gb=round(mps_gb, 4),
            memory_percentage_of_available=round((mps_gb / hardware.available_ram_gb) * 100.0, 2),
            estimated_sim_time_sec=round(mps_time, 2),
            explanation=mps_expl,
            is_recommended=False
        )

        # Selection Strategy & Recommendation
        recommended_method: Optional[str] = None
        can_simulate = False

        if metrics.is_clifford_only:
            recommended_method = "stabilizer"
            assessments["stabilizer"].is_recommended = True
            can_simulate = True
            summary_verdict = "LOCAL SIMULATION HIGHLY RECOMMENDED: 100% Clifford circuit. Run with AerSimulator(method='stabilizer')."
            guidance.append("Circuit contains only Clifford operations (H, S, CX, CZ, measurements). Local simulation is practically instantaneous.")
        elif sv_feas == SimFeasibility.FEASIBLE:
            recommended_method = "statevector"
            assessments["statevector"].is_recommended = True
            can_simulate = True
            summary_verdict = f"LOCAL SIMULATION RECOMMENDED: {n} active qubits easily fit in RAM ({sv_gb:.2f} GB required vs {hardware.available_ram_gb:.2f} GB available)."
            guidance.append(f"Ideal for pre-execution debugging, noise analysis, and parameter tuning in Qiskit Aer.")
        elif mps_feas == SimFeasibility.FEASIBLE and n > 28:
            recommended_method = "matrix_product_state"
            assessments["matrix_product_state"].is_recommended = True
            can_simulate = True
            summary_verdict = f"LOCAL SIMULATION FEASIBLE VIA MPS: Statevector OOM ({sv_gb:.1f} GB), but low/medium entanglement allows Matrix Product State simulation."
            guidance.append(f"Configure AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension={bond_dim}).")
        elif assessments["extended_stabilizer"].feasibility == SimFeasibility.FEASIBLE:
            recommended_method = "extended_stabilizer"
            assessments["extended_stabilizer"].is_recommended = True
            can_simulate = True
            summary_verdict = f"LOCAL SIMULATION FEASIBLE VIA EXTENDED STABILIZER: Circuit has low T-count ({t}), fitting Bravyi stabilizer decomposition."
            guidance.append(f"Use AerSimulator(method='extended_stabilizer') for approximate output sampling.")
        elif sv_feas == SimFeasibility.WARNING_HIGH_RAM:
            recommended_method = "statevector"
            assessments["statevector"].is_recommended = True
            can_simulate = True
            summary_verdict = f"LOCAL SIMULATION POSSIBLE WITH WARNING: Requires {sv_gb:.2f} GB ({sv_pct:.1f}% of RAM). Close heavy background apps before running."
            guidance.append("Consider setting precision='single' (complex64) in Qiskit Aer to cut RAM usage by 50%.")
        else:
            recommended_method = None
            can_simulate = False
            summary_verdict = (
                f"LOCAL CLASSICAL SIMULATION NOT FEASIBLE: Circuit with {n} highly-entangled qubits "
                f"requires {sv_gb:.2f} GB RAM for statevector, exceeding host capacity ({hardware.available_ram_gb:.2f} GB). "
                f"Direct execution on IBM Quantum Hardware (QPU) is required."
            )
            guidance.append("To simulate locally, reduce active qubits to <= 28 or synthesize circuit to Clifford+T structure.")

        return LocalSimulationReport(
            hardware=hardware,
            assessments=assessments,
            can_simulate_locally=can_simulate,
            recommended_method=recommended_method,
            summary_verdict=summary_verdict,
            guidance_notes=guidance
        )


# =====================================================================
# 5. IBM Quantum Runtime, Latency & Cost Calculator
# =====================================================================

@dataclass
class QPUEstimateResult:
    """Estimated execution metrics on an IBM Quantum Processor."""
    qpu_name: str
    vendor: str
    modality: str
    topology: str
    supported_qubits: int
    is_qubit_count_supported: bool
    
    # Timing breakdown
    t_single_shot_us: float          # Microseconds per single shot
    effective_gates_2q: int          # 2Q gates including topology routing (SWAP) overhead
    routing_overhead_factor: float
    raw_qpu_time_sec: float          # Pure QPU execution time for all shots
    control_batching_overhead_sec: float
    total_estimated_time_sec: float  # QPU time + initialization + control latency
    
    # Cost breakdown
    estimated_cost_usd: float
    pricing_model_desc: str
    cost_details: str
    
    # Feasibility flag
    hardware_notes: List[str]


class RuntimeAndCostCalculator:
    """Evaluates circuit execution time and financial cost across IBM QPUs."""

    @classmethod
    def evaluate_qpu(
        cls,
        metrics: CircuitMetrics,
        qpu: QPUProfile,
        shots: int = 1000
    ) -> QPUEstimateResult:
        """Calculate runtime, routing penalties, and IBM cloud fees for a given QPU."""
        notes: List[str] = []
        is_qubit_supported = metrics.active_qubits <= qpu.total_qubits
        
        if not is_qubit_supported:
            notes.append(
                f"ERROR: Circuit active qubits ({metrics.active_qubits}) exceed "
                f"QPU physical capacity ({qpu.total_qubits})."
            )

        # IBM Heavy-Hex topology routing overhead
        routing_factor = qpu.avg_swap_overhead_factor
        effective_2q_gates = int(metrics.gates_2q * routing_factor)
        effective_depth_2q = int(metrics.depth_2q * routing_factor)
        
        # Timing calculation per shot (in seconds)
        # Time = (1Q depth * t_1q) + (2Q depth * t_2q) + Readout + Reset
        t_1q_s = (metrics.depth - metrics.depth_2q) * (qpu.t_1q_ns * 1e-9)
        t_2q_s = effective_depth_2q * (qpu.t_2q_ns * 1e-9)
        t_meas_s = (qpu.t_readout_ns * 1e-9) if metrics.num_measurements > 0 else (qpu.t_readout_ns * 1e-9)
        t_reset_s = qpu.t_reset_ns * 1e-9
        
        t_per_shot_s = t_1q_s + t_2q_s + t_meas_s + t_reset_s
        t_per_shot_us = t_per_shot_s * 1e6

        # Total raw QPU execution time
        raw_qpu_sec = shots * t_per_shot_s
        
        # Superconducting microwave waveform playback & FPGA FIFO buffer latency
        control_latency = 0.5 + (shots * 1e-5)
        total_time_sec = qpu.job_init_overhead_sec + raw_qpu_sec + control_latency

        # Cost calculation based on IBM Quantum model
        cost_usd = 0.0
        cost_desc = ""
        cost_details = ""

        if qpu.pricing_model == PricingModelType.RUNTIME_PAY_AS_YOU_GO:
            billable_seconds = max(1.0, raw_qpu_sec)
            cost_usd = billable_seconds * qpu.price_per_runtime_sec
            cost_desc = f"${qpu.price_per_runtime_sec:.2f}/sec of QPU runtime"
            cost_details = f"{billable_seconds:.2f} billable secs × ${qpu.price_per_runtime_sec:.2f}/s = ${cost_usd:.2f} USD"

        elif qpu.pricing_model == PricingModelType.OPEN_PLAN_FREE:
            cost_usd = 0.0
            cost_desc = "IBM Quantum Open Plan (Included in 10 min/mo free allocation)"
            cost_details = f"{raw_qpu_sec:.2f} seconds consumed from free monthly quota (Balance: 600s max)"

        notes.append(f"Heavy-Hex Routing: ~{routing_factor:.1f}x 2Q gates estimated after transpilation.")

        return QPUEstimateResult(
            qpu_name=qpu.name,
            vendor=qpu.vendor,
            modality=qpu.modality.value,
            topology=qpu.topology.value,
            supported_qubits=qpu.total_qubits,
            is_qubit_count_supported=is_qubit_supported,
            t_single_shot_us=round(t_per_shot_us, 2),
            effective_gates_2q=effective_2q_gates,
            routing_overhead_factor=routing_factor,
            raw_qpu_time_sec=round(raw_qpu_sec, 3),
            control_batching_overhead_sec=round(control_latency, 2),
            total_estimated_time_sec=round(total_time_sec, 2),
            estimated_cost_usd=round(cost_usd, 3),
            pricing_model_desc=cost_desc,
            cost_details=cost_details,
            hardware_notes=notes
        )


# =====================================================================
# 6. Main Orchestrator: QuantumCircuitEstimator
# =====================================================================

@dataclass
class FullAnalysisReport:
    """Consolidated analysis report encompassing resources, simulation, and QPU estimates."""
    circuit_metrics: CircuitMetrics
    shots: int
    local_simulation: LocalSimulationReport
    qpu_estimates: List[QPUEstimateResult]

    def print_summary(self) -> None:
        """Print an executive, formatted summary table to stdout."""
        print(self.to_formatted_string())

    def to_formatted_string(self) -> str:
        """Render a clean ASCII report for terminal / logs."""
        lines = []
        lines.append("=" * 80)
        lines.append("        QISKIT QUANTUM CIRCUIT EXECUTION & RESOURCE ESTIMATOR")
        lines.append("=" * 80)
        
        # 1. Circuit Profile
        m = self.circuit_metrics
        lines.append("\n[1] CIRCUIT RESOURCE PROFILE")
        lines.append(f"  • Total Declared Qubits : {m.num_qubits}")
        lines.append(f"  • Active Qubits Used    : {m.active_qubits} (Idle: {m.num_qubits - m.active_qubits})")
        lines.append(f"  • Classical Bits        : {m.num_clbits}")
        lines.append(f"  • Total Circuit Depth   : {m.depth} (2Q Critical Depth: {m.depth_2q})")
        lines.append(f"  • Total Gate Count      : {m.total_gates} (1Q: {m.gates_1q}, 2Q Entangling: {m.gates_2q})")
        lines.append(f"  • Measurements / Resets : {m.num_measurements} / {m.num_resets}")
        lines.append(f"  • Clifford Status       : {'100% Clifford (Stabilizer)' if m.is_clifford_only else f'Non-Clifford (T-count: {m.t_count}, Total Non-Clifford: {m.non_clifford_count})'}")
        lines.append(f"  • Entangling Ratio      : {m.entangling_ratio * 100.0:.1f}%")
        
        # 2. Local Hardware & Classical Simulation
        hw = self.local_simulation.hardware
        lines.append("\n[2] LOCAL HARDWARE & SIMULATION FEASIBILITY")
        lines.append(f"  • Host System           : {hw.platform_os} | CPU: {hw.logical_cores} Cores ({hw.physical_cores} Physical)")
        lines.append(f"  • System RAM            : {hw.total_ram_gb:.2f} GB Total | {hw.available_ram_gb:.2f} GB Available")
        gpu_str = f"Yes ({hw.gpu_name}, {hw.gpu_vram_gb:.1f} GB VRAM)" if hw.gpu_available else "Not detected / CPU only"
        lines.append(f"  • GPU Acceleration      : {gpu_str}")
        lines.append(f"  • Simulation Verdict    : {self.local_simulation.summary_verdict}")
        
        lines.append("\n  Detailed Simulation Method Breakdown:")
        lines.append("  " + "-" * 76)
        lines.append(f"  {'Method':<24} | {'Feasibility':<18} | {'RAM Required':<14} | {'Est. Time':<10}")
        lines.append("  " + "-" * 76)
        for _, asm in self.local_simulation.assessments.items():
            ram_str = f"{asm.required_memory_gb:.3f} GB" if asm.required_memory_gb > 0 else "< 1 MB"
            time_str = f"{asm.estimated_sim_time_sec:.2f}s" if asm.estimated_sim_time_sec > 0 else "N/A"
            rec_tag = " ★ (BEST)" if asm.is_recommended else ""
            lines.append(f"  {asm.method_name[:24]:<24} | {asm.feasibility.value[:18]:<18} | {ram_str:<14} | {time_str:<10}{rec_tag}")
        lines.append("  " + "-" * 76)
        for note in self.local_simulation.guidance_notes:
            lines.append(f"  Guidance: {note}")

        # 3. IBM Quantum Execution & Cost Comparison
        lines.append(f"\n[3] IBM QUANTUM HARDWARE ESTIMATION (Shots: {self.shots:,})")
        lines.append("  " + "-" * 76)
        lines.append(f"  {'Target Processor':<24} | {'Raw QPU Time':<13} | {'Total Wall Time':<15} | {'Cost (USD)':<12}")
        lines.append("  " + "-" * 76)
        for res in self.qpu_estimates:
            name_str = f"{res.qpu_name}"
            if not res.is_qubit_count_supported:
                lines.append(f"  {name_str:<24} | {'UNSUPPORTED':<13} | {'(Too many qubits)':<15} | {'N/A':<12}")
                continue
            raw_t = f"{res.raw_qpu_time_sec:.3f} s"
            tot_t = f"{res.total_estimated_time_sec:.2f} s"
            cost_str = f"${res.estimated_cost_usd:.2f}" if res.estimated_cost_usd > 0 else "Free (Open Plan)"
            lines.append(f"  {name_str:<24} | {raw_t:<13} | {tot_t:<15} | {cost_str:<12}")
        lines.append("  " + "-" * 76)
        
        lines.append("\n  IBM Quantum Architectural Notes:")
        lines.append("  • IBM Heron: Tunable-coupler architecture with fast CZ gates (~60 ns) and low crosstalk.")
        lines.append("  • IBM Eagle: Fixed-frequency transmon with cross-resonance ECR gates (~350 ns). Available in Open Plan.")
        lines.append("  • Heavy-Hex Topology: Minimizes frequency collisions; requires SWAP routing for non-nearest-neighbor gates.")
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_markdown_table(self) -> str:
        """Export comparison table as Markdown format."""
        md = []
        md.append("| IBM Processor | Architecture | Qubits | Raw QPU Time | Total Job Time | Est. Cost | Qubits Fit? |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for res in self.qpu_estimates:
            fit_icon = "Yes" if res.is_qubit_count_supported else "No (Exceeds QPU)"
            raw_t = f"{res.raw_qpu_time_sec:.3f} s" if res.is_qubit_count_supported else "N/A"
            tot_t = f"{res.total_estimated_time_sec:.2f} s" if res.is_qubit_count_supported else "N/A"
            cost = f"${res.estimated_cost_usd:.2f}" if res.estimated_cost_usd > 0 else "Free (Open Plan)"
            if not res.is_qubit_count_supported:
                cost = "N/A"
            md.append(f"| **{res.qpu_name}** | {res.modality} | {res.supported_qubits} | {raw_t} | {tot_t} | {cost} | {fit_icon} |")
        return "\n".join(md)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire analysis report to a JSON-serializable dictionary."""
        return {
            "circuit_metrics": dataclasses.asdict(self.circuit_metrics),
            "shots": self.shots,
            "local_simulation": {
                "hardware": dataclasses.asdict(self.local_simulation.hardware),
                "assessments": {k: dataclasses.asdict(v) for k, v in self.local_simulation.assessments.items()},
                "can_simulate_locally": self.local_simulation.can_simulate_locally,
                "recommended_method": self.local_simulation.recommended_method,
                "summary_verdict": self.local_simulation.summary_verdict,
                "guidance_notes": self.local_simulation.guidance_notes
            },
            "qpu_estimates": [dataclasses.asdict(e) for e in self.qpu_estimates]
        }


class QuantumCircuitEstimator:
    """
    Main public interface for evaluating Quantum Circuits.
    Designed for seamless integration into Qiskit pipelines.
    
    Parameters
    ----------
    target_qpu : str or list of str or None, default="ibm_heron"
        Target IBM Quantum processor to evaluate. Defaults to 'ibm_heron' (Heron r2).
        Can be:
        - "ibm_heron" (or "heron", "heleron")
        - "ibm_eagle" (or "eagle")
        - "ibm_eagle_open" (or "open_plan", "open")
        - "ibm_falcon" (or "falcon")
        - "ibm_osprey" (or "osprey")
        - "all" or None (evaluates all registered IBM processors)
        - A list of processor keys, e.g. ["ibm_heron", "ibm_eagle"]
    custom_qpus : list of QPUProfile, optional
        Custom QPU profiles to register into the calculator.
        
    Example:
    >>> from qiskit_circuit_estimator import QuantumCircuitEstimator
    >>> # Default: Evaluates on IBM Heron
    >>> estimator = QuantumCircuitEstimator()
    >>> report = estimator.evaluate(circuit, shots=1000)
    >>> report.print_summary()
    >>>
    >>> # Specify IBM Eagle
    >>> eagle_estimator = QuantumCircuitEstimator(target_qpu="ibm_eagle")
    >>> eagle_report = eagle_estimator.evaluate(circuit, shots=1000)
    """

    def __init__(
        self,
        target_qpu: Union[str, List[str], None] = "ibm_heron",
        custom_qpus: Optional[List[QPUProfile]] = None
    ):
        self.qpus = dict(STANDARD_QPUS)
        if custom_qpus:
            for q in custom_qpus:
                key = q.name.lower().replace(" ", "_")
                self.qpus[key] = q
        
        self.default_target_qpu = target_qpu

    def _resolve_qpu_keys(self, target_spec: Union[str, List[str], None]) -> List[str]:
        """Resolve QPU names and aliases to canonical dictionary keys."""
        if target_spec is None or target_spec == "all":
            return list(self.qpus.keys())
        
        if isinstance(target_spec, str):
            target_list = [target_spec]
        else:
            target_list = list(target_spec)

        resolved_keys = []
        for item in target_list:
            cleaned = item.strip().lower().replace("-", "_").replace(" ", "_")
            canonical = QPU_ALIASES.get(cleaned, cleaned)
            if canonical in self.qpus:
                if canonical not in resolved_keys:
                    resolved_keys.append(canonical)
            else:
                # Fuzzy fallback matching
                matched = False
                for k in self.qpus.keys():
                    if cleaned in k or k in cleaned:
                        if k not in resolved_keys:
                            resolved_keys.append(k)
                        matched = True
                        break
                if not matched:
                    # Default fallback to ibm_heron if unrecognized
                    resolved_keys.append("ibm_heron")
                    
        return resolved_keys if resolved_keys else ["ibm_heron"]

    def evaluate(
        self,
        circuit: Any,
        shots: int = 1000,
        target_qpu: Union[str, List[str], None] = None
    ) -> FullAnalysisReport:
        """
        Perform circuit resource assessment, local classical simulation viability check,
        and execution runtime/cost estimation on the target IBM processor(s).
        
        Parameters
        ----------
        circuit : QuantumCircuit or duck-typed circuit
            The quantum circuit to evaluate.
        shots : int, default=1000
            Number of measurement shots.
        target_qpu : str or list of str or None, optional
            Overrides the default target QPU specified during __init__.
        """
        # 1. Analyze circuit resources
        metrics = CircuitAnalyzer.analyze(circuit)
        
        # 2. Inspect host machine & evaluate classical simulation
        sim_report = LocalHardwareInspector.evaluate_simulation_viability(metrics, shots=shots)
        
        # 3. Determine selected IBM processors
        chosen_spec = target_qpu if target_qpu is not None else self.default_target_qpu
        target_keys = self._resolve_qpu_keys(chosen_spec)
        selected_qpus = [self.qpus[k] for k in target_keys if k in self.qpus]

        qpu_results = [
            RuntimeAndCostCalculator.evaluate_qpu(metrics, qpu, shots=shots)
            for qpu in selected_qpus
        ]

        return FullAnalysisReport(
            circuit_metrics=metrics,
            shots=shots,
            local_simulation=sim_report,
            qpu_estimates=qpu_results
        )


# =====================================================================
# 7. Built-in Lightweight Quantum Circuit (Stand-Alone Execution)
# =====================================================================

class StandaloneCircuit:
    """
    A lightweight, zero-dependency circuit representation conforming
    to the Qiskit QuantumCircuit API. Allows running and testing this
    module even without `qiskit` installed in the local environment.
    """
    def __init__(self, num_qubits: int, num_clbits: int = 0, name: str = "qc"):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.name = name
        self.data: List[Any] = []
        self._ops_count: Dict[str, int] = {}
        self.qubits = list(range(num_qubits))

    def _add_gate(self, name: str, qubits: List[int]):
        class Instruction:
            def __init__(self, op_name: str, q_list: List[int]):
                self.operation = type("Op", (), {"name": op_name})()
                self.qubits = q_list
        self.data.append(Instruction(name, qubits))
        self._ops_count[name] = self._ops_count.get(name, 0) + 1

    def h(self, q: int): self._add_gate("h", [q]); return self
    def x(self, q: int): self._add_gate("x", [q]); return self
    def y(self, q: int): self._add_gate("y", [q]); return self
    def z(self, q: int): self._add_gate("z", [q]); return self
    def s(self, q: int): self._add_gate("s", [q]); return self
    def t(self, q: int): self._add_gate("t", [q]); return self
    def rz(self, theta: float, q: int): self._add_gate("rz", [q]); return self
    def rx(self, theta: float, q: int): self._add_gate("rx", [q]); return self
    def cx(self, q_ctrl: int, q_trgt: int): self._add_gate("cx", [q_ctrl, q_trgt]); return self
    def cz(self, q_ctrl: int, q_trgt: int): self._add_gate("cz", [q_ctrl, q_trgt]); return self
    def swap(self, q1: int, q2: int): self._add_gate("swap", [q1, q2]); return self
    def measure(self, q: int, c: int): self._add_gate("measure", [q]); return self
    def barrier(self): self._add_gate("barrier", list(range(self.num_qubits))); return self

    def count_ops(self) -> Dict[str, int]:
        return dict(self._ops_count)

    def depth(self) -> int:
        qubit_depths = [0] * self.num_qubits
        for item in self.data:
            q_list = item.qubits
            if not q_list:
                continue
            cur_max = max(qubit_depths[q] for q in q_list)
            new_depth = cur_max + 1
            for q in q_list:
                qubit_depths[q] = new_depth
        return max(qubit_depths) if qubit_depths else 0


# =====================================================================
# 8. Self-Testing & CLI Demonstration
# =====================================================================

def run_demonstration():
    """Run an automated demonstration with multiple quantum circuit archetypes."""
    print("Initializing QuantumCircuitEstimator Demo (IBM Quantum Edition)...\n")

    # 1. Default instance: uses IBM Heron
    print("--- [Demo 1] Default initialization: QuantumCircuitEstimator() (Targets IBM Heron) ---")
    estimator_default = QuantumCircuitEstimator()
    
    qcA = StandaloneCircuit(num_qubits=12, num_clbits=12)
    for i in range(12):
        qcA.h(i)
    for i in range(11):
        qcA.cx(i, i + 1)
        qcA.rz(0.35, i + 1)
    for i in range(12):
        qcA.measure(i, i)

    reportA = estimator_default.evaluate(qcA, shots=2000)
    reportA.print_summary()

    print("\n" + "#" * 80 + "\n")

    # 2. Explicit processor selection: IBM Eagle (Open Plan)
    print("--- [Demo 2] Explicit selection: QuantumCircuitEstimator(target_qpu='ibm_eagle_open') ---")
    estimator_eagle = QuantumCircuitEstimator(target_qpu="ibm_eagle_open")
    report_eagle = estimator_eagle.evaluate(qcA, shots=2000)
    report_eagle.print_summary()

    print("\n" + "#" * 80 + "\n")

    # 3. All IBM Processors comparative evaluation
    print("--- [Demo 3] Comparative Evaluation: QuantumCircuitEstimator(target_qpu='all') ---")
    estimator_all = QuantumCircuitEstimator(target_qpu="all")
    qcB = StandaloneCircuit(num_qubits=35, num_clbits=35)
    for i in range(35):
        qcB.h(i)
    for i in range(0, 34, 2):
        qcB.cx(i, i + 1)
        qcB.cz(i + 1, (i + 2) % 35)
    for i in range(35):
        qcB.measure(i, i)

    reportB = estimator_all.evaluate(qcB, shots=4000)
    reportB.print_summary()


if __name__ == "__main__":
    run_demonstration()