from qiskit import QuantumCircuit
from qiskit_circuit_estimator import analyze_circuit_hardware
# 1. Define a test quantum circuit
qc = QuantumCircuit(4)
qc.h(0)
for i in range(3):
    qc.cx(i, i+1)
qc.measure_all()

results = analyze_circuit_hardware(
    circuit=qc,
    csv_path='data.csv',
    shots=1024,
    precision='single',  # Halves the required RAM and VRAM
    target_device='auto',
    show_plots=True,
    show_cost=True,      # cost per QPU (optional)
    cost_plan='payg'     # 'open', 'payg' ($96/min), 'flex' ($72/min) or 'premium' ($48/min)
)

print(f"Local simulation possible?: {results['local_feasible']}")
print(f"Device used: {results['device_used']}")
print(f"Simulation method: {results['sim_method']} (Clifford circuit: {results['is_clifford']})")
print(f"Memory by method (GB): {results['mem_by_method_gb']}")
print(f"Precision: {results['precision']}")
print(f"Required RAM: {results['ram_required_gb']} GB")
print(f"Required VRAM: {results['vram_required_gb']} GB")
print(f"Available RAM: {results['ram_available_gb']} GB")
print(f"Available VRAM: {results['vram_available_gb']} GB")

print("QPU comparison table:")
print(f"Logical circuit: {results['logical_metrics']}")
print(results['qpu_summary'][['QPU', 'Type', 'Topology_Source', 'Transpiled_2Q', 'Layers_2Q',
                              'MCPS', 'QPU_Time_s', 'Estimated_Fidelity', 'Estimated_Cost_USD']].to_string(index=False))

print("\nLocally obtained shot distribution:")
print(results['counts'])
