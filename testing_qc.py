from qiskit import QuantumCircuit 
from qiskit_circuit_estimator import analyze_circuit_hardware
# 1. Definir un circuito cuántico de prueba
qc = QuantumCircuit(4)
qc.h(0)
for i in range(3):
    qc.cx(i, i+1)
qc.measure_all()

resultados = analyze_circuit_hardware(
    circuit=qc,
    csv_path='data.csv',
    shots=1024,
    precision='single',  # Reduce la RAM y VRAM requerida al 50%
    target_device='auto',
    show_plots=True
)

print(f"¿Posible local?: {resultados['es_posible_local']}")
print(f"Dispositivo utilizado: {resultados['dispositivo_usado']}")
print(f"Precisión: {resultados['precision']}")
print(f"Memoria RAM requerida: {resultados['ram_requerida_gb']} GB")
print(f"Memoria VRAM requerida: {resultados['vram_requerida_gb']} GB")
print(f"Memoria RAM disponible: {resultados['ram_disponible_gb']} GB")
print(f"Memoria VRAM disponible: {resultados['vram_disponible_gb']} GB")

print("Tabla comparativa de QPUs:")
print(resultados['resumen_qpus'][['QPU', 'Tipo', 'MCPS', 'Tiempo_QPU_s', 'Fidelidad_Estimada']])

print("\nDistribución de los shots obtenidos localmente:")
print(resultados['distribucion_shots'])