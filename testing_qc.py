from qiskit import QuantumCircuit
from qiskit_circuit_estimator import QuantumCircuitEstimator

# 1. Definir o transpolar el circuito
qc = QuantumCircuit(15, 15)
for i in range(15):
    qc.h(i)
for i in range(14):
    qc.cx(i, i + 1)
qc.measure_all()

# 2. Inicializar la calculadora y evaluar
# Evaluar en el plan abierto gratuito (IBM Eagle Open Plan)
estimator_eagle = QuantumCircuitEstimator()
report = estimator_eagle.evaluate(qc, shots=2000)
report.print_summary()


# 4. También puedes exportar en Markdown, dict o JSON
print(report.to_markdown_table())
data = report.to_dict()
