# Qiskit Circuit Hardware Estimator

Herramienta de análisis para evaluar la viabilidad de simulación local de circuitos cuánticos (CPU/GPU) y comparar tiempos de ejecución y fidelidad estimada frente a hardware cuántico (QPU) real de IBM Quantum.

## Características

* **Estimación de Memoria (RAM/VRAM):** Calcula los recursos necesarios para simular circuitos por vector de estado, con soporte para precisión `single` (complex64) o `double` (complex128).
* **Detección Automática de Hardware:** Identifica memoria RAM disponible (`psutil`) y VRAM de GPU (vía `PyTorch` o `nvidia-smi`), seleccionando automáticamente el dispositivo óptimo si `qiskit-aer-gpu` está disponible.
* **Proyección en Hardware Real:** Calcula el tiempo de ejecución físico y estima la fidelidad (ESP) basándose en las métricas de compuertas de 2 qubits y readout de las QPUs de IBM.
* **Actualización Dinámica de QPUs:** Incluye un script de *web scraping* basado en Playwright para mantener actualizada la base de datos de los procesadores de IBM Quantum activos.

## Archivos del Proyecto

* `qiskit_circuit_estimator.py`: Módulo principal que contiene la función `analyze_circuit_hardware` para evaluar los circuitos cuánticos.
* `actualizacion_data_QPU.py`: Script auxiliar que raspa la plataforma de IBM Cloud para generar/actualizar el archivo `data.csv` con el hardware activo actual.
* `data.csv`: Base de datos local (generada automáticamente) con la topología, métricas de error y estado de las QPUs.

## Instalación y Requisitos

Asegúrate de tener instalado Python 3.8+ y las siguientes dependencias:

```bash
pip install qiskit qiskit-aer pandas numpy matplotlib psutil