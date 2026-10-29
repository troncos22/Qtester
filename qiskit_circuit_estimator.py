import re
import shutil
import subprocess
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

def _parse_mcps_to_hz(val: str) -> float:
    """Convierte frecuencias ('109.7 kHz', '250 Hz') a Hz numérico."""
    s = str(val).strip().lower()
    match = re.search(r"([0-9.]+)\s*([a-z]*)", s)
    if not match:
        return np.nan
    num, unit = float(match.group(1)), match.group(2)
    multipliers = {'mhz': 1e6, 'khz': 1e3, 'hz': 1.0}
    return num * multipliers.get(unit, 1.0)

def _get_gpu_memory_info() -> dict:
    """Consulta VRAM libre y total (en GB) vía PyTorch o nvidia-smi."""
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
    precision: str = 'double',  # 'single' (complex64: 8 bytes) o 'double' (complex128: 16 bytes)
    target_device: str = 'auto',  # 'auto', 'GPU' o 'CPU'
    only_online: bool = True,
    show_plots: bool = True,
    max_parallel_threads: int = None,
    max_parallel_experiments: int = None,
    max_parallel_shots: int = None,
) -> dict:
    """
    Evalúa la viabilidad local (RAM/VRAM en precisión single/double) y compara con QPUs.
    """
    # 1. Normalizar y validar precisión numérica
    prec = str(precision).strip().lower()
    if prec not in ['single', 'double']:
        raise ValueError("El parámetro 'precision' debe ser 'single' o 'double'.")

    # complex64 = 8 bytes por amplitud | complex128 = 16 bytes por amplitud
    bytes_per_amplitude = 8 if prec == 'single' else 16

    # 2. Cargar y preparar base de datos de QPUs
    df = pd.read_csv(csv_path)
    df['MCPS_Hz'] = df['MCPS'].apply(_parse_mcps_to_hz)
    df['Online_Bool'] = df['Estado'].str.strip().str.lower().str.startswith('online')
    df_qpu = df[df['Online_Bool']].copy() if only_online else df.copy()

    # 3. Requerimientos de Memoria (RAM / VRAM)
    n_qubits = circuit.num_qubits
    bytes_statevector = (2 ** n_qubits) * bytes_per_amplitude
    mem_req_gb = (bytes_statevector / (1024 ** 3)) * 1.2  # 20% margen para buffers

    # Diagnóstico de CPU
    ram_local = psutil.virtual_memory()
    ram_disp_gb = ram_local.available / (1024 ** 3)
    es_posible_cpu = mem_req_gb <= ram_disp_gb

    # Diagnóstico de GPU
    gpu_info = _get_gpu_memory_info()
    vram_disp_gb = gpu_info["vram_free_gb"]
    es_posible_gpu_hw = gpu_info["detected"] and (mem_req_gb <= vram_disp_gb)

    # 4. Selección del backend y dispositivo de cómputo
    dispositivo_elegido = None
    aer_gpu_soportada = False

    try:
        from qiskit_aer import AerSimulator
        aer_gpu_soportada = 'GPU' in AerSimulator().available_devices()
    except ImportError:
        pass

    es_posible_gpu = es_posible_gpu_hw and aer_gpu_soportada

    if target_device.upper() == 'GPU':
        es_posible_local = es_posible_gpu
        dispositivo_elegido = 'GPU' if es_posible_local else None
    elif target_device.upper() == 'CPU':
        es_posible_local = es_posible_cpu
        dispositivo_elegido = 'CPU' if es_posible_local else None
    else:  # 'auto'
        if es_posible_gpu:
            es_posible_local = True
            dispositivo_elegido = 'GPU'
        elif es_posible_cpu:
            es_posible_local = True
            dispositivo_elegido = 'CPU'
        else:
            es_posible_local = False
            dispositivo_elegido = None

    # 5. Métricas del Circuito para Hardware Cuántico
    ops = circuit.count_ops()
    n_2q = sum(ops.get(g, 0) for g in ['cx', 'ecr', 'cz', 'swap'])
    n_measures = ops.get('measure', n_qubits)

    resultados_qpu = []
    for _, row in df_qpu.iterrows():
        compatible = (n_qubits <= row['Qubits'])
        t_estimado_s = (shots / row['MCPS_Hz']) if (compatible and row['MCPS_Hz'] > 0) else np.nan

        err_2q = row.get('2Q Error (Layered)', row.get('2Q Error (Best)', 0.0))
        err_ro = row.get('Readout Error', 0.0)
        fidelidad_esp = ((1.0 - err_2q) ** n_2q) * ((1.0 - err_ro) ** n_measures) if compatible else np.nan

        resultados_qpu.append({
            'QPU': row['QPU'],
            'Tipo': row['Tipo'],
            'Qubits_Chip': row['Qubits'],
            'Compatible': compatible,
            'MCPS': row['MCPS'],
            'Tiempo_QPU_s': round(t_estimado_s, 6) if not np.isnan(t_estimado_s) else None,
            'Fidelidad_Estimada': round(fidelidad_esp, 4) if not np.isnan(fidelidad_esp) else None,
            'Estado': row['Estado'].replace('\n', ' ')
        })

    df_resultado = pd.DataFrame(resultados_qpu).sort_values(by='Tiempo_QPU_s', ascending=True)

    # 6. Simulación y Distribución de Shots
    distribucion_shots = {}
    fig_shots = None

    if es_posible_local:
        try:
            from qiskit_aer import AerSimulator
            sim_options = {
                'device': dispositivo_elegido,
                'precision': prec
            }
            if max_parallel_threads is not None:
                sim_options['max_parallel_threads'] = max_parallel_threads
            if max_parallel_experiments is not None:
                sim_options['max_parallel_experiments'] = max_parallel_experiments
            if max_parallel_shots is not None:
                sim_options['max_parallel_shots'] = max_parallel_shots

            sim = AerSimulator(**sim_options)
            circ_sim = circuit.copy()
            if not any(inst.operation.name == 'measure' for inst in circ_sim.data):
                circ_sim.measure_all()

            job = sim.run(circ_sim, shots=shots, seed_simulator=seed_sim)
            distribucion_shots = job.result().get_counts()
        except Exception:
            from qiskit.primitives import StatevectorSampler
            circ_sim = circuit.copy()
            if not any(inst.operation.name == 'measure' for inst in circ_sim.data):
                circ_sim.measure_all()
            sampler = StatevectorSampler()
            job = sampler.run([circ_sim], shots=shots)
            distribucion_shots = job.result()[0].data.meas.get_counts()

        if show_plots and distribucion_shots:
            fig_shots = plot_histogram(
                distribucion_shots,
                title=f"Distribución de {shots} Shots ({dispositivo_elegido} - {prec})"
            )
    else:
        print(f"Simulación local NO viable ({prec}): Requiere {mem_req_gb:.2f} GB | "
              f"RAM libre: {ram_disp_gb:.2f} GB | VRAM libre: {vram_disp_gb:.2f} GB")

    # 7. Gráficos Comparativos
    fig_hw, (ax_hw, ax_qpu) = plt.subplots(1, 2, figsize=(15, 5))

    etiquetas = [f'Requerida\n({prec})', 'RAM Disp.\n(CPU)']
    valores = [mem_req_gb, ram_disp_gb]
    colores = ['#e67e22', '#3498db']

    if gpu_info["detected"]:
        etiquetas.append(f"VRAM Disp.\n({gpu_info['name'][:10]}...)")
        valores.append(vram_disp_gb)
        colores.append('#2ecc71' if es_posible_gpu else '#95a5a6')

    barras = ax_hw.bar(etiquetas, valores, color=colores, width=0.45)
    ax_hw.set_ylabel("Gigabytes (GB)")
    estado_txt = f"POSIBLE ({dispositivo_elegido})" if es_posible_local else "INSUFICIENTE"
    ax_hw.set_title(f"Viabilidad ({n_qubits} Qs, {prec}): {estado_txt}")
    for b in barras:
        h = b.get_height()
        ax_hw.annotate(f"{h:.3f} GB" if h < 1 else f"{h:.2f} GB",
                       xy=(b.get_x() + b.get_width() / 2, h),
                       xytext=(0, 4), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')

    df_validos = df_resultado[df_resultado['Tiempo_QPU_s'].notnull()]
    if not df_validos.empty:
        barras_qpu = ax_qpu.barh(df_validos['QPU'], df_validos['Tiempo_QPU_s'], color='#9b59b6')
        ax_qpu.set_xlabel("Tiempo estimado de ejecución (segundos)")
        ax_qpu.set_title(f"Tiempo de Ejecución por QPU ({shots} shots)")
        for b in barras_qpu:
            w = b.get_width()
            ax_qpu.annotate(f"{w:.4f} s",
                            xy=(w, b.get_y() + b.get_height() / 2),
                            xytext=(5, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=9)
    else:
        ax_qpu.text(0.5, 0.5, "Ninguna QPU compatible activa", ha='center', va='center')

    plt.tight_layout()
    if show_plots:
        plt.show()

    # Se retornan explícitamente ambas claves para compatibilidad total con scripts
    return {
        "es_posible_local": es_posible_local,
        "es_posible_cpu": es_posible_cpu,
        "es_posible_gpu": es_posible_gpu,
        "dispositivo_usado": dispositivo_elegido,
        "precision": prec,
        "gpu_detectada": gpu_info["name"] if gpu_info["detected"] else None,
        "ram_requerida_gb": round(mem_req_gb, 4),
        "vram_requerida_gb": round(mem_req_gb, 4),
        "ram_disponible_gb": round(ram_disp_gb, 4),
        "vram_disponible_gb": round(vram_disp_gb, 4),
        "resumen_qpus": df_resultado,
        "distribucion_shots": distribucion_shots,
        "fig_hardware": fig_hw,
        "fig_shots": fig_shots
    }