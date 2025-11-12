import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import collections
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere, plot_bloch_multivector

SHOTS = 1024
SEED = 151936
OUTPUT_DIR = "results/"
N_QUBITS = 2
TARGET_QUBIT = 0
EXECUTIONS = 20
QUANTUM_THRESHOLD = 2
BACKEND = AerSimulator()

def run_measurement(circ, simulator, shots=SHOTS, seed=SEED):
    transpiled = transpile(circ, simulator)
    result = simulator.run(transpiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    total = sum(counts.values()) if counts else shots
    probs = {k: v / total for k, v in counts.items()}
    return counts, probs

def prepare_and_measure(prep_circ, measure_rotation=None, measure_qubits=None):
    if measure_qubits is None:
        measure_qubits = list(range(N_QUBITS))
    circ = QuantumCircuit(N_QUBITS, len(measure_qubits))
    circ.compose(prep_circ, inplace=True)
    if measure_rotation is not None:
        circ.compose(measure_rotation, inplace=True)
    for i, q in enumerate(measure_qubits):
        circ.measure(q, i)
    return circ

def rotation_for_basis(basis, target_qubit=TARGET_QUBIT, withRYP=True):
    r = QuantumCircuit(N_QUBITS)
    if withRYP:
        r.ry(np.pi/2, target_qubit)
        r.p(np.pi/2, target_qubit)

    match basis:
        case 'X':
            r.h(target_qubit)
        case 'Y':
            r.sdg(target_qubit)
            r.h(target_qubit)
        case 'Z':
            pass
        case 'W':
            r.s(target_qubit)
            r.h(target_qubit)
            r.t(target_qubit)
            r.h(target_qubit)
        case 'V':
            r.s(target_qubit)
            r.h(target_qubit)
            r.tdg(target_qubit)
            r.h(target_qubit)
    
    return r

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_grouped_counts(counts_per_run, title, fname, shots=SHOTS):
    ensure_dir(os.path.dirname(fname) or ".")
    if not counts_per_run or all(not c for c in counts_per_run):
        fig = plt.figure(figsize=(6,4))
        plt.text(0.5,0.5,"no data", ha='center')
        fig.savefig(fname + ".png", bbox_inches='tight')
        plt.close(fig)
        return

    all_states = sorted({s for c in counts_per_run for s in c.keys()}, key=lambda b: int(b, 2))
    n_states = len(all_states)
    n_runs = len(counts_per_run)

    data = np.zeros((n_runs, n_states), dtype=int)
    for i, cdict in enumerate(counts_per_run):
        for j, state in enumerate(all_states):
            data[i, j] = int(cdict.get(state, 0))

    ind = np.arange(n_states)
    width = 0.8 / max(1, n_runs)

    fig_width = max(10, n_states * 0.5)
    fig, (ax_counts, ax_probs) = plt.subplots(1, 2, figsize=(fig_width, 5), gridspec_kw={'width_ratios':[1,1]})

    colors = plt.colormaps.get_cmap('tab10').colors
    bars_list = []
    for i in range(n_runs):
        positions = ind - 0.4 + width/2 + i * width
        bars = ax_counts.bar(positions, data[i], width, label=f"Run {i+1}", color=colors[i % len(colors)])
        bars_list.append(bars)
        for rect, val in zip(bars, data[i]):
            height = rect.get_height()
            ax_counts.text(rect.get_x() + rect.get_width()/2.0, height + max(1, 0.01 * max(data.max(), 1)),
                           f"{int(val)}", ha='center', va='bottom', fontsize=8, rotation=0)

    ax_counts.set_xticks(ind)
    ax_counts.set_xticklabels(all_states, rotation=90)
    ax_counts.set_xlabel("quantum state")
    ax_counts.set_ylabel("counts")
    ax_counts.set_title(title + " — counts")
    ax_counts.legend()

    for i in range(n_runs):
        positions = ind - 0.4 + width/2 + i * width
        probs = data[i] / float(shots)
        bars_p = ax_probs.bar(positions, probs, width, label=f"Run {i+1}", color=colors[i % len(colors)])
        for rect, p in zip(bars_p, probs):
            height = rect.get_height()
            ax_probs.text(rect.get_x() + rect.get_width()/2.0, height + 0.005,
                          f"{p:.3f}", ha='center', va='bottom', fontsize=8, rotation=0)

    ax_probs.set_xticks(ind)
    ax_probs.set_xticklabels(all_states, rotation=90)
    ax_probs.set_xlabel("quantum state")
    ax_probs.set_ylabel("probability")
    ax_probs.set_title(title + " — probability")
    ax_probs.legend()

    fig.tight_layout()
    fig.savefig(fname + ".png", bbox_inches='tight')
    plt.close(fig)

def twoBasisMeasurement(circ, bases, target_qubits=TARGET_QUBIT):
    r = QuantumCircuit(N_QUBITS)
    for b, q in zip(bases[::-1], target_qubits):
        print(f"Applying rotation for basis {b} on qubit {q}")
        r.compose(rotation_for_basis(b, q, withRYP=False), inplace=True)
    circ_meas = prepare_and_measure(circ, measure_rotation=r, measure_qubits=target_qubits)
    return circ_meas


def createBellStatePsiMinus():
    circ = QuantumCircuit(N_QUBITS)
    circ.x(0)
    circ.h(0)
    circ.x(1)
    circ.cx(0, 1)
    return circ

def runCHSHExperiment(seed=SEED):
    bellCirc = createBellStatePsiMinus()
    measurements = ['XW', 'XV', 'ZW', 'ZV']
    counts_per_run = {}
    probs_per_run = {}
    expectations_per_run = {}

    for meas in measurements:
        meas_circ = twoBasisMeasurement(bellCirc, meas, target_qubits=[0, 1])
        counts, probs = run_measurement(meas_circ, BACKEND, seed=seed)
        counts_per_run[meas] = counts
        probs_per_run[meas] = probs

        table_rows = []
        E = 0.0

        for bitstring, p in probs.items():
            bits = bitstring[::-1]
            x = 1 if bits[0] == '0' else -1
            y = 1 if bits[1] == '0' else -1
            product_xy = x * y
            weighted = product_xy * p
            E += weighted

            table_rows.append({
                "xy": bitstring,
                "Pdim1(x)": x,
                "Pdim2(y)": y,
                "Pdim1(x)*Pdim2(y)": product_xy,
                "p(x,y)": round(p, 4),
                "Pdim1(x)*Pdim2(y)*p(x,y)": round(weighted, 4)
            })

        expectations_per_run[meas] = E
        df = pd.DataFrame(table_rows).sort_values("xy", ascending=True)
        print(f"\n=== Measurement: {meas} ===")
        print(df.to_string(index=False))
        print(f"Expectation value E({meas}) = {E:.4f}\n")

        df.to_csv(os.path.join(OUTPUT_DIR, f"table_{meas}.csv"), index=False)

    for meas in measurements:
        plot_grouped_counts([counts_per_run[meas]], f"Measurement in bases {meas}", 
                            os.path.join(OUTPUT_DIR, f"task1_measurement_{meas}"), shots=SHOTS)
        

    S = expectations_per_run['XW'] - expectations_per_run['XV'] + expectations_per_run['ZW'] + expectations_per_run['ZV']

    print("=== Summary ===")
    for meas, E in expectations_per_run.items():
        print(f"{meas}: E = {E:.4f}")
    print(f"\nCHSH parameter S = {S:.4f}")
    if abs(S) > QUANTUM_THRESHOLD:
        print("The CHSH inequality is violated, indicating quantum entanglement.")

    return expectations_per_run, S

def task1():
    ensure_dir(OUTPUT_DIR)
    expectations, S = runCHSHExperiment()

def task2():
    SValues = []
    for exec_num in range(EXECUTIONS):
        seed_run = SEED + exec_num * 12345
        print(f"\n--- Execution {exec_num + 1} ---")
        _, S = runCHSHExperiment(seed=seed_run)
        SValues.append(S)

    S_arr = np.array(SValues)
    mean_S = float(S_arr.mean())
    std_S = float(S_arr.std(ddof=1)) if EXECUTIONS > 1 else 0.0
    stderr_S = std_S / np.sqrt(EXECUTIONS) if EXECUTIONS > 0 else 0.0

    print("\n=== Summary over executions ===")
    print(f"S values: {['{:.6f}'.format(s) for s in SValues]}")
    print(f"Mean S = {mean_S:.6f}")
    print(f"Sample standard deviation (std) = {std_S:.6f}")
    print(f"Standard error of the mean (std / sqrt(N)) = {stderr_S:.6f}")
    print(f"N executions = {EXECUTIONS}")

    df_S = pd.DataFrame({"S": SValues})
    df_S.to_csv(os.path.join(OUTPUT_DIR, "SValues.csv"), index=False)

    plt.figure(figsize=(6,4))
    plt.boxplot(SValues, vert=True, patch_artist=True, labels=['S'],
                boxprops=dict(facecolor='C0', color='black'),
                medianprops=dict(color='firebrick'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='C0', markersize=5, markeredgecolor='black'))
    plt.ylabel('S')
    plt.title(f'Boxplot of S over {EXECUTIONS} executions\nmean={mean_S:.4f}, std={std_S:.4f}')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "S_boxplot.png"), bbox_inches='tight')
    plt.close()

def main():
    task1()
    task2()

if __name__ == "__main__":
    main()