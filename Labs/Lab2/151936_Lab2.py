import numpy as np
import os
import matplotlib.pyplot as plt
import collections
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere, plot_bloch_multivector

SHOTS = 2048
SEED = 151936
OUTPUT_DIR = "results/"
N_QUBITS = 2
TARGET_QUBIT = 0
EXECUTIONS = 3
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

    if basis == 'X':
        r.h(target_qubit)
    if basis == 'Y':
        r.sdg(target_qubit)
        r.h(target_qubit)
    if basis == 'Z':
        pass
    return r
    
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

def plot_grouped_bell_counts(all_counts, basis, fname_prefix, shots=SHOTS):
    ensure_dir(os.path.dirname(fname_prefix) or ".")
    bell_names = list(all_counts.keys())
    n_states = 0

    all_states = sorted({
        s for counts_list in all_counts.values() for cdict in counts_list for s in cdict.keys()
    }, key=lambda b: int(b, 2))
    n_states = len(all_states)
    ind = np.arange(n_states)
    n_bells = len(bell_names)
    colors = plt.colormaps.get_cmap('tab10').colors

    fig_c, axs_c = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axs_c = axs_c.flatten()
    if n_bells == 1:
        axs_c = [axs_c]

    for i, bell_name in enumerate(bell_names):
        ax = axs_c[i]
        counts_per_run = all_counts[bell_name]
        n_runs = len(counts_per_run)
        width = 0.8 / max(1, n_runs)

        data = np.zeros((n_runs, n_states), dtype=int)
        for r, cdict in enumerate(counts_per_run):
            for j, state in enumerate(all_states):
                data[r, j] = int(cdict.get(state, 0))

        for r in range(n_runs):
            pos = ind - 0.4 + width/2 + r*width
            bars = ax.bar(pos, data[r], width, color=colors[r % len(colors)], label=f"Run {r+1}")

            for rect in bars:
                height = rect.get_height()
                if height > 0:
                    ax.text(
                        rect.get_x() + rect.get_width()/2.0,
                        height + max(1, 0.01 * height),
                        f"{int(height)}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=0
                    )

        axs_c[i].set_title(bell_name)
        axs_c[i].set_xticks(ind)
        axs_c[i].set_xticklabels(all_states, rotation=90)
        axs_c[i].set_xlabel("Quantum state")
        if i == 0:
            axs_c[i].set_ylabel("Counts")

    axs_c[0].legend()
    fig_c.suptitle(f"Bell State Measurements in {basis} basis — Counts", fontsize=14)
    fig_c.tight_layout(rect=[0, 0, 1, 0.95])
    fig_c.savefig(fname_prefix + "_counts.png", bbox_inches='tight')
    plt.close(fig_c)

    fig_p, axs_p = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    axs_p = axs_p.flatten()
    if n_bells == 1:
        axs_p = [axs_p]

    for i, bell_name in enumerate(bell_names):
        counts_per_run = all_counts[bell_name]
        n_runs = len(counts_per_run)
        width = 0.8 / max(1, n_runs)

        data = np.zeros((n_runs, n_states), dtype=int)
        for r, cdict in enumerate(counts_per_run):
            for j, state in enumerate(all_states):
                data[r, j] = int(cdict.get(state, 0))

        probs_data = data / float(shots)
        for r in range(n_runs):
            pos = ind - 0.4 + width/2 + r*width
            axs_p[i].bar(pos, probs_data[r], width, color=colors[r % len(colors)], label=f"Run {r+1}")

        axs_p[i].set_title(bell_name)
        axs_p[i].set_xticks(ind)
        axs_p[i].set_xticklabels(all_states, rotation=90)
        axs_p[i].set_xlabel("Quantum state")
        if i == 0:
            axs_p[i].set_ylabel("Probability")

    axs_p[0].legend()
    fig_p.suptitle(f"Bell State Measurements in {basis} basis — Probabilities", fontsize=14)
    fig_p.tight_layout(rect=[0, 0, 1, 0.95])
    fig_p.savefig(fname_prefix + "_probabilities.png", bbox_inches='tight')
    plt.close(fig_p)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def createBellStateFiPlus():
    circ = QuantumCircuit(N_QUBITS)
    circ.h(0)
    circ.cx(0, 1)
    return circ

def createBellStateFiMinus():
    circ = QuantumCircuit(N_QUBITS)
    circ.x(0)
    circ.h(0)
    circ.cx(0, 1)
    return circ

def createBellStatePsiPlus():
    circ = QuantumCircuit(N_QUBITS)
    circ.h(0)
    circ.x(1)
    circ.cx(0, 1)
    return circ

def createBellStatePsiMinus():
    circ = QuantumCircuit(N_QUBITS)
    circ.x(0)
    circ.h(0)
    circ.x(1)
    circ.cx(0, 1)
    return circ

def twoBasisMeasurement(circ, bases, target_qubits=TARGET_QUBIT):
    r = QuantumCircuit(N_QUBITS)
    for b, q in zip(bases[::-1], target_qubits):
        print(f"Applying rotation for basis {b} on qubit {q}")
        r.compose(rotation_for_basis(b, q, withRYP=False), inplace=True)
    circ_meas = prepare_and_measure(circ, measure_rotation=r, measure_qubits=target_qubits)
    return circ_meas

def task1():
    prep_circ = createBellStateFiPlus()
    circ = prepare_and_measure(prep_circ)
    print(circ.draw())

    counts_per_run = []
    for run in range(EXECUTIONS):
        counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
        counts_per_run.append(counts)
        print(f"Run {run+1} counts: {counts}")
        print(f"Run {run+1} probabilities: {probs}")

    plot_grouped_counts(counts_per_run, "Bell State Measurement(Fi+)", os.path.join(OUTPUT_DIR, "bell_state_measurement(Fi+)"), shots=SHOTS)

def task2():
    prep_circ = createBellStateFiMinus()
    circ = prepare_and_measure(prep_circ)
    print(circ.draw())

    counts_per_run = []
    for run in range(EXECUTIONS):
        counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
        counts_per_run.append(counts)
        print(f"Run {run+1} counts: {counts}")
        print(f"Run {run+1} probabilities: {probs}")

    plot_grouped_counts(counts_per_run, "Bell State Measurement(Fi-)", os.path.join(OUTPUT_DIR, "bell_state_measurement(Fi-)"), shots=SHOTS)

def task3():
    prep_circ = createBellStatePsiPlus()
    circ = prepare_and_measure(prep_circ)
    print(circ.draw())
    counts_per_run = []
    for run in range(EXECUTIONS):
        counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
        counts_per_run.append(counts)
        print(f"Run {run+1} counts: {counts}")
        print(f"Run {run+1} probabilities: {probs}")
        
    plot_grouped_counts(counts_per_run, "Bell State Measurement(Psi+)", os.path.join(OUTPUT_DIR, "bell_state_measurement(Psi+)"), shots=SHOTS)

def task4():
    prep_circ = createBellStatePsiMinus()
    circ = prepare_and_measure(prep_circ)
    print(circ.draw())
    counts_per_run = []
    for run in range(EXECUTIONS):
        counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
        counts_per_run.append(counts)
        print(f"Run {run+1} counts: {counts}")
        print(f"Run {run+1} probabilities: {probs}")
        
    plot_grouped_counts(counts_per_run, "Bell State Measurement(Psi-)", os.path.join(OUTPUT_DIR, "bell_state_measurement(Psi-)"), shots=SHOTS)

def task5():
    bases = 'XX'
    bell_creators = {
        "Fi+": createBellStateFiPlus,
        "Fi-": createBellStateFiMinus,
        "Psi+": createBellStatePsiPlus,
        "Psi-": createBellStatePsiMinus,
    }
    all_counts = {}

    for name, creator in bell_creators.items():
        prep_circ = creator()
        circ = twoBasisMeasurement(prep_circ, bases, target_qubits=[0, 1])
        print(f"\n{name} — measurement in {''.join(bases)} basis:")
        print(circ.draw())

        counts_per_run = []
        for run in range(EXECUTIONS):
            counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
            counts_per_run.append(counts)
            print(f"Run {run+1} counts: {counts}")
        all_counts[name] = counts_per_run

    plot_grouped_bell_counts(all_counts, ''.join(bases), os.path.join(OUTPUT_DIR, f"bell_measurements_{''.join(bases)}"))

def task6():
    bases = 'YY'
    bell_creators = {
        "Fi+": createBellStateFiPlus,
        "Fi-": createBellStateFiMinus,
        "Psi+": createBellStatePsiPlus,
        "Psi-": createBellStatePsiMinus,
    }
    all_counts = {}

    for name, creator in bell_creators.items():
        prep_circ = creator()
        circ = twoBasisMeasurement(prep_circ, bases, target_qubits=[0, 1])
        print(f"\n{name} — measurement in {''.join(bases)} basis:")
        print(circ.draw())

        counts_per_run = []
        for run in range(EXECUTIONS):
            counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
            counts_per_run.append(counts)
            print(f"Run {run+1} counts: {counts}")
        all_counts[name] = counts_per_run

    plot_grouped_bell_counts(all_counts, ''.join(bases), os.path.join(OUTPUT_DIR, f"bell_measurements_{''.join(bases)}"))

def task7():
    bases = 'XZ'
    bell_creators = {
        "Fi+": createBellStateFiPlus,
        "Fi-": createBellStateFiMinus,
        "Psi+": createBellStatePsiPlus,
        "Psi-": createBellStatePsiMinus,
    }
    all_counts = {}

    for name, creator in bell_creators.items():
        prep_circ = creator()
        circ = twoBasisMeasurement(prep_circ, bases, target_qubits=[0, 1])
        print(f"\n{name} — measurement in {''.join(bases)} basis:")
        print(circ.draw())

        counts_per_run = []
        for run in range(EXECUTIONS):
            counts, probs = run_measurement(circ, BACKEND, shots=SHOTS, seed=SEED + run)
            counts_per_run.append(counts)
            print(f"Run {run+1} counts: {counts}")
        all_counts[name] = counts_per_run

    plot_grouped_bell_counts(all_counts, ''.join(bases), os.path.join(OUTPUT_DIR, f"bell_measurements_{''.join(bases)}"))



def main():
    np.random.seed(SEED)

    ensure_dir(OUTPUT_DIR)
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    task7()


if __name__ == "__main__":
    main()