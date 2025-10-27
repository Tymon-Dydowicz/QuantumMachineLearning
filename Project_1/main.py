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
N_QUBITS = 4
TARGET_QUBIT = 0
EXECUTIONS = 3

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


def rotation_for_basis(basis, target_qubit=TARGET_QUBIT):
    r = QuantumCircuit(N_QUBITS)
    if basis == 'X':
        r.ry(np.pi/2, target_qubit)
        r.p(np.pi/2, target_qubit)
        r.h(target_qubit)
        return r
    if basis == 'Y':
        r.ry(np.pi/2, target_qubit)
        r.p(np.pi/2, target_qubit)
        r.sdg(target_qubit)
        r.h(target_qubit)
        return r
    if basis == 'Z':
        r.ry(np.pi/2, target_qubit)
        r.p(np.pi/2, target_qubit)
        return r

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_plot(fig, fname):
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

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

if __name__ == "__main__":
    np.random.seed(SEED)
    ensure_dir(OUTPUT_DIR)
    simulator = AerSimulator()

    prep_zero = QuantumCircuit(N_QUBITS)
    prep_x = QuantumCircuit(N_QUBITS)
    prep_x.x(TARGET_QUBIT)
    prep_h = QuantumCircuit(N_QUBITS)
    prep_h.h(TARGET_QUBIT)

    agg_z = collections.defaultdict(int)
    agg_x = collections.defaultdict(int)
    agg_h = collections.defaultdict(int)
    agg_tomo = {b: collections.defaultdict(int) for b in ['X', 'Y', 'Z']}

    z_runs = []
    x_runs = []
    h_runs = []
    tomo_runs = {b: [] for b in ['X', 'Y', 'Z']}

    for run_idx in range(1, EXECUTIONS + 1):
        run_seed = SEED + run_idx


        z_circ = prepare_and_measure(prep_zero, measure_rotation=None, measure_qubits=None)
        z_counts, z_probs = run_measurement(z_circ, simulator, shots=SHOTS, seed=run_seed)
        for k, v in z_counts.items():
            agg_z[k] += v
        z_runs.append(dict(z_counts))

        x_circ = prepare_and_measure(prep_x, measure_rotation=None, measure_qubits=None)
        x_counts, x_probs = run_measurement(x_circ, simulator, shots=SHOTS, seed=run_seed)
        for k, v in x_counts.items():
            agg_x[k] += v
        x_runs.append(dict(x_counts))

        h_circ_meas = prepare_and_measure(prep_h, measure_rotation=None, measure_qubits=None)
        h_counts, h_probs = run_measurement(h_circ_meas, simulator, shots=SHOTS, seed=run_seed)
        for k, v in h_counts.items():
            agg_h[k] += v
        h_runs.append(dict(h_counts))

        bases = ['X', 'Y', 'Z']
        for basis in bases:
            rot = rotation_for_basis(basis, target_qubit=TARGET_QUBIT)
            circ = prepare_and_measure(prep_zero, measure_rotation=rot, measure_qubits=None)
            counts_b, probs_b = run_measurement(circ, simulator, shots=SHOTS, seed=run_seed)
            for k, v in counts_b.items():
                agg_tomo[basis][k] += v
            tomo_runs[basis].append(dict(counts_b))


    ensure_dir(OUTPUT_DIR)

    plot_grouped_counts(z_runs, f"Z-measure |0000> — runs grouped (full state)", os.path.join(OUTPUT_DIR, "grouped_z_full"), shots=SHOTS)
    plot_grouped_counts(x_runs, f"After X on q{TARGET_QUBIT} — runs grouped (full state)", os.path.join(OUTPUT_DIR, "grouped_x_full"), shots=SHOTS)
    plot_grouped_counts(h_runs, f"After H on q{TARGET_QUBIT} — runs grouped (full state)", os.path.join(OUTPUT_DIR, "grouped_h_full"), shots=SHOTS)

    for basis in ['X', 'Y', 'Z']:
        plot_grouped_counts(tomo_runs[basis], f"Tomography {basis}-basis — runs grouped (full state)", os.path.join(OUTPUT_DIR, f"grouped_tomography_{basis.lower()}"), shots=SHOTS)

    vis_circs = {"zero": prep_zero, "x": prep_x, "h": prep_h}
    for name, circ in vis_circs.items():
        SV = StatevectorSimulator()
        result = SV.run(circ).result()
        psi = result.get_statevector()

        plot_state_city(psi, title=f"State City |{name}>", filename=os.path.join(OUTPUT_DIR, f"state_city_{name}.png"))
        plot_state_hinton(psi, title=f"State Hinton |{name}>", filename=os.path.join(OUTPUT_DIR, f"state_hinton_{name}.png"))
        plot_bloch_multivector(psi, title=f"State Bloch Multivector |{name}>", filename=os.path.join(OUTPUT_DIR, f"state_bloch_{name}.png"))
        plot_qsphere = plot_state_qsphere(psi)
        plot_qsphere.savefig(os.path.join(OUTPUT_DIR, f"state_qsphere_{name}.png"))


        