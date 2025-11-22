import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import collections
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere, plot_bloch_multivector

SHOTS = 1024
SEED = 151936
OUTPUT_DIR = "results/"
SAMPLES_TO_TEST = [16, 32, 64, 128, 256, 512, 1024]
EXECUTIONS = 5
BACKEND = AerSimulator()
N_QUBITS = 4

def createBB84Circuit():
    q0 = QuantumRegister(N_QUBITS, 'q')
    c0 = ClassicalRegister(N_QUBITS, 'c')

    all_qubits = [q0[i] for i in range(N_QUBITS)]

    circ = QuantumCircuit(q0, c0)
    circ.reset(all_qubits)
    circ.h(q0[1])
    circ.measure(q0[1], c0[1])
    circ.h(q0[2])
    circ.measure(q0[2], c0[2])
    circ.barrier(all_qubits)

    circ.cx(q0[1], q0[0])
    circ.ch(q0[2], q0[0])
    circ.barrier(all_qubits)

    circ.h(q0[3])
    circ.measure(q0[3], c0[3])
    circ.barrier(all_qubits)

    circ.ch(q0[3], q0[0])
    circ.measure(q0[0], c0[0])
    circ.barrier(all_qubits)

    return circ

def testSample(circ, sample, verbose=False):
    bits = []
    for _ in range(sample):
        compiled_circuit = transpile(circ, BACKEND)
        job_sim = BACKEND.run(compiled_circuit, shots=1)
        sim_result = job_sim.result()
        counts = sim_result.get_counts(circ)

        xA = int(list(counts.keys())[0][2])
        yA = int(list(counts.keys())[0][1])
        yB = int(list(counts.keys())[0][0])
        xB = int(list(counts.keys())[0][3])

        if verbose:
            print(f"Alice bits: xA={xA}, yA={yA} | Bob bits: yB={yB}, xB={xB}")
            print([xA, yA, yB, xB])
        bits.append([xA, yA, yB, xB])

    return bits

def siftKey(bits):
    keyA = []
    keyB = []
    for bit in bits:
        xA, yA, yB, xB = bit
        if yA == yB:
            keyA.append(xA)
            keyB.append(xB)
    return keyA, keyB

def runExperiment(sample, verbose=False):
    circ = createBB84Circuit()
    bits = testSample(circ, sample)
    keyA, keyB = siftKey(bits)
    if verbose:
        print(f"Alice's key: {keyA}")
        print(f"Bob's key:   {keyB}")
    return keyA, keyB

def visualizeResults(results, save_path=OUTPUT_DIR):
    stats = results.groupby("Sample Size")["Key Length"].agg(["mean", "std"])
    stats["percent"] = (stats["mean"] / stats.index) * 100

    x_labels = stats.index.astype(str)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(
        x_labels,
        stats["mean"],
        yerr=stats["std"],
        capsize=5,
        label="Mean Key Length",
        width=0.6
    )
    plt.plot(
        x_labels,
        stats["mean"],
        color="orange",
        marker="o",
        label="Trend Line"
    )

    plt.xlabel("Sample Size")
    plt.ylabel("Key Length")
    plt.title("Average Key Length with Error Bars and Trend Line")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "keylength_errorbars_trend.png"), dpi=300)

    plt.figure(figsize=(10, 6))
    plt.bar(
        x_labels,
        stats["percent"],
        width=0.6,
        label="Key Length % of Sample Size"
    )

    plt.axhline(50, color='red', linestyle='--', linewidth=1)
    plt.text(0.98, 50 + 1.0, '50%', color='red', ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())


    plt.xlabel("Sample Size")
    plt.ylabel("Key Length as % of Sample Size")
    plt.title("Key Length / Sample Size (%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "keylength_percentage.png"), dpi=300)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    rows = []

    for sample in SAMPLES_TO_TEST:
        for exec_num in range(EXECUTIONS):
            print(f"Running experiment with sample size {sample}, execution {exec_num+1}/{EXECUTIONS}")
            keyA, keyB = runExperiment(sample)
            key_length = len(keyA)
            print(f"Final sifted key length: {key_length}")
            rows.append({'Sample Size': sample, 'Execution': exec_num + 1, 'Key Length': key_length})

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUTPUT_DIR, 'bb84_key_lengths.csv'), index=False)
    
    visualizeResults(results, save_path=OUTPUT_DIR)

if __name__ == "__main__":
    main()