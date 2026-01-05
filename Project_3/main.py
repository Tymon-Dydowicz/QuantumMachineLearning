import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.visualization import plot_state_city, plot_state_hinton, plot_state_qsphere, plot_bloch_multivector

SHOTS = 1
SINGLETS = 1024
SEED = 151936
OUTPUT_DIR = "results/"
BACKEND = AerSimulator()
BASIS_MAP = {'X': 1, 'Y': 2, 'Z': 3}
CHSH_LABELS = {(1,1): "X ⊗ W", (1,3): "X ⊗ V", (3,1): "Z ⊗ W", (3,3): "Z ⊗ V"}

def compareBasis(basis1, basis2):
    basis1_num = BASIS_MAP.get(basis1.upper())
    basis2_num = BASIS_MAP.get(basis2.upper())
    return (basis1_num == 2 and basis2_num == 1) or (basis1_num == 3 and basis2_num == 2)

def prepareSinglet():
    qr = QuantumRegister(2, 'qr')
    cr = ClassicalRegister(4, 'cr')
    circ = QuantumCircuit(qr, cr)

    circ.x(0)
    circ.x(1)
    circ.h(0)
    circ.cx(0, 1)

    return circ

def measureAlice(circ, basis, target_qubit=0, classical_bit=0):
    formated_basis = basis.upper()
    match formated_basis:
        case 'X':
            circ.h(target_qubit)
        case 'Y':
            circ.s(target_qubit)
            circ.h(target_qubit)
            circ.t(target_qubit)
            circ.h(target_qubit)
        case 'Z':
            pass
        case _:
            raise ValueError("Basis must be 'X', 'Y', or 'Z'")
        
    circ.measure(target_qubit, classical_bit)
        
def measureBob(circ, basis, target_qubit=1, classical_bit=1):
    formated_basis = basis.upper()
    match formated_basis:
        case 'X':
            circ.s(target_qubit)
            circ.h(target_qubit)
            circ.t(target_qubit)
            circ.h(target_qubit)
        case 'Y':
            pass
        case 'Z':
            circ.s(target_qubit)
            circ.h(target_qubit)
            circ.tdg(target_qubit)
            circ.h(target_qubit)
        case _:
            raise ValueError("Basis must be 'X', 'Y', or 'Z'")
        
    circ.measure(target_qubit, classical_bit)

def measureSinglets(num_singlets=SINGLETS, shots=SHOTS, seed=SEED, backend=BACKEND):
    random.seed(seed)

    bases = ['X', 'Y', 'Z']

    b = [random.choice(bases) for _ in range(num_singlets)]
    b_prime = [random.choice(bases) for _ in range(num_singlets)]

    results = []

    for i in tqdm(range(num_singlets)):
        circ = prepareSinglet()

        measureAlice(circ, b[i], target_qubit=0, classical_bit=0)
        measureBob(circ, b_prime[i], target_qubit=1, classical_bit=1)

        tcirc = transpile(circ, backend)
        job = backend.run(tcirc, shots=shots)
        counts = job.result().get_counts()

        bitstring = list(counts.keys())[0]

        alice_bit = bitstring[-1]
        bob_bit = bitstring[-2]

        results.append({
            "alice_basis": b[i],
            "bob_basis": b_prime[i],
            "alice_bit": int(alice_bit),
            "bob_bit": int(bob_bit)
        })

    print("Measurement Results:")
    for res in results[:10]:
        print(res)
    return b, b_prime, results

def revealBasis(results):
    alice_key = []
    bob_key = []
    mismatch_count = 0

    for i in range(len(results)):
        alices_basis, bob_basis, alice_bit, bob_bit = results[i].values()
        matching = compareBasis(alices_basis, bob_basis)
        if matching:
            alice_key.append(alice_bit)
            bob_bit_corrected = 1 - bob_bit
            bob_key.append(bob_bit_corrected)

            if alice_bit != bob_bit_corrected:
                mismatch_count += 1

    return alice_key, bob_key, mismatch_count

def groupResults(results):
    grouped = defaultdict(list)

    for r in results:
        b = BASIS_MAP[r["alice_basis"]]
        b_p = BASIS_MAP[r["bob_basis"]]
        a = 1 - 2 * r["alice_bit"]
        a_p = 1 - 2 * r["bob_bit"]

        grouped[(b, b_p)].append((a, a_p))

    return grouped

def countGroup(group):
    counts = {
        (1,1): 0,
        (1,-1): 0,
        (-1,1): 0,
        (-1,-1): 0
    }

    for pair in group:
        a, a_p = pair
        counts[(a, a_p)] += 1

    return counts

def calculateExpectation(counts):
    total = sum(counts.values())
    expectation = 0.0

    for (a, a_p), count in counts.items():
        expectation += a * a_p * (count / total)

    return expectation


def calculateCHSH(results):
    grouped = groupResults(results)

    E_XW = calculateExpectation(countGroup(grouped[(1,1)]))
    E_XV = calculateExpectation(countGroup(grouped[(1,3)]))
    E_ZW = calculateExpectation(countGroup(grouped[(3,1)]))
    E_ZV = calculateExpectation(countGroup(grouped[(3,3)]))

    print(f"E(1,1): {E_XW}")
    print(f"E(1,3): {E_XV}")
    print(f"E(3,1): {E_ZW}")
    print(f"E(3,3): {E_ZV}")

    S = E_XW - E_XV + E_ZW + E_ZV
    
    print(f"CHSH S value: {S} | {-2*np.sqrt(2)} Target Value")
    print("Is CHSH Violated", abs(S) > 2)
    return S

def buildCSHSTable(results):
    grouped = groupResults(results)
    table = []

    for setting, label in CHSH_LABELS.items():
        pairs = grouped.get(setting, [])
        counts = countGroup(pairs)
        total = sum(counts.values())

        expectation = calculateExpectation(counts)

        for (a, a_p), n in counts.items():
            if total == 0:
                continue

            p = n / total
            contribution = p * (a * a_p)

            table.append({
                "Measurement": label,
                "(b,b')": setting,
                "(a,a')": (a, a_p),
                "n_ij": n,
                "p_ij": round(p, 4),
                "p·(a·a')": round(contribution, 4),
                "Expectation": round(expectation, 4)
            })

    return table

def main():
    b, b_prime, results = measureSinglets()
    alice_key, bob_key, mismatches = revealBasis(results)
    print(f"Alice Key length: {len(alice_key)}")
    print(f"Bob Key length: {len(bob_key)}")
    print(f"Number of mismatches: {mismatches}")
    print(f"Alice's key: {alice_key}")
    print(f"Bob's key: {bob_key}")

    S = calculateCHSH(results)
    table = buildCSHSTable(results)
    df = pd.DataFrame(table)
    print("\nCHSH Detailed Table:")
    print(df)

if __name__ == "__main__":
    main()