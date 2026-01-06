# Grover's Diffusion Algorithm

**Author:** Tymon Dydowicz\
**Environment:** Qiskit, Python 3.10.11\
**Student ID:** 151936

------------------------------------------------------------------------

## Project Overview

This project implements and evaluates a testing environment for Grover's Diffusion Alogrithm. As well as some explicit form of matrices for XOR/Phase oracles


------------------------------------------------------------------------

## Experiment Parameters
  **Simulator**               Qiskit AerSimulator  
  **Qubits Used**             3  
  **Shots**                   1024  
  **Random Seed**             151936

------------------------------------------------------------------------

## Important Code
```python
def createXOROracle(f, n):
    dim = 2 ** (n + 1)
    U = np.zeros((dim, dim), dtype=complex)

    for y in (0, 1):
        for x in range(2**n):
            input_index = (y << n) | x
            new_y = y ^ int(f.compare(x))
            output_index = (new_y << n) | x
            U[output_index, input_index] = 1.0

    return U

def createPhaseOracle(f, n):
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        U[x, x] = (-1) ** f.compare(x)

    return U
```

```python
def createUf(f, n):
    qc = QuantumCircuit(n, name="Uf")
    for x in range(2**n):
        if f.compare(x):
            x_bin = format(x, f'0{n}b')[::-1]
            for i, bit in enumerate(x_bin):
                if bit == '0':
                    qc.x(i)

            qc.h(n-1)
            qc.mcx(list(range(n-1)), n-1)
            qc.h(n-1)

            for i, bit in enumerate(x_bin):
                if bit == '0':
                    qc.x(i)

    return qc

def createGroverDiffusion(n):
    qc = QuantumCircuit(n, name="W")

    qc.h(range(n))
    qc.x(range(n))

    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)

    qc.x(range(n))
    qc.h(range(n))

    return qc

def groverAlgorithm(f, n, iterations=None):
    qc = QuantumCircuit(n, n)

    qc.h(range(n))

    if iterations is None:
        iterations = int(np.floor(np.pi/4 * np.sqrt(2**n)))

    for _ in range(iterations):
        qc.compose(createUf(f, n), inplace=True)
        qc.compose(createGroverDiffusion(n), inplace=True)

    qc.measure(range(n), range(n))

    return qc
```

------------------------------------------------------------------------