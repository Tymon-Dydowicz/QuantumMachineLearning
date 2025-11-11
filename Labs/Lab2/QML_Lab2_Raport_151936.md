# Quantum Bell States Measurement and Visualization Project

**Author:** *Tymon Dydowicz*  
**Environment:** Qiskit, Python 3.10.11  
**Student ID:** 151936  

---

## Project Overview

This project demonstrates **quantum entanglement preparation, measurement, and visualization** for **Bell states** using Qiskit.  

The Bell states (`Φ⁺`, `Φ⁻`, `Ψ⁺`, `Ψ⁻`) are fundamental examples of **maximally entangled two-qubit states**, and their behavior under different measurement bases (XX, YY, XZ).

### **Tasks Overview**
| Task | Description |
|------|--------------|
| **1–4** | Preparation and measurement of Bell states `Φ⁺`, `Φ⁻`, `Ψ⁺`, `Ψ⁻` |
| **5–7** | Measurement of all Bell states in **XX**, **YY**, and **XZ** bases |

All simulations use:
- **2 qubits**
- **2048 shots per run**
- **3 executions per configuration**
- **Random seed:** 151936  
- **Simulator:** Qiskit Aer

---

## Bell State Circuits

| Bell State | Quantum Circuit |
|-------------|-----------------|
| **Φ⁺ (Fi⁺)** | ![Φ⁺ circuit](./results/screenshots/circuit_fi+.png) |
| **Φ⁻ (Fi⁻)** | ![Φ⁻ circuit](./results/screenshots/cricuit_fi-.PNG) |
| **Ψ⁺ (Psi⁺)** | ![Ψ⁺ circuit](./results/screenshots/cricuit_psi+.PNG) |
| **Ψ⁻ (Psi⁻)** | ![Ψ⁻ circuit](./results/screenshots/cricuit_psi-.PNG) |

---

## Task 1–4: Bell State Preparation and Measurement

Each Bell state was prepared, measured in the computational basis (Z), and run three times.  
The histograms below show both **counts** and **probabilities**, with labels indicating exact values.

| Bell State | Counts & Probabilities |
|-------------|------------------------|
| **Φ⁺ (Fi⁺)** | ![Φ⁺ measurement](./results/bell_state_measurement(Fi+).png) |
| **Φ⁻ (Fi⁻)** | ![Φ⁻ measurement](./results/bell_state_measurement(Fi-).png) |
| **Ψ⁺ (Psi⁺)** | ![Ψ⁺ measurement](./results/bell_state_measurement(Psi+).png) |
| **Ψ⁻ (Psi⁻)** | ![Ψ⁻ measurement](./results/bell_state_measurement(Psi-).png) |

---

## Task 5–7: Bell State Measurements in Different Bases

The Bell states were measured in **XX**, **YY**, and **XZ** bases to demonstrate quantum correlations in rotated measurement spaces.  
Each plot aggregates data from all four Bell states in a **2×2 grid** layout.

### **Task 5 — Measurement in XX Basis**
| Type | Image |
|------|--------|
| **Counts** | ![XX Counts](./results/bell_measurements_XX_counts.png) |
| **Probabilities** | ![XX Probabilities](./results/bell_measurements_XX_probabilities.png) |

### **Task 6 — Measurement in YY Basis**
| Type | Image |
|------|--------|
| **Counts** | ![YY Counts](./results/bell_measurements_YY_counts.png) |
| **Probabilities** | ![YY Probabilities](./results/bell_measurements_YY_probabilities.png) |

### **Task 7 — Measurement in XZ Basis**
| Type | Image |
|------|--------|
| **Counts** | ![XZ Counts](./results/bell_measurements_XZ_counts.png) |
| **Probabilities** | ![XZ Probabilities](./results/bell_measurements_XZ_probabilities.png) |

---

## Additional Paper Calculations

![Calculations](./results//screenshots/calculations.jpg)

## Conclusion

This experiment demonstrates:
- The creation of **entangled Bell states** using simple quantum circuits.  
- The dependence of measured outcomes on **measurement basis**.  
- **Clear quantum correlations** that distinguish the four Bell states.  
- Reproducibility across multiple runs, confirming simulation stability.

---
