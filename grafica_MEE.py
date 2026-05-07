#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gráficas a partir de MEE_simulation_results.csv
Genera las figuras del manuscrito basadas en los resultados de la simulación Monte Carlo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Cargar los datos
# ------------------------------------------------------------
df = pd.read_csv("MEE_simulation_results.csv")

# ------------------------------------------------------------
# 2. Figura 1: False Positive Rate para log‑normal (CV = 1.5)
# ------------------------------------------------------------
# Filtrar log-normal con CV = 1.5 (la que más se acerca a 1.2 del manuscrito)
df_fp_lognorm = df[(df['Distribution'] == 'lognormal') & (df['CV'] == 1.5)]
# Ordenar por N
df_fp_lognorm = df_fp_lognorm.sort_values('N')

plt.figure(figsize=(6, 4))
rules = df_fp_lognorm['Rule'].unique()
for rule in rules:
    data = df_fp_lognorm[df_fp_lognorm['Rule'] == rule]
    plt.plot(data['N'], data['FalsePositiveRate'], marker='o', label=rule)
plt.axhline(0.05, linestyle='--', color='gray', label='α = 0.05')
plt.xlabel('Sample size N')
plt.ylabel('False positive rate')
plt.title('Figure 1: False positive rates – log‑normal (CV = 1.5)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig1_false_positives.pdf', bbox_inches='tight')
plt.close()
print("Figura 1 guardada: fig1_false_positives.pdf")

# ------------------------------------------------------------
# 3. Figura 2: Sesgo en la estimación de la media (datos exponenciales)
# ------------------------------------------------------------
df_bias = df[(df['Distribution'] == 'exponential') & (df['CV'] == 1.0)]
df_bias = df_bias.sort_values('N')

plt.figure(figsize=(6, 4))
for rule in rules:
    data = df_bias[df_bias['Rule'] == rule]
    plt.plot(data['N'], data['BiasMean'], marker='s', label=rule)
plt.axhline(0, linestyle='-', color='black')
plt.xlabel('Sample size N')
plt.ylabel('Relative bias in estimated mean')
plt.title('Figure 2: Bias in β̂ (mean) – exponential data')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig2_bias_beta.pdf', bbox_inches='tight')
plt.close()
print("Figura 2 guardada: fig2_bias_beta.pdf")

# ------------------------------------------------------------
# 4. Figura 3: Potencia (True Positive Rate) para datos exponenciales
# ------------------------------------------------------------
df_power = df[(df['Distribution'] == 'exponential') & (df['CV'] == 1.0)]
df_power = df_power.sort_values('N')

plt.figure(figsize=(6, 4))
for rule in rules:
    data = df_power[df_power['Rule'] == rule]
    plt.plot(data['N'], data['TruePositiveRate'], marker='^', label=rule)
plt.axhline(0.95, linestyle='--', color='gray', label='95% power')
plt.xlabel('Sample size N')
plt.ylabel('True positive rate (power)')
plt.title('Figure 3: Power – exponential data')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig3_power.pdf', bbox_inches='tight')
plt.close()
print("Figura 3 guardada: fig3_power.pdf")

# ------------------------------------------------------------
# 5. Figura adicional (opcional): False Positive Rate para gamma (CV = 2.0)
# ------------------------------------------------------------
df_fp_gamma = df[(df['Distribution'] == 'gamma') & (df['CV'] == 2.0)]
if not df_fp_gamma.empty:
    df_fp_gamma = df_fp_gamma.sort_values('N')
    plt.figure(figsize=(6, 4))
    for rule in rules:
        data = df_fp_gamma[df_fp_gamma['Rule'] == rule]
        plt.plot(data['N'], data['FalsePositiveRate'], marker='D', label=rule)
    plt.axhline(0.05, linestyle='--', color='gray')
    plt.xlabel('Sample size N')
    plt.ylabel('False positive rate')
    plt.title('False positive rates – gamma distribution (CV = 2.0)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('fig_supp_gamma_fpr.pdf', bbox_inches='tight')
    plt.close()
    print("Figura suplementaria guardada: fig_supp_gamma_fpr.pdf")
else:
    print("No hay datos para gamma con CV=2.0, se omite figura adicional.")

# ------------------------------------------------------------
# 6. Árbol de decisión (texto)
# ------------------------------------------------------------
with open('fig_decision_tree.txt', 'w') as f:
    f.write("Decision tree for binning rule selection (based on simulation results):\n")
    f.write("1. If N < 300: use Sturges (default) and sensitivity check with Freedman-Diaconis.\n")
    f.write("2. If 300 <= N < 2000:\n")
    f.write("   - If CV_m < 0.8: use Scott\n")
    f.write("   - If CV_m >= 0.8: use Freedman-Diaconis\n")
    f.write("3. If N >= 2000: use Freedman-Diaconis unconditionally.\n")
    f.write("4. Always report r, Delta_m, KS statistic, AICc difference under at least two rules.\n")
print("Árbol de decisión guardado: fig_decision_tree.txt")

# ------------------------------------------------------------
# (Opcional) Generar una figura simple del árbol de decisión con matplotlib
# ------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(5, 9, "Start", ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    ax.text(5, 7.5, "N < 300 ?", ha='center', fontsize=11, bbox=dict(boxstyle="round", facecolor='lightgray'))
    ax.text(2, 6, "Yes: Sturges + check FD", ha='center', fontsize=10)
    ax.text(8, 6, "No", ha='center', fontsize=10)
    ax.text(5, 4.5, "N < 2000 ?", ha='center', fontsize=11, bbox=dict(boxstyle="round", facecolor='lightgray'))
    ax.text(2, 3, "Yes: check CV_m", ha='center', fontsize=10)
    ax.text(8, 3, "No: Freedman-Diaconis", ha='center', fontsize=10)
    ax.text(3, 1.5, "CV < 0.8: Scott", ha='center', fontsize=10, bbox=dict(boxstyle="round", facecolor='lightgreen'))
    ax.text(7, 1.5, "CV >= 0.8: Freedman-Diaconis", ha='center', fontsize=10, bbox=dict(boxstyle="round", facecolor='lightgreen'))
    plt.title("Figure: Decision tree for binning rule selection")
    plt.savefig('fig_decision_tree.pdf', bbox_inches='tight')
    plt.close()
    print("Figura del árbol de decisión guardada: fig_decision_tree.pdf")
except Exception as e:
    print(f"No se pudo generar la figura del árbol: {e}")
