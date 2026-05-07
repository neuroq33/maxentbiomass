#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera las figuras faltantes para el manuscrito:
- Figura de potencia (True Positive Rate) vs N para datos exponenciales.
- Figura de sesgo relativo en k̂ (Shannon index) vs N (requiere calcular k̂ estimado vs teórico).
Nota: El CSV actual no contiene la columna Bias_k. Se estima a partir de los datos simulados
(si no se dispone, se puede omitir la figura de sesgo de k̂, o calcularla teóricamente).
Para este script, asumimos que el CSV contiene una columna 'BiasK' (si no, se puede generar
a partir de los datos crudos de simulación). Como no la tenemos, se generará una figura
de potencia y se sugerirá cómo añadir sesgo de k̂ si se dispone de los datos completos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar datos
df = pd.read_csv("MEE_simulation_results.csv")

# ============================================================================
# Figura de potencia (True Positive Rate) para datos exponenciales
# ============================================================================
df_power = df[(df['Distribution'] == 'exponential') & (df['CV'] == 1.0)]
df_power = df_power.sort_values('N')

plt.figure(figsize=(6, 4))
rules = df_power['Rule'].unique()
for rule in rules:
    data = df_power[df_power['Rule'] == rule]
    plt.plot(data['N'], data['TruePositiveRate'], marker='o', label=rule)

plt.axhline(0.95, linestyle='--', color='gray', label='95% power')
plt.xlabel('Sample size N')
plt.ylabel('True Positive Rate (power)')
plt.title('Figure: Statistical power for exponential data')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('fig_power1.pdf', bbox_inches='tight')
plt.close()
print("Figura de potencia guardada: fig_power.pdf")
