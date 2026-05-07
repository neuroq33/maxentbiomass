"""
Código completo para el modelo de diversidad de biomasa (Lurie & Wagensberg, 1983)
Aplicado a datos de semillas de la Gran Sabana, Venezuela.
Incluye gráficas: histograma + exponencial, m_i vs P_i, y m_i vs -ln(P_i).
"""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# 1. FUNCIONES DEL MODELO
# ============================================================================

def calculate_delta_m_sturges(masses):
    min_m = np.min(masses)
    max_m = np.max(masses)
    N = len(masses)
    return (max_m - min_m) / (1 + np.log2(N))


def biomass_diversity_algorithm(masses, delta_m=None):
    masses = np.array(masses)
    N = len(masses)
    
    if N == 0:
        return None
    
    m_mean = np.mean(masses)
    m_std = np.std(masses, ddof=1)
    m_min = np.min(masses)
    m_max = np.max(masses)
    m_median = np.median(masses)
    m_q25 = np.percentile(masses, 25)
    m_q75 = np.percentile(masses, 75)
    
    if delta_m is None:
        delta_m = calculate_delta_m_sturges(masses)
    
    bins = np.arange(m_min, m_max + delta_m, delta_m)
    if len(bins) < 2:
        bins = np.linspace(m_min, m_max, min(10, N))
        delta_m = bins[1] - bins[0]
    
    counts, bin_edges = np.histogram(masses, bins=bins)
    P = counts / N
    m_i = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    valid = P > 0
    if np.sum(valid) >= 2:
        x = -np.log(P[valid])
        y = m_i[valid]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    else:
        slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
    
    fit_ok = False
    if not np.isnan(slope):
        fit_ok = np.isclose(slope, m_mean, rtol=0.2)
    
    if np.sum(P > 0) > 0:
        mu_bar = -np.sum(P[P > 0] * np.log2(P[P > 0] * m_mean / delta_m))
    else:
        mu_bar = np.nan
    mu_teo = 1 / np.log(2)
    k = -np.sum(P[P > 0] * np.log2(P[P > 0]))
    r_eff = np.sum(P > 0)
    
    return {
        'N': N, 'm_mean': m_mean, 'm_std': m_std, 'm_min': m_min, 'm_max': m_max,
        'm_median': m_median, 'm_q25': m_q25, 'm_q75': m_q75, 'delta_m': delta_m,
        'r_eff': r_eff, 'slope': slope, 'intercept': intercept, 'correlation': r_value,
        'p_value': p_value, 'std_err': std_err, 'fit_ok': fit_ok,
        'mu_bar': mu_bar, 'mu_teo': mu_teo, 'k': k,
        'masses': masses, 'P': P, 'm_i': m_i, 'bin_edges': bin_edges
    }


def plot_and_save(results, ecosistema_name, save_path):
    """
    Genera tres gráficos:
    1) Histograma de masas + ajuste exponencial.
    2) m_i vs P_i con curva teórica exponencial.
    3) m_i vs -ln(P_i) con regresión lineal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ----- Gráfico 1: Histograma y ajuste exponencial -----
    ax1 = axes[0]
    masses = results['masses']
    m_mean = results['m_mean']
    
    ax1.hist(masses, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Datos observados')
    m_range = np.linspace(0, np.max(masses), 200)
    f_exp = (1 / m_mean) * np.exp(-m_range / m_mean)
    ax1.plot(m_range, f_exp, 'r-', linewidth=2, label=r'Ajuste exponencial $\lambda = 1/\bar{m}$ = ' + f'{1/m_mean:.2f}')
    ax1.set_xlabel('Masa (g)')
    ax1.set_ylabel('Densidad de probabilidad')
    ax1.set_title(f'{ecosistema_name}\nHistograma y ajuste exponencial')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ----- Gráfico 2: m_i vs P_i (curva exponencial) -----
    ax2 = axes[1]
    valid = results['P'] > 0
    m_i_vals = results['m_i'][valid]
    P_vals = results['P'][valid]
    
    ax2.scatter(m_i_vals, P_vals, c='steelblue', alpha=0.7, s=50, label='Datos discretizados')
    
    # Curva teórica: P(m) = (Δm / m̄) * exp(-m / m̄)
    delta_m = results['delta_m']
    m_theo = np.linspace(0, np.max(masses), 200)
    P_theo = (delta_m / m_mean) * np.exp(-m_theo / m_mean)
    ax2.plot(m_theo, P_theo, 'r-', linewidth=2, label=rf'Teórica $P(m) = \frac{{\Delta m}}{{\bar{{m}}}} e^{{-m/\bar{{m}}}}$ ($\bar{{m}}$={m_mean:.3f}, $\Delta m$={delta_m:.4f})')
    
    ax2.set_xlabel('$m_i$ (g)')
    ax2.set_ylabel('$P_i$')
    ax2.set_title(f'{ecosistema_name}\nProbabilidad vs masa')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ----- Gráfico 3: m_i vs -ln(P_i) con regresión lineal -----
    ax3 = axes[2]
    x = -np.log(P_vals)
    y = m_i_vals
    ax3.scatter(x, y, c='steelblue', alpha=0.7, s=50, label='Datos discretizados')
    
    if not np.isnan(results['slope']):
        x_line = np.array([min(x), max(x)])
        y_line = results['slope'] * x_line + results['intercept']
        ax3.plot(x_line, y_line, 'r-', linewidth=2,
                label=rf'Regresión lineal $m_i = {results["slope"]:.3f} (-\ln P_i) + {results["intercept"]:.3f}$, $r={results["correlation"]:.3f}$')
    
    ax3.set_xlabel(r'$-\ln(P_i)$')
    ax3.set_ylabel('$m_i$ (g)')
    ax3.set_title(f'{ecosistema_name}\nRelación lineal del modelo')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfico guardado: {save_path}")


# ============================================================================
# 2. EXTRACCIÓN DIRECTA DE DATOS
# ============================================================================

def extract_data_direct():
    primary_data = [
        (1, 0.174), (4, 0.108), (5, 0.297), (3, 0.801), (11, 0.29),
        (13, 0.383), (15, 0.348), (3, 0.268), (3, 0.155), (7, 0.285),
        (8, 0.19), (45, 0.164), (20, 0.0956), (4, 0.5852), (155, 3.12),
        (66, 0.0405), (48, 0.0412), (137, 0.075), (36, 0.2417), (104, 0.1572),
        (144, 0.0647), (58, 0.1089), (138, 0.081), (47, 0.0226), (53, 0.073),
        (5, 0.085), (7, 0.059), (8, 0.2189)
    ]
    secondary_data = [
        (21, 0.411), (2, 0.314), (22, 0.508), (26, 0.727), (24, 0.741),
        (5, 0.819), (21, 0.879), (16, 0.336), (10, 0.681), (12, 0.739),
        (21, 0.159), (12, 0.3365), (63, 1.702), (155, 2.68), (3, 0.1635),
        (171, 0.072), (10, 0.0149), (5, 5.31), (100, 0.0428), (31, 1.21),
        (29, 0.0421), (23, 0.062), (64, 0.138), (12, 0.042), (27, 0.1044),
        (38, 0.1065)
    ]
    scrub_data = [
        (5, 0.132), (134, 0.254), (60, 0.217), (51, 0.241), (49, 0.155),
        (12, 0.284), (17, 0.385), (35, 0.367), (93, 0.744), (26, 0.241),
        (7, 0.456), (86, 0.136), (28, 0.1063), (27, 0.0226), (49, 0.083),
        (4, 0.117), (15, 0.1461), (33, 0.181), (65, 0.1725), (25, 0.1331),
        (46, 0.3933), (50, 0.5213), (850, 1.19), (434, 2.7), (368, 5.11),
        (345, 3.57), (359, 5.89), (277, 3.15), (231, 0.7)
    ]
    savanna_data = [
        (70, 3.25), (23, 0.021), (7, 0.041), (1, 0.0038), (4, 0.0058),
        (4, 0.023), (246, 0.1292), (241, 0.016), (1, 0.269), (55, 0.25)
    ]
    return {
        'primary_forest': primary_data,
        'secondary_forest': secondary_data,
        'scrub': scrub_data,
        'savanna': savanna_data
    }


def expand_masses(data_list):
    masses = []
    for n, p in data_list:
        if n > 0 and p > 0:
            m_individual = p / n
            masses.extend([m_individual] * int(n))
    return np.array(masses)


# ============================================================================
# 3. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    output_file = "biomass-v00E.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MODELO DE DIVERSIDAD DE BIOMASA (LURIE & WAGENSBERG, 1983)\n")
        f.write("Aplicado a datos de semillas de la Gran Sabana, Venezuela\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        print("=" * 70)
        print("MODELO DE DIVERSIDAD DE BIOMASA")
        print("Incluye gráficas: histograma + exponencial, m_i vs P_i, m_i vs -ln(P_i)")
        print("=" * 70)
        
        ecosistemas_data = extract_data_direct()
        nombres = {
            'primary_forest': 'Bosque Natural (Primario)',
            'secondary_forest': 'Bosque Secundario',
            'scrub': 'Matorral',
            'savanna': 'Sabana'
        }
        
        resultados = {}
        
        for eco_key, data_list in ecosistemas_data.items():
            nombre = nombres.get(eco_key, eco_key)
            print(f"\n--- {nombre} ---")
            f.write(f"\n--- {nombre} ---\n")
            
            masses = expand_masses(data_list)
            print(f"  Semillas: {len(masses)}")
            f.write(f"  Semillas: {len(masses)}\n")
            
            results = biomass_diversity_algorithm(masses)
            resultados[eco_key] = results
            
            lambda_exp = 1 / results['m_mean']
            print(f"  Masa media (m̄): {results['m_mean']:.6f} g")
            print(f"  Parámetro λ = 1/m̄: {lambda_exp:.4f} g⁻¹")
            print(f"  Delta_m (Sturges): {results['delta_m']:.6f} g")
            print(f"  Correlación lineal: {results['correlation']:.4f}" if not np.isnan(results['correlation']) else "  Correlación: N/A")
            print(f"  Shannon k: {results['k']:.4f} bits/ind")
            print(f"  μ̄ normalizada: {results['mu_bar']:.4f} bits/ind (teórico: {results['mu_teo']:.4f})")
            print(f"  Ajuste exponencial: {'Válido' if results['fit_ok'] else 'No válido'}")
            
            f.write(f"  Masa media: {results['m_mean']:.8f} g\n")
            f.write(f"  Parámetro λ: {lambda_exp:.6f} g⁻¹\n")
            f.write(f"  Delta_m: {results['delta_m']:.8f} g\n")
            f.write(f"  Correlación (r): {results['correlation']:.6f}\n")
            f.write(f"  Shannon k: {results['k']:.6f} bits/ind\n")
            f.write(f"  μ̄ normalizada: {results['mu_bar']:.6f} bits/ind\n")
            f.write(f"  Ajuste exponencial: {results['fit_ok']}\n")
            
            plot_and_save(results, nombre, f"biomass_{eco_key}.png")
        
        # Tabla comparativa
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA")
        print("=" * 70)
        print("\n| Ecosistema | N | m̄ (g) | λ (g⁻¹) | k (bits) | r | Ajuste |")
        print("|------------|-----|--------|---------|----------|-----|--------|")
        
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("TABLA COMPARATIVA\n")
        f.write("=" * 70 + "\n\n")
        f.write("| Ecosistema | N | m̄ (g) | λ (g⁻¹) | k (bits) | r | Ajuste |\n")
        f.write("|------------|-----|--------|---------|----------|-----|--------|\n")
        
        for eco_key, res in resultados.items():
            nombre = nombres.get(eco_key, eco_key)
            N = res['N']
            m_mean = f"{res['m_mean']:.4f}"
            lam = f"{1/res['m_mean']:.4f}"
            k = f"{res['k']:.4f}"
            r = f"{res['correlation']:.3f}" if not np.isnan(res['correlation']) else "N/A"
            ajuste = "Sí" if res['fit_ok'] else "No"
            print(f"| {nombre} | {N} | {m_mean} | {lam} | {k} | {r} | {ajuste} |")
            f.write(f"| {nombre} | {N} | {m_mean} | {lam} | {k} | {r} | {ajuste} |\n")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    print("\nArchivos generados:")
    print("  - biomass-v00E.txt (resultados detallados)")
    print("  - biomass_primary_forest.png (3 gráficas)")
    print("  - biomass_secondary_forest.png")
    print("  - biomass_scrub.png")
    print("  - biomass_savanna.png")


if __name__ == "__main__":
    main()
