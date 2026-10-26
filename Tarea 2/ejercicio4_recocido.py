"""
Ejercicio 4: Ajuste de parámetros (Simulated Annealing para TSP)

Recocido simulado con 2-opt como operador de vecindad, partiendo de un tour aleatorio.
Se comparan tres esquemas de enfriamiento: lineal, exponencial y logarítmico.
"""
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ejercicio3_2opt import (dist_matrix, random_cities, tour_length, two_opt,
                             two_opt_delta, two_opt_move)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------- esquemas de enfriamiento: T(t) para t = 0..N-1 ----------
def linear(t, N, T0, Tf):
    return T0 - (T0 - Tf) * t / (N - 1)


def exponential(t, N, T0, Tf):
    return T0 * (Tf / T0) ** (t / (N - 1))


def logarithmic(t, N, T0, Tf):
    # T = T0 / (1 + c ln(1+t)), con c calibrada para que T(N-1) = Tf
    c = (T0 / Tf - 1) / math.log(N)
    return T0 / (1 + c * math.log(1 + t))


SCHEDULES = {"Lineal": linear, "Exponencial": exponential, "Logarítmico": logarithmic}


def simulated_annealing(D, schedule, N, T0, Tf, rng, record_every=0):
    """
    SA con N iteraciones. Movimiento: 2-opt aleatorio (i<k).
    Devuelve (mejor tour, mejor longitud, historial) donde el historial es una lista
    de (iteración, temperatura, costo actual, mejor costo) si record_every > 0.
    """
    n = len(D)
    tour = list(range(n))
    rng.shuffle(tour)
    cost = tour_length(D, tour)
    best, best_cost = tour[:], cost
    history = []
    for t in range(N):
        T = schedule(t, N, T0, Tf)
        i = rng.randint(1, n - 2)
        k = rng.randint(i + 1, n - 1)
        delta = two_opt_delta(D, tour, i, k)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            tour = two_opt_move(tour, i, k)
            cost += delta
            if cost < best_cost - 1e-12:
                best, best_cost = tour[:], cost
        if record_every and t % record_every == 0:
            history.append((t, T, cost, best_cost))
    return best, best_cost, history


def calibrate_temperatures(D):
    """T0 ~ longitud media de arista (acepta casi todo al inicio); Tf muy pequeña."""
    n = len(D)
    avg_edge = sum(D[i][j] for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)
    return avg_edge, avg_edge * 1e-3


def main():
    n_cities, n_runs = 40, 10
    budgets = [2_000, 10_000, 50_000]      # iteraciones (para el análisis tiempo vs calidad)
    conv_N = 20_000                        # presupuesto usado en las curvas de convergencia
    rng = random.Random(42)
    D = dist_matrix(random_cities(n_cities, rng))
    T0, Tf = calibrate_temperatures(D)
    print(f"n = {n_cities} ciudades, T0 = {T0:.2f}, Tf = {Tf:.4f}, {n_runs} corridas por configuración")

    # Referencia: 2-opt de búsqueda local desde inicios aleatorios
    ls = []
    for _ in range(n_runs):
        t = list(range(n_cities))
        rng.shuffle(t)
        ls.append(two_opt(D, t)[1])
    print(f"Referencia 2-opt (búsqueda local): media = {np.mean(ls):.1f}, mejor = {np.min(ls):.1f}\n")

    # ---- Curvas de convergencia (presupuesto fijo conv_N) ----
    conv = {}
    for name, sch in SCHEDULES.items():
        hist_runs = []
        for r in range(n_runs):
            _, _, h = simulated_annealing(D, sch, conv_N, T0, Tf, random.Random(1000 + r), record_every=50)
            hist_runs.append(h)
        conv[name] = np.array(hist_runs)   # (runs, puntos, 4)

    # ---- Tiempo vs calidad (varios presupuestos) ----
    results = {name: {} for name in SCHEDULES}
    print(f"{'Esquema':<13}{'Iter':>8}{'Tiempo (s)':>12}{'Costo medio':>13}{'Mejor':>9}{'Desv.':>8}")
    for name, sch in SCHEDULES.items():
        for N in budgets:
            costs, times = [], []
            for r in range(n_runs):
                t0 = time.perf_counter()
                _, c, _ = simulated_annealing(D, sch, N, T0, Tf, random.Random(5000 + r))
                times.append(time.perf_counter() - t0)
                costs.append(c)
            results[name][N] = (np.mean(times), np.mean(costs), np.std(costs), np.min(costs))
            print(f"{name:<13}{N:>8,}{np.mean(times):>12.3f}{np.mean(costs):>13.1f}"
                  f"{np.min(costs):>9.1f}{np.std(costs):>8.1f}")

    # ---- Gráfica 1: convergencia ----
    colors = {"Lineal": "tab:blue", "Exponencial": "tab:green", "Logarítmico": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for name, arr in conv.items():
        its = arr[0, :, 0]
        ax.plot(its, arr[:, :, 2].mean(axis=0), color=colors[name], alpha=0.35, linewidth=1)
        ax.plot(its, arr[:, :, 3].mean(axis=0), color=colors[name], linewidth=2, label=name)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Costo (longitud del tour)")
    ax.set_title("Costo vs. iteraciones\n(línea gruesa = mejor; tenue = actual)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, arr in conv.items():
        ax.plot(arr[0, :, 1], arr[:, :, 2].mean(axis=0), color=colors[name], label=name)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Temperatura (escala log, de alta a baja)")
    ax.set_ylabel("Costo actual")
    ax.set_title("Costo vs. temperatura")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    its = np.arange(conv_N)
    for name, sch in SCHEDULES.items():
        ax.plot(its, [sch(t, conv_N, T0, Tf) for t in its], color=colors[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Temperatura (log)")
    ax.set_title("Perfil de cada esquema de enfriamiento")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Ejercicio 4: curvas de convergencia del recocido simulado")
    fig.tight_layout()
    out1 = RESULTS / "ejercicio4_convergencia.png"
    fig.savefig(out1, dpi=150)

    # ---- Gráfica 2: tiempo vs calidad ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in SCHEDULES:
        ts = [results[name][N][0] for N in budgets]
        cs = [results[name][N][1] for N in budgets]
        es = [results[name][N][2] for N in budgets]
        ax.errorbar(ts, cs, yerr=es, fmt="o-", color=colors[name], capsize=3, label=name)
        for N, t_, c_ in zip(budgets, ts, cs):
            ax.annotate(f"{N:,}", (t_, c_), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.axhline(np.mean(ls), color="gray", linestyle="--", label="2-opt búsqueda local (media)")
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo de ejecución (s, escala log)")
    ax.set_ylabel("Costo final medio (± desv. estándar)")
    ax.set_title("Ejercicio 4: tiempo vs. calidad (etiquetas = iteraciones)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out2 = RESULTS / "ejercicio4_tiempo_vs_calidad.png"
    fig.savefig(out2, dpi=150)
    print(f"\nGráficas guardadas en {out1} y {out2}")


if __name__ == "__main__":
    main()
