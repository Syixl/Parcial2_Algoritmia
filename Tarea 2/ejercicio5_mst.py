"""
Ejercicio 5: Razón de aproximación (MST 2-aproximación para TSP)

100 instancias aleatorias de 9 ciudades. Se calcula el óptimo exacto (Held-Karp),
la aproximación MST (Prim + recorrido en preorden), MST + 2-opt, 2-opt desde inicio
aleatorio (ejercicio 3) y recocido simulado (ejercicio 4).
"""
import itertools
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ejercicio3_2opt import dist_matrix, random_cities, tour_length, two_opt
from ejercicio4_recocido import calibrate_temperatures, exponential, simulated_annealing

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def mst_prim(D):
    """Prim desde la ciudad 0; devuelve el árbol como {nodo: hijos}."""
    n = len(D)
    best = {v: (D[0][v], 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)
        for u in best:
            if D[v][u] < best[u][0]:
                best[u] = (D[v][u], v)
    return children


def preorder(children, root=0):
    return [root] + [c for ch in children[root] for c in preorder(children, ch)]


def mst_weight(D):
    ch = mst_prim(D)
    return sum(D[p][c] for p, cs in ch.items() for c in cs)


def held_karp(D):
    """Óptimo exacto por programación dinámica, O(n^2 2^n). Suficiente para n = 9."""
    n = len(D)
    INF = float("inf")
    dp = {(1 << j, j): D[0][j] for j in range(1, n)}
    for size in range(2, n):
        for subset in itertools.combinations(range(1, n), size):
            mask = sum(1 << j for j in subset)
            for j in subset:
                prev = mask ^ (1 << j)
                dp[(mask, j)] = min(dp[(prev, k)] + D[k][j] for k in subset if k != j)
    full = sum(1 << j for j in range(1, n))
    return min(dp[(full, j)] + D[j][0] for j in range(1, n))


def brute_force(D):
    n = len(D)
    return min(tour_length(D, (0,) + p) for p in itertools.permutations(range(1, n)))


def main():
    n_cities, n_instances, sa_iters = 9, 100, 4_000
    rng = random.Random(2024)

    # Verificación de Held-Karp contra fuerza bruta en unas instancias
    for _ in range(3):
        Dt = dist_matrix(random_cities(n_cities, rng))
        assert abs(held_karp(Dt) - brute_force(Dt)) < 1e-9
    print("Held-Karp verificado contra fuerza bruta (8! = 40,320 tours).\n")

    res = {k: [] for k in ["opt", "mst", "mst2opt", "ls", "sa"]}
    for _ in range(n_instances):
        D = dist_matrix(random_cities(n_cities, rng, size=10))
        opt = held_karp(D)

        mst_tour = preorder(mst_prim(D))
        L_mst = tour_length(D, mst_tour)
        assert L_mst <= 2 * mst_weight(D) + 1e-9 <= 2 * opt + 2e-9   # garantía teórica

        _, L_mst2, _ = two_opt(D, mst_tour)

        t = list(range(n_cities))
        rng.shuffle(t)
        _, L_ls, _ = two_opt(D, t)

        T0, Tf = calibrate_temperatures(D)
        _, L_sa, _ = simulated_annealing(D, exponential, sa_iters, T0, Tf, rng)

        for k, v in zip(res, [opt, L_mst, L_mst2, L_ls, L_sa]):
            res[k].append(v)
    res = {k: np.array(v) for k, v in res.items()}

    ratios = {k: res[k] / res["opt"] for k in res if k != "opt"}
    labels = {"mst": "MST puro", "mst2opt": "MST + 2-opt", "ls": "2-opt (inicio aleat.)",
              "sa": "Recocido simulado"}

    print(f"{n_instances} instancias de {n_cities} ciudades")
    print(f"{'Método':<24}{'Razón media':>12}{'Mediana':>9}{'Máx':>8}{'% óptimo':>10}")
    for k, name in labels.items():
        r = ratios[k]
        print(f"{name:<24}{r.mean():>12.3f}{np.median(r):>9.3f}{r.max():>8.3f}{np.mean(r < 1 + 1e-9):>10.0%}")
    print(f"\nLímite teórico = 2.0; razón máxima observada con MST = {ratios['mst'].max():.3f}")
    print(f"Costo medio óptimo = {res['opt'].mean():.2f}")

    # ---- Gráfica 1: razón de aproximación del MST ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.hist(ratios["mst"], bins=20, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(2.0, color="red", linestyle="--", linewidth=2, label="Límite teórico = 2")
    ax.axvline(ratios["mst"].mean(), color="black", linestyle=":", label=f"Media = {ratios['mst'].mean():.3f}")
    ax.set_xlim(0.95, 2.1)
    ax.set_xlabel("Razón $C_{obtenido} / C_{óptimo}$")
    ax.set_ylabel("Nº de instancias")
    ax.set_title("Histograma de la razón de aproximación (MST)")
    ax.legend()
    ax = axes[1]
    ax.scatter(res["opt"], ratios["mst"], alpha=0.7)
    ax.axhline(2.0, color="red", linestyle="--", label="Límite teórico = 2")
    ax.axhline(1.0, color="green", linestyle="--", label="Óptimo = 1")
    ax.set_ylim(0.9, 2.1)
    ax.set_xlabel("Costo óptimo de la instancia")
    ax.set_ylabel("Razón $C_{MST} / C_{óptimo}$")
    ax.set_title("Dispersión de la razón por instancia")
    ax.legend()
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle("Ejercicio 5: razón de aproximación del algoritmo basado en MST")
    fig.tight_layout()
    out1 = RESULTS / "ejercicio5_razon_aproximacion.png"
    fig.savefig(out1, dpi=150)

    # ---- Gráfica 2: comparación de métodos ----
    order = ["mst", "mst2opt", "ls", "sa"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, data, ylabel, title in [
        (axes[0], [res[k] for k in order] + [res["opt"]], "Costo (longitud del tour)", "Costo obtenido"),
        (axes[1], [ratios[k] for k in order], "Razón $C / C_{óptimo}$", "Razón respecto al óptimo"),
    ]:
        names = [labels[k] for k in order] + (["Óptimo"] if len(data) == 5 else [])
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["tab:red", "tab:orange", "tab:blue", "tab:green", "gray"]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.3)
    axes[1].axhline(1.0, color="green", linestyle="--")
    fig.suptitle(f"Ejercicio 5: MST, MST + 2-opt, búsqueda local y recocido ({n_instances} instancias, n = {n_cities})")
    fig.tight_layout()
    out2 = RESULTS / "ejercicio5_comparacion_metodos.png"
    fig.savefig(out2, dpi=150)
    print(f"\nGráficas guardadas en {out1} y {out2}")


if __name__ == "__main__":
    main()
