"""
Ejercicio 3: Búsqueda local (2-opt para TSP)

Implementa 2-opt (primera mejora) y mide cuántas mejoras se necesitan para llegar
a un óptimo local desde (1) un tour aleatorio y (2) Vecino Más Cercano (NN) con
nodo inicial aleatorio.
"""
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ---------- utilidades TSP (también las importan los ejercicios 4 y 5) ----------
def random_cities(n: int, rng: random.Random, size: float = 100.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def dist_matrix(points):
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def tour_length(D, tour) -> float:
    n = len(tour)
    return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))


def nearest_neighbor(D, start: int):
    n = len(D)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: D[last][j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def two_opt_delta(D, tour, i, k) -> float:
    """Cambio de longitud al invertir tour[i..k] (quita aristas (i-1,i) y (k,k+1))."""
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[k], tour[(k + 1) % n]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]


def two_opt_move(tour, i, k):
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]


# ---------- 2-opt: primera mejora ----------
def two_opt(D, tour, eps: float = 1e-9):
    """
    Aplica 2-opt hasta alcanzar un óptimo local.
    Devuelve (tour, longitud final, número de mejoras aplicadas).
    """
    tour = tour[:]
    n = len(tour)
    improvements = 0
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                if two_opt_delta(D, tour, i, k) < -eps:
                    tour = two_opt_move(tour, i, k)
                    improvements += 1
                    improved = True
    return tour, tour_length(D, tour), improvements


def experiment(n_cities=30, n_instances=100, seed=0):
    rng = random.Random(seed)
    rows = {"Aleatorio": [], "Vecino Más Cercano": []}
    for _ in range(n_instances):
        D = dist_matrix(random_cities(n_cities, rng))
        # 1) inicio aleatorio
        t = list(range(n_cities))
        rng.shuffle(t)
        L0 = tour_length(D, t)
        _, L, k = two_opt(D, t)
        rows["Aleatorio"].append((L0, L, k))
        # 2) NN desde un nodo aleatorio
        t = nearest_neighbor(D, rng.randrange(n_cities))
        L0 = tour_length(D, t)
        _, L, k = two_opt(D, t)
        rows["Vecino Más Cercano"].append((L0, L, k))
    return {m: np.array(v) for m, v in rows.items()}


def main():
    n_cities, n_instances = 30, 100
    data = experiment(n_cities, n_instances)

    print(f"{n_instances} instancias aleatorias de {n_cities} ciudades (misma instancia para ambos inicios)\n")
    print(f"{'Inicio':<20}{'L inicial':>11}{'Mejoras (media)':>17}{'Mejoras (mediana)':>19}{'L final (media)':>17}")
    for m, a in data.items():
        print(f"{m:<20}{a[:, 0].mean():>11.1f}{a[:, 2].mean():>17.1f}{np.median(a[:, 2]):>19.1f}{a[:, 1].mean():>17.1f}")

    fewer = np.mean(data["Vecino Más Cercano"][:, 2] < data["Aleatorio"][:, 2])
    diff = data["Vecino Más Cercano"][:, 1] / data["Aleatorio"][:, 1]
    print(f"\nNN necesita menos mejoras que aleatorio en {fewer:.0%} de las instancias")
    print(f"Razón de longitud final NN/aleatorio: media = {diff.mean():.3f} "
          f"(<1 significa que NN termina en un óptimo local mejor)")

    # ---- Gráficas ----
    labels = list(data)
    colors = ["tab:red", "tab:blue"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    titles = ["Nº de mejoras hasta el óptimo local", "Longitud del tour inicial",
              "Longitud del óptimo local alcanzado"]
    cols = [2, 0, 1]
    for ax, title, c in zip(axes, titles, cols):
        bp = ax.boxplot([data[m][:, c] for m in labels], tick_labels=labels, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mejoras (movimientos 2-opt aplicados)")
    axes[1].set_ylabel("Longitud")
    axes[2].set_ylabel("Longitud")
    fig.suptitle(f"Ejercicio 3: 2-opt desde inicio aleatorio vs. Vecino Más Cercano "
                 f"({n_instances} instancias, n = {n_cities})")
    fig.tight_layout()
    out = RESULTS / "ejercicio3_2opt.png"
    fig.savefig(out, dpi=150)
    print(f"\nGráfica guardada en {out}")


if __name__ == "__main__":
    main()
