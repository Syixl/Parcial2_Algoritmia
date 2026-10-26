"""
Ejercicio 2: Análisis algorítmico (Greedy Knapsack)

Compara el voraz clásico (mayor valor/peso primero) con la variante que prioriza
los objetos de MENOR PESO, y construye una instancia donde menor-peso es peor.
Se incluye el óptimo por programación dinámica como referencia.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def greedy(items, capacity, key):
    """items: lista de (nombre, valor, peso). Recorre en el orden dado por key y toma si cabe."""
    chosen, used, total = [], 0, 0
    for name, v, w in sorted(items, key=key):
        if used + w <= capacity:
            chosen.append(name)
            used += w
            total += v
    return chosen, used, total


def greedy_ratio(items, capacity):
    """Original: mayor valor/peso primero."""
    return greedy(items, capacity, key=lambda it: -it[1] / it[2])


def greedy_min_weight(items, capacity):
    """Modificación pedida: menor peso primero (desempate: mayor valor)."""
    return greedy(items, capacity, key=lambda it: (it[2], -it[1]))


def knapsack_dp(items, capacity):
    """Óptimo 0/1 por programación dinámica."""
    n = len(items)
    K = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        _, v, w = items[i - 1]
        for c in range(capacity + 1):
            K[i][c] = K[i - 1][c]
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(items[i - 1][0])
            c -= items[i - 1][2]
    used = sum(w for n_, _, w in items if n_ in chosen)
    return chosen[::-1], used, K[n][capacity]


def compare(title, items, capacity):
    methods = {
        "Valor/peso": greedy_ratio(items, capacity),
        "Menor peso": greedy_min_weight(items, capacity),
        "Óptimo (DP)": knapsack_dp(items, capacity),
    }
    print(f"\n=== {title} (capacidad = {capacity}) ===")
    print("Objetos (nombre, valor, peso):", items)
    print(f"{'Método':<14}{'Objetos':<22}{'Capacidad usada':<18}{'Valor total'}")
    for m, (chosen, used, total) in methods.items():
        print(f"{m:<14}{str(sorted(chosen)):<22}{f'{used}/{capacity}':<18}{total}")
    return methods


def plot(instances):
    fig, axes = plt.subplots(len(instances), 2, figsize=(11, 4.2 * len(instances)))
    axes = np.atleast_2d(axes)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for row, (title, capacity, methods) in zip(axes, instances):
        names = list(methods)
        values = [methods[m][2] for m in names]
        used = [methods[m][1] for m in names]
        bars = row[0].bar(names, values, color=colors)
        row[0].bar_label(bars, padding=2)
        row[0].set_title(f"{title}\nValor total")
        row[0].set_ylabel("Valor")
        bars = row[1].bar(names, used, color=colors)
        row[1].bar_label(bars, padding=2)
        row[1].axhline(capacity, color="red", linestyle="--", label=f"Capacidad = {capacity}")
        row[1].set_title(f"{title}\nCapacidad usada")
        row[1].set_ylabel("Peso usado")
        row[1].legend(loc="lower right")
        for ax in row:
            ax.grid(axis="y", alpha=0.3)
            for i, m in enumerate(names):
                ax.text(i, 0, "\n" + ", ".join(sorted(methods[m][0])), ha="center",
                        va="top", fontsize=8)
    fig.tight_layout()
    out = RESULTS / "ejercicio2_mochila.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en {out}")


def main():
    # Instancia de las notas de clase
    items_a = [("A", 60, 10), ("B", 100, 20), ("C", 120, 30)]
    cap_a = 50
    res_a = compare("Instancia de clase", items_a, cap_a)

    # Instancia diseñada: menor-peso es mucho peor que valor/peso
    #   - X pesa 10 (llena la mochila sola) pero vale 100 (razón 10).
    #   - a y b son ligeros pero casi no valen nada (razón 1). Menor peso los toma
    #     primero y X ya no cabe.
    items_b = [("X", 100, 10), ("a", 4, 4), ("b", 5, 5)]
    cap_b = 10
    res_b = compare("Instancia diseñada (menor peso falla)", items_b, cap_b)

    assert res_b["Menor peso"][2] < res_b["Valor/peso"][2]
    print(f"\nMenor peso obtiene {res_b['Menor peso'][2]} vs {res_b['Valor/peso'][2]} "
          f"de valor/peso: el criterio ignora por completo el valor.")

    plot([("Instancia de clase", cap_a, res_a),
          ("Instancia diseñada", cap_b, res_b)])


if __name__ == "__main__":
    main()
