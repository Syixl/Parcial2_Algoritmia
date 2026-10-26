"""
Ejercicio 1: Aceleración paralela (Parallel Speedup)

Ejecuta estimate_pi con 1, 2, 4 y 8 procesos y grafica el speedup
(S_p = T_1 / T_p) contra el número de procesos, junto con la línea ideal S_p = p.
"""
import multiprocessing as mp
import random
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

N_TASKS = 8             # tareas independientes (una semilla cada una)
SAMPLES = 500_000       # muestras por tarea
REPEATS = 3             # repeticiones por configuración (se usa la mediana)
PROCESS_COUNTS = [1, 2, 4, 8]


def estimate_pi(seed: int, samples: int = SAMPLES) -> float:
    """Monte Carlo: fracción de puntos aleatorios dentro del cuarto de círculo unitario."""
    rng = random.Random(seed)
    inside = sum(rng.random() ** 2 + rng.random() ** 2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def run_with_processes(p: int, seeds: list[int]) -> tuple[float, float]:
    """Ejecuta todas las tareas con p procesos. Devuelve (tiempo, estimación de pi)."""
    t0 = time.perf_counter()
    if p == 1:
        # Con 1 proceso se ejecuta en serie (sin Pool): es la línea base T_1 real.
        results = [estimate_pi(s) for s in seeds]
    else:
        with mp.Pool(processes=p) as pool:
            results = pool.map(estimate_pi, seeds, chunksize=1)
    elapsed = time.perf_counter() - t0
    return elapsed, sum(results) / len(results)


def amdahl(p: int, serial_fraction: float) -> float:
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / p)


def estimate_serial_fraction(procs, speedups) -> float:
    """Ajuste por mínimos cuadrados (búsqueda en rejilla) de la fracción serial f de Amdahl."""
    best_f, best_err = 0.0, float("inf")
    for i in range(0, 1001):
        f = i / 1000
        err = sum((amdahl(p, f) - s) ** 2 for p, s in zip(procs, speedups))
        if err < best_err:
            best_f, best_err = f, err
    return best_f


def main():
    seeds = list(range(N_TASKS))
    print(f"CPUs lógicas detectadas: {mp.cpu_count()}")
    print(f"{N_TASKS} tareas x {SAMPLES:,} muestras, {REPEATS} repeticiones por configuración\n")

    times = {}
    for p in PROCESS_COUNTS:
        runs = []
        for r in range(REPEATS):
            t, pi_est = run_with_processes(p, seeds)
            runs.append(t)
            print(f"  p = {p}, repetición {r + 1}/{REPEATS}: {t:.2f} s", flush=True)
        times[p] = statistics.median(runs)
        print(f"p = {p}: mediana = {times[p]:.3f} s   (pi ~ {pi_est:.5f})")

    t1 = times[1]
    speedup = {p: t1 / times[p] for p in PROCESS_COUNTS}
    efficiency = {p: speedup[p] / p for p in PROCESS_COUNTS}

    print("\n p | tiempo (s) | speedup | eficiencia")
    for p in PROCESS_COUNTS:
        print(f"{p:2d} | {times[p]:10.3f} | {speedup[p]:7.2f} | {efficiency[p]:9.1%}")

    f = estimate_serial_fraction(PROCESS_COUNTS, [speedup[p] for p in PROCESS_COUNTS])
    print(f"\nFracción serial efectiva estimada (Amdahl): f = {f:.3f}")

    # ---- Gráfica ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(PROCESS_COUNTS, PROCESS_COUNTS, "k--", label="Speedup ideal (S = p)")
    ax.plot(PROCESS_COUNTS, [speedup[p] for p in PROCESS_COUNTS], "o-", color="tab:blue",
            linewidth=2, label="Speedup medido")
    ps = [1, 2, 3, 4, 5, 6, 7, 8]
    ax.plot(ps, [amdahl(p, f) for p in ps], ":", color="tab:red",
            label=f"Ajuste de Amdahl (f = {f:.3f})")
    ax.axvline(mp.cpu_count(), color="gray", alpha=0.4)
    ax.text(mp.cpu_count(), 0.5, f" {mp.cpu_count()} CPUs lógicas", color="gray", fontsize=8)
    ax.set_xlabel("Número de procesos")
    ax.set_ylabel("Speedup  $T_1 / T_p$")
    ax.set_title("Ejercicio 1: speedup de estimate_pi (Monte Carlo)")
    ax.set_xticks(PROCESS_COUNTS)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "ejercicio1_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"\nGráfica guardada en {out}")


if __name__ == "__main__":
    main()
