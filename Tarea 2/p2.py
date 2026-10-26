import multiprocessing as mp
import random
import time

def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fraction of random points that fall inside the unit quarter circle."""
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

if __name__ == "__main__":
    seeds = list(range(8))  # 8 independent runs, one seed each

    # 1. Sequential Looping
    t0 = time.perf_counter()
    seq_results = [estimate_pi(s) for s in seeds]
    t_seq = time.perf_counter() - t0

    # 2. Parallel Pool Execution
    t0 = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        par_results = pool.map(estimate_pi, seeds)
    t_par = time.perf_counter() - t0

    print(f"Sequential wall-clock: {t_seq:.3f} s")
    print(f"Parallel wall-clock:   {t_par:.3f} s (Speedup: {t_seq/t_par:.2f}x)")
    print(f"pi estimate (mean of {len(seeds)} runs): {sum(par_results) / len(par_results):.5f}")

# Destructive State Mutation (Unsafe for Concurrency)
global_accumulator = 0
def impure_sink(val: int) -> int:
    global global_accumulator
    global_accumulator += val  # Shared mutable state: Race condition under threads
    return global_accumulator

# Pure Mathematical Function (Thread-Safe & Distributable)
def pure_transform(val: int) -> int:
    return val * 2  # No side effects, deterministic output

data = [1, 2, 3, 4, 5]
transformed = list(map(pure_transform, data))
print(f"Transformed stream: {transformed}")

from typing import Iterator, Callable

def event_source() -> Iterator[int]:
    """Simulates an asynchronous stream of arriving events."""
    for item in range(1, 10):
        yield item

def stream_filter(predicate: Callable[[int], bool], stream: Iterator[int]) -> Iterator[int]:
    for event in stream:
        if predicate(event):
            yield event

def stream_map(transform: Callable[[int], int], stream: Iterator[int]) -> Iterator[int]:
    for event in stream:
        yield transform(event)

# Constructing the reactive stream
raw_stream = event_source()
evens = stream_filter(lambda x: x % 2 == 0, raw_stream)
scaled = stream_map(lambda x: x * 10, evens)

print("Reactive output events:", list(scaled))

import math

cities = {"A": (0, 0), "B": (0, 3), "E": (2, 5), "D": (4, 3), "C": (4, 0)}

def tour_length(route: list[str]) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

route = ["A", "D", "E", "B", "C"]
new_route = two_opt_move(route, 1, 3)
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")

import math
import random

def hire(candidates: list[int], r: int) -> int:
    """Observe the first r candidates, then hire the first better than all of them."""
    best_seen = max(candidates[:r], default=-1)
    return next((c for c in candidates[r:] if c > best_seen), candidates[-1])

n, trials = 100, 20_000
rules = {"Greedy (r = 0)": 0, "1/e rule (r = 37)": round(n / math.e)}
for name, r in rules.items():
    wins = sum(hire(random.sample(range(n), n), r) == n - 1 for _ in range(trials))
    print(f"{name:18s} P(best) = {wins / trials:.3f}")
# Pseaudo code for SA
# function SIMULATED-ANNEALING(problem, schedule) returns a state
#   current = problem.INITIAL_STATE
#   for t = 1 to infinity do
#     T = schedule(t)
#     if T = 0 then return current
#     next = RANDOM_SUCCESSOR(current)       # Diff 1: Random pick (not best)
#     delta_E = next.VALUE - current.VALUE
#     if delta_E > 0 then
#       current = next
#     else
#       current = next with probability e^(delta_E / T) # Diff 2: Accept bad moves

import random

# Each operator improves the solution with an unknown probability.
success_rate = {"swap": 0.10, "insert": 0.20, "2-opt": 0.35}
reward_sum = {op: 0.0 for op in success_rate}
uses = {op: 0 for op in success_rate}
epsilon = 0.1  # 10% exploration, 90% exploitation

for step in range(2_000):
    if random.random() < epsilon or step < len(success_rate):
        op = random.choice(list(success_rate))                               # explore
    else:
        op = max(success_rate, key=lambda o: reward_sum[o] / max(uses[o], 1))  # exploit
    improved = random.random() < success_rate[op]  # apply operator to the solution
    uses[op] += 1
    reward_sum[op] += improved

for op in success_rate:
    print(f"{op:7s} used {uses[op]:5d} times, learned success = {reward_sum[op] / max(uses[op], 1):.2f}")

import itertools
import math
import random

def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    """Prim's algorithm from city 0; returns the tree as {node: children}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)          # add the cheapest edge to the tree
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children

def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Walk around the tree; visiting each city once = shortcutting repeated cities."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]

def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

random.seed(7)
points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(9)]

approx = preorder(mst_prim(points))
optimal = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
              key=lambda t: tour_length(points, t))   # brute force: 8! = 40,320 tours

print(f"MST approximation: {approx}  L = {tour_length(points, approx):.2f}")
print(f"Optimal (brute):   {list(optimal)}  L = {tour_length(points, optimal):.2f}")
print(f"Ratio = {tour_length(points, approx) / tour_length(points, optimal):.3f}  (guaranteed <= 2)")

import math
import random

random.seed(1)
for n in [10**2, 10**3, 10**4, 10**5, 10**6]:
    inside = sum(random.random()**2 + random.random()**2 <= 1.0 for _ in range(n))
    estimate = 4 * inside / n
    print(f"N = {n:>9,}   pi ~ {estimate:.5f}   error = {abs(estimate - math.pi):.5f}   1/sqrt(N) = {1 / math.sqrt(n):.5f}")

import random

def is_probable_prime(n: int, k: int = 20) -> bool:
    """Miller-Rabin: 'False' is always right; 'True' is wrong with probability <= 4^-k."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for _ in range(k):
        x = pow(random.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False          # found a witness: n is certainly composite
    return True                   # probably prime

for n in [561, 7919, 2**61 - 1, 2**61 + 1]:
    print(f"{n:>22,}: {'probably prime' if is_probable_prime(n) else 'composite'}")

import math
import random
import sys

sys.setrecursionlimit(10_000)

def quicksort(a: list[int], pick_pivot, counter: list[int]) -> list[int]:
    if len(a) <= 1:
        return a
    p = pick_pivot(a)
    counter[0] += len(a) - 1                         # comparisons against the pivot
    left = [x for x in a if x < p]
    right = [x for x in a if x > p]
    return quicksort(left, pick_pivot, counter) + [p] + quicksort(right, pick_pivot, counter)

n = 2_000
data = list(range(n))                                # already sorted: worst case for a fixed pivot
for name, pick in {"First element": lambda a: a[0], "Random pivot": random.choice}.items():
    counter = [0]
    assert quicksort(data, pick, counter) == data    # Las Vegas: the answer is always correct
    print(f"{name:14s} comparisons = {counter[0]:>9,}")
print(f"{'n log2 n':14s}             = {round(n * math.log2(n)):>9,}")

calls = {"naive": 0, "memoized": 0}

def fib_naive(n: int) -> int:
    calls["naive"] += 1
    return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)

memo: dict[int, int] = {}          # the extra space we pay

def fib_memoized(n: int) -> int:
    calls["memoized"] += 1
    if n < 2:
        return n
    if n not in memo:
        memo[n] = fib_memoized(n - 1) + fib_memoized(n - 2)
    return memo[n]

n = 30
print(f"fib({n}) = {fib_naive(n)} with {calls['naive']:,} calls (naive)")
print(f"fib({n}) = {fib_memoized(n)} with {calls['memoized']:,} calls and {len(memo)} stored values (memoized)")

def knapsack(values: list[int], weights: list[int], W: int):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]   # K[i][c]: best value with items 1..i and capacity c
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]                                   # skip item i
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])         # take item i
    chosen, c = [], W                           # walk back through the table
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return K, K[n][W], chosen[::-1]

items, values, weights = ["A", "B", "C"], [60, 100, 120], [10, 20, 30]
K, best, chosen = knapsack(values, weights, 50)
print("capacity:", *[f"{c:4d}" for c in range(0, 51, 10)])
for i, row in enumerate(K):
    print(f"{'-' if i == 0 else '+' + items[i - 1]:>8s}:", *[f"{row[c]:4d}" for c in range(0, 51, 10)])
print(f"Best value = {best} with items {[items[i] for i in chosen]}  (greedy found 160)")
