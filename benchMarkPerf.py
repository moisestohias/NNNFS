"""A place to quickly benchmark different implementations """
import numpy as np
from time import perf_counter

np.random.seed(12)
Z = np.random.randn(10,10,160,160) #.astype(np.float32)
W = np.random.randn(10,10,3,3) #.astype(np.float32)

B = perf_counter()
for i in range(100): f1()
A = perf_counter()
print(f"Elapsed {A-B:.4f}")

B = perf_counter()
for i in range(100): f2()
A = perf_counter()
print(f"Elapsed {A-B:.4f}")

