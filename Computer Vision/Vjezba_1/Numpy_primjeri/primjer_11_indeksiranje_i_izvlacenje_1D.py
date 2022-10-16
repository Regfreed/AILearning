import numpy as np

a = np.array([1, 4, 5, 2, 6, 3, 6])
print(a[4])
a[6] = 7
print(a)
print(a[2:6])
a[:4] = 0
print(a)