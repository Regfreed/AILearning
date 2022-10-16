import numpy as np

a = np.array([[31,  42, 63,34],
              [52,   6,  7,28],
              [669,103,113,12]])

print(a.min())
print(a.argmin())
print(a.max())
print(a.argmax())
print(a.sum())
print(a.mean())
print(a.prod())
a.sort()
print(a)