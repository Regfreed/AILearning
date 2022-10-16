import numpy as np

a = np.array([[1, 2, 3],
             [5, 60, 7]])

if a.mean()>10:
    print(a.sum())
else:
    print(a.max())