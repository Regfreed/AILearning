import numpy as np

a = np.array([5, 4, 6])
b = np.array([1, 2, 3])
print("a + b = " + str(a + b))
print("a - b = " + str(a - b))
print("a * b = " + str(a * b))
print("a / b = " + str(a / b))

a = np.array([[1, 2],
              [4, 5]])
b = np.array([[7, 8],
              [10, 11]])

print("a * b = " + str(a * b))

c = np.matmul(a, b)
print("a x b = " + str(c))