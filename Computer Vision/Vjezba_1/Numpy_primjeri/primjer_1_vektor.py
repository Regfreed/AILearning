import numpy as np

python_1D_polje = [5, 6, 13]

np_1D_polje = np.array(python_1D_polje)

print("Broj osi (dimenzija): " + str(np_1D_polje.ndim))
print("Dimenzije polja: " + str(np_1D_polje.shape))
print("Broj elemenata u polju: " + str(np_1D_polje.size))
print("Tip elemenata: " + str(np_1D_polje.dtype))

np_1D_polje.dtype = np.float32
print("Novi tip elemenata: " + str(np_1D_polje.dtype))