import numpy as np

python_2D_polje = [[3,7,1],[4,5,6]]
np_2D_polje = np.array(python_2D_polje)
print("Broj osi (dimenzija): " + str(np_2D_polje.ndim))
print("Dimenzije polja: " + str(np_2D_polje.shape))
print("Broj elemenata u polju: " + str(np_2D_polje.size))
print("Tip elemenata: " + str(np_2D_polje.dtype))

np_2D_polje = np.array(python_2D_polje, dtype=np.int16)
print("Tip elemenata drugog polja: " + str(np_2D_polje.dtype))
