import numpy as np

python_3D_polje = [[(1,2),(2,3),(3,4),(4,5)],[[1,2],[2,3],[3,4],[4,5]],[[1,2],[2,3],[3,4],[4,5]]]

np_3D_polje = np.array(python_3D_polje, dtype=np.float64)


print("Broj osi (dimenzija): " + str(np_3D_polje.ndim))
print("Dimenzije polja: " + str(np_3D_polje.shape))
print("Broj elemenata u polju: " + str(np_3D_polje.size))
print("Tip elemenata: " + str(np_3D_polje.dtype))
print(np_3D_polje)

