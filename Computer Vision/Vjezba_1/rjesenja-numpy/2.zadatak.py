import numpy as np
python_3D_polje = [[(1,2),(2,3),(3,4),(4,5)],[[1,2],[2,3],[3,4],[4,5]],[[1,2],[2,3],[3,4],[4,5]]]

np_3D_polje = np.array(python_3D_polje)

print("Tip elemenata: " + str(np_3D_polje.dtype))

np_3D_polje.dtype=np.float16
print("Tip elemenata: " + str(np_3D_polje.dtype))
