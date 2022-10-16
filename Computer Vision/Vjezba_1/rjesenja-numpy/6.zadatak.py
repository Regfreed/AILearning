import numpy as np

polje1 = np.eye(4)
polje2 = np.full((4,4),9)

print(polje1)
print(polje2)

print('zbroj \n'+str(polje1+polje2))
print('umnožak \n'+str(polje1*polje2))
print('razlika \n'+str(polje1-polje2))
print('dijeljenje \n'+str(polje1/polje2))