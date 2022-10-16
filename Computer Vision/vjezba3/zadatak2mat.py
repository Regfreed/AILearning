#Funkcije iz prethodnog zadatka prikažite na dva odvojena grafa

import numpy as np
import matplotlib.pyplot as plt

x_vrijednosti = np.arange(0,10,0.1)
cos_vrijednosti = np.cos(x_vrijednosti)
sin_vrijednosti = np.sin(x_vrijednosti)
print(x_vrijednosti)

fig, axes = plt.subplots(2)
axes[0].plot(x_vrijednosti,cos_vrijednosti)
axes[0].set_title('cosinus')
axes[0].set(xlabel='x', ylabel='y')
axes[1].plot(x_vrijednosti,sin_vrijednosti)
axes[1].set_title('sinus')
axes[1].set(xlabel='x', ylabel='y')

plt.legend()
plt.show()