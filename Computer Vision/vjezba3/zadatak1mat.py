'''Prikažite sinus i kosinus vrijednosti u rasponu [0, 10]. Dodjelite odgovarajuće nazive
koordinatnim osima, te odgovarajući naslov grafa. Prikažite legendu. Obje funkcije prikažite na
istom grafu.'''

import numpy as np
import matplotlib.pyplot as plt

x_vrijednosti = np.arange(0,10,0.1)
cos_vrijednosti = np.cos(x_vrijednosti)
sin_vrijednosti = np.sin(x_vrijednosti)
print(x_vrijednosti)

plt.plot(x_vrijednosti, cos_vrijednosti, label='cos')
plt.plot(x_vrijednosti,sin_vrijednosti,label='sin')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('graf')
plt.show()