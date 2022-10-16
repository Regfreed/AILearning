import matplotlib.pyplot as plt
import numpy as np

x_vrijednosti = np.linspace(0,2,50)
y_vrijednosti_linearno = x_vrijednosti
y_vrijednosti_kvadrati = np.power(x_vrijednosti, 2)

plt.plot(x_vrijednosti, y_vrijednosti_linearno, label='linearno')
plt.plot(x_vrijednosti, y_vrijednosti_kvadrati, label='kvadratno')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('graf')
plt.show()