import pandas as pd
import matplotlib.pyplot as plt

covid = pd.read_csv('covid-data.csv')
hrv = covid[covid['iso_code']=='HRV']

hrv.plot(x='date', y='new_cases')
plt.show()