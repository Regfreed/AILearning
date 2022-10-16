import pandas as pd
import matplotlib.pyplot as plt

covid = pd.read_csv('covid-data.csv')
hrv = covid[covid['iso_code']=='HRV']
ma = hrv['new_cases'].max()
print(hrv['date'] hrv['new_cases']==ma)