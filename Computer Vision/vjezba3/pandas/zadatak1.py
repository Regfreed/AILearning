import pandas as pd

covid = pd.read_csv('covid-data.csv')
prvi = covid[covid['iso_code']=='DEU']
prvi.sort_values(by=['date'])
print(prvi.head(1))