import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

pd.set_option("display.max_rows", None, "display.max_columns", None)

data = pd.read_csv(r'C:\Users\Matteo\Desktop\CISC 595 project\Border_Crossing_Entry_Data.csv')

data['Date'] = pd.to_datetime(data['Date'])

dataset2 = pd.read_excel(r'C:\Users\Matteo\Desktop\CISC 595 project\CivilianLaborForceLevel.xlsx', sheet_name='Sheet1')

dataset3 = pd.read_excel(r'C:\Users\Matteo\Desktop\CISC 595 project\TXUR.xlsx')

                         
'--------- Pie Chart showcasing people entering from south and northern borders ----------------------'

people = data[data['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]

people_borders = people[['Border','Value']].groupby('Border').sum()

values = people_borders.values.flatten()
labels = people_borders.index
fig = go.Figure(data=[go.Pie(labels = labels, values=values)])
fig.update(layout_title_text='Immigrants entered within USA since 1996')

fig.show()

'--------- Pie Chart showcasing mode of transportation across border ----------------------'

people_measure = people[['Measure','Value']].groupby('Measure').sum()
values = people_measure.values.flatten()
labels = people_measure.index
fig = go.Figure(data=[go.Pie(labels = labels, values=values)])
fig.update(layout_title_text='Total inbound persons, since 1996')

fig.show()


'------------------------Pie Chart showcasing immigration across states-------------------------------'

start_year = 1996
end_year = 2018

people = data[data['Measure'].isin(['Personal Vehicle Passengers', 'Bus Passengers','Pedestrians', 'Train Passengers'])]


p_states = people[['Date','State','Value']].set_index('Date')
p_states = p_states.groupby([p_states.index.year, 'State']).sum()

p_states = p_states.loc(axis=0)[start_year:end_year,:].groupby('State').mean()

p_states = p_states['Value'].sort_values()

rest = p_states[p_states < p_states.sum()*.04].sum()
p_states = p_states[p_states > p_states.sum()*.04].append(pd.Series({'Rest' : rest}))

p_states_tot = people[['State','Value']].groupby('State').sum()
p_states_tot = p_states_tot['Value'].sort_values()
rest_tot = p_states_tot[p_states_tot < p_states_tot.sum()*.04].sum()
p_states_tot = p_states_tot[p_states_tot > p_states_tot.sum()*.04].append(pd.Series({'Rest' : rest_tot}))


f,ax = plt.subplots(ncols=2, nrows=1)

p_states_tot.plot.pie( ax = ax[0], autopct = '%1.1f%%')

ax[0].set(title = 'Immigrants by State, since 1996', ylabel = '')
f.show()


'--------- Linear Regression of years vs people coming in ----------------------'

data2 = data

data2['Date'] = pd.to_datetime(data['Date'], errors = 'coerce')
data2['Year'] = data2['Date'].dt.year
data2.sort_values('Year')

a = 1996
averages = []
years = list(set(data2['Year'].tolist()))
years.append(None)
years.append(None)

while a < 2022:
    data3 = data2.loc[data2['Year'] == a]    
    b = data3['Value'].mean()
    averages.append(b)
    a += 1

data4 = pd.DataFrame()
data4['Years'] = years
data4['Values'] = averages

sns.pairplot(data4, x_vars='Years', y_vars='Values', height=10, aspect=1, kind='reg')
plt.ylabel('Immigrants')
plt.show()

'----------------------Linear Regression of years vs labor force in USA ---------------------------------'

CivilianLaborForce = dataset2['Average'].tolist()

data4['LaborForce'] = CivilianLaborForce

sns.pairplot(data4, x_vars='Years', y_vars='LaborForce', height=10, aspect=1, kind='reg')
plt.show()

'----------------------Linear Regression of years vs UR in TX ---------------------------------'

data5 = dataset3

data5['Date'] = pd.to_datetime(data5['DATE'], errors = 'coerce')
data5['Year'] = data5['Date'].dt.year
data5.sort_values('Year')

a = 2010
averages = []
years = list(set(data5['Year'].tolist()))
years = [x for x in years if x > 2009]

while 2009 < a < 2021:
    data6 = data5.loc[data5['Year'] == a]    
    b = data6['TXUR'].mean()
    averages.append(b)
    a += 1

data7 = pd.DataFrame()
data7['Years'] = years
data7['Unemployment Rate in Texas'] = averages

sns.pairplot(data7, x_vars='Years', y_vars='Unemployment Rate in Texas', height=10, aspect=1, kind='reg')
plt.show()

'--------- Linear Regression of years vs people coming in ----------------------'

data2 = data

data2['Date'] = pd.to_datetime(data['Date'], errors = 'coerce')
data2['Year'] = data2['Date'].dt.year
data2.sort_values('Year')

a = 2010
averages = []
years = list(set(data2['Year'].tolist()))
years = [x for x in years if x > 2009]

while a < 2022:
    data3 = data2.loc[data2['Year'] == a]    
    b = data3['Value'].mean()
    averages.append(b)
    a += 1

averages.pop()
averages.pop()
data4 = pd.DataFrame()
data4['Years'] = years
data4['Values'] = averages

sns.pairplot(data4, x_vars='Years', y_vars='Values', height=10, aspect=1, kind='reg')
plt.ylabel('Immigrants')
plt.show()


