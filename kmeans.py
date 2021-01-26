import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
import os
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

# Data trimming notes
# 2008 recession from dec 2007 - june 2009
# februrary 20 2020
# 2009-jun-1: 548
# 2009-jul-1: 570
# 2020-feb-3: 3335
# JULY 1 2009 - FEB 20 2020

# Current results:
# Default initialization, 1000 tries, 2000 max it, average inertia = 121.2806240364205
# Random initialization, 1000 tries, 2000 max it, average inertia = 125.1204598038364

# Treasury data from https://fred.stlouisfed.org/series/DGS10
# API for: https://financialmodelingprep.com/api/v3/financials/cash-flow-statement/AAPL?period=quarter

os.chdir('S:\Things\Thesis')

# import normalized movements as dataframe
movements = pd.read_excel('232components_norm.xlsx', index_col = 0)

nrow = movements.shape[0]
ncol = movements.shape[1]

# For some reason the imported dataframe has a new, useless column, we delete that
#movements.drop(["Unnamed: 0"], axis = 1)

# import historical s&p 500 data as data frame
sp500_hist = pd.read_csv('sp500_historical.csv')

# import dictionary of components as dataframe
dictionary = pd.read_excel('232components_dict.xlsx')
current_sp500 = {}
for i in range(movements.shape[0]):
    current_sp500[dictionary.loc[i, "company"]] = dictionary.loc[i, "ticker"]


# ==================================================================================
# First, conduct a K-means with 11 clusters
# This is to follow S&P 500's own sector breakdown
# More information can be found here: https://us.spindices.com/indices/equity/sp-500
# https://financialmodelingprep.com/developer/docs/

best_inertia = 1000
best = ''
for i in range(0, 100):
    kmeans_11 = KMeans(n_clusters = 11, max_iter = 2000)
    kmeans_11.fit(movements)
    if kmeans_11.inertia_ < best_inertia:
        best_inertia = kmeans_11.inertia_
        best = kmeans_11

# Predict the data and put them into a dataframe for viewing
predicted = pd.DataFrame(index = range(nrow), columns = ['company', 'ticker', 'group'])

i = 0
for key in current_sp500.keys():
    current_prediction = best.predict([list(movements.loc[i])])[0]
    predicted.loc[i, 'company'] = key
    predicted.loc[i, 'ticker'] = current_sp500[key]
    predicted.loc[i, 'group'] = current_prediction
    i = i + 1

predicted.sort_values(by = ['group'], inplace = True)

# Extract the predicted dataframe to Excel
pd.DataFrame(predicted).to_excel(r'test.xlsx', header = True)

#predicted_clusters=11_default_it=2000

# TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST
# This is the test with Kmeans initial clusters set to 'random'
# This is still with 11 clusters

kmeans_11_2 = KMeans(n_clusters=11, init = 'random', max_iter = 2000)
kmeans_11_2.fit(movements)

predicted2 = pd.DataFrame(index = range(nrow), columns = ['company', 'ticker', 'group'])

i = 0
for key in current_sp500.keys():
    current_prediction = kmeans_11_2.predict([list(movements.loc[i])])[0]
    predicted2.loc[i, 'company'] = key
    predicted2.loc[i, 'ticker'] = current_sp500[key]
    predicted2.loc[i, 'group'] = current_prediction
    i = i + 1

predicted2.sort_values(by = ['group'], inplace = True)

pd.DataFrame(predicted2).to_excel(r'predicted_clusters=11_random_it=2000.xlsx', header = True)


# It seems that the random centroid initialization method has a more even
# distribution of number of points within clusters
# Next, I should test each one's effectiveness by repeating it a few times and 
# taking the best sample, then compare it to the index

# ==================================================================================
# Choosing k with elbow method

# Test elbow method with default initialization
WSS = []
k_range = range(5,50)
for k in k_range:
    curr_kmeans = KMeans(n_clusters = k, max_iter = 1000)
    curr_kmeans.fit(movements)
    WSS += [curr_kmeans.inertia_]

plt.plot(k_range, WSS, 'bx-')
plt.xlabel('k')
plt.ylabel('Within Cluster Sum of Squares')
plt.title('Elbow Method: Plot of K vs WSS')
plt.show()

# Test elbow method with random initialization
WSS = []
for k in k_range:
    curr_kmeans = KMeans(n_clusters = k, init = 'random', max_iter = 1000)
    curr_kmeans.fit(movements)
    WSS += [curr_kmeans.inertia_]

plt.plot(k_range, WSS, 'bx-')
plt.xlabel('k')
plt.ylabel('Within Cluster Sum of Squares')
plt.title('Elbow Method: Plot of K vs WSS')
plt.show()

# ==================================================================================
# Choosing k with Silhouette method
from sklearn.metrics import silhouette_score
import random

curr_kmeans = KMeans(n_clusters = 8, max_iter = 1000)
curr_labels = curr_kmeans.fit(movements).labels_
sil = []
for i in range(10):
    random.seed(i)
    sil += [(silhouette_score(movements, curr_labels, metric = 'euclidean'))]

fin_data = pd.read_excel('data_roa_and_turnover.xlsx')
fin_data = pd.read_excel('movements_no_res.xlsx')
fin_data = fin_data.drop(['Unnamed: 0'], axis = 1)

# test Silhouette method with default initialization
sil = []
k_range = range(4,12)
for k in k_range:
    curr_sil = []
    for i in range(50):
        curr_kmeans = KMeans(n_clusters = k, max_iter = 1000)
        curr_labels = curr_kmeans.fit(fin_data).labels_
        curr_sil += [(silhouette_score(fin_data, curr_labels, metric = 'euclidean'))]
    sil += [np.mean(curr_sil)]

plt.plot(k_range, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette')
plt.title('Silhouette Method: Plot of K vs Silhouette')
plt.show()

# test Silhouette method with random initialization
sil = []
k_range = range(2,25)
for k in k_range:
  curr_kmeans = KMeans(n_clusters = k, init = 'random', max_iter = 1000)
  curr_labels = curr_kmeans.fit(movements).labels_
  sil += [(silhouette_score(movements, curr_labels, metric = 'euclidean'))]

plt.plot(k_range, sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette')
plt.title('Silhouette Method: Plot of K vs Silhouette')
plt.show()

'''
plt.figure(figsize=(18,16))
ax1 = plt.subplot(221)
plt.plot(new[0][:])
plt.title(companies[0])

for i in range(1, len(companies_dict)): 
    plt.subplot(222, sharey=ax1)
    plt.plot(new[i][:])
    plt.title(companies[i])
plt.show()
'''

# predict cluster labels
labels = pipeline.predict(movements)
 
# create a DataFrame aligning labels & companies
df = pd.DataFrame({'labels': labels, 'companies': companies})

# display df sorted by cluster labels
print(df.sort_values('labels'))

# display WCSS
print(kmeans_11.inertia_)


# Do a histogram on the density
import matplotlib.pyplot as plt

plt.hist(list(predicted2.loc[:,'group']), bins = 11)
plt.show()

color_cycle = ['blue', 'red', 'green', 'yellow', 'cyan', 'orange', 'black', 'magenta', 'blue', 'red', 'green', 'yellow', 'cyan', 'orange', 'black', 'magenta']
for i in range(7):
    plt.figure(i)
    plt.plot(list(movements.loc[i,:]), color = color_cycle[i], linewidth = 0.2)
    plt.show()


# trimming the data to from Jul-1 2009 to Feb-3-2020
movements_trim = movements
for i in range(571):
    movements_trim = movements_trim.drop([i], axis = 1)

for i in range(3335, movements.shape[1]):
    movements_trim = movements_trim.drop([i], axis = 1)

movements_trim.to_excel(r'trimmed_data.xlsx', header = True)


test = KMeans(n_clusters = 5, max_iter = 2000)
test.fit(movements_trim)

predicted = pd.DataFrame(index = range(movements_trim.shape[0]), columns = ['company', 'ticker', 'group'])

i = 0
for key in current_sp500.keys():
    current_prediction = test.predict([list(movements_trim.loc[i])])[0]
    predicted.loc[i, 'company'] = key
    predicted.loc[i, 'ticker'] = current_sp500[key]
    predicted.loc[i, 'group'] = current_prediction
    i = i + 1

plt.hist(list(predicted.loc[:,'group']), bins = 5)
plt.show()


# Conduct PCA then map data to 2d

movements_trim = pd.read_excel('trimmed_data.xlsx')
movements_trim_stand = movements_trim

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


fin_data = pd.read_excel('data_roa_and_turnover.xlsx')

fin_data = fin_data.drop(['Unnamed: 0'], axis = 1)

fin_data_stand = StandardScaler().fit_transform(fin_data)

pca = PCA(n_components = 3)

principalComponents = pca.fit_transform(fin_data_stand)
principal_df = pd.DataFrame(principalComponents)

labels = pd.read_excel('main_kmean2_finance_7_sumd_27.6499.xls')

groups = pd.concat([principal_df, labels], axis = 1)

groups = groups.groupby('label')


color = ['magenta', 'cyan', 'red', 'blue', 'black', 'orange', 'green', 'purple']
marker = ['.', 'o', '^', 's', 'p', '*', 'x', '+']

fig = plt.figure().gca(projection = '3d')
i = 0
legend_list = []
for name, group in groups:
    a = fig.scatter(group[0], group[1], group[2], label = name, color = color[i], marker = marker[i], s = 40)
    i += 1
fig.set_xlabel('Principal Component 1')
fig.set_ylabel('Principal Component 2')
fig.set_zlabel('Principal Component 3')


# Read treasury rate from csv

treasury = pd.read_csv('USTREASURY-YIELD_no_res.csv')
treasury = treasury.loc[:,'10 YR']
treasury = treasury[::-1]

# Note that the indexes are reversed, so we use a loop to iterate through it

ex_return = movements_trim
ex_return.drop(columns = ['Unnamed: 0'])


# change the indexes from 571-3334 to 0-2763
for i in range(571, 3335):
    ex_return = ex_return.rename(columns = {i : (i-571)})


treasury_return = []
i = 2647
while i >= 0:
    treasury_return += [treasury[i] - treasury[i + 1]]
    i -=1

name_list = []
for i in range(114):
    curr_ticker = fin_data.loc[i,0]
    curr_name = ''
    for key, value in current_sp500.items():
        if value == curr_ticker:
            curr_name = key
    name_list += [curr_name]

d = {'label': labels['label'], 'company': name_list}
df = pd.DataFrame(data = d)
df.to_excel('main_kmean2_finance_4_sumd.xls')

# LUV removed
# MDT removed
