import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
import os

os.chdir('S:\Things\Thesis') # choose the directory of your data

changes = pd.read_excel('sp500change.xlsx') # import s&p 500 historical change data
constituent = pd.read_csv('sp-500-index-04-15-2020.csv') # import s&p500 constituent data on apr/15/2020

# Create a dictionary with the current S&P500 components (as of 4/14/2020)
# Length of dictionary is 505
# Company names are keys and company tickers are values
current_sp500 = {}
for i in range(len(constituent)):
    current_sp500[constituent['Name'][i]] = constituent['Symbol'][i]

# Define a function to find keys corresponding to a given value
def getKeysByValue(dictOfElements, valueToFind):
    '''
    Get a list of keys from dictionary which has the given value
    '''
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

# Delete all entries that appear in the historical changes from the dict
for i in changes['Ticker']:
    current_key = getKeysByValue(current_sp500, i)
    if len(current_key) != 0:
        current_sp500.pop(current_key[0])

# Replace ticker symbol "." with "-", as data reader have a problem with "."
for key in current_sp500:
    value = current_sp500[key]
    if '.' in value:
        new_value = value.replace('.', '-')
        current_sp500.pop(key)
        current_sp500[key] = new_value

data_source = 'yahoo' # Define which online source to use
start_date = '2006-10-31' # Define start and end dates
end_date = '2020-04-14'

# Use pandas_datareader.data.DataReader to load the desired data list(companies_dict.values()) used for python 3 compatibility
panel_data = web.DataReader(list(current_sp500.values()), data_source, start_date, end_date)

# Export to Excel, result is a 3385x1392 matrix
panel_data.to_excel(r'232components.xlsx', header = True)

# Find Stock Open and Close Values
stock_close = panel_data['Close']
stock_open = panel_data['Open']

# Calculate daily stock returns
# When I wrote this code, I forgot the word "return" and instead used "movements" for "daily percentage movements"
# So bear with me on the bad variable names -_-
stock_close = np.array(stock_close).T
stock_open = np.array(stock_open).T
row, col = stock_close.shape

movements = np.zeros([row, col]) # create movements dataset filled with 0's

for i in range(0, row):
    movements[i,:] = np.subtract(stock_close[i,:], stock_open[i,:])

# normalize
norm_movements = np.zeros([row, col])

for i in range(0, row):
    norm_movements[i,:] = np.divide(movements[i,:], stock_open[i,:])

# Export to Excel, result is a 232x3385 matrix
# Here, each rows is a company, and each column is a daily return
(pd.DataFrame(norm_movements)).to_excel(r'norm_232components.xlsx', header = True)

