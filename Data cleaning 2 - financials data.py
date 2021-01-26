import pandas_datareader.data as web
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
import requests
import os

os.chdir('S:\Things\Thesis')

dictionary = pd.read_excel('232components_dict.xlsx') # read in list of companies
current_sp500 = {}
for i in range(dictionary.shape[0]):
    current_sp500[dictionary.loc[i, "company"]] = dictionary.loc[i, "ticker"]


# Next, we use an API to access companies' historical financial data
def get_code(ticker):
    """
    Given a ticker, this function returns the code needed to download a company's financial data.
    It then outputs the code as a list of strings.
    Where is_code is code used to get income statement data.
    bs_code is code used to get balance sheet data.
    """
    is_prefix = 'https://financialmodelingprep.com/api/v3/financials/income-statement/'
    bs_prefix = 'https://financialmodelingprep.com/api/v3/financials/balance-sheet-statement/'
    suffix = '?period=quarter'
    is_link = is_prefix + ticker + suffix
    bs_link = bs_prefix + ticker + suffix
    is_code = str(requests.get(is_link).content)
    bs_code = str(requests.get(bs_link).content)
    is_code = is_code.replace('\\n', '')[2:-1]
    bs_code = bs_code.replace('\\n', '')[2:-1]
    return [is_code, bs_code]

# Download company financial data
# Structure the data as a dictionary in the following form:
# {'AAPL': [ [date1, roa1, turnover1], [date2, roa2, turnover2], ...],
#  'AMZN': [ [date1, roa1, turnover1], [date2, roa2, turnover2], ...] }
#  Where roa is return on assets (net income over total assets)
#  and turnover is asset turnover (revenue over total assets)

# Check for if companies 
null_count = 0
fin_data_dict_no_null = {}

for comp in current_sp500:
    ticker = current_sp500[comp]

    # Run the strings of codes to get IS and BS data
    exec('curr_is =' + get_code(ticker)[0])
    exec('curr_bs =' + get_code(ticker)[1])

    curr_company_data = []

    null_tick = False

    # iterate through both list of dictionaries at same time
    for (is_dict, bs_dict) in zip(curr_is['financials'], curr_bs['financials']):
        curr_date = is_dict['date']
        if curr_date != bs_dict['date']:
            print(ticker + ' date mismatch.    INDEX:' + str(i) + '    IS_DATE:' + curr_date + '    BS_DATE:' + bs_dict['date'])
        curr_rev = (is_dict['Revenue'])
        curr_net_income = is_dict['Net Income Com']
        curr_tot_assets = float(bs_dict['Total assets'])

        if curr_tot_assets == 0:
            null_tick = True
            curr_company_data += [[curr_date, 0, 0]]
        else:
            # Calculate return on assets
            if curr_net_income != '':
                curr_net_income = float(curr_net_income)
                curr_roa = curr_net_income/curr_tot_assets
            else:
                null_tick = True
                curr_roa = 0
            # Calculate asset turnover
            if curr_rev != '':
                curr_rev = float(curr_rev)
                curr_turnover = curr_rev/curr_tot_assets
            else:
                null_tick = True
                curr_turnover = 0
            # Add current quarterly data to curr_company_data 
            # in the form of [date, roa, turnover]
            curr_company_data += [[curr_date, curr_roa, curr_turnover]]
    
    if null_tick == True:
        null_count += 1
    else:
        # add current company's data to dictionary
        fin_data_dict_no_null[ticker] = curr_company_data

    # indexing for debugging
    #i += 1


# Save data to file
data_array = []
for ticker in fin_data_dict_no_null:
    current_list = [ticker] + fin_data_dict_no_null[ticker]
    data_array += [current_list]

fin_data_df = pd.DataFrame(data_array)
fin_data_df.to_excel(r"fin_data_no_null_v1.xlsx")



# Calculate the ROA and turnover data.
# And then roughly match each company's data's timeframe.
# Note that completely matching dates are impossible since each company has their own financial reporting cycle.
fin_data_df = pd.read_excel('fin_data_no_null_v2.xlsx')

roa_data = []
turnover_data = []
for row in range(fin_data_df.shape[0]):
    # Add tickers as first column
    curr_roa_data = [fin_data_df.loc[row, 0]]
    curr_turnover_data = [fin_data_df.loc[row, 0]]
    # Add remaining columns
    for col in range(1, 42):
        exec('curr_data = ' + fin_data_df.loc[row, col])
        curr_roa_data += [curr_data[1]]
        curr_turnover_data += [curr_data[2]]
    roa_data += [curr_roa_data]
    turnover_data += [curr_turnover_data]

roa_df = pd.DataFrame(roa_data)
turnover_df = pd.DataFrame(turnover_data)

roa_df.to_excel(r"data_return_on_assets.xlsx")
turnover_df.to_excel(r"data_asset_turnover.xlsx")



# Re-read stock data with date
import statistics

movements = pd.read_excel('movements_no_res.xlsx')
treasury = pd.read_excel('DGS10.xls')
treasury = pd.read_csv('USTREASURY-YIELD_no_res.csv')

sharpe_list = []
curr_diff = []
# Calculate sharpe for each company
treasury_data = list(treasury.loc[:,'DGS10'])
for i in range(movements.shape[0]):
    curr_comp_data = list(movements.loc[i,:])[19:]
    #curr_diff = []
    for j in range(len(curr_comp_data)):
        curr_diff += [curr_comp_data[j] - treasury_data[j]]
    curr_sharpe = np.mean(curr_diff)/np.std(curr_diff)
    sharpe_list += [curr_sharpe]


sharpe_list = []
curr_diff = []
# Calculate sharpe for each company
treasury_data = list(treasury.loc[0:2670,'1 MO'])
list.reverse(treasury_data)
for i in range(movements.shape[0]):
    curr_comp_data = list(movements.loc[i,:])[19:]
    last_rtn = 1
    curr_rtn = 1
    #curr_diff = []
    for j in range(len(curr_comp_data)):
        curr_rtn = curr_rtn * (1 + curr_comp_data[j])
        if (j != 0) and (j % 30 == 0) and (j - 30 <= len(treasury_data)):
            curr_diff += [curr_rtn - last_rtn - treasury_data[j-30]/12]
            last_rtn = curr_rtn
    curr_sharpe = np.mean(curr_diff)/np.std(curr_diff)
    sharpe_list += [curr_sharpe]


comp_list = []
for key in current_sp500:
    comp_list += [key]

sharpe_df = {'Company':comp_list, 'Sharpe': sharpe_list}
sharpe_df = pd.DataFrame(sharpe_df)