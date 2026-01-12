# py_scratch.py #

    """
    This IS NOT a standalone script; it is a scratchpad for Python code snippets, which I have written, 'borrowed' and duct-taped together during my time as a data-scientist and fraud fighter, while battling the evil scoundrels and villains whose MO is cyber-enabled and cyber-dependant fraud. (As well as through some empassioned tinkering and bamboozlery among so-called 'toy' datasets).
    
    Feel free to use, borrow, take, steal, adapt, modify, and repurpose any of the code snippets herein, as you see fit. If you do so, I would appreciate a shout-out or a mention in your acknowledgements. Happy coding!
    
    P.S. "Choose not thy thugg-lyfe and/or or bedevilment, but per chance a path less eville." - Me
    """

# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Misc pandas 🐼🐼🐼 to live by # 
# ----------------------------------------------------------------------------------------------- #

# Datetime ops
from datetime import datetime

# Convert a UNIX datetime to a datetime64[ns] in pandas (useful in Etherscan and Kaiko datasets;)
df['datetime'] = pd.to_datetime(df['unix_timestamp'], unit='ms')

# Then take the newly created 'datetime' column and set it as the index, and sort the index in ascending order ;)
df.set_index('datetime', inplace=True).sort_index(ascending=True, inplace=True)

# Set an obj to datetime and round to nearest minute
dt['datetime'] = pd.to_datetime(dt['datetime']).dt.round('min')

# Set an obj to datetime and round to nearest day
dt['datetime'] = pd.to_datetime(dt['datetime']).dt.round('D')

# Moar datetime ops:
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f')  # Convert to (e.g.) '2024-10-01 12:34:56.789123' format

df['datetime'] = pd.to_datetime(df['transaction_created_at'], utc=True).dt.tz_convert(None)  # Convert to UTC datetime


# datetime for saving files/ experiments/ etc. with timestamps
# Set up a 'saved_models' directory (if it doesnt already exist):
from datetime import datetime

save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)  # creates it if it doesn't exist

# Add the current date to the filename
current_date = datetime.now().strftime("%Y%m%d")
model_save_path = os.path.join(save_dir, f"efficientnet_b0_pizza_steak_sushi_{current_date}.pth")

# Can easily adjust the '.now' output for more/ less granularity, diff't formats, etc.:
# e.g., for a current date/time of 2025-10-24 15:30 (local):
datetime.now().strftime("%Y%m%d") # '20251024'
datetime.now().strftime("%Y-%m-%d") # '2025-10-24'
datetime.now().strftime("%Y%m%d_%H%M") # '20251024_1530'
datetime.now().strftime("%Y%m%d-%H%ML") # '20251024-1530L' (with 'L' appended for local time)

# date string to UNIX timestamp at midnight function #

from datetime import datetime, timezone

# Define the function
def date_to_unix_midnight(date_string):

    try:
        date_obj = datetime.strptime(date_string, '%Y-%m-%d')
        
        # Set the timezone to UTC (UNIX time is based on UTC)
        date_obj = date_obj.replace(tzinfo=timezone.utc)
        
        # Convert the datetime object to a UNIX timestamp
        unix_timestamp = int(date_obj.timestamp())
        
        return unix_timestamp

    except ValueError:
        # Handle invalid date format
        raise ValueError("Invalid date format. Please use 'yyyy-mm-dd'.") 


# Custom function to convert the 'birthdate' (if a string or some other non-datetime format) column into a datetime variable type:

def date_formatter(date):
    '''takes a string (or other non-dt obj)and converts into a datetime object'''
    date_str = str(date)  # Ensure the input is a string
    if date_str.find('-') > -1:
        return pd.to_datetime(date_str, format='%Y-%m-%d') # Adjust format as needed (this is YYYY-MM-DD)
    elif date_str.find('/') > -1:
        return pd.to_datetime(date_str, format='%m/%d/%Y') # Adjust format as needed (this is MM/DD/YYYY)
    else :
        return pd.to_datetime(np.nan)  # Return NaN for unrecognized formats


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #

# Float decimal points (several options n trix)

pd.options.display.float_format = '{:.2f}'.format  # Set global float format to 2 decimal places

df['column_name'] = df['column_name'].round(4)  # Round specific column to 4 decimal places

df['column_name'] = df['column_name'].apply(lambda x: ':,.4f').format # Format specific column to 4 decimal places as string

# Set a float to something ridiculously small, like 10^-12 (useful in sh*tcoin datasets;)
df['price'] = df['price'].astype(float).round(12)

# ----------------------------------------------------------------------------------------------- #

# Multi-column conversion functions

# Convert multiple columns to float (two options)
# Option1
def convert_columns_to_float(df, columns):
    for col in columns:
        df[col] = df[col].astype(float)
    return df

# Option2
def convert_columns_to_float(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        
# Sample usage:
convert_columns_to_float(df, ['col1', 'col2', 'col3'])
df.info()



# Convert multi-columns to int64:
def convert_columns_to_int64(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
        
        
# Example usage - on strings in Etherscan data:
convert_columns_to_int64(df, ['blockNumber', 'transactionIndex', 'gasUsed', 'gasPrice', 'cumulativeGasUsed', 'confirmations'])
print(df.info())
df.head(3)

# Simple string representation of int change to int (e.g., remove the ',' and convert to int):

# Remove commas from the "y" column and convert it to integer type
# Keep the exact behavior as in your notebook
# df['y'].str.replace(',', '').astype(int)  # alternative kept as a comment

df["y"] = df["y"].str.replace(",", "").astype(int)
df.head()

# ----------------------------------------------------------------------------------------------- #

# Hex conversions!

# This function is for string SLICES from longer hex strings (e.g., eventLogs, etc.) {NOTE: must not be a '0x' prefix!}

def hex_to_string(hex_input):
    try:
        return bytes.fromhex(hex_input).decode('utf-8')
    except ValueError:
        return "Invalid hex input"
    
# Example usage:
# Step 1: slice the portion of the long hex string for conversion;
df['raw_dest_addr_hex'] = df['data'].str[578:646]  # Slice the hex string

# Step 2: apply the function and check .value_counts() or .head() on the df to make sure the output is coherent;
df['trx_dest_addr'] = df['raw_dest_addr_hex'].apply(hex_to_string)
df['trx_dest_addr'].value_counts()

# ----------------------------------------------------------------------------------------------- #

# Hex to int:

#Option 1 (Pythonic version):
def hex_to_int(hex_str):
    try:
        return int(hex_str, 16)
    except ValueError:
        return "Invalid hex input"  # or handle the error as needed

# Example usage:
df['blockNumber_int'] = df['blockNumber'].apply(hex_to_int)

# Option 2a (Lambda version ~ for single use cases):
hex_to_int = lambda x: int(x, 16)
# Example usage:
hex_to_int('F4240')

# Option 2B (Lambda version ~ in a df; with a check for '0x' prefix):
hex_to_int = lambda x: int(x, 16) if isinstance(x, str) and x.startswith('0x') else None
# Example usage:
df['blockNumber_int'] = df['blockNumber'].apply(hex_to_int)


# ----------------------------------------------------------------------------------------------- #

# Drops, replaces, renames:

# Drop the really annoying 'Unnamed: 0' column that appears when you read a CSV with an index column:
df.drop(['Unnamed: 0'], axis=1, inplace=True)  # Drop a column

# If excessive values in a col are 'NaN' and this makes them irrelevant, can drop the rows with NaN in that col:
df.dropna(
    subset=['column_name']
    , inplace=True
    )  # Drop rows with NaN in 'column_name'

# Check for NaNs remaining:
df['column_name'].isna().any()
# or
df['column_name'].isnull().any()

# dropnas
df2 = df.replace(to_replace=[None], value=np.nan).dropna()

# Assuming your DataFrame is named 'df'
# Insert a '-' between the base and quote currency in the 'symbol' column
df['instrument'] = df['symbol'].str.replace(
    r'([a-z]+)(usd|usdt)'
    , r'\1-\2'
    , regex=True
    )


# ----------------------------------------------------------------------------------------------- #
# Non-statistical normalization of strings (e.g., column names):

# Normalizing column names to lowercase and replacing spaces with underscores:
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Normalizing column names of trading pairs (e.g., 'BTC/USD' to 'btc-usd'):
import re

# option 1 
def normalize_trading_pair(trading_pair):
    return re.sub(r'/', '-', trading_pair.lower())

# Example usage:
df.columns = [normalize_trading_pair(col) for col in df.columns]

# or 

df['instrument'] = df['trading_pair'].apply(normalize_trading_pair)

# option 2
def normalize_trading_pair(col_name):
    col_name = col_name.lower()  # Convert to lowercase
    col_name = re.sub(r'[^a-z0-9]', '-', col_name)  # Replace non-alphanumeric chars with hyphen
    col_name = re.sub(r'-+', '-', col_name)  # Replace multiple hyphens with single hyphen
    col_name = col_name.strip('-')  # Remove leading/trailing hyphens
    return col_name 

# ----------------------------------------------------------------------------------------------- #
# Dealing with NaNs, nulls, etc.:

# Neat little function to calculate, ratio and tabulate nulls for data cleaning:

def missing_percentage(df):
    
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100,2)
    
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



# ----------------------------------------------------------------------------------------------- #

# Nice little function:

"""This function takes in a dataframe and a column 
    and finds the percentage of the value_counts"""

def percent_value_counts(df, feature):
    
    percent = pd.DataFrame(
        round(df.loc[:,feature].value_counts(
            dropna=False, normalize=True)*100,2))

    ## creating a df with th
    total = pd.DataFrame(
        df.loc[:,feature].value_counts(dropna=False))

    ## concating percent and total dataframe
    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# ----------------------------------------------------------------------------------------------- #

# Rename cols
df.rename(
    columns={
        'old_name1': 'new_name1'
        , 'old_name2': 'new_name2'
        }, inplace=True
    )

# Specific example:
df.rename(
    columns={
        'asset_amt_ABS': 'amount'
        , 'USD_Amt_ABS': 'total_trade_amt'
        }, inplace=True
    )

# ----------------------------------------------------------------------------------------------- #

# mapping to json (specifically for Kaiko data w/ dictionary of exchange names and their codes saved as 'exch_dict_all.json'):

import json

with open('exch_dict_all.json', 'r') as f:
    exch_dict = json.load(f)

# Example usage:
df['exchange_name'] = df['exchange'].map(exch_dict)
df.head(3)


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

# Concatenations: 


# Concatenate dataframes vertically (stacking rows):
df_combined = pd.concat([df1, df2], ignore_index=True)




# ----------------------------------------------------------------------------------------------- #
# RegEx #

# Finding crypto addresses in dataframe strings (BTC, ETH, TRON, SOL, etc.):
import pandas as pd
import numpy as np
import re

# Example patterns:
pattern_evm_wallet = r'\b0x[a-fA-F0-9]{40}\b'
pattern_btc_wallet = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}|\b[bB][cC]1[pPqP][a-zA-Z0-9]{38,58}'
pattern_tron_wallet = r'T[a-zA-Z0-9]{33}'
pattern_sol_wallet = r'\b(?=.*[1-9A-HJ-NP-Za-km-z])([1-9A-HJ-NP-Za-km-z]{32,44})\b' # Need to double check and validate this in testing;
# ^^^ Need to add txn hash patterns too (ETH, BTC, etc.);

def find_regex_matches(df, pattern):
    # Compile the regex pattern
    regex = re.compile(pattern)
    
    # Function to check if the pattern exists in any cell in a row:
    def row_contains_pattern(row):
        return any(regex.search(str(cell)) for cell in row)
    
    # Filter the dataframe to keep ONLY rows that contain the pattern:
    matched_rows = df[df.apply(row_contains_pattern, axis=1)]
    return matched_rows

# Example usage:
results_evm_df = find_regex_matches(df, pattern_evm_wallet)

# Optional - save results to csv:
results_evm_df.to_csv('evm_wallet_matches.csv', index=False)

# ----------------------------------------------------------------------------------------------- #
# Several ways to find a specific string/term in a dataframe:

# EXAMPLE 1 (Multiple versions): Using 'lambda' and 'apply' to search across all columns in each row:
# EXAMPLE 1a:
search_term = 'specific_string'

df[df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)]

# EXAMPLE 1b:
df[df.apply(lambda row: row.str.contains(search_term).any(), axis=1)]

# EXAMPLE 1c:
df[df.apply(lambda row: row.str.contains('specific_string').any(), axis=1)]

# ------ #
# EXAMPLE 2: Using 'isin' to check for exact matches across all columns in each row:
search_term = 'specific_string'
df[df.isin([search_term]).any(axis=1)]

# or # 
df[df.isin(['specific_string']).any(axis=1)]

# ------ #
# EXAMPLE 3: Using a 'mask' to filter rows containing the search term:
search_term = 'specific_string'
mask - df.applymap(lambda x: search_term in str(x)).lower()
df[df[mask.any(axis=1)]]

# Can also spin up a masked df (from above):
filtered_df = df[mask.any(axis=1)]

# e.g., ::
abc_map = df.applymap(lambda x: 'abc' in str(x).lower())
df[abc_map.any(axis=1)]


# NEED TO ADD IN THE MULTI-TERM SEARCH;

search_terms = ['abc', 'def', 'xyz']

# EXAMPLE 1a: Case-insensitive search
df[df.apply(lambda row: row.astype(str)\
    .str.contains('|'.join(search_terms), case=False)\
        .any(), axis=1)]

# EXAMPLE 1b: Case-sensitive search
df[df.apply(lambda row: row.astype(str)\
    .str.contains('|'.join(search_terms))\
        .any(), axis=1)]

# EXAMPLE 2: Using isin to check for exact matches for any of multiple strings
search_terms = ['abc', 'def', 'xyz']
df[df.isin(search_terms).any(axis=1)]

# EXAMPLE 3: Using a mask to filter rows containing any of the search terms
search_terms = ['abc', 'def', 'xyz']
mask = df.applymap(lambda x: any(term in str(x).lower() for term in search_terms))
df[mask.any(axis=1)]


# BONUS EXAMPLE: Using regex for more complex pattern matching
search_terms = ['abc', 'def', 'xyz']
pattern = '|'.join(search_terms)
df[df.apply(lambda row: row.astype(str)\
    .str.contains(pattern, case=False).any(), axis=1)]


# BONUS EXAMPLE 2: Using numpy for logical OR across columns (efficient for large DataFrames)

import numpy as np

search_terms = ['abc', 'def', 'xyz']
mask = np.column_stack([
    df[col].astype(str).str.contains('|'.join(search_terms)
                                     , case=False) for col in df]
                       )
df[mask.any(axis=1)]


# SPECIFIC EXAMPLES:

# Find all rows where 'transaction_hash' col contains the any transaction hashes from a list:

# Simple list search (case sensitive!):

hashes_for_search = ['0xabc123...', '0xdef456...', '0xghi789...']  # Example list of transaction hashes
df[df['transaction_hash'].isin(hashes_for_search)]


# Case insensitive option (IDEAL OPTION GIVEN UNPREDICTABLE DATA CASES FORMAT):

hashes_for_search_lower = [h.lower() for h in [
    '0xabc123...' , '0xdef456...', '0xghi789...']
                           ]  

df[df['transaction_hash'].isin(hashes_for_search)]




# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Globbing-Globlins #
# ----------------------------------------------------------------------------------------------- #

# To grab specified rows/headers from the first tab in a sheet, for every excel file in a given directory, and concatenate them into a single dataframe:

import glob
import pandas as pd

path = './test/*.xlsx'  # Adjust the path and file extension as needed

combined_data = pd.DataFrame()

for file in glob.glob(path):
    df = pd.read_excel(file, header=3, nrows=2, usecols="A:H") # Adjust parameters as needed
    combined_data = pd.concat([combined_data, df], ignore_index=True)
    
print(combined_data.info())
combined_data.head(10)

# ----------------------------------------------------------------------------------------------- #

# Recursive directory file searches for filenames (explicit, wildcard(s), numeric ranges;)

import glob
import os

# For explicitly named files (e.g., a specific file):

print('Named explicitly:')
for name in glob.glob('/path/to/directory/specific_filename_2024.xlsx', recursive=True):
    print(name)

# Greedy wildcards: {Using '*' pattern}
print("Named with wildcards:")
for name in glob.glob('./folder/*01*.*'): # (File agnostic) Example finds all files in the dir w/ '01' in the filename;
    print(name)

# Greedy wildcard, across directory tree (e.g., all folders recursively from 'jupyterNotebooks' down, for any file containting the term (string) 'Voting'):

for name in glob.glob('./jupyterNotebooks/**/*Voting*', recursive=True):
    print(name)
    
# Now the same search but case insensitive:
# Option 1 (still using glob only):
import glob

pattern = 'Voting'.lower()
for name in glob.glob('./jupyterNotebooks/**/*Voting*', recursive=True):
    if pattern in name.lower():
        print(name)

# Option 2:
from pathlib import Path

root = Path('./jupyterNotebooks')
pattern = 'Voting'.lower()

for path in root.rglob('*'):
    if pattern in path.name.lower():
        print(path)

# Option 3 (havent actually tested this; it was auto-generated by VS Code...;):
import fnmatch
for dirpath, dirnames, filenames in os.walk('./jupyterNotebooks/'):
    for filename in fnmatch.filter(filenames, '*voting*'):
        print(os.path.join(dirpath, filename))
        
        
# Using '?' pattern:
print("Named with single-character wildcard:")
for name in glob.glob('./folder/file_?.txt'):  # Example finds files like 'file_A.txt', 'file_1.txt', etc.
    print(name)
    
# Using numeric ranges with '[0-9]' pattern (for numerics occurring anywhere in the filename):
print("Named with numeric range wildcard:")
for name in glob.glob('./folder/*[0-5]*,*'):  # Example finds files like 'file_0.txt' to 'file_5.txt'
    print(name)
    
# Using numeric ranges with '[0-9]' pattern (for filenames ending with a numeric
print("Name ending with numeric range wildcard:")
for name in glob.glob('./folder/*[0-5].*'):  # Example finds files like 'file_0.txt' to 'file_5.txt'
    print(name)

# Specific file extensions:
# *e.g., all Jupyter Notebook files containing the word 'regression' in the filename, in the 'jupyter_notebooks_' directory on down:*
for name in glob.glob('./jupyter_notebooks_/**/*regression*.ipynb', recursive=True):
    print(name)
    
# *e.g., all csv files containing the word 'analysis' in the filename, in the 'jupyter_notebooks_' directory on down:*
for name in glob.glob('./jupyter_notebooks_/**/*analysis*.csv', recursive=True):
    print(name)
    
    
    
# ----------------------------------------------------------------------------------------------- #
# Delete a dataframe and purge from memory (with garbage collection):

import gc
del df  # Delete the dataframe
gc.collect()  # Force garbage collection to free up memory
    
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #

### GROUPBY Ops ###

# Single groupby(s):

# Simple= 1-col w/ calc:
df_grp = df.groupby('datetime')['volume'].sum().to_frame().sort_values('datetime', ascending=True).reset_index()
df_grp.head(3)

# Simple= 2-col w/ calc:
df_grp = df.groupby('datetime')['volume', 'count']\
            .sum()\
                .to_frame()\
                    .sort_values('datetime', ascending=True).reset_index()
df_grp.head(3)


# Real-world example:
df_grp = df.groupby('Address')['Amount_USD'].sum()\
            .to_frame()\
                .sort_values('Amount_USD', ascending=False)\
                    .reset_index()


# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------- #
# Clearing the cache in PyTorch #

# Example: When you're using a custom helper script, which you have modifed in real-time, and you need to clear and reload latest version is loaded (instead of a cached version):

import importlib
from helper_scripts import t0rch_h3lp3r_0x01
importlib.reload(t0rch_h3lp3r_0x01)
from helper_scripts.t0rch_h3lp3r_0x01 import plot_training_curves

# ^^^ In the above, I updated a few lines in the 'plot_training_curves' function in the 't0rch_h3lp3r_0x01.py' script, and wanted to implement the changes without starting over (and retraining a model from scratch!)




# ----------------------------------------------------------------------------------------------- #

# duckdb 
import duckdb
import pandas as pd
import os
import glob
import sys
# (and any others specific to the data, task, use-case, etc.)

# Create a DuckDB connection (in-memory by default)
conn = duckdb.connect()

# Convert a large CSV file to Parquet format using DuckDB:
conn.execute("""
    COPY (
        SELECT *
        FROM read_csv_auto('large_file.csv')
    ) TO 'large_file.parquet' (FORMAT 'parquet')
""")

print("Conversion complete!")


# To view the first few rows of a Parquet file (just like a .head()) using DuckDB:
duckdb.sql("""
    SELECT *
    FROM read_parquet('large_file.parquet')
    LIMIT 5
""").df()

# Can also run dataframe queries directly using DuckDB (+ '.df()'):

duckdb.sql("""
    SELECT * FROM read_parquet('large_file.parquet') LIMIT 5
""").df().shape


# Miscellaneous DuckDB query example (group by, avg, where, order by):
duckdb.sql("""
    SELECT column1, column2, AVG(column3) AS avg_col3
    FROM df
    WHERE column4 > 100
    GROUP BY column1, column2
    ORDER BY avg_col3 DESC
""").df()

# ----------------------------------------------------------------------------------------------- #

# Statistical Related good-to-know code...

# Highlighted Pearson Correlation matrix (with self correlations removed;) --> THE EASY WAY!!! 

df.corr(numeric_only=True, method='pearson')\
    .replace(1.0, np.nan)\
    .style\
    .highlight_max(color='green')\
    .highlight_min(color='red')\
    .format("{:.4f}")

#  ^^^ Remove self-correlations; Only need to call the Styler object once (not before each 'highlight_...' method);

# For Spearman Correlation (*Spearman Corr is ideal for Time Series Analysis;)

df.corr(numeric_only=True, method='spearman')\
    .replace(1.0, np.nan)\
    .style\
    .highlight_max(color='green')\
    .highlight_min(color='red')\
    .format("{:.4f}")

# ----------------------------------------------------------------------------------------------- #

# Plotting (evil)...#

# Create side-by-side subplots with shared y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Box plots
ax1.boxplot(df1["price"], vert=True)
ax2.boxplot(df2["price"], vert=True)

# Titles
ax1.set_title("DF1 Box Plot")
ax2.set_title("DF2 Box Plot")

# Y-axis label (only needed once when shared)
ax1.set_ylabel("Price")

# Force 9 decimal places and disable scientific notation
formatter = mticker.FormatStrFormatter('%.9f')
ax1.yaxis.set_major_formatter(formatter)

# Optional: rotate y tick labels slightly if crowded
plt.setp(ax1.get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show();


# Now here with from statsmodels.graphics.tsaplots 'quarter_plot'(s):
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Quarter plots
quarter_plot(df['y'].resample('QE').mean(), ax=ax1)
quarter_plot(df['y'].resample('QE').median(), ax=ax2)

# Titles
ax1.set_title("Quarter Plot – Mean")
ax2.set_title("Quarter Plot – Median")

# Y-axis label (shared)
ax1.set_ylabel("y")

plt.tight_layout()
plt.show();

# ----------------------------------------------------------------------------------------------- # When working w/ dt data and need to run a groupby but aggregating by a shorter/longer timeframe

# (e.g.) OHLCVWAP date on a 1min scale, but you need an hourly or daily mean (or just need it organized better):
# Tip: Easier to groupby w/ dt data when dt vals NOT index; (reset_index() if they are!)
# Example 1: Simple groupby on dt for hourly:
df_grp = df.groupby(df['datetime'].dt. floor('H"))['price'].mean().to_frame()

df_grp
# Example 2: More complex agg groupby data w/ mean price and sum volume by hour:
df_grp2 = df.groupby(df['datetime'].dt. floor('H"))['price', 'volume'].agg({'price': "mean', "volume': 'sum'})
df_grp2
# NOTE: the 'dt.floor' can also be applied in plotting:
fig_box_d = px.box(df
, x=df['datetime'].dt.floor('D') # For the boxplot, converting out to 'D" , y='price'
color _discrete_sequence=['green']
width=900, height=700
fig_box_d.show()
# Can also use numerical values (e.g. '4H', '2H', '24H', '2D', etc.)
px.box(df
, x=df[ 'datetime'] -dt. floor('4H')
, y='price'
color_discrete_sequence=[ 'green' ] width=1200, height=700
#
#
#
# UNZIPPING MULTIPLE GZ (GUNZIP) FILES IN THE SAME DIRECTORY:
