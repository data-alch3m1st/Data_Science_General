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
df.dropna(subset=['column_name'], inplace=True)  # Drop rows with NaN in 'column_name'

# Check for NaNs remaining:
df['column_name'].isna().any()
# or
df['column_name'].isnull().any()



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