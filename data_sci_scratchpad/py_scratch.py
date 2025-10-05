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

# ----------------------------------------------------------------------------------------------- #




# ----------------------------------------------------------------------------------------------- #




# ----------------------------------------------------------------------------------------------- #




# ----------------------------------------------------------------------------------------------- #




# ----------------------------------------------------------------------------------------------- #
