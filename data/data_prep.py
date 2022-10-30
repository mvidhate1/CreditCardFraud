import pandas as pd
import numpy as np
from datetime import datetime

def data_clean(df):
    '''
    Clean the dataset:
        - remove attributes that are empty
        - drop duplicated entries (no duplicates)
        - replace attribute null values with unknown
    :param
        df: dataframe to be cleaned and preprocessed.
    :return:
        cleaned dataframe.
    '''

    # checking for column-wise null values
    print('\nNULL VALUES IN ATTRIBUTES: \n', df.isnull().sum())
    # attributes that are entirely null:
    #   echoBuffer
    #   merchantCity
    #   merchantState
    #   merchantZip
    #   posOnPremises
    #   recurringAuthInd

    # drop columns with all null values
    null_features = ['echoBuffer',
                     'merchantCity',
                     'merchantState',
                     'merchantZip',
                     'posOnPremises',
                     'recurringAuthInd']
    df = df.drop(null_features, axis=1)
    print('\nATTRIBUTES LEFT: ', df.columns.to_list())

    # drop duplicate entries
    print('\nSHAPE BEFORE DROPPING DUPLICATES: ', df.shape)
    df = df.drop_duplicates()
    print('\nSHAPE AFTER DROPPING DUPLICATES: ', df.shape)
    # no entries dropped therefore no duplicates

    # checking if we can remove entries with null attribute values
    temp = df.copy()
    temp['null_value'] = temp.isnull().sum(axis=1)
    print('\nATTRIBUTE NULL VALUES: \n', temp.isnull().sum())
    print('\nNULL VALUES CLASS DISTRIBUTION: \n', temp.loc[temp.null_value >= 1].groupby('isFraud').count()['null_value'])
    # cannot remove all entries with null attribute values
    # False - 9244 (95.35%)
    # True  -  451 (04.65%)
    # the dataset has 98.48, 1.52 imbalance and
    # we will be removing more from True than False percentage wise.
    # There will be a information loss.

    # Replace NAs with unknown
    df = df.fillna(value='unknown')
    print('\nATTRIBUTE NULL VALUES (after replacing with "unknown"): \n', df.isnull().sum())

    return df

def data_prep(df):
    '''
    Preprocess of the dataset.
        - insert new columns (overDraw, overSeas, cvvMatch)
        -
    :param
        df: dataframe to be cleaned and preprocessed.
    :return:
        preprocessed dataframe.
    '''

    # adding new features
    new_features = []

    new_features.append('overDraw')
    df.insert(4, 'overDraw', 0)
    df['overDraw'] = np.where(df['availableMoney'] - df['transactionAmount'] <= 0, 1, df['overDraw'])

    # convert transactionDateTime from object type to datetime type
    df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])

    # splitting transaction datetime to
    #   month - maybe there is thieving season?
    #   date  - maybe scammers know when people get paid?
    #   hour  - maybe they like doing shady stuff at night?
    #   not using year, minute and sec because that doesn't make any sense
    new_features.extend(['transactionMonth','transactionDate','transactionHour'])
    df.insert(6, 'transactionMonth', df['transactionDateTime'].dt.month)
    df.insert(7, 'transactionDate', df['transactionDateTime'].dt.day)
    df.insert(8, 'transactionHour', df['transactionDateTime'].dt.hour)

    new_features.append('storeID')
    df.insert(11, 'storeID', 'unknown')
    for i in range(0, len(df)):
        item = df.loc[i]
        merchantName = item['merchantName']
        merchantName = merchantName.split(' #')
        if len(merchantName) == 2:
            df.at[i, 'storeID'] = merchantName[1]
        df.at[i, 'merchantName'] = str(merchantName[0])

    new_features.append('overSeas')
    df.insert(14, 'overSeas', 0)
    df['overSeas'] = np.where(df['acqCountry'] != df['merchantCountryCode'], 1, df['overSeas'])

    new_features.append('cvvMatch')
    df.insert(23, 'cvvMatch', 0)
    df['cvvMatch'] = np.where(df['cardCVV'] == df['enteredCVV'], 1, df['cvvMatch'])

    # don't need to do this
    # df['cardPresent'] = df['cardPresent'].astype(int)
    # df['expirationDateKeyInMatch'] = df['expirationDateKeyInMatch'].astype(int)
    # df['isFraud'] = df['isFraud'].astype(int)

    print('New Features : ', new_features)
    return df

# not using because it would increase attribute number insanely
def add_dummies(df, cat_vars):

    columns = df.columns.values
    # cat_vars = everything you want to create a dummy for
    for var in cat_vars:
        # print(var)
        cat_list = 'var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        # print(cat_list)
        # print('-----')
        df = df.join(cat_list)

    data_vars = df.columns.values.tolist()
    to_keep = [i for i in data_vars if i not in cat_vars]
    dummies = [i for i in data_vars if i not in columns]
    df_final = df[to_keep]

    return df_final, dummies

def main(df):
    wd = '/data/datasets/'

    # data cleaning
    new_df = data_clean(df)

    # NOT USING RIGHT NOW
    # data preprocessing
    new_df = data_prep(new_df)
    # cat_vars = ['acqCountry', 'merchantCountryCode']
    # new_df, dummies = add_dummies(new_df, cat_vars)

    # store as csv (for quick access)
    new_df.to_csv('.' + wd + 'transactions_cleaned.csv')

    new_df = pd.read_csv('.' + wd + 'transactions_cleaned.csv', index_col=[0])
    print('\nATTRIBUTES AFTER PREPROCESSING: ', new_df.columns.values)

    return new_df
