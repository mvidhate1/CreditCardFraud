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
        - insert new columns
        - remove columns that make no sense for classification
    :param
        df: dataframe
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
    new_features.extend(['transactionMonth','transactionDay','transactionHour'])
    df.insert(6, 'transactionMonth', df['transactionDateTime'].dt.month)
    df.insert(7, 'transactionDay', df['transactionDateTime'].dt.day)
    df.insert(8, 'transactionHour', df['transactionDateTime'].dt.hour)
    # having an exact date is not good classification
    df = df.drop('transactionDateTime', axis=1)

    # new_features.append('storeID')
    # df.insert(10, 'storeID', 'unknown')
    # for i in range(0, len(df)):
    #     item = df.loc[i]
    #     merchantName = item['merchantName']
    #     merchantName = merchantName.split(' #')
    #     if len(merchantName) == 2:
    #         df.at[i, 'storeID'] = merchantName[1]
    #     df.at[i, 'merchantName'] = str(merchantName[0])
    # not looking at particular merchants right now. only the merchant industry.
    df = df.drop('merchantName', axis=1)

    new_features.append('overSeas')
    df.insert(11, 'overSeas', 0)
    df['overSeas'] = np.where(df['acqCountry'] != df['merchantCountryCode'], 1, df['overSeas'])

    new_features.append('cvvMatch')
    df.insert(20, 'cvvMatch', 0)
    df['cvvMatch'] = np.where(df['cardCVV'] == df['enteredCVV'], 1, df['cvvMatch'])

    # don't need to do this
    # df['cardPresent'] = df['cardPresent'].astype(int)
    # df['expirationDateKeyInMatch'] = df['expirationDateKeyInMatch'].astype(int)
    # df['isFraud'] = df['isFraud'].astype(int)

    # dropping columns that don't are not relevant
    df = df.drop(['currentExpDate','accountOpenDate','dateOfLastAddressChange'], axis=1)

    print('New Features : ', new_features)

    # Dropping columns that make no sense in classification
    useless_features = ['accountNumber',
                        'customerId']
    df = df.drop(useless_features, axis=1)
    print('\nATTRIBUTES LEFT: ', df.columns.to_list())

    return df

def main(df):
    datasets = '/data/datasets/'

    # data cleaning
    new_df = data_clean(df)

    # data preprocessing
    new_df = data_prep(new_df)

    # get dummies - need it for classification
    vars = ['acqCountry', 'merchantCountryCode',
            'posEntryMode', 'posConditionCode',
            'merchantCategoryCode', 'transactionType']
    new_df = pd.get_dummies(new_df, columns=vars, drop_first=True)

    # store as csv (for quick access)
    new_df.to_csv('.' + datasets + 'transactions_cleaned.csv')

    print('\nATTRIBUTES AFTER PREPROCESSING: ', new_df.columns.values)

    return new_df
