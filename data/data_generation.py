import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

def main(df, type, features=None, target=None):
    print('\nBEFORE DATASET BALANCING\n', df.groupby(['isFraud']).count().iloc[:, 0])
    datasets = '/data/datasets/'

    # data duplication
    if type == 1:
        fraud = df.loc[df.isFraud == 1]
        count_valid = df.groupby(['isFraud']).count().iloc[0, 0]
        count_fraud = df.groupby(['isFraud']).count().iloc[1, 0]

        multiple = int(count_valid / count_fraud)

        new_df = df.copy()
        for i in range(multiple - 1):
            new_df = pd.concat([new_df, fraud])

        print('\nAFTER DATA DUPLICATION:\n', new_df.groupby(['isFraud']).count().iloc[:, 0])

        # store as csv (for quick access)
        new_df.to_csv('.' + datasets + 'transactions_balanced_duplication.csv')

    # random over-sampling
    elif type == 2:
        ros = RandomOverSampler(random_state=42)
        x_ros, y_ros = ros.fit_resample(df[features], df[target])

        new_df = x_ros
        new_df[target] = y_ros
        print('\nAFTER RANDOM OVER-SAMPLING:\n', new_df.groupby(['isFraud']).count().iloc[:, 0])

        # store as csv (for quick access)
        new_df.to_csv('.' + datasets + 'transactions_balanced_ros.csv')

    # random under-sampling
    elif type == 3:
        rus = RandomUnderSampler(random_state=42, replacement=True)
        x_rus, y_rus = rus.fit_resample(df[features], df[target])

        new_df = x_rus
        new_df[target] = y_rus
        print('\nAFTER RANDOM UNDER-SAMPLING:\n', new_df.groupby(['isFraud']).count().iloc[:, 0])

        # store as csv (for quick access)
        new_df.to_csv('.' + datasets + 'transactions_balanced_rus.csv')

    # SMOTE
    elif type == 4:
        smote = SMOTE()
        x_smote, y_smote = smote.fit_resample(df[features], df[target])

        new_df = x_smote
        new_df[target] = y_smote
        print('\nAFTER SMOTE:\n', new_df.groupby(['isFraud']).count().iloc[:, 0])

        # store as csv (for quick access)
        new_df.to_csv('.' + datasets + 'transactions_balanced_smote.csv')


    return new_df