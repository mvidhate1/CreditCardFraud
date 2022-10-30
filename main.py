import pandas as pd
from data import get_data as gd
from eda import eda
from data import data_prep as dp
from data import data_generation as dg

def main():
    # # get dataframe
    # df = gd.main()
    #
    # # using intermediary file for easy & quick reading
    # df = pd.read_csv('./data/datasets/transactions.csv', index_col=[0])
    #
    # # quicklook at dataset summary
    # # eda.main(df)
    #
    # # data cleaning & preprocessing
    # df = dp.main(df)

    # using intermediary file for easy & quick reading
    # df = pd.read_csv('./data/datasets/transactions_cleaned.csv', index_col=[0])

    # dataset balancing
    # features = df.columns.tolist()[:-1]
    # target = 'isFraud'
    # # data duplication
    # df = dg.main(df, 1)
    df = pd.read_csv('./data/datasets/transactions_balanced_duplication.csv', index_col=[0])
    print(df.shape)
    print(df.columns.tolist())

    # random under-sampling
    # df = dg.main(df, 3, features=features, target=target)
    # df = pd.read_csv('./data/datasets/transactions_balanced_rus.csv', index_col=[0])
    # print(df.shape)
    # print(df.columns.tolist())


if __name__ == "__main__":
    main()