import pandas as pd
from data import get_data as gd
from eda import eda
from data import data_prep as dp

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
    df = pd.read_csv('./data/datasets/transactions_cleaned.csv', index_col=[0])


if __name__ == "__main__":
    main()