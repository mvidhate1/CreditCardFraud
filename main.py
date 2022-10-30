import pandas as pd
from data import get_data as gd
from eda import eda

def main():
    # get dataframe
    # df = gd.main()

    # using intermediary for easy & quick reading
    df = pd.read_pickle('./data/datasets/transactions.pkl')

    # quicklook at dataset summary
    eda.main(df)


if __name__ == "__main__":
    main()