import pandas as pd
pd.options.mode.chained_assignment = None
from data import get_data as gd
from eda import eda
from data import data_prep as dp
from data import data_generation as dg
import balanced_main as bm
from classification import classifiers, performance
from sklearn.feature_selection import chi2, f_classif
from sklearn.model_selection import train_test_split


def main():
    # # get dataframe
    # df = gd.main()

    # # using intermediary file for easy & quick reading
    # df = pd.read_csv('./data/datasets/transactions.csv', index_col=[0])
    # print(df.shape)

    # # quicklook at dataset summary
    # eda.main(df)

    # # data cleaning & preprocessing
    # df = dp.main(df)
    # print(df.dtypes)
    # print(df.shape)

    # # using intermediary file for easy & quick reading
    df = pd.read_csv('./data/datasets/transactions_cleaned.csv', index_col=[0])
    # print(df.shape)
    # print(df.dtypes)

    # dataset balancing (select only one method at a time. comment out the other method)
    temp = df.copy()
    features = df.columns.tolist()
    features.remove('isFraud')
    target = 'isFraud'
    predicted = 'pred_isFraud'

    # # data duplication
    print("\nDATA DUPLICATION")
    # dd_df = dg.main(temp, 1)
    dd_df = pd.read_csv('./data/datasets/transactions_balanced_duplication.csv', index_col=[0])
    # print(dd_df.shape)
    # bm.main(dd_df, features, target, predicted)


    # # random over-sampling
    print("\nRANDOM OVER-SAMPLING")
    # ros_df = dg.main(temp, 2, features=features, target=target)
    ros_df = pd.read_csv('./data/datasets/transactions_balanced_ros.csv', index_col=[0])
    # print(ros_df.shape)
    # bm.main(ros_df, features, target, predicted, ['kbest'])


    # # random under-sampling
    print("\nRANDOM UNDER-SAMPLING")
    # rus_df = dg.main(temp, 3, features=features, target=target)
    rus_df = pd.read_csv('./data/datasets/transactions_balanced_rus.csv', index_col=[0])
    print(rus_df.shape)
    bm.main(rus_df, features, target, predicted, ['kbest-f_classif', 'kbest-mutual', 'rfecv', 'sfm', 'sfs'])


    # # synthetic minority oversampling technique
    print("\nSYNTHETIC MINORITY OVERSAMPLING TECHNIQUE")
    # smote_df = dg.main(temp, 4, features=features, target=target)
    smote_df = pd.read_csv('./data/datasets/transactions_balanced_smote.csv', index_col=[0])
    # print(smote_df.shape)


if __name__ == "__main__":
    main()