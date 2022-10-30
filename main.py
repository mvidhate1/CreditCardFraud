import pandas as pd
pd.options.mode.chained_assignment = None
from data import get_data as gd
from eda import eda
from data import data_prep as dp
from data import data_generation as dg
from classification import classifiers, performance, feature_selection
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split

def choose_model(df, model, split_type, feature_method, features, target, predicted, random_state, additional):
    X_test, y_test = classifiers.classify(df=df, split_type=split_type, feature_method=feature_method, model=model,
                                          features=features, target=target, predicted=predicted,
                                          random_state=random_state, additional=additional)

    conf_mat = performance.main(X_test, y_test, target, predicted)


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
    # print(df.shape)

    # # using intermediary file for easy & quick reading
    # df = pd.read_csv('./data/datasets/transactions_cleaned.csv', index_col=[0])
    # print(df.shape)
    # print(df.dtypes)

    # # dataset balancing (select only one method at a time. comment out the other method)
    # features = df.columns.tolist()
    # features.remove('isFraud')
    target = 'isFraud'
    predicted = 'pred_isFraud'

    # # random under-sampling
    print('RANDOM UNDER-SAMPLING')
    # df = dg.main(df, 3, features=features, target=target)
    df = pd.read_csv('./data/datasets/transactions_balanced_rus.csv', index_col=[0])
    print(df.shape)
    features = df.columns.to_list()
    features.remove('isFraud')

    # # classification
    # choose_model(df=df, split_type=0, feature_method=None,
    #              model='logreg', features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=0, feature_method=None,
    #              model='kNN', features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=0, feature_method=None,
    #              model='gnbayes', features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=0, feature_method=None,
    #              model='decitree', features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=0, feature_method=None,
    #              model='randfor', features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])

    # data split (66-34)
    # feature method = kbest (using k=10)
    print('\nFEATURE SELECTION - K BEST')
    choose_model(df=df, split_type=1, feature_method='kbest', model='logreg',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='kbest', model='kNN',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='kbest', model='gnbayes',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='kbest', model='decitree',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='kbest', model='randfor',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])

    # NEED TO FIX
    # feature method = rfe (using min_features=10)
    print('\nFEATURE SELECTION - RFE')
    choose_model(df=df, split_type=1, feature_method='rfe', model='logreg',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])
    # choose_model(df=df, split_type=1, feature_method='rfe', model='kNN',
    #              features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=1, feature_method='rfe', model='gnbayes',
    #              features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='rfe', model='decitree',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])
    choose_model(df=df, split_type=1, feature_method='rfe', model='randfor',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])

    # feature method = select from model (using min_features=10)
    print('\nFEATURE SELECTION - SELECT FROM MODEL')
    choose_model(df=df, split_type=1, feature_method='rfe', model='logreg',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])
    # choose_model(df=df, split_type=1, feature_method='rfe', model='kNN',
    #              features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    # choose_model(df=df, split_type=1, feature_method='rfe', model='gnbayes',
    #              features=features, target=target, predicted=predicted,
    #              random_state=100, additional=[10])
    choose_model(df=df, split_type=1, feature_method='rfe', model='decitree',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])
    choose_model(df=df, split_type=1, feature_method='rfe', model='randfor',
                 features=features, target=target, predicted=predicted,
                 random_state=100, additional=[7])


    # # data duplication
    # df = dg.main(df, 1)
    # df = pd.read_csv('./data/datasets/transactions_balanced_duplication.csv', index_col=[0])


if __name__ == "__main__":
    main()