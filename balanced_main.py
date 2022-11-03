import pandas as pd
pd.options.mode.chained_assignment = None
from classification import classifiers, performance

def choose_model(df, model, split_type, feature_method, features, target, predicted, random_state, additional):
    X_test, y_test = classifiers.classify(df=df, split_type=split_type, feature_method=feature_method, model=model,
                                          features=features, target=target, predicted=predicted,
                                          random_state=random_state, additional=additional)

    conf_mat = performance.main(X_test, y_test, target, predicted)

def main(df, features, target, predicted, feature_methods):
    # classification (stratified k-fold without feature selection)
    print("\n Classifier alone - Stratified 10-fold Cross-validation - No feature selection")
    choose_model(df=df, split_type=0, feature_method=None,
                 model='logreg', features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=0, feature_method=None,
                 model='kNN', features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=0, feature_method=None,
                 model='gnbayes', features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=0, feature_method=None,
                 model='decitree', features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])
    choose_model(df=df, split_type=0, feature_method=None,
                 model='randfor', features=features, target=target, predicted=predicted,
                 random_state=100, additional=[10])

    # data split (66-34)
    for method in feature_methods:
        print('\n Classifier - train-test - feature selection method - ', method)
        choose_model(df=df, split_type=1, feature_method=method, model='logreg',
                     features=features, target=target, predicted=predicted,
                     random_state=100, additional=[10])
        choose_model(df=df, split_type=1, feature_method=method, model='etc',
                     features=features, target=target, predicted=predicted,
                     random_state=100, additional=[10])
        choose_model(df=df, split_type=1, feature_method=method, model='decitree',
                     features=features, target=target, predicted=predicted,
                     random_state=100, additional=[10])
        choose_model(df=df, split_type=1, feature_method=method, model='randfor',
                     features=features, target=target, predicted=predicted,
                     random_state=100, additional=[10])
        choose_model(df=df, split_type=1, feature_method=method, model='lda',
                     features=features, target=target, predicted=predicted,
                     random_state=100, additional=[10])

        # Used for testing only
        # choose_model(df=df, split_type=1, feature_method=method, model='gnbayes',
        #              features=features, target=target, predicted=predicted,
        #              random_state=100, additional=[10])
        # choose_model(df=df, split_type=1, feature_method=method, model='qda',
        #              features=features, target=target, predicted=predicted,
        #              random_state=100, additional=[10])
