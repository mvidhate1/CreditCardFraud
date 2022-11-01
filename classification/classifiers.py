from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif,\
                                      RFECV,\
                                      SelectFromModel,\
                                      SequentialFeatureSelector
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn


def split(X, y, split_type, random_state, additional=None):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # for the stratified k fold
    if split_type == 0:
        skf = StratifiedKFold(n_splits=additional[0], shuffle=True, random_state=random_state)
        for train_index, test_index in skf.split(X, y):
            X_train.append(X.iloc[train_index])
            X_test.append(X.iloc[test_index])
            y_train.append(y.iloc[train_index])
            y_test.append(y.iloc[test_index])
    # for the train-test with feature selection
    if split_type == 1:
        X_trn, X_tst, y_trn, y_tst = train_test_split(X, y,
                                                      test_size=0.34,
                                                      random_state=random_state,
                                                      stratify=y)
        X_train.append(X_trn)
        X_test.append(X_tst)
        y_train.append(y_trn)
        y_test.append(y_tst)

    return X_train, X_test, y_train, y_test

def select_features(X_train, X_test, y_train, y_test, method, additional):

    # k best feature selection - f_classif scorer
    if method == 'kbest-f_classif':
        for ind in range(len(X_train)):
            X_trn = X_train[ind]
            X_tst = X_test[ind]
            y_trn = y_train[ind]
            y_tst = y_test[ind]

            # classif - ANOVA f-test values
            fs = SelectKBest(k=additional[0], score_func=f_classif)
            fs.fit(X_trn, y_trn)
            selected_features = fs.get_feature_names_out()
            # reduce train and test dataset
            X_train[ind] = X_train[ind][selected_features]
            X_test[ind] = X_test[ind][selected_features]

            print('\n features selected:\n', selected_features)

    # k best feature selection - mutual_info_classif scorer
    if method == 'kbest-mutual':
        for ind in range(len(X_train)):
            X_trn = X_train[ind]
            X_tst = X_test[ind]
            y_trn = y_train[ind]
            y_tst = y_test[ind]

            # classif - ANOVA f-test values
            fs = SelectKBest(k=additional[0], score_func=mutual_info_classif)
            fs.fit(X_trn, y_trn)
            selected_features = fs.get_feature_names_out()
            # reduce train and test dataset
            X_train[ind] = X_train[ind][selected_features]
            X_test[ind] = X_test[ind][selected_features]

            print('\n features selected:\n', selected_features)

    # recursive feature elimination with cross validation
    if method == 'rfecv':
        for ind in range(len(X_train)):
            X_trn = X_train[ind]
            X_tst = X_test[ind]
            y_trn = y_train[ind]
            y_tst = y_test[ind]

            # classif - ANOVA f-test values
            # cv = 5 (default 5 fold validation)
            fs = RFECV(estimator=additional[1], step=5, min_features_to_select=additional[0])
            fs.fit(X_trn, y_trn)
            selected_features = fs.get_feature_names_out()
            # reduce train and test dataset
            X_train[ind] = X_train[ind][selected_features]
            X_test[ind] = X_test[ind][selected_features]

            print('\n features selected:\n', selected_features)

    # select from model
    if method == 'sfm':
        for ind in range(len(X_train)):
            X_trn = X_train[ind]
            X_tst = X_test[ind]
            y_trn = y_train[ind]
            y_tst = y_test[ind]

            # classif - ANOVA f-test values
            fs = SelectFromModel(estimator=additional[1], prefit=False)
            fs.fit(X_trn, y_trn)
            selected_features = fs.get_feature_names_out()
            X_train[ind] = X_train[ind][selected_features]
            X_test[ind] = X_test[ind][selected_features]

            print('\n features selected:\n', selected_features)

    # sequential feature selection
    if method == 'sfs':
        for ind in range(len(X_train)):
            X_trn = X_train[ind]
            X_tst = X_test[ind]
            y_trn = y_train[ind]
            y_tst = y_test[ind]

            # classif - ANOVA f-test values
            fs = SequentialFeatureSelector(estimator=additional[1], n_features_to_select=additional[0])
            fs.fit(X_trn, y_trn)
            selected_features = fs.get_feature_names_out()
            X_train[ind] = X_train[ind][selected_features]
            X_test[ind] = X_test[ind][selected_features]

            print('\n features selected:\n', selected_features)

    return X_train, X_test, y_train, y_test, fs

def classify(df, model, features, target, predicted, random_state, split_type=None, additional=None, feature_method=None):
    '''
    Classification rules
    :param
        df: dataframe on which classification is to be done.
        type: classifier type
        feature: features used for classification
        target: target feature to be used for training
        predicted: predicted column
        random_state: for dataframe spliting
        additional: additional information to be passed for particular classifiers
        no_split: bool to forgo splitting
    :return:
        new dataframe with predicted class features
    '''

    # split data
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = split(X, y, split_type, random_state, additional=additional)


    # Logistic Regression
    if model == 'logreg':
        print('\nLOGISTIC REGRESSION')
        for ind in range(len(X_train)):
            logreg = LogisticRegression(random_state=random_state, solver='lbfgs', max_iter=10000)
            logreg.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(logreg)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method, additional=additional)

            logreg.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = logreg.predict(X_test[ind])

    # kNN Classifier
    elif model == 'kNN':
        print('\nK-NEAREST NEIGHBORS')
        for ind in range(len(X_train)):
            neigh = knn()
            neigh.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(neigh)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method, additional=additional)

            # knn doesnt need training but have to use fit
            # to follow standard format for classifiers
            neigh.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = neigh.predict(X_test[ind])

    # Gaussian Naive Bayes
    elif model == 'gnbayes':
        print('\nGAUSSIAN NAÏVE BAYES')
        for ind in range(len(X_train)):
            gnb = GaussianNB()
            gnb.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(gnb)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method, additional=additional)

            gnb.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = gnb.predict(X_test[ind])

    # Decision Tree
    elif model == 'decitree':
        print('\nDECISION TREE')
        for ind in range(len(X_train)):
            dt = DecisionTreeClassifier(criterion='entropy',
                                        random_state=random_state)
            dt.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(dt)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method, additional=additional)

            dt.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = dt.predict(X_test[ind])

    # Random Forest
    elif model == 'randfor':
        print('\nRANDOM FOREST')
        for ind in range(len(X_train)):
            rf = RandomForestClassifier(10, random_state=random_state)
            rf.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(rf)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method, additional=additional)

            rf.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = rf.predict(X_test[ind])

    # Extra Trees Classifier
    elif model == 'etc':
        print('\nEXTRA TREE CLASSIFIER')
        for ind in range(len(X_train)):
            dt = ExtraTreesClassifier(criterion='entropy',
                                        random_state=random_state)
            dt.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(dt)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method,
                                                                       additional=additional)

            dt.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = dt.predict(X_test[ind])

    # Linear Discriminant Analysis
    elif model == 'lda':
        print('\nLINEAR DISCRIMINANT ANALYSIS')
        for ind in range(len(X_train)):
            lda = LDA()
            lda.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(lda)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method,
                                                                       additional=additional)

            lda.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = lda.predict(X_test[ind])

    # Quadratic Discriminant Analysis
    elif model == 'qda':
        print('\nQUADRATIC DISCRIMINANT ANALYSIS')
        for ind in range(len(X_train)):
            qda = QDA()
            qda.fit(X_train[ind], y_train[ind])

            # feature selection
            if feature_method != None:
                additional.append(qda)
                X_train, X_test, y_train, y_test, fs = select_features(X_train, X_test, y_train, y_test, feature_method,
                                                                       additional=additional)

            qda.fit(X_train[ind], y_train[ind])
            X_test[ind][predicted] = qda.predict(X_test[ind])

    return X_test, y_test
