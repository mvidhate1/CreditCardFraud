import pandas as pd
from data import get_data as gd
from eda import eda
from data import data_prep as dp
# from data import data_generation as dg
# from classification import classifiers, performance

# def choose_model(df, model, features, target, predicted, random_state):
#     print(model)
#
#     classifier = classifiers.classifier(df, model, features, target, predicted, random_state)
#
#     tp, fn, fp, tn = performance.performance(classifier, target, predicted)
#     print("\n Accuracy: ", round((tp+tn)/(tp+tn+fp+fn)*100.0, 2),
#           "\n Precision: ", round(tp/(tp+fp)*100.0, 2),
#           "\n Sensitvity/Recall: ", round(tp/(tp+fn)*100.0, 2),
#           "\n Specificity: ", round(tn/(tn+fp)*100.0, 2))

def main():
    # get dataframe
    # df = gd.main()

    # using intermediary for easy & quick reading
    df = pd.read_pickle('./data/datasets/transactions.pkl')



if __name__ == "__main__":
    main()