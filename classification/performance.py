import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, confusion_matrix

def main(X_test, y_test, true, predicted):
    '''
    Check performance of model used
    :param
        df: Dataframe to be used
        true: True label feature in the dataframe
        predicted: Predicted label feature in the dataframe
    :return: confusion matrix
    '''

    conf_mats = []
    accuracy = []
    tnr = []
    precision = []
    recall = []
    for ind in range(len(X_test)):
        # print(true, '-', predicted)

        true = y_test[ind].to_numpy()
        pred = X_test[ind][predicted].to_numpy()

        tp, fn, fp, tn = confusion_matrix(true, pred).ravel()
        accuracy.append(round(accuracy_score(true, pred), 4))
        tnr.append(round(tn/(fp+tn), 4))
        # precision.append(precision_score(true, pred))
        precision.append(round(tp/(tp+fp), 4))
        # recall.append(recall_score(true, pred))
        recall.append(round(tp/(tp+fn), 4))
        # print(confusion_matrix(y_test[ind].to_numpy(), X_test[ind][predicted].to_numpy()))
        # print(classification_report(y_test[ind].to_numpy(), X_test[ind][predicted].to_numpy()))
        conf_mats.append([tp, fn, fp, tn])


    print(' ACCURACY = ', round(np.mean(accuracy),4))
    print(' PRECISION = ', round(np.mean(precision),4))
    print(' RECALL = ', round(np.mean(recall),4))
    print(' TRUE NEGATIVE RATE = ', round(np.mean(tnr),4))

    return conf_mats
