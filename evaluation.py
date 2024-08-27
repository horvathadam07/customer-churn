import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def find_threshold_cv(model, X, y, cv):

    thr = []
    auc = []
    acc = []
    prc = []
    rec = []
    f1s = []
    f1w = []

    for (train_index, test_index) in cv.split(X):

        train_index = pd.Index(train_index)
        test_index = pd.Index(test_index)

        X_train = X.loc[X.index.isin(train_index)]
        X_test = X.loc[X.index.isin(test_index)]

        y_train = y.loc[y.index.isin(train_index)]
        y_test = y.loc[y.index.isin(test_index)]

        pred_train = model.predict_proba(X_train)[:,1]
        pred_test = model.predict_proba(X_test)[:,1]

        _t = np.arange(0.01, 1, 0.01)[np.argmax([f1_score(y_train, np.where(pred_train >= i, 1, 0), average='weighted') for i in np.arange(0.01, 1, 0.01)])].round(2)
        thr.append(_t)
        auc.append(roc_auc_score(y_test, pred_test).round(4))
        acc.append(round(accuracy_score(y_test, np.where(pred_test >= _t, 1, 0)),4))
        prc.append(precision_score(y_test, np.where(pred_test >= _t, 1, 0)).round(4))
        rec.append(recall_score(y_test, np.where(pred_test >= _t, 1, 0)).round(4))
        f1s.append(f1_score(y_test, np.where(pred_test >= _t, 1, 0)).round(4))
        f1w.append(f1_score(y_test, np.where(pred_test >= _t, 1, 0), average='weighted').round(4))

    return(pd.DataFrame({'Threshold': thr, 'Area Under Curve': auc, 'Accuracy': acc,
                        'Precision': prc, 'Recall': rec,
                        'F1-score': f1s, 'Weighted F1-score': f1w}))





def test_threshold(model, X, y, threshold):

        pred = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, pred).round(4)
        acc = round(accuracy_score(y, np.where(pred >= threshold, 1, 0)),4)
        prc = precision_score(y, np.where(pred >= threshold, 1, 0)).round(4)
        rec = recall_score(y, np.where(pred >= threshold, 1, 0)).round(4)
        f1s = f1_score(y, np.where(pred >= threshold, 1, 0)).round(4)
        f1w = f1_score(y, np.where(pred >= threshold, 1, 0), average='weighted').round(4)

        return(pd.DataFrame({'Area Under Curve': auc, 'Accuracy': acc,
                             'Precision': prc, 'Recall': rec,
                             'F1-score': f1s, 'Weighted F1-score': f1w}, index=[0]))