import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline

def fisher_score(X, y):
    labels = np.unique(y)
    scores = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        m_total = np.mean(feature)
        num, den = 0, 0
        for label in labels:
            X_c = feature[y == label]
            m_c = np.mean(X_c)
            v_c = np.var(X_c)
            num += len(X_c) * (m_c - m_total)**2
            den += len(X_c) * v_c
        scores.append(num / den if den != 0 else 0)
    return np.array(scores)

class HierarchicalClassifier:
    def __init__(self):
        self.pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])

    def evaluate(self, X, y, task_name="Task"):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(self.pipe, X, y, cv=cv)
        y_prob = cross_val_predict(self.pipe, X, y, cv=cv, method='predict_proba')[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)
        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0])>0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
        
        return {"acc": acc, "auc": auc, "sens": sens, "spec": spec}
