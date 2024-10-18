import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score,  f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(MLPBinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  
            nn.BatchNorm1d(64),       
            nn.ReLU(), 
            nn.Dropout(0.5),  

            # nn.Linear(128, 64),       
            # nn.BatchNorm1d(64),  
            # nn.ReLU(),
            # nn.Dropout(0.5),  

            nn.Linear(64, 32),       
            nn.BatchNorm1d(32),  
            nn.ReLU(),
            nn.Dropout(0.5), 

            nn.Linear(32, 1),           
            nn.Sigmoid()                
        )
    
    def forward(self, x):
        return self.model(x)




class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity  
        out = self.relu(out)
        return out

class ResNetClassifier(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=64):
        super(ResNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.resblock = ResidualBlock(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.resblock(x)
        x = self.sigmoid(self.fc2(x))
        return x







def train_random_forest(train_features, train_labels, test_features, test_labels):
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    
    random_forest.fit(train_features, train_labels)
    y_pred = random_forest.predict(test_features)

    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_pred = random_forest.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_pred):.4f}")
    f1 = f1_score(train_labels, y_pred)
    print(f"Train F1 Score: {f1:.4f}")

    return random_forest


def train_logistic_regression(train_features, train_labels, test_features, test_labels):
    weights = {0: 1, 1: 1.4}
    logistic_regression = LogisticRegression(C=0.5, max_iter=10000000, class_weight=weights)
    
    logistic_regression.fit(train_features, train_labels)

    y_pred = logistic_regression.predict(test_features)
    
  
    
    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred)}")
    f1 = f1_score(test_labels, y_pred)
    print(f"validation F1 Score: {f1:.4f}")



    y_pred = logistic_regression.predict(train_features)
    print(f"train Accuracy: {accuracy_score(train_labels, y_pred)}")
    f1 = f1_score(train_labels, y_pred)
    print(f"train F1 Score: {f1:.4f}")


    return logistic_regression


def train_xgboost(train_features, train_labels, test_features, test_labels):
    xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.6,
    scale_pos_weight=1,
    gamma=2,
    random_state=42,
    # reg_alpha=0.1, 
    # reg_lambda=1.0
    )
    
    xgb_model.fit(train_features, train_labels)

    y_pred = xgb_model.predict(test_features)
    
    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_pred = xgb_model.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_pred):.4f}")
    f1 = f1_score(train_labels, y_pred)
    print(f"Train F1 Score: {f1:.4f}")

    return xgb_model




def train_svm_rbf(train_features, train_labels, test_features, test_labels):
    weights = {0: 1, 1: 1.2}
    svm_model = SVC(kernel='poly', C=0.1, max_iter=1000000, class_weight=weights)
    
    svm_model.fit(train_features, train_labels)


    y_pred = svm_model.predict(test_features)
    
    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_pred = svm_model.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_pred):.4f}")
    f1 = f1_score(train_labels, y_pred)
    print(f"Train F1 Score: {f1:.4f}")

    return svm_model


def smoteboost(train_features, train_labels, test_features, test_labels):

    smote_boost = SMOTEBoost(n_estimators=100, random_state=42)


    smote_boost.fit(train_features, train_labels)


    y_pred = smote_boost.predict(test_features)


    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_pred = smote_boost.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_pred):.4f}")
    f1 = f1_score(train_labels, y_pred)
    print(f"Train F1 Score: {f1:.4f}")

    return smote_boost



def balancebag(train_features, train_labels, test_features, test_labels):

    balanced_bagging = BalancedBaggingClassifier(
        estimator=DecisionTreeClassifier(),  
        sampling_strategy='auto',
        random_state=42,
        n_estimators=50
        )


    balanced_bagging.fit(train_features, train_labels)


    y_pred = balanced_bagging.predict(test_features)


    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_pred = balanced_bagging.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_pred):.4f}")
    f1 = f1_score(train_labels, y_pred)
    print(f"Train F1 Score: {f1:.4f}")

    return balanced_bagging





def smote_adaboost(train_features, train_labels, test_features, test_labels):

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(train_features, train_labels)

    ada_boost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), 
        n_estimators=10000,
        learning_rate=0.1,
        random_state=42
    )


    ada_boost.fit(X_resampled, y_resampled)


    y_pred = ada_boost.predict(test_features)

    print(f"Validation Accuracy: {accuracy_score(test_labels, y_pred):.4f}")
    f1 = f1_score(test_labels, y_pred)
    print(f"Validation F1 Score: {f1:.4f}")

    y_train_pred = ada_boost.predict(train_features)
    print(f"Train Accuracy: {accuracy_score(train_labels, y_train_pred):.4f}")
    f1_train = f1_score(train_labels, y_train_pred)
    print(f"Train F1 Score: {f1_train:.4f}")

    return ada_boost







