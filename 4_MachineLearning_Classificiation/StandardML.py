import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def main():
    # Load datasets.
    df_train = pd.read_csv('train.csv', index_col=0)
    df_test = pd.read_csv('test.csv', index_col=0)
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Support Vector Classifier
    model = SVC(C=0.1, kernel='rbf').fit(X_train, y_train)
    # Gradient Boosting Classifier
    # model = GradientBoostingClassifier().fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    # y_predicted = clf.predict_proba(X_test)[:, 1]

    # Evaluate accuracy.
    score = accuracy_score(y_test, y_predicted)
    print("\nTest Accuracy: {0:f}\n".format(score))


if __name__ == "__main__":
    main()
