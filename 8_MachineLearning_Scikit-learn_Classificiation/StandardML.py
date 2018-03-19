# StandardML.py
# 
# Copyright (C) 2017  Yangang Chen
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# 
# 
# 
# Standard machine learning classifiers (e.g. Support Vector Classifier, 
# Gradient Boosting Classifier, etc) for the preprocessed data "train.csv" and "test.csv"

################################

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
    # model = SVC(C=0.1, kernel='rbf').fit(X_train, y_train)
    # Gradient Boosting Classifier
    model = GradientBoostingClassifier().fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    # y_predicted = clf.predict_proba(X_test)[:, 1]

    # Evaluate accuracy.
    score = accuracy_score(y_test, y_predicted)
    print("\nTest Accuracy: {0:f}\n".format(score))


if __name__ == "__main__":
    main()
