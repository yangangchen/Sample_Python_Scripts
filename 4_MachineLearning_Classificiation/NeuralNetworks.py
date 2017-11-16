from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf


def main():
    tf.set_random_seed(100)

    # Load datasets.
    df_train = pd.read_csv('train.csv', index_col=0)
    df_test = pd.read_csv('test.csv', index_col=0)
    numeric_features = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education-num', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']
    label = 'income-class'
    features = numeric_features + categorical_features
    df_train[categorical_features] = df_train[categorical_features].applymap(np.int64)
    df_test[categorical_features] = df_test[categorical_features].applymap(np.int64)

    print(pd.DataFrame([df_train.min(), df_train.max()], index=['min', 'max']))

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column(k) for k in features]

    # features = categorical_features
    # # feature_columns = [tf.feature_column.categorical_column_with_identity(k, int(df_train[k].max())) for k in features]
    # feature_columns = [tf.feature_column.embedding_column(
    #     categorical_column=tf.feature_column.categorical_column_with_identity(k, int(df_train[k].max())),
    #     dimension=int(df_train[k].max()))
    #                    for k in features]

    # feature_columns_numeric = [tf.feature_column.numeric_column(k) for k in numeric_features]
    # feature_columns_categorical = [tf.feature_column.embedding_column(
    #     categorical_column=tf.feature_column.categorical_column_with_identity(k, int(df_train[k].max())),
    #     dimension=int(df_train[k].max()))
    #                                for k in categorical_features]
    # feature_columns = feature_columns_numeric + feature_columns_categorical

    # dff = df_train.head()
    # for i in range(len(df_train.columns) - 1):
    #     print(dff.loc[:, features[i]])
    #     fc_tensor = feature_columns[i]._transform_feature(dff)
    #     sess = tf.InteractiveSession()
    #     print(sess.run(fc_tensor))
    #     print(i)
    #     input('check')

    # Build 2 layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 10],
                                            n_classes=2,
                                            model_dir='/tmp/yangang_dnnclassifier')
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: df_train[k].values for k in features}),
        y=pd.Series(df_train[label].values),
        num_epochs=None,
        shuffle=True)

    # Train model.
    classifier.train(input_fn=train_input_fn, steps=50000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: df_test[k].values for k in features}),
        y=pd.Series(df_test[label].values),
        num_epochs=1,
        shuffle=False)

    # Predictions.
    predictions = list(classifier.predict(input_fn=test_input_fn))
    predictions = [p["classes"][0] for p in predictions]
    print(predictions)

    # Evaluate accuracy.
    score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}\n".format(score))


if __name__ == "__main__":
    main()
