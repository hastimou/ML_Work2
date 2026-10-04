from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


#FUNCTION TO READ AN ARFF FILE
def read_arff(file_path):
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Convert byte strings to normal Python strings
    for column in df.columns:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8")
            if isinstance(x, bytes)
            else x
        )

    # Convert '?' into missing values
    df = df.replace("?", np.nan)

    return df







def preprocess(fold):
        # File names
        train_file = (
            f"soybean/soybean.fold.{fold:06d}.train.arff"
        )

        test_file = (
            f"soybean/soybean.fold.{fold:06d}.test.arff"
        )


        # Read training and testing data
        train_df = read_arff(train_file)
        test_df = read_arff(test_file)

        print("Training shape:", train_df.shape)
        print("Testing shape :", test_df.shape)

        # Separate features and target
        X_train = train_df.drop(columns=["class"])
        y_train = train_df["class"]

        X_test = test_df.drop(columns=["class"])
        y_test = test_df["class"]

        # Check missing values before preprocessing
        missing_train = X_train.isna().sum().sum()
        missing_test = X_test.isna().sum().sum()

        print("Missing values in training:", missing_train)
        print("Missing values in testing :", missing_test)

        # HANDLE MISSING VALUES
        # Because soybean features are categorical,
        # use the most frequent value (mode).

        imputer = SimpleImputer(strategy="most_frequent")

        X_train_imputed = imputer.fit_transform(X_train)

        X_test_imputed = imputer.transform(X_test)

        # ENCODE CATEGORICAL FEATURES
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        X_train_encoded = encoder.fit_transform(X_train_imputed)

        X_test_encoded = encoder.transform(X_test_imputed)

        print("Encoded training shape:", X_train_encoded.shape)
        print("Encoded testing shape :", X_test_encoded.shape)

        return X_train_encoded,y_train,X_test_encoded,y_test


