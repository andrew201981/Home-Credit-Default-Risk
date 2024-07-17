from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # Initialize the encoders
    ordinal_encoder = OrdinalEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    # Categorical column lists
    binary_cols = ["NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "EMERGENCYSTATE_MODE"]  # Sustituye con las columnas binarias
    multi_cat_cols = ["CODE_GENDER", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE"]  # Sustituye con las columnas con más de 2 categorías

    # Fit and transform binary columns
    for col in binary_cols:
        ordinal_encoder.fit(working_train_df[[col]])
        working_train_df[col] = ordinal_encoder.transform(working_train_df[[col]])
        working_val_df[col] = ordinal_encoder.transform(working_val_df[[col]])
        working_test_df[col] = ordinal_encoder.transform(working_test_df[[col]])


    # Adjust and transform columns with multiple categories
    for col in multi_cat_cols:
        onehot_encoder.fit(working_train_df[[col]])
        train_encoded = onehot_encoder.transform(working_train_df[[col]])
        val_encoded = onehot_encoder.transform(working_val_df[[col]])
        test_encoded = onehot_encoder.transform(working_test_df[[col]])

        # Add the encoded columns to the original DataFrame
        col_names = onehot_encoder.get_feature_names_out([col])
        working_train_df[col_names] = train_encoded
        working_val_df[col_names] = val_encoded
        working_test_df[col_names] = test_encoded

    # Delete the original categorical columns
    working_train_df.drop(columns=multi_cat_cols, inplace=True)
    working_val_df.drop(columns=multi_cat_cols, inplace=True)
    working_test_df.drop(columns=multi_cat_cols, inplace=True)

    print("Output train data shape: ", working_train_df.shape)
    print("Output val data shape: ", working_val_df.shape)
    print("Output test data shape: ", working_test_df.shape, "\n")


    # Initialize the imputer with the median strategy
    imputer = SimpleImputer(strategy="median")

    # Adjust and transform training data
    imputer.fit(working_train_df)
    working_train_df = imputer.transform(working_train_df)

    # Transform validation and test data using the same imputer
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)
    

    # Initialize the Min-Max Scaler
    scaler = MinMaxScaler()

    # Adjust and transform training data
    scaler.fit(working_train_df)
    working_train_df = scaler.transform(working_train_df)

    # Transform validation and test data using the same scaler
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)


    # Convert DataFrames to ndarrays
    train = working_train_df
    val = working_val_df
    test = working_test_df

    return train, val, test


