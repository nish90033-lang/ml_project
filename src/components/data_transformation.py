
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "artifacts", "preprocessor.pkl"
    )


class DataTransformation:
    """
    This class is responsible for data transformation tasks,
    including preprocessing and feature engineering.
    """

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            # Define categorical and numerical features
            cat_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_features = [
                'reading_score',
                'writing_score'
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            logging.info(
                "Numerical and categorical pipelines created successfully"
            )

            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            logging.info("Preprocessor object created successfully")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            # Separate target variable
            target_colum_name = 'math_score'

            input_feature_train_df = train_df.drop(
                columns=[target_colum_name]
            )

            target_colum_name_train_df = train_df[target_colum_name]

            input_feature_test_df = test_df.drop(
                columns=[target_colum_name]
            )

            target_colum_name_test_df = test_df[target_colum_name]

            logging.info(
                "Applying preprocessing object on training and testing datasets"
            )

            # Fit on training data and transform
            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )

            # Transform test data using training parameters
            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            # Combine transformed features with target variable
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_colum_name_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_colum_name_test_df)
            ]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
