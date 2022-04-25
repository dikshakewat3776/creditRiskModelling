from celery.exceptions import CeleryError
from creditRiskModelling.celery import app

import os
from django.conf import settings
import numpy as np
import pandas as pd

# for pre-processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# for machine learning modelling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix


# for ignoring warnings
import warnings
warnings.filterwarnings("ignore")


@app.task(bind=True)
def home_loan_credit_model(self):
    """
    Ref: https://www.kaggle.com/code/wssamhassan/home-credit-default-risk-wssam-hassan/notebook
    Impl: https://colab.research.google.com/drive/1632OS5AOg8rALyQCpktsAxIPa6onLwKY?usp=sharing
    """
    try:
        dirname = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/datasets/home-credit-default-risk")

        # training dataset
        train_df = pd.read_csv(f"{dirname}/application_train.csv", index_col='SK_ID_CURR')
        # print(test_df.head())

        # testing dataset
        test_df = pd.read_csv(f"{dirname}/application_test.csv", index_col='SK_ID_CURR')
        # print(test_df.head())

        # datasets sizes
        print(f'Training dataset contains {train_df.shape[0]} records and {train_df.shape[1]} columns.')
        print(f'Testing dataset contains {test_df.shape[0]} records and {test_df.shape[1]} columns.')

        """
        Data Cleaning and Pre-processing
        """
        # create copy of datasets
        train_copy = train_df.copy()
        test_copy = test_df.copy()

        # Drop Columns with >40% NaNs

        # Only columns with NaNs count and percentage
        columns = train_df.isnull().sum()[train_df.isnull().sum() != 0].keys()
        nans_count = train_df.isnull().sum()[train_df.isnull().sum() != 0].values
        nans_percentage = train_df.isnull().sum()[train_df.isnull().sum() != 0].values / train_df.shape[0]

        # create a dataframe from the extracted info.
        nans_df = pd.DataFrame(
            {'Column': columns, 'No. of NaNs': nans_count, '% of NaNs in Column': nans_percentage * 100})
        nans_df = nans_df.sort_values(by='% of NaNs in Column', ascending=False)
        # print(nans_df)

        # extract these columns from nans_df
        drop_cols = nans_df[nans_df['% of NaNs in Column'] > 40]['Column'].tolist()
        keep_cols = [col for col in train_df.columns if col not in drop_cols]

        # extract the new train dataframe
        train_df = train_df[keep_cols]

        # remove target from keep_cols and create the new test dataframe
        keep_cols.remove('TARGET')
        test_df = test_df[keep_cols]

        # check the new datasets shapes
        # print(train_df.shape)
        # print(test_df.shape)

        # Drop XNA records from CODE_GENDER column
        train_df = train_df[train_df['CODE_GENDER'] != 'XNA']

        # check
        train_df['CODE_GENDER'].value_counts()
        # print(train_df)

        # Drop the wrong value in AMT_INCOME_TOTAL column
        train_df = train_df[train_df['AMT_INCOME_TOTAL'] != 117000000.0]

        # Change the wrong value in DAYS_EMPLOYED and DAYS_LAST_PHONE_CHANGE columns
        train_df['DAYS_EMPLOYED'] = train_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)
        test_df['DAYS_EMPLOYED'] = test_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)

        # check
        # print(train_df['DAYS_EMPLOYED'].max())
        # print(test_df['DAYS_EMPLOYED'].max())

        # DAYS_LAST_PHONE_CHANGE column
        train_df['DAYS_LAST_PHONE_CHANGE'] = train_df['DAYS_LAST_PHONE_CHANGE'].apply(
            lambda x: np.nan if x == 0.0 else x)
        test_df['DAYS_LAST_PHONE_CHANGE'] = test_df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: np.nan if x == 0.0 else x)

        # check
        # print(train_df['DAYS_LAST_PHONE_CHANGE'].max())
        # print(test_df['DAYS_LAST_PHONE_CHANGE'].max())

        # NaNs Imputation
        # Catagorical Features Encoding : Instead of doing both label encoding for features with 2 unique categories and
        # one hot encoding for the rest, we can do one hot encoding for all features with and drop the first outcome column, as:
        # it will do it for us in one step
        # decrease the no. of features to prevent increasing dimensions and prevent overfitting
        # MinMax Scaling : get rid of the outliers

        # create a pipeline to deal with numerical features
        # 1- impute with median as most of the features contain outliers
        # 2- apply Min-Max Scaler get rid of the outliers
        numeric_transformer = Pipeline(
            steps=[("num_imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )

        # create a pipeline to deal with categorical features
        # 1- impute with the most frequent class "mode"
        # 2- apply One-Hot Encoding
        categorical_transformer = Pipeline(
            steps=[("cat_imputer", SimpleImputer(strategy="most_frequent")),
                   ("encoder", OneHotEncoder(handle_unknown='ignore', drop='first'))]
        )

        # create a column transformer instant
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
                ("cat", categorical_transformer, make_column_selector(dtype_include="object")),
            ]
        )

        """
        Modelling
        """

        # create a function for trained models evaluation
        def evaluate_model(model_pipeline):
            # prediction
            train_pred = model_pipeline.predict(X_train)
            test_pred = model_pipeline.predict(X_val)

            train_pred_proba = model_pipeline.predict_proba(X_train)
            test_pred_proba = model_pipeline.predict_proba(X_val)

            # evaluations
            print('Training & Validation ROC AUC Scores:\n', '-' * 40)
            print('Training   roc auc score= {:.4f}'.format(roc_auc_score(y_train, train_pred_proba[:, 1])))
            print('Validation roc auc score= {:.4f}'.format(roc_auc_score(y_val, test_pred_proba[:, 1])))
            print('')
            print('Training & Validation Confusion Metrices:')
            print('Training   confusion matrix:\n', confusion_matrix(y_train, train_pred))
            print('Validation confusion matrix:\n', confusion_matrix(y_val, test_pred))

        # return the train_df and test_df from their copies
        train_df = train_copy.copy()
        test_df = test_copy.copy()

        # data cleaning
        train_df = train_df[train_df['CODE_GENDER'] != 'XNA']

        train_df = train_df[train_df['AMT_INCOME_TOTAL'] != 117000000.0]

        train_df['DAYS_EMPLOYED'] = train_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)
        test_df['DAYS_EMPLOYED'] = test_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)

        train_df['DAYS_LAST_PHONE_CHANGE'] = train_df['DAYS_LAST_PHONE_CHANGE'].apply(
            lambda x: np.nan if x == 0.0 else x)
        test_df['DAYS_LAST_PHONE_CHANGE'] = test_df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: np.nan if x == 0.0 else x)

        # separate target out of features "predictors"
        X = train_df.drop('TARGET', axis=1)
        y = train_df['TARGET']

        # data splitting
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)

        """
        Feature engineering
        """
        # column represent the credit/income percentage
        X_train['Credit/Income'] = X_train['AMT_CREDIT'] / X_train['AMT_INCOME_TOTAL']
        X_val['Credit/Income'] = X_val['AMT_CREDIT'] / X_val['AMT_INCOME_TOTAL']
        test_df['Credit/Income'] = test_df['AMT_CREDIT'] / test_df['AMT_INCOME_TOTAL']

        # column represent the annuity/income percentage
        X_train['Annuity/Income'] = X_train['AMT_ANNUITY'] / X_train['AMT_INCOME_TOTAL']
        X_val['Annuity/Income'] = X_val['AMT_ANNUITY'] / X_val['AMT_INCOME_TOTAL']
        test_df['Annuity/Income'] = test_df['AMT_ANNUITY'] / test_df['AMT_INCOME_TOTAL']

        # column represent days employed percentage
        X_train['Employed/Birth'] = X_train['DAYS_EMPLOYED'] / X_train['DAYS_BIRTH']
        X_val['Employed/Birth'] = X_val['DAYS_EMPLOYED'] / X_val['DAYS_BIRTH']
        test_df['Employed/Birth'] = test_df['DAYS_EMPLOYED'] / test_df['DAYS_BIRTH']

        # flag represents if he's greater than 32 or not
        X_train['Flag_Greater_32'] = (X_train['DAYS_BIRTH'] / -365.25).apply(lambda x: 1 if x > 32 else 0)
        X_val['Flag_Greater_32'] = (X_val['DAYS_BIRTH'] / -365.25).apply(lambda x: 1 if x > 32 else 0)
        test_df['Flag_Greater_32'] = (test_df['DAYS_BIRTH'] / -365.25).apply(lambda x: 1 if x > 32 else 0)

        # flag represents if his employmeny years is greater than 5 or not
        X_train['Flag_Employment_Greater_5'] = (X_train['DAYS_EMPLOYED'] / -365.25).apply(lambda x: 1 if x > 5 else 0)
        X_val['Flag_Employment_Greater_5'] = (X_val['DAYS_EMPLOYED'] / -365.25).apply(lambda x: 1 if x > 5 else 0)
        test_df['Flag_Employment_Greater_5'] = (test_df['DAYS_EMPLOYED'] / -365.25).apply(lambda x: 1 if x > 5 else 0)

        # flag represents if his income is greater than the loan or not
        X_train['Flag_Income_Greater_Credit'] = X_train['AMT_INCOME_TOTAL'] > X_train['AMT_CREDIT']
        X_val['Flag_Income_Greater_Credit'] = X_val['AMT_INCOME_TOTAL'] > X_val['AMT_CREDIT']
        test_df['Flag_Income_Greater_Credit'] = test_df['AMT_INCOME_TOTAL'] > test_df['AMT_CREDIT']

        # create polynomial features of the top 3 pos & neg features with target
        cols = ['DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
                'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1']

        for col in cols:
            for i in [2, 3]:
                X_train[f'{col}_power_{i}'] = X_train[col] ** i
                X_val[f'{col}_power_{i}'] = X_val[col] ** i
                test_df[f'{col}_power_{i}'] = test_df[col] ** i

        # As in our target major is 91% and minor is 9%, we can't use either oversampling only as the minor is very small or
        # downsampling only as we will lose alot of our data, so we will apply both oversample on minor class firstly, then downsample the major one.

        # create oversampler, downsampler instants
        oversampler = SMOTE(sampling_strategy=0.25)  # minor/major = 1/4
        undersampler = RandomUnderSampler(sampling_strategy=0.75)  # minor/major = 3/4

        """
        RandomForestClassifier
        """
        # create pipeline
        print("RandomForestClassifier------------------------------------------------------------------------------------>")
        rf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)
        steps = [('preprocessor', preprocessor), ('oversampler', oversampler), ('undersampler', undersampler),
                 ('model', rf)]
        rf_pipeline = Pipeline(steps=steps)

        # train
        rf_pipeline.fit(X_train, y_train)

        # evaluate
        evaluate_model(rf_pipeline)

        """
        AdaBoostClassifier
        """
        print("AdaBoostClassifier------------------------------------------------------------------------------------>")
        # create pipeline
        adaboost = AdaBoostClassifier(n_estimators=200, random_state=42)
        steps = [('preprocessor', preprocessor), ('oversampler', oversampler), ('undersampler', undersampler),
                 ('model', adaboost)]
        ada_pipeline = Pipeline(steps=steps)

        # train
        ada_pipeline.fit(X_train, y_train)

        # evaluate
        evaluate_model(ada_pipeline)

        """
        LGBMClassifier
        """
        print("LGBMClassifier------------------------------------------------------------------------------------>")
        # create pipeline
        lgbm = LGBMClassifier(n_estimators=500, num_leaves=36, random_state=42)
        steps = [('preprocessor', preprocessor), ('oversampler', oversampler), ('undersampler', undersampler),
                 ('model', lgbm)]
        lgbm_pipeline = Pipeline(steps=steps)

        # train
        lgbm_pipeline.fit(X_train, y_train)

        # evaluate
        evaluate_model(lgbm_pipeline)

        """
        Prediction
        """

        res = pd.DataFrame({'SK_ID_CURR': test_df.index,
                            'TARGET': lgbm_pipeline.predict_proba(test_df)[:, 1]})
        print(res)

        return True
    except CeleryError as ce:
        print(ce)
        return False
