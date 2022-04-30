from celery.exceptions import CeleryError
from creditRiskModelling.celery import app
import os
from django.conf import settings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import  make_column_selector
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import pickle
import time
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  PowerTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# for ignoring warnings
import warnings
warnings.filterwarnings("ignore")


@app.task(bind=True)
def home_loan_credit_model(self):
    """
    Impl: https://colab.research.google.com/drive/1632OS5AOg8rALyQCpktsAxIPa6onLwKY?usp=sharing
    """
    try:
        dirname = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/datasets/home-credit-default-risk")

        # training dataset
        train_df = pd.read_csv(f"{dirname}/application_train.csv", index_col='SK_ID_CURR')

        # testing dataset
        test_df = pd.read_csv(f"{dirname}/application_test.csv", index_col='SK_ID_CURR')

        # datasets sizes
        print(f'Training dataset contains {train_df.shape[0]} records and {train_df.shape[1]} columns.')
        print(f'Testing dataset contains {test_df.shape[0]} records and {test_df.shape[1]} columns.')

        """
        Data Cleaning and Pre-processing
        """
        # create copy of datasets
        train_copy = train_df.copy()
        test_copy = test_df.copy()

        # Drop Columns with >40% NaNs.
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

        # Drop XNA records from CODE_GENDER column
        train_df = train_df[train_df['CODE_GENDER'] != 'XNA']

        # Drop the wrong value in AMT_INCOME_TOTAL column
        train_df = train_df[train_df['AMT_INCOME_TOTAL'] != 117000000.0]

        # Change the wrong value in DAYS_EMPLOYED and DAYS_LAST_PHONE_CHANGE columns
        train_df['DAYS_EMPLOYED'] = train_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)
        test_df['DAYS_EMPLOYED'] = test_df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)

        # DAYS_LAST_PHONE_CHANGE column
        train_df['DAYS_LAST_PHONE_CHANGE'] = train_df['DAYS_LAST_PHONE_CHANGE'].apply(
            lambda x: np.nan if x == 0.0 else x)
        test_df['DAYS_LAST_PHONE_CHANGE'] = test_df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: np.nan if x == 0.0 else x)

        # NaNs Imputation
        # Categorical Features Encoding : Instead of doing both label encoding for features with 2 unique categories and
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
        print("RandomForestClassifier------------------------------------------------------------------------------->")
        rf = RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42)
        steps = [('preprocessor', preprocessor), ('oversampler', oversampler), ('undersampler', undersampler),
                 ('model', rf)]
        rf_pipeline = Pipeline(steps=steps)

        # train
        rf_pipeline.fit(X_train, y_train)

        # evaluate
        evaluate_model(rf_pipeline)

        # pickle dump model
        # pickle.dump(rf_pipeline, open('hlcrm_random_forest.sav', 'wb'))

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

        # pickle dump model
        pickle.dump(ada_pipeline, open('hlcrm_ada_boost.sav','wb'))

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

        # pickle dump model
        pickle.dump(lgbm_pipeline, open('hlcrm_lgbm.sav','wb'))

        """
        Prediction
        """

        res = pd.DataFrame({'SK_ID_CURR': test_df.index,
                            'TARGET': rf_pipeline.predict_proba(test_df)[:, 1]})
        return res
    except CeleryError as ce:
        print(ce)
        return False


@app.task(bind=True)
def probability_of_default_credit_model(self):
    """
    Impl: https://colab.research.google.com/drive/1WmE0mjAU8obAMppL1GWTjpCxU7C0DrkI?usp=sharing
          https://www.kaggle.com/code/dikshak123/give-me-some-credit-fp/edit
    """
    try:
        dirname = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/datasets/customer-default-probability")

        # training dataset
        train_df = pd.read_csv(f"{dirname}/cs-training.csv")

        # testing dataset
        test_df = pd.read_csv(f"{dirname}/cs-test.csv")

        target_variable = "SeriousDlqin2yrs"

        desc1 = train_df.describe(percentiles=[.25, .5, .75, .9, .95, .99, .999])
        print(desc1)

        # handle NA values
        features_with_na = train_df.isna().sum()[train_df.isna().sum() > 0]
        print(features_with_na.sort_values(ascending=False))

        (train_df.isna().sum(axis=1)[train_df.isna().sum(axis=1) > 0]
         .reset_index().rename(columns={0: 'number_of_na'})
         .groupby('number_of_na')
         .count().rename(columns={'index': 'number_of_observations'}))

        # Detecting outliers
        # Many of the financial features have outliers (extreme outliers = > 3* interquartile range), with a right-tail skew.
        # Hence, we should either use models that are robust to outliers (e.g. tree-based models), or transform the features accordingly (e.g. Box-Cox transformation).

        def get_outlier_counts(df, outlier_threshold=1.5):
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1

            outlier_counts = ((df < (Q1 - outlier_threshold * IQR)) | (df > (Q3 + outlier_threshold * IQR))).sum()
            return outlier_counts[outlier_counts > 0].sort_values(ascending=False)

        cnts = get_outlier_counts(train_df, outlier_threshold=3)
        print(cnts)

        desc2 = train_df.describe(percentiles=[.25,.5,.75,.9,.95,.99,.999])
        print(desc2)

        # Handling duplicates
        (train_df.groupby(train_df.columns.tolist(), as_index=False)
         .size()['size'].value_counts())

        # Train test split
        X, y = train_df.iloc[:, 1:], train_df.loc[:, target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Data Preprocessing
        # Based on EDA, we have found the need for imputation and scaling. Hence, we assemble a pipeline of imputation (median) and scaling (Box-Cox via PowerTransformer),
        # testing out different classifiers to compare model performance.
        # Based on experimentation, model performance when training with SMOTE for synthetic oversampling of the minority class did not significantly improve performance.

        def get_pipeline(classifier, random_seed=42):
            """Takes in a classifier, returns the entire data preprocessing pipeline (imblearn pipeline)"""
            return Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                # SMOTE(sampling_strategy='minority', random_state=random_seed), -> removed due to poor performance
                ('scaler', PowerTransformer()),
                ('clf', classifier)
            ])

        # We train several baseline models using different learning algorithms to get a sense of what learning algorithm best performs during k-fold cross validation.
        # In the interest of time, we perform model selection with minimal hyperparameter tuning,
        #  and select the model with best performance on cross-validation score. Ideally,this selection process happens after hyperparameter tuning for all of the models.

        def train_models(classifiers, num_folds=3, random_seed=42):
            # models = dict()
            results = dict()
            for classifier in classifiers:
                curr_time = time.time()
                model = get_pipeline(classifier, random_seed=random_seed)
                kfold = KFold(n_splits=num_folds)
                print(f'Training {classifier.__class__.__name__}')
                score = cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
                print("%0.2f AUC with a standard deviation of %0.2f" % (score.mean(), score.std()))
                # models[classifier.__class__.__name__] = model
                results[classifier.__class__.__name__] = score
                print(f'Training took {round(time.time() - curr_time)} seconds')
            print('Training complete')
            return results

        classifiers = [
            GaussianNB(),
            KNeighborsClassifier(5),
            LogisticRegression(),
            RandomForestClassifier(),
            XGBClassifier(),
        ]
        results = train_models(classifiers)
        print(results)

        # Based on the baseline model performances,
        # XGBoost algorithm performed the best (highest AUC with lowest standard deviation across 3 folds).
        #  To improve performance, we tune the model hyperparameters using randomized search, which does random search over the range of hyperparameters.
        #   This is faster than grid search, which exhaustively searches over all possible hyperparameters in the range provided.

        model = get_pipeline(XGBClassifier(n_jobs=-1))

        xgb_hyperparams = {
            'clf__max_depth': np.arange(4, 10, 1),
            'clf__learning_rate': [0.01, 0.1, 0.2, 0.3],
            'clf__n_estimators': np.arange(400, 1000, 100),
            'clf__subsample': np.arange(0.5, 1, 0.1),
            'clf__scale_pos_weight': [10, 15, 20]  # Imbalanced ratio of 94:6, hence add weight to negative class

        }

        clf = RandomizedSearchCV(estimator=model,
                                 param_distributions=xgb_hyperparams,
                                 scoring='roc_auc',
                                 n_iter=20,
                                 verbose=2)

        clf.fit(X_train, y_train)

        print("Best parameters:", clf.best_params_)
        print("Best score: ", clf.best_score_)

        model = get_pipeline(XGBClassifier(max_depth=4,
                                           learning_rate=0.01,
                                           n_estimators=800,
                                           subsample=0.5,
                                           scale_pos_weight=20,
                                           n_jobs=-1))

        model.fit(X_train, y_train)

        pickle.dump(model, open('default_probability_xgb.sav', 'wb'))

        def get_output(model, test_df):
            output_predictions = model.predict_proba(test_df.iloc[:, 2:])
            output_df = pd.DataFrame(output_predictions[:, 1]).rename(columns={0: 'probability'})
            output_df['id'] = range(1, len(output_df) + 1)
            output_df = output_df.loc[:, ['id', 'probability']]
            return output_df

        output_df = get_output(model, test_df)
        return output_df
    except CeleryError as ce:
        print(ce)
        return False


