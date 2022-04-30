import numpy as np


def home_loan_credit_risk_model(df):
    """
    Data cleaning and data preprocessing
    """
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: np.nan if x == 365243 else x)
    df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: np.nan if x == 0.0 else x)
    df['Credit/Income'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['Annuity/Income'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['Employed/Birth'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['Flag_Greater_32'] = (df['DAYS_BIRTH'] / -365.25).apply(lambda x: 1 if x > 32 else 0)
    df['Flag_Employment_Greater_5'] = (df['DAYS_EMPLOYED'] / -365.25).apply(lambda x: 1 if x > 5 else 0)
    df['Flag_Income_Greater_Credit'] = df['AMT_INCOME_TOTAL'] > df['AMT_CREDIT']
    cols = ['DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
            'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1']
    for col in cols:
        for i in [2, 3]:
            df[f'{col}_power_{i}'] = df[col] ** i
    return df
