import traceback

import pandas as pd
from django.conf import settings
import pickle
import os
import datetime
import xgboost as xgb


def calculate_age(birth_date):
    b_date = datetime.datetime.strptime(birth_date, "%d-%m-%Y")
    age = int((datetime.datetime.today() - b_date).days/365)
    return age


def customer_segment(data):
    try:
        new_data = list()
        result = dict()
        op_fields = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                     'Credit_History', 'Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes',
                     'Property_Area_Semiurban', 'Property_Area_Urban']
        d = dict([(x, 0) for x in op_fields])

        d['Dependents'] = data.get('dependents')
        d['ApplicantIncome'] = round(data.get('annual_income') / 12, 2)
        d['LoanAmount'] = round(data.get('loan_amount_required') / 1000, 2)
        d['Loan_Amount_Term'] = data.get('loan_tenure')
        d['Credit_History'] = 1 if data.get('existing_loan_flag') is True else 0
        d['Gender_Male'] = 1 if data.get('gender') == "MALE" else 0
        d['Married_Yes'] = 1 if data.get('marital_status') == "MARRIED" else 0
        d['Education_Not Graduate'] = 1 if data.get('education_type') == "GRADUATE" else 0
        d['Self_Employed_Yes'] = 1 if data.get('self_employed_flag') is True else 0
        d['Property_Area_Semiurban'] = 1 if data.get('residential_area_type') == "SEMIURBAN" else 0
        d['Property_Area_Urban'] = 1 if data.get('residential_area_type') == "URBAN" else 0

        result['customer_eligibility_data'] = d

        new_data.append(d)
        df = pd.DataFrame(new_data)
        model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/customer_seg_svc.sav")
        model = pickle.load(open(model_path, 'rb'))
        prediction = model.predict(df)
        print(prediction)

        if prediction[0] == 1:
            result['customer_eligibility_flag'] = True
        else:
            result['customer_eligibility_flag'] = False
        return result
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {
            "flag": False,
            "customer_eligibility_data": None
        }


def default_probability(data):
    new_data = list()
    op_fields = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse',
                 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                 'NumberOfDependents']
    d = dict([(x, 0) for x in op_fields])

    d['age'] = calculate_age(data.get('date_of_birth'))
    d['NumberOfDependents'] = data.get('dependents')
    d['MonthlyIncome'] = data.get('annual_income') / 12
    d['NumberOfOpenCreditLinesAndLoans'] = data.get('existing_loans_count')
    d['NumberRealEstateLoansOrLines'] = 1 if data.get('existing_home_loan_flag') is True else 0
    d['NumberRealEstateLoansOrLines'] = 1 if data.get('existing_home_loan_flag') is True else 0

    new_data.append(d)
    df = pd.DataFrame(new_data)
    model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/default_probability_xgb.sav")
    model = pickle.load(open(model_path, 'rb'))
    prediction = model.predict(df)

    # model2 = xgb.Booster()
    # model2.load_model(model_path)
    # prediction = model2.predict(df)

    return prediction


def scorecard_signals(scorecard_data):
    try:
        data = {k: v for d in scorecard_data for k, v in d.items()}
        print(data)

        # TODO: BUILD SCORECARD

        # TODO: RULE ENGINE BASED CHECKS
        """
        CHECK 1 : MIN QUALIFICATION CRITERIA/ QUALIFIER
        
        - Vintage Customer	
        - Customer Demographics - Marital Status, Residential, Credit History, Family
        - Min Transacting products

        """
        # TODO : MIN QUALIFICATION CRITERIA/ QUALIFIER SIGNAL

        """
        CHECK 2 : BEHAVIOURAL ELIMINATION	
        
        - Behaviour on MATM 
        - Cash withdrawal to Balance enquiry ratio
        - Debt ratio
        - Delinquency identifiers - Open credit lines, real estate trade lines
        
        """
        # TODO : BEHAVIOURAL ELIMINATION SIGNAL

        """
        CHECK 3: INTENT ELIMINATION	

        - Transaction location
        - Fraud hot listing
        - Fraudulent transactions
        - Balance Enquiries

        """
        # TODO : INTENT ELIMINATION	 SIGNAL

        """
        CHECK 4: LOCATION BASED ELIMINATION	

        - Most prominent location
        - Negative district list
        - Negative pincode list
        """
        # TODO : LOCATION BASED ELIMINATION SIGNAL

        """
        CHECK 5: BUREAU ELIMINATION
        
        - Loan enquiries
        - Current Loans
        - Overdue
        - Outstanding
        - Bureau score
        """
        # TODO : BUREAU ELIMINATION SIGNAL
    except Exception:
        traceback.print_exc()

