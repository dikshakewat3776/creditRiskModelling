import traceback
import pandas as pd
import numpy as np
from django.conf import settings
import pickle
import os
import datetime
import xgboost as xgb
import regex as re
import psycopg2
import json
from sqlalchemy import create_engine


STATE_MASTER_LIST = [
    {"state_code": "01", "state": "JAMMU and KASHMIR"},
    {"state_code": "02", "state": "HIMACHAL PRADESH"},
    {"state_code": "03", "state": "PUNJAB"},
    {"state_code": "04", "state": "CHANDIGARH"},
    {"state_code": "05", "state": "UTTRANCHAL"},
    {"state_code": "06", "state": "HARAYANA"},
    {"state_code": "07", "state": "DELHI"},
    {"state_code": "08", "state": "RAJASTHAN"},
    {"state_code": "09", "state": "UTTAR PRADESH"},
    {"state_code": "10", "state": "BIHAR"},
    {"state_code": "11", "state": "SIKKIM"},
    {"state_code": "12", "state": "ARUNACHAL PRADESH"},
    {"state_code": "13", "state": "NAGALAND"},
    {"state_code": "14", "state": "MANIPUR"},
    {"state_code": "15", "state": "MIZORAM"},
    {"state_code": "16", "state": "TRIPURA"},
    {"state_code": "17", "state": "MEGHALAYA"},
    {"state_code": "18", "state": "ASSAM"},
    {"state_code": "19", "state": "WEST BENGAL"},
    {"state_code": "20", "state": "JHARKHAND"},
    {"state_code": "21", "state": "ORRISA"},
    {"state_code": "22", "state": "CHHATTISGARH"},
    {"state_code": "23", "state": "MADHYA PRADESH"},
    {"state_code": "24", "state": "GUJARAT"},
    {"state_code": "25", "state": "DAMAN and DIU"},
    {"state_code": "26", "state": "DADARA and NAGAR HAVELI"},
    {"state_code": "27", "state": "MAHARASHTRA"},
    {"state_code": "28", "state": "ANDHRA PRADESH"},
    {"state_code": "29", "state": "KARNATAKA"},
    {"state_code": "30", "state": "GOA"},
    {"state_code": "31", "state": "LAKSHADWEEP"},
    {"state_code": "32", "state": "KERALA"},
    {"state_code": "33", "state": "TAMIL NADU"},
    {"state_code": "34", "state": "PONDICHERRY"},
    {"state_code": "35", "state": "ANDAMAN and NICOBAR ISLANDS"},
    {"state_code": "36", "state": "TELANGANA"}
]


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

        if prediction[0] == 1:
            result['customer_eligibility_flag'] = True
        else:
            result['customer_eligibility_flag'] = False
        return result
    except Exception:
        traceback.print_exc()
        return {
            "customer_eligibility_flag": False,
            "customer_eligibility_data": None
        }


def get_prob_default(model, test_df):
    output_predictions = model.predict_proba(test_df.iloc[:, :])
    output_df = pd.DataFrame(output_predictions[:, :]).rename(columns={0: 'probability'})
    prob = output_df['probability'].to_dict().get(0)
    return prob


def default_probability_risk_v1(data):
    """
    Probability of Default (PD) tells us the likelihood that a borrower will default on the debt (loan or credit card).
    In simple words, it returns the expected probability of customers fail to repay the loan.
    """
    try:
        new_data = list()
        result = dict()
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

        result['customer_default_probability_data_v1'] = d

        new_data.append(d)
        df = pd.DataFrame(new_data)
        model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/default_probability_xgb_3.pkl")
        model = pickle.load(open(model_path, 'rb'))

        default_probability = round(get_prob_default(model, df), 2)

        if default_probability > 0.65:
            result['probability_of_default_flag_v1'] = True
            result['probability_of_default_score_v1'] = default_probability
        else:
            result['probability_of_default_flag_v1'] = False
            result['probability_of_default_score_v1'] = default_probability
        return result
    except Exception:
        traceback.print_exc()
        return {
            "probability_of_default_flag_v1": False,
            "probability_of_default_score_v1": 0,
            "customer_default_probability_data_v1": {}
        }


def compute_credit_score(input_df):
    try:
        scorecard_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/datasets/lending-club/df_scorecard.csv")
        df_scorecard = pd.read_csv(scorecard_path)
        inp = input_df.to_dict(orient='records')
        scorecard_scores = list()

        for k, v in inp[0].items():
            if v == 1:
                scores = df_scorecard[df_scorecard['Feature name'] == k]['Score - Final'].to_dict()
                if scores:
                    for key in scores.keys():
                        scorecard_scores.append(scores[key])
        credit_score = sum(scorecard_scores)
        return credit_score
    except Exception:
        traceback.print_exc()
        return 0


def default_probability_risk_v2(data):
    """
    Probability of Default (PD) tells us the likelihood that a borrower will default on the debt (loan or credit card).
    In simple words, it returns the expected probability of customers fail to repay the loan.
    """
    try:
        new_data = list()
        result = dict()
        op_fields = ['grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F',
                     'home_ownership:OWN', 'home_ownership:MORTGAGE',
                     'addr_state:NM_VA', 'addr_state:NY', 'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA',
                     'addr_state:UT_KY_AZ_NJ', 'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN',
                     'addr_state:GA_WA_OR', 'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
                     'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID', 'verification_status:Not Verified', 'verification_status:Source Verified',
                     'purpose:credit_card', 'purpose:debt_consolidation', 'purpose:oth__med__vacation', 'purpose:major_purch__car__home_impr',
                     'initial_list_status:w', 'term:36', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6', 'emp_length:7-9', 'emp_length:10',
                     'months_since_issue_d:<38', 'months_since_issue_d:38-39', 'months_since_issue_d:40-41', 'months_since_issue_d:42-48',
                     'months_since_issue_d:49-52', 'months_since_issue_d:53-64', 'months_since_issue_d:65-84', 'int_rate:<9.548', 'int_rate:9.548-12.025',
                     'int_rate:12.025-15.74', 'int_rate:15.74-20.281', 'months_since_earliest_cr_line:141-164', 'months_since_earliest_cr_line:165-247',
                     'months_since_earliest_cr_line:248-270', 'months_since_earliest_cr_line:271-352', 'months_since_earliest_cr_line:>352',
                     'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'acc_now_delinq:>=1', 'annual_inc:20K-30K', 'annual_inc:30K-40K',
                     'annual_inc:40K-50K', 'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K', 'annual_inc:80K-90K', 'annual_inc:90K-100K',
                     'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K', 'dti:<=1.4', 'dti:1.4-3.5', 'dti:3.5-7.7', 'dti:7.7-10.5',
                     'dti:10.5-16.1', 'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35', 'mths_since_last_delinq:Missing',
                     'mths_since_last_delinq:4-30', 'mths_since_last_delinq:31-56', 'mths_since_last_delinq:>=57', 'mths_since_last_record:Missing',
                     'mths_since_last_record:3-20', 'mths_since_last_record:21-31', 'mths_since_last_record:32-80', 'mths_since_last_record:81-86',
                     'mths_since_last_record:>86']

        d = dict([(x, 0) for x in op_fields])

        d['grade:A'] = 1 if data.get('grade') == 'A' else 0
        d['grade:B'] = 1 if data.get('grade') == 'B' else 0
        d['grade:C'] = 1 if data.get('grade') == 'C' else 0
        d['grade:D'] = 1 if data.get('grade') == 'D' else 0
        d['grade:E'] = 1 if data.get('grade') == 'E' else 0
        d['grade:F'] = 1 if data.get('grade') == 'F' else 0

        d['home_ownership:OWN'] = 1 if data.get('home_ownership') == 'OWN' else 0
        d['home_ownership:MORTGAGE'] = 1 if data.get('home_ownership') == 'MORTGAGE' else 0

        d['purpose:credit_card'] = 1 if data.get('loan_purpose') == 'CREDIT CARD' else 0
        d['purpose:debt_consolidation'] = 1 if data.get('loan_purpose') == 'DEBT' else 0
        d['purpose:oth__med__vacation'] = 1 if data.get('loan_purpose') == 'OTHER' else 0
        d['purpose:major_purch__car__home_impr'] = 1 if data.get('loan_purpose') == 'PURCHASE' else 0

        d['term:36'] = 1 if data.get('loan_tenure') > 100 else 0

        d['emp_length:1'] = 1 if data.get('employment_length') == 1 else 0
        d['emp_length:2-4'] = 1 if 2 <= data.get('employment_length') <= 4 else 0
        d['emp_length:5-6'] = 1 if 5 <= data.get('employment_length') <= 6 else 0
        d['emp_length:7-9'] = 1 if 7 <= data.get('employment_length') <= 9 else 0
        d['emp_length:10'] = 1 if data.get('employment_length') == 10 else 0

        d['int_rate:<9.548'] = 1 if data.get('interest_rate') <= 9.5 else 0
        d['int_rate:9.548-12.025'] = 1 if 9.5 <= data.get('interest_rate') <= 12.0 else 0
        d['int_rate:12.025-15.74'] = 1 if 12.1 <= data.get('interest_rate') <= 15.8 else 0
        d['int_rate:15.74-20.281'] = 1 if 15.9 <= data.get('interest_rate') <= 21.5 else 0

        d['annual_inc:20K-30K'] = 1 if 20000 <= data.get('annual_income')/76 <= 30000 else 0
        d['annual_inc:30K-40K'] = 1 if 30001 <= data.get('annual_income')/76 <= 40000 else 0
        d['annual_inc:40K-50K'] = 1 if 40001 <= data.get('annual_income')/76 <= 50000 else 0
        d['annual_inc:50K-60K'] = 1 if 50001 <= data.get('annual_income')/76 <= 60000 else 0
        d['annual_inc:60K-70K'] = 1 if 60001 <= data.get('annual_income')/76 <= 70000 else 0
        d['annual_inc:70K-80K'] = 1 if 70001 <= data.get('annual_income')/76 <= 80000 else 0
        d['annual_inc:80K-90K'] = 1 if 80001 <= data.get('annual_income')/76 <= 90000 else 0
        d['annual_inc:90K-100K'] = 1 if 90001 <= data.get('annual_income')/76 <= 100000 else 0
        d['annual_inc:100K-120K'] = 1 if 100001 <= data.get('annual_income')/76 <= 120000 else 0
        d['annual_inc:120K-140K'] = 1 if 120001 <= data.get('annual_income')/76 <= 140000 else 0
        d['annual_inc:>140K'] = 1 if data.get('annual_income')/76 >= 140001 else 0

        d['mths_since_last_delinq:Missing'] = 1 if data.get('months_since_last_delinquency') is None and \
                                                   data.get('existing_loan_flag') is True else 0
        d['mths_since_last_delinq:4-30'] = 1 if 4 <= data.get('months_since_last_delinquency') <= 30 else 0
        d['mths_since_last_delinq:31-56'] = 1 if 31 <= data.get('months_since_last_delinquency') <= 56 else 0
        d['mths_since_last_delinq:>=57'] = 1 if data.get('months_since_last_delinquency') <= 57 else 0

        result['customer_default_probability_data_v2'] = d

        new_data.append(d)
        df = pd.DataFrame(new_data)
        model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/pd_model_deployment.sav")
        model = pickle.load(open(model_path, 'rb'))

        default_probability = round(model.predict_proba(df)[:][:, 1][0], 2)

        credit_score = compute_credit_score(df)

        if default_probability > 0.65:
            result['probability_of_default_flag_v2'] = True
            result['credit_score'] = credit_score
            result['probability_of_default_score_v2'] = default_probability
        else:
            result['probability_of_default_flag_v2'] = False
            result['credit_score'] = credit_score
            result['probability_of_default_score_v2'] = default_probability
        return result
    except Exception:
        traceback.print_exc()
        return {
            "probability_of_default_flag_v2": False,
            "probability_of_default_score_v2": 0,
            "credit_score": 0,
            "customer_default_probability_data_v2": {}
        }


def loss_given_default(data):
    """
    Loss Given Default (LGD) is a proportion of the total exposure when borrower defaults. It is calculated by (1 - Recovery Rate).
    """
    try:
        new_data = list()
        op_fields = ['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment',
                     'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
                     'issue_d', 'loan_status', 'pymnt_plan', 'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state', 'dti',
                     'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
                     'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'out_prncp',
                     'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
                     'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
                     'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code', 'application_type', 'annual_inc_joint',
                     'dti_joint', 'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_il_6m',
                     'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
                     'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m']
        d = dict([(x, 0) for x in op_fields])

        d['term'] = data.get('loan_tenure')
        d['int_rate'] = data.get('interest_rate')
        d['loan_status'] = "Current" if data.get('existing_loan_flag') is True else ""
        d['emp_length'] = data.get('employment_length')
        d['installment'] = data.get('monthly_loan_installment_amount')
        d['expected_payment'] = data.get('monthly_loan_installment_amount')
        d['total_pymnt'] = data.get('existing_loans_amount')

        new_data.append(d)
        df = pd.DataFrame(new_data)
        columns = ["annual_inc", "delinq_2yrs", "dti", "emp_length", "grade", "home_ownership", "installment", "int_rate", "loan_amnt", "loan_status", "purpose", "term", "total_acc", "total_pymnt"]
        df = df[columns]
        df['term'] = df['term'].astype('float')
        df['int_rate'] = pd.to_numeric(df['int_rate'])
        df['loan_status'] = df['loan_status'].astype('category')
        df['emp_length'] = df['emp_length'].astype('float')
        df = df.dropna()

        df["expected_payment"] = df["installment"] * df["term"]
        df["recovery_rate"] = df["total_pymnt"] / df["expected_payment"] * 100
        df.drop(['installment', 'term'], axis=1)

        recovery_rate = df.to_dict(orient="records")[0].get('recovery_rate')
        lgd = 1 - recovery_rate

        if recovery_rate:
            return {
                'recovery_rate': round(recovery_rate, 2),
                'lgd': lgd
            }
        else:
            return {
                'recovery_rate': 0,
                'lgd': 0
            }
    except Exception:
        traceback.print_exc()
        return {
            'recovery_rate': 0,
            'lgd': 0
        }


def exposure_at_default(data):
    """
    Exposure at Default (EAD) is the amount that the borrower has to pay the bank at the time of default.
    """
    try:
        new_data = list()
        result = dict()
        op_fields = ['grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F',
                     'home_ownership:OWN', 'home_ownership:MORTGAGE',
                     'addr_state:NM_VA', 'addr_state:NY', 'addr_state:OK_TN_MO_LA_MD_NC', 'addr_state:CA',
                     'addr_state:UT_KY_AZ_NJ', 'addr_state:AR_MI_PA_OH_MN', 'addr_state:RI_MA_DE_SD_IN',
                     'addr_state:GA_WA_OR', 'addr_state:WI_MT', 'addr_state:TX', 'addr_state:IL_CT',
                     'addr_state:KS_SC_CO_VT_AK_MS', 'addr_state:WV_NH_WY_DC_ME_ID', 'verification_status:Not Verified',
                     'verification_status:Source Verified',
                     'purpose:credit_card', 'purpose:debt_consolidation', 'purpose:oth__med__vacation',
                     'purpose:major_purch__car__home_impr',
                     'initial_list_status:w', 'term:36', 'emp_length:1', 'emp_length:2-4', 'emp_length:5-6',
                     'emp_length:7-9', 'emp_length:10',
                     'months_since_issue_d:<38', 'months_since_issue_d:38-39', 'months_since_issue_d:40-41',
                     'months_since_issue_d:42-48',
                     'months_since_issue_d:49-52', 'months_since_issue_d:53-64', 'months_since_issue_d:65-84',
                     'int_rate:<9.548', 'int_rate:9.548-12.025',
                     'int_rate:12.025-15.74', 'int_rate:15.74-20.281', 'months_since_earliest_cr_line:141-164',
                     'months_since_earliest_cr_line:165-247',
                     'months_since_earliest_cr_line:248-270', 'months_since_earliest_cr_line:271-352',
                     'months_since_earliest_cr_line:>352',
                     'inq_last_6mths:0', 'inq_last_6mths:1-2', 'inq_last_6mths:3-6', 'acc_now_delinq:>=1',
                     'annual_inc:20K-30K', 'annual_inc:30K-40K',
                     'annual_inc:40K-50K', 'annual_inc:50K-60K', 'annual_inc:60K-70K', 'annual_inc:70K-80K',
                     'annual_inc:80K-90K', 'annual_inc:90K-100K',
                     'annual_inc:100K-120K', 'annual_inc:120K-140K', 'annual_inc:>140K', 'dti:<=1.4', 'dti:1.4-3.5',
                     'dti:3.5-7.7', 'dti:7.7-10.5',
                     'dti:10.5-16.1', 'dti:16.1-20.3', 'dti:20.3-21.7', 'dti:21.7-22.4', 'dti:22.4-35',
                     'mths_since_last_delinq:Missing',
                     'mths_since_last_delinq:4-30', 'mths_since_last_delinq:31-56', 'mths_since_last_delinq:>=57',
                     'mths_since_last_record:Missing',
                     'mths_since_last_record:3-20', 'mths_since_last_record:21-31', 'mths_since_last_record:32-80',
                     'mths_since_last_record:81-86',
                     'mths_since_last_record:>86']

        d = dict([(x, 0) for x in op_fields])

        d['grade:A'] = 1 if data.get('grade') == 'A' else 0
        d['grade:B'] = 1 if data.get('grade') == 'B' else 0
        d['grade:C'] = 1 if data.get('grade') == 'C' else 0
        d['grade:D'] = 1 if data.get('grade') == 'D' else 0
        d['grade:E'] = 1 if data.get('grade') == 'E' else 0
        d['grade:F'] = 1 if data.get('grade') == 'F' else 0

        d['home_ownership:OWN'] = 1 if data.get('home_ownership') == 'OWN' else 0
        d['home_ownership:MORTGAGE'] = 1 if data.get('home_ownership') == 'MORTGAGE' else 0

        d['purpose:credit_card'] = 1 if data.get('loan_purpose') == 'CREDIT CARD' else 0
        d['purpose:debt_consolidation'] = 1 if data.get('loan_purpose') == 'DEBT' else 0
        d['purpose:oth__med__vacation'] = 1 if data.get('loan_purpose') == 'OTHER' else 0
        d['purpose:major_purch__car__home_impr'] = 1 if data.get('loan_purpose') == 'PURCHASE' else 0

        d['term:36'] = 1 if data.get('loan_tenure') > 100 else 0

        d['emp_length:1'] = 1 if data.get('employment_length') == 1 else 0
        d['emp_length:2-4'] = 1 if 2 <= data.get('employment_length') <= 4 else 0
        d['emp_length:5-6'] = 1 if 5 <= data.get('employment_length') <= 6 else 0
        d['emp_length:7-9'] = 1 if 7 <= data.get('employment_length') <= 9 else 0
        d['emp_length:10'] = 1 if data.get('employment_length') == 10 else 0

        d['int_rate:<9.548'] = 1 if data.get('interest_rate') <= 9.5 else 0
        d['int_rate:9.548-12.025'] = 1 if 9.5 <= data.get('interest_rate') <= 12.0 else 0
        d['int_rate:12.025-15.74'] = 1 if 12.1 <= data.get('interest_rate') <= 15.8 else 0
        d['int_rate:15.74-20.281'] = 1 if 15.9 <= data.get('interest_rate') <= 21.5 else 0

        d['annual_inc:20K-30K'] = 1 if 20000 <= data.get('annual_income') / 76 <= 30000 else 0
        d['annual_inc:30K-40K'] = 1 if 30001 <= data.get('annual_income') / 76 <= 40000 else 0
        d['annual_inc:40K-50K'] = 1 if 40001 <= data.get('annual_income') / 76 <= 50000 else 0
        d['annual_inc:50K-60K'] = 1 if 50001 <= data.get('annual_income') / 76 <= 60000 else 0
        d['annual_inc:60K-70K'] = 1 if 60001 <= data.get('annual_income') / 76 <= 70000 else 0
        d['annual_inc:70K-80K'] = 1 if 70001 <= data.get('annual_income') / 76 <= 80000 else 0
        d['annual_inc:80K-90K'] = 1 if 80001 <= data.get('annual_income') / 76 <= 90000 else 0
        d['annual_inc:90K-100K'] = 1 if 90001 <= data.get('annual_income') / 76 <= 100000 else 0
        d['annual_inc:100K-120K'] = 1 if 100001 <= data.get('annual_income') / 76 <= 120000 else 0
        d['annual_inc:120K-140K'] = 1 if 120001 <= data.get('annual_income') / 76 <= 140000 else 0
        d['annual_inc:>140K'] = 1 if data.get('annual_income') / 76 >= 140001 else 0

        d['mths_since_last_delinq:Missing'] = 1 if data.get('months_since_last_delinquency') is None and \
                                                   data.get('existing_loan_flag') is True else 0
        d['mths_since_last_delinq:4-30'] = 1 if 4 <= data.get('months_since_last_delinquency') <= 30 else 0
        d['mths_since_last_delinq:31-56'] = 1 if 31 <= data.get('months_since_last_delinquency') <= 56 else 0
        d['mths_since_last_delinq:>=57'] = 1 if data.get('months_since_last_delinquency') <= 57 else 0

        result['customer_default_probability_data_v2'] = d

        new_data.append(d)
        df = pd.DataFrame(new_data)
        model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/ead_reg_deployment.sav")
        model = pickle.load(open(model_path, 'rb'))

        ead_probability = round(model.predict_proba(df)[:][:, 1][0], 2)
        return ead_probability
    except Exception:
        traceback.print_exc()
        return 0


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


def rule_engine(scorecard_data, type):
    try:
        # TODO: BUILD RULE ENGINE - based on various credit parameters flowing via bureau/ bank statements/cohorts, etc.
        data = {k: v for d in scorecard_data for k, v in d.items()}

        # TODO: RULE ENGINE BASED CHECKS
        """
        CHECK 1 : MIN QUALIFICATION CRITERIA/ QUALIFIER

        - Vintage Customer	
        - Customer Demographics - Marital Status, Residential, Credit History, Family
        - Min Transacting products
        - Customer Employment
        - Customer Resident
        - Mobile

        """
        if type == 1:
            min_qualification_criteria = dict()
            min_qualification_criteria['vintage_customer_check_flag'] = False
            min_qualification_criteria['customer_demographics_check_flag'] = False
            min_qualification_criteria['min_transacting_trade_lines_flag'] = False
            min_qualification_criteria['customer_employment_check_flag'] = False
            min_qualification_criteria['customer_mobile_check_flag'] = False
            min_qualification_criteria['overall_min_qualification_check_success'] = False

            if re.compile('^[A-Z]{5}[0-9]{4}[A-Z]{1}$').match(data.get('pan')) and \
                    re.compile('^(0/91)?[6-9][0-9]{9}$').match(data.get('mobile')):
                min_qualification_criteria['overall_min_qualification_check_success'] = True

            return min_qualification_criteria

        """
        CHECK 2 : BEHAVIOURAL ELIMINATION	

        - Behaviour on MATM 
        - Cash withdrawal to Balance enquiry ratio
        - Debt ratio
        - Delinquency identifiers - Open credit lines, real estate trade lines
        """
        if type == 2:
            behavioural_elimination_criteria = dict()
            behavioural_elimination_criteria['monthly_matm_transactions_flag'] = False
            behavioural_elimination_criteria['monthly_cash_withdrawal_flag'] = False
            behavioural_elimination_criteria['debt_flag'] = False
            behavioural_elimination_criteria['delinquency_flag'] = False
            behavioural_elimination_criteria['existing_loans_flag'] = False
            behavioural_elimination_criteria['existing_trade_lines_flag'] = False
            behavioural_elimination_criteria['overall_behavioural_check_success'] = True

            if data.get('months_since_last_delinquency') >= 5:
                behavioural_elimination_criteria['delinquency_flag'] = True
                behavioural_elimination_criteria['overall_behavioural_check_success'] = False

            if data.get('existing_loan_flag') >= 1:
                behavioural_elimination_criteria['existing_loans_flag'] = True
                behavioural_elimination_criteria['existing_trade_lines_flag'] = True

            if data.get('existing_loans_count') >= 1:
                behavioural_elimination_criteria['debt_flag'] = True

            return behavioural_elimination_criteria

        """
        CHECK 3: INTENT ELIMINATION	

        - Transaction location
        - Overdue, wilful defaulter
        - Fraudulent transactions, lawsuits
        - Balance Enquiries
        """
        if type == 3:
            intent_elimination_criteria = dict()
            intent_elimination_criteria['transaction_location_check_flag'] = False
            intent_elimination_criteria['fraudulent_transactions_flag'] = False
            intent_elimination_criteria['wilful_defaulter_flag'] = False
            intent_elimination_criteria['lawsuits_flag'] = False
            intent_elimination_criteria['balance_enquiries_exceeds'] = False
            intent_elimination_criteria['overall_intent_elimination_success'] = False

            return intent_elimination_criteria

        """
        CHECK 4: LOCATION BASED ELIMINATION	

        - Most prominent location
        - Negative district list
        - Negative pincode list
        """
        if type == 4:
            location_elimination_criteria = dict()
            location_elimination_criteria['pincode_check_flag'] = False
            location_elimination_criteria['city_check_flag'] = False
            location_elimination_criteria['state_check_flag'] = False
            location_elimination_criteria['address_check_flag'] = False
            location_elimination_criteria['overall_location_elimination_success'] = False

            if re.compile('^[1-9][0-9]{5}$').match(data.get('pincode')) :
                location_elimination_criteria['pincode_check_flag'] = True

            state_master_list = list(i['state'] for i in STATE_MASTER_LIST)
            if data.get('state').upper() in state_master_list:
                location_elimination_criteria['city_check_flag'] = True
                location_elimination_criteria['state_check_flag'] = True
                location_elimination_criteria['address_check_flag'] = True
                location_elimination_criteria['overall_location_elimination_success'] = True

            return location_elimination_criteria

        """
        CHECK 5: BUREAU ELIMINATION
        - Loan enquiries
        - Current Loans
        - Overdue
        - Outstanding
        - Bureau score
        - credit score
        """

        if type == 5:
            bureau_elimination_criteria = dict()
            bureau_elimination_criteria['current_loans_exceeds_flag'] = False
            bureau_elimination_criteria['current_overdue_exceeds_flag'] = False
            bureau_elimination_criteria['current_outstanding_exceeds_flag'] = False
            bureau_elimination_criteria['current_balance_check_flag'] = False
            bureau_elimination_criteria['bureau_score_check_flag'] = False
            bureau_elimination_criteria['credit_score_check_flag'] = False
            bureau_elimination_criteria['overall_bureau_elimination_success'] = False

            if data.get('credit_score') > 450:
                bureau_elimination_criteria['overall_bureau_elimination_success'] = True
            return  bureau_elimination_criteria
        return True
    except Exception as e:
        print(e)
        return False, {}


def test(pan):
    data = {'FFRPK9830A': {"customer_eligibility_flag":True, "probability_of_default_flag_v1":False, "probability_of_default_score_v1":0.1, "probability_of_default_flag_v2":False, "probability_of_default_score_v2":0.3, "loss_given_default":0.2, "loss_given_default_recovery_rate":66, "exposure_at_default":0.3, "expected_loss_v1":0.36,"expected_loss_v2":0.02, "credit_score":468, "loan_eligibility_flag":True},
           'FFRPK9830B': {"customer_eligibility_flag":True, "probability_of_default_flag_v1":False, "probability_of_default_score_v1":0.1, "probability_of_default_flag_v2":False, "probability_of_default_score_v2":0.3, "loss_given_default":0.667, "loss_given_default_recovery_rate":40, "exposure_at_default":0.7, "expected_loss_v1":0.76,"expected_loss_v2":0.02, "credit_score":218, "loan_eligibility_flag":True},
           'FFRPK9830C': {"customer_eligibility_flag":True, "probability_of_default_flag_v1":False, "probability_of_default_score_v1":0.1, "probability_of_default_flag_v2":False, "probability_of_default_score_v2":0.3, "loss_given_default":0.2, "loss_given_default_recovery_rate":28, "exposure_at_default":0.3, "expected_loss_v1":0.36,"expected_loss_v2":0.02, "credit_score":468, "loan_eligibility_flag":True}}
    res = data.get(pan) if data.get(pan) else {}
    return res


def get_overall_data():
    uri = "postgresql+psycopg2://postgres:postgres@localhost:5432/test"
    engine = create_engine(uri, echo=False)
    conn = engine.connect()
    query = """ SELECT * FROM escm"""
    df = pd.read_sql(query, con=conn)

    overall_count = df.shape[0]
    credit_worthy_count = len(df.loc[df['status'] == "Credit Worthy"])
    defaulted_count = len(df.loc[df['status'] == "Defaulted"])
    upcoming_defaults_count = len(df.loc[df['status'] == "Upcoming Default"])

    data = {
        "credit_worthy_count": credit_worthy_count,
        "defaulted_count": defaulted_count,
        "upcoming_defaults_count": upcoming_defaults_count,
        "overall_count": overall_count,
        "table_data": df.to_dict(orient="records")
    }
    with open("/home/diksha/Projects/creditRiskModelling/Frontend/escm.json", "w") as outfile:
        outfile.write(json.dumps(data))
    return data


def save_data(record_to_insert):
    try:

        connection = psycopg2.connect(user="postgres", password="postgres", host="127.0.0.1", port="5432", database="test")
        cursor = connection.cursor()

        postgres_insert_query = """ INSERT INTO escm (pan, first_name, last_name, input_data, result_data, status) VALUES (%s,%s,%s,%s,%s,%s)"""
        cursor.execute(postgres_insert_query, record_to_insert)
        connection.commit()
        connection.close()
        print("Record inserted successfully!!!")

        get_overall_data()
    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into the table!!!", error)







