import pandas as pd
from django.conf import settings
import pickle
import os


def customer_segment(data):
    new_data = list()
    op_fields = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                 'Credit_History', 'Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes',
                 'Property_Area_Semiurban', 'Property_Area_Urban']
    d = dict([(x, 0) for x in op_fields])

    d['Dependents'] = data.get('dependents')
    d['ApplicantIncome'] = data.get('annual_income') / 12
    d['LoanAmount'] = data.get('loan_amount_required') / 1000
    d['Loan_Amount_Term'] = data.get('loan_tenure')
    d['Credit_History'] = 1 if data.get('existing_loan_flag') is True else 0
    d['Gender_Male'] = 1 if data.get('gender') == "MALE" else 0
    d['Married_Yes'] = 1 if data.get('marital_status') == "MARRIED" else 0
    d['Education_Not Graduate'] = 1 if data.get('education_type') == "GRADUATE" else 0
    d['Self_Employed_Yes'] = 1 if data.get('self_employed_flag') is True else 0
    d['Property_Area_Semiurban'] = 1 if data.get('residential_area_type') == "SEMIURBAN" else 0
    d['Property_Area_Urban'] = 1 if data.get('residential_area_type') == "URBAN" else 0

    new_data.append(d)
    df = pd.DataFrame(new_data)
    model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModels/customer_seg_svc.sav")
    model = pickle.load(open(model_path, 'rb'))
    prediction = model.predict(df)

    if prediction[0] == 1:
        return True
    else:
        return False
