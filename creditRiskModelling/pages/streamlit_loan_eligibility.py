"""
This file is to identify the customers segments for credit risk where company wants to automate the loan eligibility process (real time)
based on customer's detail provided while filling online application, and classify those are eligible for loan amount
so that they can specifically target these customers
"""

import streamlit as st
from utils import default_probability_risk_v1,default_probability_risk_v2, loss_given_default, exposure_at_default, \
    home_loan_credit_risk_model, rule_engine, save_data
import json


def app():
    # ---------------------------------#
    st.markdown("## Loan Eligibility Credit Engine ")

    st.write("""
    Credit risk measures the probabilities of borrowers fail to pay back the debt and thus default on their obligations.
    There are three key components of credit risk measurement:
    1) probability of default (PD), usually defined as likelihood of default over a period of time
    2) loss given default (LGD), typically referred to as the amount that can not be recovered after the borrower defaults.
    3) exposure at default (EAD), is the amount that the borrower has to pay the bank at the time of default.
    The multiplication of these two components gives one the expected loss
    """)

    # Main panel
    # Take request parameters
    scorecard_data = list()
    final_eligibility = list()
    status = None

    st.subheader('1. Customer Details')
    first_name = st.text_input('first name')
    last_name = st.text_input('last name')
    pan = st.text_input('pan')
    gender = st.selectbox('gender', ('MALE', 'FEMALE'))
    marital_status = st.selectbox('marital_status', ('MARRIED', 'UNMARRIED'))
    education_type = st.selectbox('education_type', ('GRADUATE', 'NONGRADUATE'))
    employment_type = st.selectbox('employment_type', ('GOVERNMMENT', 'HOUSEWIFE', 'MILITARY', 'PRIVATE SECTOR',
                                                      'PUBLIC SECTOR', 'SELF EMPLOYED', 'STUDENT', 'UNEMPLOYED','OTHERS'))
    date_of_birth = st.text_input('date_of_birth')
    # date_of_birth = datetime.datetime.strptime(str(date_of_birth), "%Y-%m-%d").strftime("%d-%m-%Y")
    dependents = st.number_input('dependents')
    mobile = st.text_input('mobile')

    if mobile:
        scorecard_data.extend([{'pan': pan, 'mobile': mobile}])
        rule_res = rule_engine(scorecard_data, type=1)
        final_eligibility.append(rule_res)
        if rule_res:
            st.warning('This is a signal:')
            st.json(rule_res)

    st.subheader('2. Customer Address Details')
    residential_area_type = st.selectbox('residential_area_type', ('RURAL', 'URBAN', 'SEMIURBAN'))
    home_ownership = st.selectbox('home_ownership', ('OWN', 'MORTGAGE', 'RENT', 'OTHER'))
    address = st.text_input('address')
    city = st.text_input('city')
    state = st.text_input('state')
    pincode = st.text_input('pincode')

    if pincode:
        scorecard_data.extend([{'city': city, 'state': state, 'pincode': pincode}])
        rule_res = rule_engine(scorecard_data, type=4)
        if rule_res:
            final_eligibility.append(rule_res)
            st.warning('This is a signal:')
            st.json(rule_res)

    st.subheader('2. Customer Employment Details')
    self_employed_flag = st.checkbox('self_employed_flag')
    annual_income = st.number_input('annual_income')
    employment_length = st.number_input('employment_length')

    st.subheader('3. Required loan Details')
    loan_amount_required = st.number_input('loan_amount_required')
    loan_tenure = st.number_input('loan_tenure')
    loan_purpose = st.selectbox('loan_purpose', ('CREDIT CARD', 'DEBT', 'PURCHASE', 'OTHER'))
    interest_rate = st.number_input('interest_rate')

    st.subheader('4. Existing loan Details')
    existing_loan_flag = st.checkbox('existing_loan_flag')
    existing_home_loan_flag = st.checkbox('existing_home_loan_flag')
    existing_loans_count = st.number_input('existing_loans_count')
    existing_loans_amount = st.number_input('existing_loans_amount')
    monthly_loan_installment_amount = st.number_input('monthly_loan_installment_amount')
    months_since_last_delinquency = st.number_input('months_since_last_delinquency')

    if months_since_last_delinquency:
        scorecard_data.extend([{'months_since_last_delinquency': months_since_last_delinquency, 'existing_loan_flag': existing_loan_flag,
                                'existing_loans_count': existing_loans_count}])
        rule_res = rule_engine(scorecard_data, type=2)
        final_eligibility.append(rule_res)
        if rule_res:
            st.warning('This is a signal:')
            st.json(rule_res)

    if existing_loans_count == 0:
        grade = "A"
    elif existing_loans_count == 1 and existing_loans_amount >= 90000:
        grade = "B"
    elif existing_loans_count >=2 and  existing_loans_amount >= 350000:
        grade = "C"
    else:
        grade = "D"

    if st.button('PREDICT'):
        data = dict()
        data['grade'] = grade
        data['pan'] = pan
        data['gender'] = gender
        data['marital_status'] = marital_status
        data['education_type'] = education_type
        data['employment_type'] = employment_type
        data['date_of_birth'] = date_of_birth
        data['dependents'] = dependents
        data['residential_area_type'] = residential_area_type
        data['home_ownership'] = home_ownership
        data['address'] = address
        data['city'] = city
        data['state'] = state
        data['pincode'] = pincode
        data = dict()
        data['pan'] = pan
        data['gender'] = gender
        data['marital_status'] = marital_status
        data['education_type'] = education_type
        data['employment_type'] = employment_type
        data['date_of_birth'] = date_of_birth
        data['dependents'] = dependents
        data['residential_area_type'] = residential_area_type
        data['home_ownership'] = home_ownership
        data['address'] = address
        data['city'] = city
        data['state'] = state
        data['pincode'] = pincode
        data['mobile'] = mobile
        data['self_employed_flag'] = self_employed_flag
        data['annual_income'] = annual_income
        data['self_employed_flag'] = self_employed_flag
        data['employment_length'] = employment_length
        data['loan_amount_required'] = loan_amount_required
        data['loan_tenure'] = loan_tenure
        data['loan_purpose'] = loan_purpose
        data['interest_rate'] = interest_rate
        data['existing_loan_flag'] = existing_loan_flag
        data['existing_home_loan_flag'] = existing_home_loan_flag
        data['existing_loans_count'] = existing_loans_count
        data['existing_loans_amount'] = existing_loans_amount
        data['monthly_loan_installment_amount'] = monthly_loan_installment_amount
        data['months_since_last_delinquency'] = months_since_last_delinquency

        # GENERATE PROBABILITY OF DEFAULT
        probability_of_default_check_v1 = default_probability_risk_v1(data=data)
        print(probability_of_default_check_v1)

        probability_of_default_check_v2 = default_probability_risk_v2(data=data)
        print(probability_of_default_check_v2)

        # GENERATE LOSS GIVEN DEFAULT
        lgd_compute = loss_given_default(data=data)
        print(lgd_compute)

        # GENERATE EXPOSURE AT DEFAULT
        ead_compute = exposure_at_default(data=data)
        print(ead_compute)

        # GENERATE EXPECTED LOSS
        expected_loss_v1 = round(
            probability_of_default_check_v1.get('probability_of_default_flag_v1') * lgd_compute.get('lgd'), 2)
        expected_loss_v2 = round(
            probability_of_default_check_v2.get('probability_of_default_flag_v2') * lgd_compute.get('lgd'), 2)

        if (expected_loss_v1 > 0.5) or (expected_loss_v2 > 0.5):
            loan_eligibility_flag = True
        else:
            loan_eligibility_flag = False

        result = {
            'probability_of_default_flag_v1': probability_of_default_check_v1.get('probability_of_default_flag_v1'),
            'probability_of_default_score_v1': probability_of_default_check_v1.get('probability_of_default_score_v1'),
            'probability_of_default_flag_v2': probability_of_default_check_v2.get('probability_of_default_flag_v2'),
            'probability_of_default_score_v2': probability_of_default_check_v2.get('probability_of_default_score_v2'),
            'loss_given_default': lgd_compute.get('lgd'),
            'loss_given_default_recovery_rate': lgd_compute.get('recovery_rate'),
            'exposure_at_default': ead_compute,
            'expected_loss_v1': expected_loss_v1,
            'expected_loss_v2': expected_loss_v2,
            'credit_score': probability_of_default_check_v2.get('credit_score'),
            'loan_eligibility_flag': loan_eligibility_flag
        }

        st.json(result)

        if result:
            if result['credit_score']:
                scorecard_data.extend([{'credit_score': result['credit_score']}])
                rule_res = rule_engine(scorecard_data, type=5)
                final_eligibility.append(rule_res)
                if rule_res:
                    st.warning('This is a signal:')
                    st.json(rule_res)

            final_eligibility_data = {k: v for d in final_eligibility for k, v in d.items()}

            status = "Credit Worthy"

            if not final_eligibility_data.get('overall_min_qualification_check_success') or \
                    final_eligibility_data.get('overall_location_elimination_success'):
                status = "Upcoming Default"

            if not final_eligibility_data.get('overall_behavioural_check_success'):
                status = "Defaulted"

            st.success(status)

            data_to_save = {
                "pan": pan,
                "first_name": first_name,
                "last_name": last_name,
                "input_data": json.dumps(data),
                "result_data": json.dumps(result),
                "status": status
            }
            save_data(record_to_insert=tuple(data_to_save.values()))



