from flask import Flask,render_template
import os
import json

TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('styles')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)


@app.route('/', methods=["GET"])
def index():
    return render_template("Index.html")


@app.route("/behavioural_dash", methods=["GET"])
def behavioural_dash():
    with open('escm.json', 'r') as openfile:
        context = json.load(openfile)
    # context = {'credit_worthy_count': 1, 'defaulted_count': 2, 'upcoming_defaults_count': 1, 'overall_count': 4, 'table_data': [{'id': 1, 'pan': 'FFRPK8787K', 'first_name': 'diksha', 'last_name': 'kewat', 'input_data': {'pan': 'FFRPK8787K', 'city': 'Mumbai', 'state': '', 'gender': 'MALE', 'mobile': '9999787878', 'address': '', 'pincode': '400081', 'dependents': 2.0, 'loan_tenure': 0.0, 'loan_purpose': 'CREDIT CARD', 'annual_income': 0.0, 'date_of_birth': '21-08-1991', 'interest_rate': 0.0, 'education_type': 'GRADUATE', 'home_ownership': 'OWN', 'marital_status': 'MARRIED', 'employment_type': 'GOVERNMMENT', 'employment_length': 0.0, 'existing_loan_flag': False, 'self_employed_flag': False, 'existing_loans_count': 0.0, 'loan_amount_required': 0.0, 'existing_loans_amount': 0.0, 'residential_area_type': 'RURAL', 'existing_home_loan_flag': False, 'months_since_last_delinquency': 0.0, 'monthly_loan_installment_amount': 0.0}, 'result_data': {'credit_score': 0, 'expected_loss_v1': None, 'expected_loss_v2': None, 'loss_given_default': None, 'exposure_at_default': 0, 'loan_eligibility_flag': False, 'probability_of_default_flag_v1': False, 'probability_of_default_flag_v2': False, 'probability_of_default_score_v1': 0, 'probability_of_default_score_v2': 0, 'loss_given_default_recovery_rate': None}, 'status': 'Defaulted'}, {'id': 2, 'pan': 'FFRPK8787K', 'first_name': 'Laxmi', 'last_name': 'Sarki', 'input_data': {'pan': 'KLZX6789', 'city': 'Mumbai', 'state': '', 'gender': 'MALE', 'mobile': '9999787878', 'address': '', 'pincode': '400081', 'dependents': 2.0, 'loan_tenure': 0.0, 'loan_purpose': 'CREDIT CARD', 'annual_income': 0.0, 'date_of_birth': '21-08-1991', 'interest_rate': 0.0, 'education_type': 'GRADUATE', 'home_ownership': 'OWN', 'marital_status': 'MARRIED', 'employment_type': 'GOVERNMMENT', 'employment_length': 0.0, 'existing_loan_flag': False, 'self_employed_flag': False, 'existing_loans_count': 0.0, 'loan_amount_required': 0.0, 'existing_loans_amount': 0.0, 'residential_area_type': 'RURAL', 'existing_home_loan_flag': False, 'months_since_last_delinquency': 0.0, 'monthly_loan_installment_amount': 0.0}, 'result_data': {'credit_score': 0, 'expected_loss_v1': None, 'expected_loss_v2': None, 'loss_given_default': None, 'exposure_at_default': 0, 'loan_eligibility_flag': False, 'probability_of_default_flag_v1': False, 'probability_of_default_flag_v2': False, 'probability_of_default_score_v1': 0, 'probability_of_default_score_v2': 0, 'loss_given_default_recovery_rate': None}, 'status': 'Upcoming Default'}, {'id': 3, 'pan': 'ABCD123c', 'first_name': 'Diksha', 'last_name': 'Kewat', 'input_data': {'pan': 'KLZX6789', 'city': 'Mumbai', 'state': '', 'gender': 'MALE', 'mobile': '9999787878', 'address': '', 'pincode': '400081', 'dependents': 2.0, 'loan_tenure': 0.0, 'loan_purpose': 'CREDIT CARD', 'annual_income': 0.0, 'date_of_birth': '21-08-1991', 'interest_rate': 0.0, 'education_type': 'GRADUATE', 'home_ownership': 'OWN', 'marital_status': 'MARRIED', 'employment_type': 'GOVERNMMENT', 'employment_length': 0.0, 'existing_loan_flag': False, 'self_employed_flag': False, 'existing_loans_count': 0.0, 'loan_amount_required': 0.0, 'existing_loans_amount': 0.0, 'residential_area_type': 'RURAL', 'existing_home_loan_flag': False, 'months_since_last_delinquency': 0.0, 'monthly_loan_installment_amount': 0.0}, 'result_data': {'credit_score': 0, 'expected_loss_v1': None, 'expected_loss_v2': None, 'loss_given_default': None, 'exposure_at_default': 0, 'loan_eligibility_flag': False, 'probability_of_default_flag_v1': False, 'probability_of_default_flag_v2': False, 'probability_of_default_score_v1': 0, 'probability_of_default_score_v2': 0, 'loss_given_default_recovery_rate': None}, 'status': 'Credit Worthy'}, {'id': 4, 'pan': 'TYUF4567', 'first_name': 'Harsha', 'last_name': 'Pawar', 'input_data': {'pan': 'KLZX6789', 'city': 'Mumbai', 'state': '', 'gender': 'MALE', 'mobile': '9999787878', 'address': '', 'pincode': '400081', 'dependents': 2.0, 'loan_tenure': 0.0, 'loan_purpose': 'CREDIT CARD', 'annual_income': 0.0, 'date_of_birth': '21-08-1991', 'interest_rate': 0.0, 'education_type': 'GRADUATE', 'home_ownership': 'OWN', 'marital_status': 'MARRIED', 'employment_type': 'GOVERNMMENT', 'employment_length': 0.0, 'existing_loan_flag': False, 'self_employed_flag': False, 'existing_loans_count': 0.0, 'loan_amount_required': 0.0, 'existing_loans_amount': 0.0, 'residential_area_type': 'RURAL', 'existing_home_loan_flag': False, 'months_since_last_delinquency': 0.0, 'monthly_loan_installment_amount': 0.0}, 'result_data': {'credit_score': 0, 'expected_loss_v1': None, 'expected_loss_v2': None, 'loss_given_default': None, 'exposure_at_default': 0, 'loan_eligibility_flag': False, 'probability_of_default_flag_v1': False, 'probability_of_default_flag_v2': False, 'probability_of_default_score_v1': 0, 'probability_of_default_score_v2': 0, 'loss_given_default_recovery_rate': None}, 'status': 'Defaulted'}]}
    return render_template("escm_dashboard.html", context=context)
