from rest_framework import serializers
import datetime
import re


class LoanEligibilityRequestSerializer(serializers.Serializer):
    pan = serializers.CharField(max_length=10, required=True, allow_null=False, allow_blank=False)
    gender = serializers.ChoiceField(required=True,  choices=['MALE', 'FEMALE'], allow_null=False, allow_blank=False)
    marital_status = serializers.ChoiceField(required=True, choices=['MARRIED', 'UNMARRIED'], allow_null=False,
                                             allow_blank=False)
    education_type = serializers.ChoiceField(required=True, choices=['GRADUATE', 'NONGRADUATE'], allow_null=False,
                                             allow_blank=False)
    employment_type = serializers.ChoiceField(required=True, choices=['GOVERNMMENT', 'HOUSEWIFE', 'MILITARY',
                                                                      'PRIVATE SECTOR', 'PUBLIC SECTOR',
                                                                      "SELF EMPLOYED', 'STUDENT', 'UNEMPLOYED",
                                                                      'OTHERS', 'NO RESPONSE'], allow_null=False,
                                              allow_blank=False)
    self_employed_flag = serializers.BooleanField(required=True)
    residential_area_type = serializers.ChoiceField(required=True, choices=['RURAL', 'URBAN', 'SEMIURBAN'],
                                                    allow_null=False, allow_blank=False)
    dependents = serializers.IntegerField(required=True)
    annual_income = serializers.IntegerField(required=True)
    loan_amount_required = serializers.IntegerField(required=True)
    loan_tenure = serializers.IntegerField(required=True)
    existing_loan_flag = serializers.BooleanField(required=True)
    existing_loans_count = serializers.IntegerField(required=True)
    existing_home_loan_flag = serializers.BooleanField(required=True)
    date_of_birth = serializers.CharField(max_length=20, allow_null=False, allow_blank=False)
    grade = serializers.ChoiceField(required=True, choices=['A', 'B', 'C', 'D', 'E', 'F'], allow_null=False,
                                    allow_blank=False)
    home_ownership = serializers.ChoiceField(required=True, choices=['OWN', 'MORTGAGE', 'RENT', 'OTHER'],
                                             allow_null=False, allow_blank=False)
    loan_purpose = serializers.ChoiceField(required=True, choices=['CREDIT CARD', 'DEBT', 'PURCHASE', 'OTHER'],
                                           allow_null=False, allow_blank=False)
    employment_length = serializers.IntegerField(required=True)
    interest_rate = serializers.FloatField(required=True, min_value=6.0, max_value=21.5)
    months_since_last_delinquency = serializers.IntegerField(required=True, min_value=0, max_value=60)
    address = serializers.CharField(max_length=255, required=True, allow_null=False, allow_blank=False)
    city = serializers.CharField(max_length=100, required=True, allow_null=False, allow_blank=False)
    state = serializers.CharField(max_length=100, required=True, allow_null=False, allow_blank=False)
    pincode = serializers.CharField(max_length=100, required=True, allow_null=False, allow_blank=False)

    def validate_pan(self, value):
        if value:
            try:
                PAN_REGEX = "[A-Za-z]{5}\d{4}[A-Za-z]{1}"
                pan_format = re.compile(PAN_REGEX)
                if pan_format.match(value):
                    return value
                else:
                    raise serializers.ValidationError("Invalid PAN Number entered.")
            except ValueError:
                raise serializers.ValidationError("Invalid PAN Number entered.")

    def validate_date_of_birth(self, value):
        if value:
            try:
                datetime.datetime.strptime(value, "%d-%m-%Y").date()
            except ValueError:
                raise serializers.ValidationError('Invalid Date format in date_of_birth. Should be in dd-mm-yyyy format')


