from rest_framework import serializers
import datetime


class LoanEligibilityRequestSerializer(serializers.Serializer):

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

    def validate_date_of_birth(self, value):
        if value:
            try:
                datetime.datetime.strptime(value, "%d-%m-%Y").date()
            except ValueError:
                raise serializers.ValidationError('Invalid Date format in date_of_birth. Should be in dd-mm-yyyy format')


