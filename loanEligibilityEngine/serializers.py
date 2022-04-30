from rest_framework import serializers


class LoanEligibilityRequestSerializer(serializers.Serializer):
    gender = serializers.ChoiceField(required=True,  choices=['MALE', 'FEMALE'], allow_null=False, allow_blank=False)
    marital_status = serializers.ChoiceField(required=True, choices=['MARRIED', 'UNMARRIED'], allow_null=False,
                                             allow_blank=False)
    education_type = serializers.ChoiceField(required=True, choices=['GRADUATE', 'NONGRADUATE'], allow_null=False,
                                             allow_blank=False)
    employment_type = serializers.CharField(required=True, max_length=100, allow_null=False, allow_blank=False)
    self_employed_flag = serializers.BooleanField(required=True)
    residential_area_type = serializers.ChoiceField(required=True, choices=['RURAL', 'URBAN', 'SEMIURBAN'],
                                                    allow_null=False, allow_blank=False)
    dependents = serializers.IntegerField(required=True)
    annual_income = serializers.IntegerField(required=True)
    loan_amount_required = serializers.IntegerField(required=True)
    loan_tenure = serializers.IntegerField(required=True)
    existing_loan_flag = serializers.BooleanField(required=True)
