from rest_framework.generics import GenericAPIView
from rest_framework import status
from rest_framework.response import Response
import traceback
from loanEligibilityEngine.serializers import LoanEligibilityRequestSerializer
from loanEligibilityEngine.utils import customer_segment, default_probability_risk_v1, \
    default_probability_risk_v2, loss_given_default, exposure_at_default

"""
Loan Eligibility Engine

Credit risk measures the probabilities of borrowers fail to pay back the debt and thus default on their obligations.
Credit risk modeling is widely adopted in banking industry for multiple applications:
- underwriting
- account management (e.g. extending line of credits)
- credit allowance
- risk management and capital planning
- regulatory capital calculation


There are two key components of credit risk measurement:
1) probability of default (PD), usually defined as likelihood of default over a period of time
2) loss given default (LGD), typically referred to as the amount that can not be recovered after the borrower defaults.
3) exposure at default (EAD), is the amount that the borrower has to pay the bank at the time of default.
The multiplication of these two components gives one the expected loss
"""


class loanEligibilityEngine(GenericAPIView):
    """
    post:
        Return all loan-eligibility data for each customer for loan offer.
    """

    serializer_class = LoanEligibilityRequestSerializer

    def post(self, request):
        try:
            # VALIDATE REQUEST PARAMETERS
            request_params = request.data
            serialized_object = LoanEligibilityRequestSerializer(data=request_params)

            if not serialized_object.is_valid():
                resp_object = {
                            "responseStatus": 'ERROR',
                            "errors": str(serialized_object.errors)
                        }
                return Response(resp_object, status=status.HTTP_400_BAD_REQUEST)


            scorecard_data = list()

            # CHECKING CUSTOMER ELIGIBILITY
            customer_eligibility_check = customer_segment(data=request_params)
            scorecard_data.append(customer_eligibility_check)
            # print(scorecard_data)

            # GENERATE PROBABILITY OF DEFAULT
            probability_of_default_check_v1 = default_probability_risk_v1(data=request_params)
            scorecard_data.append(probability_of_default_check_v1)

            probability_of_default_check_v2 = default_probability_risk_v2(data=request_params)
            scorecard_data.append(probability_of_default_check_v2)

            # GENERATE LOSS GIVEN DEFAULT
            lgd_compute = loss_given_default(data=request_params)

            # GENERATE EXPOSURE AT DEFAULT
            ead_compute = exposure_at_default(data=request_params)

            # GENERATE EXPECTED LOSS
            expected_loss_v1 = round(probability_of_default_check_v1.get('probability_of_default_flag_v1') * lgd_compute.get('lgd'), 2)
            expected_loss_v2 = round(probability_of_default_check_v2.get('probability_of_default_flag_v2') * lgd_compute.get('lgd'), 2)

            if expected_loss_v1 or expected_loss_v2 > 0.5:
                loan_eligibility_flag = True
            else:
                loan_eligibility_flag = False

            resp_object = {
                "responseStatus": 'SUCCESS',
                "data": {
                    'customer_eligibility_flag': customer_eligibility_check.get('customer_eligibility_flag'),
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
            }
            return Response(resp_object, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            resp_object = {
                "responseStatus": 'INTERNAL_SERVER_ERROR',
                "errors": str(e)
            }
            return Response(resp_object, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
