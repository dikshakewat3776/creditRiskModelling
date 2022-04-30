from django.shortcuts import render
from rest_framework.generics import GenericAPIView
from rest_framework import status
from rest_framework.response import Response
import traceback
import numpy as np
import json
import os
from django.conf import settings
import pickle
import pandas as pd
from loanEligibilityEngine.serializers import LoanEligibilityRequestSerializer
from loanEligibilityEngine.utils import customer_segment, default_probability


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
            print(scorecard_data)

            # GENERATE PROBABILITY OF DEFAULT
            # probability_of_default = default_probability(data=request_params)
            # print(probability_of_default)

            # TODO : GENERATE LOAN ELIGIBILITY FLAG

            # TODO : GENERATE LOAN ELIGIBILITY SCORE

            # TODO : GENERATE RISK SCORE

            # TODO : RULE BASED ENGINE

            # TODO : CHECKS - SIGNALS

            # TODO : BUILD SCORECARD

            resp_object = {
                "responseStatus": 'SUCCESS',
                "data": {
                    'customer_eligibility_flag': customer_eligibility_check.get('customer_eligibility_flag'),
                    'loan_eligibility_flag': True,
                    'loan_eligibility_score': 0,
                    'customer_risk_score': 0
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
