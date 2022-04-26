import uuid
from rest_framework.generics import GenericAPIView
from creditRiskModelling.creditRiskModels.creditModels import home_loan_credit_risk_model
from rest_framework.response import Response
from rest_framework import status
import pickle
from django.conf import settings
import os
import pandas as pd
import traceback


class RunHomeLoanCreditRiskModel(GenericAPIView):
    def post(self, request):
        try:
            # home_loan_credit_model.apply_async()
            query_params = request.GET
            model_type = query_params.get('modelType')

            ndata = list()
            request_params = request.data
            ndata.append(request_params)
            df = pd.DataFrame(ndata)

            ndf = home_loan_credit_risk_model(df)

            # returns the number of votes for each class (each tree in the forest makes its own decision and
            # chooses exactly one class). Hence, the precision is exactly 1/n_estimators

            if model_type.upper() == "RDF":
                model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/creditRiskModels/hlcrm_random_forest.sav")
                model = pickle.load(open(model_path, 'rb'))
                prediction = model.predict_proba(ndf)[:, 1]
                res = {'id': uuid.uuid4(),
                       'prediction_probability': prediction[0]}
                return Response({'responseStatus': 'SUCCESS', 'data': res}, status=status.HTTP_200_OK)
            elif model_type.upper() == "ADA":
                model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/creditRiskModels/hlcrm_ada_boost.sav")
                model = pickle.load(open(model_path, 'rb'))
                res = {'id': uuid.uuid4(),
                       'prediction_probability': model.predict_proba(ndf)[:, 1][0]}
                return Response({'responseStatus': 'SUCCESS', 'data': res}, status=status.HTTP_200_OK)
            # elif model_type.upper() == "LGBM":
            #     model_path = os.path.join(f"{settings.BASE_DIR}/creditRiskModelling/creditRiskModels/hlcrm_lgbm.sav")
            #     model = pickle.load(open(model_path, 'rb'))
            #     res = {'ID': uuid.uuid4(),
            #            'RESULT': model.predict_proba(ndf)[:, 1][0]}
            #     return Response({'responseStatus': 'SUCCESS', 'data': res}, status=status.HTTP_200_OK)
            else:
                return Response({'responseStatus': 'ERROR', 'data': 'Invalid modelType'}, status=status.HTTP_200_OK)
        except Exception as e:
            traceback.print_exc()
            resp_object = {
                "responseStatus": 'INTERNAL_SERVER_ERROR',
                "errors": str(e)
            }
            return Response(resp_object,
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
