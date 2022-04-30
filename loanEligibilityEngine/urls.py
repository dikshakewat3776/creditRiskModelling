from django.urls import path
from loanEligibilityEngine.views import loanEligibilityEngine

urlpatterns = [
    path("", loanEligibilityEngine.as_view(), name='loan-eligibility')
]
