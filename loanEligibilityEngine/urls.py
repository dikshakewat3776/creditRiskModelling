from django.urls import path
from loanEligibilityEngine.views import loanEligibilityEngine, HomeLoanEligibilityEngine

urlpatterns = [
    path("", loanEligibilityEngine.as_view(), name='loan-eligibility'),
    path("home-loan-eligibility-engine/", HomeLoanEligibilityEngine.as_view(), name='home-loan-eligibility')
]
