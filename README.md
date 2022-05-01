# Early_Signaling_for_Credit_Model (ESCM)

<b><i>Early_Signaling_for_Credit_Model (ESCM)</i></b> :- Proactive tool to manage credit risk.

Talking about “Credit Risk model” estimates financial losses that firm might suffer if the borrower defaults or does not repay the loan amount to the institute or firm.

So Financial firms with the help of historical data generated try to proactively look for those credit risks.

As these credit risks models are run once in a whole financial year so the activeness to take proactive measures is lost in the journey.

So we decided to trigger early warning signals promptly with the help of the machine learning model/ rule engine model to the Financial firm with a probabilistic value of the user getting default.
<i> Example </i>
Let’s say the user misses 1 payment due there’s the likelihood of missing 2nd. and if this is informed to the financial firm. They will start taking action on it promptly rather than waiting till a whole financial year and then taking action on it.

<b> To use our application we have 3 steps:</b>
<ul>
<li> Check Loan Eligibility </li>
<li>Probability value of (PD/EAD/LGD/ECL)</li>
<li> Early warning signaling Dashboard</li>
  </ul>
  
  <li><b>Check Loan Eligibility</b></li></br>
 <img src = "https://user-images.githubusercontent.com/94001814/166158353-a6515301-729a-4045-a60e-344bce9061ab.png" width=75% height=75%>
  </br></br>
   <img src = "https://user-images.githubusercontent.com/94001814/166158610-768c5788-549d-48a1-80a4-a85b772d9d8b.png"  width=75% height=75%>
  
  <li> <b>Probability value of (PD/EAD/LGD/ECL)</b></li></br>
  <img src = "https://user-images.githubusercontent.com/94001814/166158921-4c77a6bd-3cff-4e09-9e99-62cbe30fb9bf.png" width=75% height=75%>
  
  <li> <b>Early warning signaling Dashboard</b></li></br>
  <img src="https://user-images.githubusercontent.com/94001814/166158698-04091b1a-6e97-4d28-a5bd-1d436a4cbfb2.png" width=75% height=75%>
  
  
 <b> Steps to run the Project </b>
 <ul>
  <li> git clone `https://github.com/dikshakewat3776/creditRiskModelling`
  <li> Cd into the project still `creditRiskModelling` </li>
     # installing dependencies
  <li> pip install -r requirements.txt </li>
 # creating new migrations based on the changes you have made to your models
 <li>python manage.py makemigrations </li>
# for applying migrations.
  <li>python manage.py migrate </li>
# starting django server
  <li>python manage.py runserver</li>
# run app server
<li> streamlit run credit_risk_app.py </li>
  </ul>
  




