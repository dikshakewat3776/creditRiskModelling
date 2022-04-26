"""
This file is to identify the customers segments for credit risk where company wants to automate the loan eligibility process (real time)
based on customer's detail provided while filling online application, and classify those are eligible for loan amount
so that they can specifically target these customers
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os


def feature_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

# ---------------------------------#
# Model building
def build_model(train_data, test_data, **kwargs):
    # CREATING A NEW COLUMN SOURCE UNDER BOTH TRAIN AND TEST DATA
    train_data["Source"] = "Train"
    test_data["Source"] = "Test"

    # COMBINE BOTH TRAIN AND TEST AS FULL DATA
    data = pd.concat([train_data, test_data])

    # THere is an invalid category as 3+
    data.Dependents = np.where(data.Dependents == '3+', 3, data.Dependents).astype(float)

    # MISSING VALUE IMPUTATION
    for col_name in list(data):
        if ((col_name not in ['Loan_ID', 'Loan_Status', 'Source']) & (data[col_name].isnull().sum() > 0)):
            if (data[col_name].dtype != object):
                temp1 = data[col_name][data.Source == "Train"].median()
                data[col_name].fillna(temp1, inplace=True)
            else:
                temp2 = data[col_name][data.Source == "Train"].mode()[0]
                data[col_name].fillna(temp2, inplace=True)

    # OUTLIER DETECTION AND CORRECTION

    # ApplicantIncome
    data[data.Source == "Train"].boxplot(column='ApplicantIncome')
    np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], [95, 96, 97, 98, 99])
    data.ApplicantIncome = np.where(
        data.ApplicantIncome > np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 99),
        np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 99), data.ApplicantIncome)
    data.ApplicantIncome = np.where(
        data.ApplicantIncome > np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 95),
        np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 95), data.ApplicantIncome)
    data.ApplicantIncome = np.where(
        data.ApplicantIncome > np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 90),
        np.percentile(data.loc[data.Source == "Train", "ApplicantIncome"], 90), data.ApplicantIncome)

    # CoapplicantIncome
    data[data.Source == "Train"].boxplot(column="CoapplicantIncome")
    np.percentile(data.loc[data.Source == "Train", "CoapplicantIncome"], 99)
    data.CoapplicantIncome = np.where(
        data.CoapplicantIncome > np.percentile(data.loc[data.Source == "Train", "CoapplicantIncome"], 99),
        np.percentile(data.loc[data.Source == "Train", "CoapplicantIncome"], 99), data.CoapplicantIncome)
    data.CoapplicantIncome = np.where(
        data.CoapplicantIncome > np.percentile(data.loc[data.Source == "Train", "CoapplicantIncome"], 95),
        np.percentile(data.loc[data.Source == "Train", "CoapplicantIncome"], 95), data.CoapplicantIncome)

    # LoanAmount
    data[data.Source == "Train"].boxplot(column="LoanAmount")
    np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 99)
    data.LoanAmount = np.where(
        data.LoanAmount > np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 99),
        np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 99), data.LoanAmount)
    data.LoanAmount = np.where(
        data.LoanAmount > np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 95),
        np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 95), data.LoanAmount)
    data.LoanAmount = np.where(
        data.LoanAmount > np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 90),
        np.percentile(data.loc[data.Source == "Train", "LoanAmount"], 90), data.LoanAmount)

    # ONE HOT ENCODING OF CATEGORICAL VARIABLES  BY CREATING dummy VARIABLES ########
    cat = data.loc[:, data.dtypes == object].columns
    dummy = pd.get_dummies(data[cat].drop(['Loan_ID', 'Source', 'Loan_Status'], axis=1), drop_first=True)
    data2 = pd.concat([data, dummy], axis=1)
    Cols_To_Drop = ['Loan_ID', 'Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    data3 = data2.drop(Cols_To_Drop, axis=1).copy()
    # Convert Dependent variable into 0,1. If Loan_Status = N, then 1 else 0
    data3.Loan_Status = np.where(data3.Loan_Status == 'N', 1, 0)

    # SAMPLING #
    # Divide the data into Train and Test based on Source column and make sure you drop the source column
    train = data3.loc[data3.Source == "Train",].drop("Source", axis=1).copy()
    test = data3.loc[data3.Source == "Test",].drop("Source", axis=1).copy()

    # DIVIDE EACH DATA SET AS INDEPENDENT AND DEPENDENT VARAIBLES
    train_X = train.drop("Loan_Status", axis=1)
    train_y = train["Loan_Status"].copy()
    test_X = test.drop("Loan_Status", axis=1)
    test_y = test["Loan_Status"].copy()

    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(train_X.shape)
    st.write('Test set')
    st.info(test_X.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable')
    st.info(list(train_data.columns))
    st.write('Y variable')
    st.info(list(test.columns))

    # SVM MODEL
    M1 = SVC()
    Model1 = M1.fit(train_X, train_y)
    Pred1 = Model1.predict(test_X)

    st.subheader('2. Model Performance')
    conf1 = confusion_matrix(test_y, Pred1)
    accuracy = ((conf1[0][0] + conf1[1][1]) / test_y.shape[0]) * 100
    report1 = classification_report(test_y, Pred1, output_dict=True)
    report_df = pd.DataFrame(report1).transpose()

    st.write('Confusion matrix')
    st.write(conf1)
    st.write('Accuracy')
    st.write(accuracy)
    st.write('Classification Report')
    st.write(report_df)

    mycost_List = []
    mygamma_List = []
    mykernel_List = []
    accuracy_List = []

    # [1, 2]
    mycost = kwargs['mycost']

    # [0.01, 0.1]
    mygamma = kwargs['mygamma']

    # ['sigmoid', 'rbf']
    mykernel = kwargs['mykernel']

    Temp_Model = SVC(C=mycost, kernel=mykernel, gamma=mygamma)
    Temp_Model = Temp_Model.fit(train_X, train_y)
    Test_Pred = Temp_Model.predict(test_X)
    Confusion_Mat = confusion_matrix(test_y, Test_Pred)
    Temp_Accuracy = ((Confusion_Mat[0][0] + Confusion_Mat[1][1]) / test_y.shape[0]) * 100
    mycost_List.append(mycost)
    mygamma_List.append(mygamma)
    mykernel_List.append(mykernel)
    accuracy_List.append(Temp_Accuracy)

    model_validation_df = pd.DataFrame({'Cost': mycost_List,
                                        'Gamma': mygamma_List,
                                        'Kernel': mykernel_List,
                                        'Accuracy': accuracy_List})

    st.write('Model Validation')
    st.write(model_validation_df)

    # from sklearn.inspection import permutation_importance
    # import matplotlib.pyplot as plt
    # perm_importance = permutation_importance(Temp_Model, test_X, test_y)
    # feature_names = list(train_X.columns.values)
    # features = np.array(feature_names)
    #
    # sorted_idx = perm_importance.importances_mean.argsort()
    # plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    # plt.xlabel("Permutation Importance")
    #
    # plt.savefig('imp.png')
    # st.image('imp.png')
    #
    # feature_importances(Temp_Model.coef_, feature_names)

    # print(Temp_Model.get_booster().feature_names)



def app():
    # ---------------------------------#
    st.markdown("## Customer Segmentation Credit Model ")

    st.write("""
    #### In this implementation, the *SVM (Support Vector Machine)* is used in this app to build a classification model 
    A credit risk model to identify the customers segments for credit risk where company wants to  automate the loan 
    eligibility process (real time) based on customer's detail provided while filling online application, 
    and classify those are eligible for loan amount so that they can specifically target these customers.
    Try adjusting the hyperparameters!
    """)

    # ---------------------------------#
    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    #     st.sidebar.markdown("""
    # [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    # """)

    # Sidebar - Specify parameter settings
    # with st.sidebar.header('2. Set Parameters'):
    #     split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    #

    with st.sidebar.subheader('2. Learning Parameters'):
        parameter_cost = st.sidebar.select_slider('Cost (penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly)', options=[1, 2])
        parameter_gamma = st.sidebar.select_slider('Gamma (gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set)', options=[0.01, 0.1])
        parameter_kernel = st.sidebar.select_slider('Kernel (kernel parameters selects the type of hyperplane used to separate the data. )', options=['sigmoid', 'rbf'])

    # with st.sidebar.subheader('2.2. General Parameters'):
    #     parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    #     parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    #     parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    #     parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    #     parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

    # Main panel

    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        train_data = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(train_data)
        build_model(train_data)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Boston housing dataset
            print(os.getcwd())
            os.chdir('/home/diksha/Projects/creditRiskModelling/creditRiskModelling/datasets/customer-segment-risk/')
            # files = os.listdir(os.curdir)
            # print(files)

            train_data = pd.read_csv("train_data.csv")
            test_data = pd.read_csv("test_data.csv")

            st.markdown('Dataset Sample.')
            st.write(train_data.head(5))

            build_model(train_data=train_data,
                        test_data=test_data,
                        mycost=parameter_cost,
                        mygamma=parameter_gamma,
                        mykernel=parameter_kernel
                        )
