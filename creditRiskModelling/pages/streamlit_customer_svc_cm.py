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
import seaborn as sns
import os
from sklearn import metrics

# ---------------------------------#
# Model building


def build_model(train_data, test_data, **kwargs):
    # CREATING A NEW COLUMN SOURCE UNDER BOTH TRAIN AND TEST DATA
    train_data["Source"] = "Train"
    test_data["Source"] = "Test"

    # EDA
    st.subheader('2. Data Exploration')

    st.write("Loan Status wise data distribution")
    plt.figure(figsize=(10, 5))
    item = train_data['Loan_Status'].value_counts()[:50]
    sns.barplot(item.values, item.index)
    sns.despine(left=True, right=True)
    st.pyplot(plt)

    st.write("Loan Status wise data distribution against each variable")
    plt.figure(figsize=(10, 20))
    sns.pairplot(train_data, hue="Loan_Status", palette='bright')
    st.pyplot(plt)

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

    st.subheader('3. Model Performance')
    conf1 = confusion_matrix(test_y, Pred1)
    accuracy = ((conf1[0][0] + conf1[1][1]) / test_y.shape[0]) * 100
    report1 = classification_report(test_y, Pred1, output_dict=True)
    report_df = pd.DataFrame(report1).transpose()

    st.write('Confusion matrix')
    st.write(conf1)
    # sns.heatmap(conf1, annot=True, cmap="gray_r", linewidth=2, linecolor='w', fmt='.0f')
    # plt.xlabel('Predicted Value')
    # plt.ylabel('True Value')
    # st.pyplot(plt)

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

    st.write('**Model Validation**')
    st.write(model_validation_df)

    st.write('**Classification accuracy**')
    TP = Confusion_Mat[0, 0]
    TN = Confusion_Mat[1, 1]
    FP = Confusion_Mat[0, 1]
    FN = Confusion_Mat[1, 0]
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    st.write(classification_accuracy)

    st.write('**Classification error**')
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    st.write(classification_error)

    p, r, f, _ = metrics.precision_recall_fscore_support(test_y, Test_Pred, average='weighted', warn_for=tuple())
    st.write('**Precision**')
    st.write('Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes.')
    st.write(p)

    st.write('**Recall**')
    st.write('Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes.')
    st.write(p)

    st.write('**F1-Score**')
    st.write('The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.')
    st.write(f)

    # from sklearn.feature_selection import SelectPercentile, f_classif
    #
    # svm_weights_selected = (Temp_Model.coef_ ** 2).sum(axis=0)
    # svm_weights_selected /= svm_weights_selected.max()
    # selector = SelectPercentile(f_classif, percentile=10)
    # X_indices = np.arange(train_X.shape[-1])
    # plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
    #         width=.2, label='SVM weights after selection', color='b')
    #
    # plt.title("Comparing feature selection")
    # plt.xlabel('Feature number')
    # plt.yticks(())
    # plt.axis('tight')
    # plt.legend(loc='upper right')
    # # plt.show()
    # plt.savefig('imp.png')
    # st.image('imp.png')

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

    with st.sidebar.subheader('2. Learning Parameters'):
        st.sidebar.write('REGULARIZATION PARAMETER :')
        parameter_cost = st.sidebar.select_slider('tells the SVM optimization how much you want to avoid miss classifying . It controls the trade off between smooth decision boundary and classifying the training points correctly', options=[0.1, 1, 1.5, 2])
        st.sidebar.write('GAMMA :')
        parameter_gamma = st.sidebar.select_slider('defines how far the influence of a single training example reaches. The higher the gamma value it tries to exactly fit the training data set', options=[0.01, 0.1])
        st.sidebar.write('KERNEL :')
        parameter_kernel = st.sidebar.select_slider('parameter selects the type of hyperplane used to separate the data.', options=['sigmoid', 'rbf'])

    with st.sidebar.subheader('3. Test with data'):
        uploaded_test_file = st.sidebar.file_uploader("Upload your test CSV file", type=["csv"])
        st.sidebar.write('Test data values:')
        st.sidebar.write("Loan_ID, Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, "
                 "CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status")


    # Main panel
    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        train_data = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(train_data)

        if uploaded_test_file is not None:
            test_data = pd.read_csv("test_data.csv")
        else:
            os.chdir('/home/diksha/Projects/creditRiskModelling/creditRiskModelling/datasets/customer-segment-risk/')
            test_data = pd.read_csv("test_data.csv")

        build_model(train_data=train_data,
                    test_data=test_data,
                    mycost=parameter_cost,
                    mygamma=parameter_gamma,
                    mykernel=parameter_kernel
                    )
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            os.chdir('/home/diksha/Projects/creditRiskModelling/creditRiskModelling/datasets/customer-segment-risk/')

            train_data = pd.read_csv("train_data.csv")
            st.markdown('Dataset Sample.')
            st.write(train_data.head(5))

            if uploaded_test_file is not None:
                test_data = pd.read_csv("test_data.csv")
            else:
                os.chdir('/home/diksha/Projects/creditRiskModelling/creditRiskModelling/datasets/customer-segment-risk/')
                test_data = pd.read_csv("test_data.csv")

            build_model(train_data=train_data,
                        test_data=test_data,
                        mycost=parameter_cost,
                        mygamma=parameter_gamma,
                        mykernel=parameter_kernel
                        )
