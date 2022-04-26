import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def isCategorical(col):
    unis = np.unique(col)
    if len(unis)<0.2*len(col):
        return True
    return False

def getColumnTypes(cols):
    Categorical=[]
    Numerical = []
    Object = []
    for i in range(len(cols)):
        if cols["type"][i]=='categorical':
            Categorical.append(cols['column_name'][i])
        elif cols["type"][i]=='numerical':
            Numerical.append(cols['column_name'][i])
        else:
            Object.append(cols['column_name'][i])
    return Categorical, Numerical, Object

def isNumerical(col):
    return is_numeric_dtype(col)

def genMetaData(df):
    col = df.columns
    ColumnType = []
    Categorical = []
    Object = []
    Numerical = []
    for i in range(len(col)):
        if isCategorical(df[col[i]]):
            ColumnType.append((col[i], "categorical"))
            Categorical.append(col[i])

        elif is_numeric_dtype(df[col[i]]):
            ColumnType.append((col[i], "numerical"))
            Numerical.append(col[i])

        else:
            ColumnType.append((col[i], "object"))
            Object.append(col[i])

    return ColumnType


def app():
    st.markdown("## Data Upload")

    # Upload the dataset and save as csv
    st.markdown("### Upload a csv file for analysis.")
    st.write("\n")

    # Code to read a single file
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    global data

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)

    ''' Load the data and save the columns with categories as a dataframe. 
        This section also allows changes in the numerical and categorical columns. '''
    if st.button("Load Data"):
        # Raw data
        st.dataframe(data)
        data.to_csv('data/main_data.csv', index=False)

        # Collect the categorical and numerical columns

        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(data.columns)) - set(numeric_cols))

        # Save the columns as a dataframe or dictionary
        columns = []

        # Iterate through the numerical and categorical columns and save in columns
        columns = genMetaData(data)

        # Save the columns as a dataframe with categories
        # Here column_name is the name of the field and the type is whether it's numerical or categorical
        columns_df = pd.DataFrame(columns, columns=['column_name', 'type'])
        columns_df.to_csv('data/metadata/column_type_desc.csv', index=False)

        # Display columns
        st.markdown("**Column Name**-**Type**")
        for i in range(columns_df.shape[0]):
            st.write(f"{i + 1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")

        st.markdown("""The above are the automated column types detected by the application in the data. 
                In case you wish to change the column types, head over to the **Column Change** section. """)