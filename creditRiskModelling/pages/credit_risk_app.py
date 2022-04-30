"""
This file is the framework for generating multiple Streamlit applications
through an object oriented framework.
"""

# Import necessary libraries
import streamlit as st
import data_upload_page
import streamlit_sample_web_app
import streamlit_customer_svc_cm


# Define the multipage class to manage the multiple apps in our program
class MultiPage:
    """Framework for combining multiple streamlit applications."""

    def __init__(self) -> None:
        """Constructor class to generate a list which will store all our applications as an instance variable."""
        self.pages = []

    def add_page(self, title, func) -> None:
        """Class Method to Add pages to the project
        Args: title ([str]): The title of page which we are adding to the list of apps
        func: Python function to render this page in Streamlit
        """

        self.pages.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Dropdown to select the page to run
        page = st.sidebar.selectbox(
            'Credit Risk Models',
            self.pages,
            format_func=lambda page: page['title']
        )
        # st.set_page_config(page_title='Credit Risk Modelling', layout='wide')

        # run the app function
        page['function']()


# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title("Credit Risk Modelling")

# Add all your applications (pages) here
# app.add_page("Upload Data", data_upload_page.app)
# app.add_page("Random Forest Regressor", streamlit_sample_web_app.app)
app.add_page("Customer Segmentation Credit Model", streamlit_customer_svc_cm.app)


# Run the Web app
app.run()
