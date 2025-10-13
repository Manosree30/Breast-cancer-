
## Breast Cancer Classification 
This interactive web application uses machine learning to classify breast cancer tumors as malignant or benign based on various tumor features. The app is built with Streamlit for an easy-to-use interface and leverages a trained classification model to make predictions from user input.

## Project Overview
Breast cancer is a common and potentially deadly disease. Early and accurate diagnosis is critical for effective treatment and prognosis. This project applies machine learning classification techniques on the well-known Breast Cancer Wisconsin (Diagnostic) dataset to build a predictive model. Users can input specific tumor features, and the app will output the predicted tumor type along with confidence probabilities.

## Features Used
The model uses key tumor attributes such as:

Radius Mean

Texture Mean

Smoothness Mean

Compactness Mean

Concavity Mean

Symmetry Mean

And others from the diagnostic dataset features

These measurable features contribute to the classification accuracy of the model.

## How to Use
Access the app online at: [streamlit app.py](https://manosree30-breast-cancer--app-i4o2on.streamlit.app/)

Input numeric values for the tumor features as prompted.

Click the "Predict" button.

View the prediction result indicating whether the tumor is malignant or benign, including the probability score.

## Technologies Used
Python

Streamlit for the web app interface

scikit-learn for machine learning model building and evaluation

pandas and numpy for data processing

Breast Cancer Wisconsin (Diagnostic) dataset from sklearn

## Installation and Local Setup (Optional)
To run this app locally:

bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
streamlit run app.py
Ensure all dependencies in requirements.txt (e.g., streamlit, scikit-learn, pandas, numpy) are installed.