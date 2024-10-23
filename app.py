from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('loan_status_model.pkl', 'rb'))

def preprocess_features(features):
    fill_values = {
    'Gender': features['Gender'].mode()[0],
    'Married': features['Married'].mode()[0],
    'Dependents': features['Dependents'].mode()[0],
    'Self_Employed': features['Self_Employed'].mode()[0],
    'LoanAmount': features['LoanAmount'].mean(),
    'Loan_Amount_Term': features['Loan_Amount_Term'].mean(),
    'Credit_History': features['Credit_History'].mode()[0]
    }

# Fill missing values using the dictionary
    features.fillna(fill_values, inplace=True)

    # Replace categorical values with numerical labels
    features.replace({
        'Married': {'No': 0, 'Yes': 1},
        'Gender': {'Male': 1, 'Female': 0},
        'Self_Employed': {'No': 0, 'Yes': 1},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Education': {'Graduate': 1, 'Not Graduate': 0}
    }, inplace=True)

    features.replace({'Dependents': {'3+': 4}}, inplace=True)

    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = request.form['Dependents']
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = int(request.form['ApplicantIncome'])
        coapplicant_income = int(request.form['CoapplicantIncome'])
        loan_amount = int(request.form['LoanAmount'])
        loan_amount_term = int(request.form['Loan_Amount_Term'])
        credit_history = int(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Create a dictionary of input data
        user_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }

        # Convert dictionary to DataFrame
        features = pd.DataFrame(user_data, index=[0])

        # Preprocess the features
        processed_data = preprocess_features(features)

        # Make prediction using the model
        prediction = model.predict(processed_data)

        # Output result based on prediction
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', prediction_text=f'Your loan is likely to be {result}.')

if __name__ == "__main__":
    app.run(debug=True)
