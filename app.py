from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained pipeline

model = pickle.load(open('Linear_Regression_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        company = request.form['company']
        name = request.form['name']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        data = pd.DataFrame(
            [[name,
            company,
            year,
            kms_driven,
            fuel_type]],
        columns=[
            'name',
            'company',
            'year',
            'kms_driven',
            'fuel_type'
        ]
    )

        prediction = model.predict(data)[0]
        return render_template(
        'index.html',
        prediction_text=f"₹ {prediction:,.0f}"
    )
    except Exception as e:

        return render_template(

        'index.html',
         prediction_text="Vehicle configuration not available in the training dataset"
    )

if __name__ == '__main__':
    app.run(debug=True)
