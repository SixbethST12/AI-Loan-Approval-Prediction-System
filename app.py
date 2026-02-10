from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    applicant_name = None
    if request.method == "POST":
        applicant_name = request.form["name"]
        income = float(request.form["income"])
        loan_amount = float(request.form["loan_amount"])
        repayment_history = request.form["repayment_history"]
        existing_loans = int(request.form["existing_loans"])
        age = int(request.form["age"])

        # Encode repayment history
        repayment_dict = {'Good': 2, 'Average': 1, 'Poor': 0}
        repayment_encoded = repayment_dict[repayment_history]

        input_data = pd.DataFrame([[income, loan_amount, repayment_encoded, existing_loans, age]],
                                  columns=['Income','Loan_Amount','Repayment_History','Existing_Loans','Age'])
        pred = model.predict(input_data)[0]
        result = "Approved" if pred == 1 else "Declined"

    return render_template("index.html", result=result, name=applicant_name)

if __name__ == "__main__":
    app.run(debug=True)
