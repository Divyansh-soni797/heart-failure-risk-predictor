from flask import Flask, render_template, request, redirect, url_for, session
#importing librareries-->
import pickle
import numpy as np
import os

# Initialize app
app = Flask(__name__)
app.secret_key = 'anyrandomsecret'  # required for session management

#Loading ML model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("home"))


@app.route("/", methods=["GET"])
def home():
    result = session.pop("prediction_text", None)  # show once
    inputs = session.pop("form_data", {})          # show once
    return render_template("index.html", prediction_text=result, inputs=inputs)


#Prediction 
@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
    data = [float(x) for x in form_values.values()]
    final_input = [np.array(data)]

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    # result
    if prediction == 1:
        result = f"⚠️ High risk of heart failure ({round(probability*100, 2)}% confidence)"
    else:
        result = f"✅ Low risk — Patient likely to survive ({round((1 - probability)*100, 2)}% confidence)"

    #Storeing result 
    session["prediction_text"] = result
    session["form_data"] = form_values

    return redirect(url_for("home"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

