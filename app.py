from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("naive_model.pkl")

html = """

<!DOCTYPE html>
<html>
<head>

<title>Naive Bayes Predictor</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:Arial,Helvetica,sans-serif;
}

body{

background:linear-gradient(135deg,#0f172a,#1e3a8a,#3b82f6);
display:flex;
justify-content:center;
align-items:center;
height:100vh;

}

.container{

width:420px;
background:white;
padding:35px;
border-radius:18px;
box-shadow:0 15px 40px rgba(0,0,0,.3);

}

h1{

text-align:center;
color:#1e3a8a;
margin-bottom:10px;

}

p{

text-align:center;
color:gray;
margin-bottom:25px;

}

label{

font-weight:bold;
color:#333;

}

input{

width:100%;
padding:12px;
margin-top:8px;
margin-bottom:18px;
border:1px solid #ccc;
border-radius:8px;
font-size:16px;

}

input:focus{

outline:none;
border-color:#2563eb;

}

button{

width:100%;
padding:14px;
background:#2563eb;
border:none;
color:white;
font-size:18px;
border-radius:8px;
cursor:pointer;
transition:.3s;

}

button:hover{

background:#1d4ed8;

}

.result{

margin-top:25px;
padding:15px;
background:#eff6ff;
border-left:6px solid #2563eb;
font-size:22px;
text-align:center;
border-radius:8px;
font-weight:bold;
color:#1e3a8a;

}

.footer{

margin-top:20px;
text-align:center;
font-size:13px;
color:gray;

}

</style>

</head>

<body>

<div class="container">

<h1>Naive Bayes Predictor</h1>

<p>Machine Learning Prediction App</p>

<form method="POST">

<label>Feature 1</label>
<input type="number" step="any" name="feature1" required>

<label>Feature 2</label>
<input type="number" step="any" name="feature2" required>

<label>Feature 3</label>
<input type="number" step="any" name="feature3" required>

<button type="submit">

Predict

</button>

</form>

{% if prediction != None %}

<div class="result">

Prediction : {{ prediction }}

</div>

{% endif %}

<div class="footer">

Made with Flask ❤️

</div>

</div>

</body>

</html>

"""

@app.route("/", methods=["GET","POST"])

def home():

    prediction = None

    if request.method == "POST":

        f1 = float(request.form["feature1"])
        f2 = float(request.form["feature2"])
        f3 = float(request.form["feature3"])

        data = np.array([[f1,f2,f3]])

        prediction = model.predict(data)[0]

    return render_template_string(html,prediction=prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
