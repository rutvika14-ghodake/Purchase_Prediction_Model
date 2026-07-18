from flask import Flask, request, render_template_string
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("naive_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Naive Bayes Prediction</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Segoe UI',sans-serif;
}

body{

background:linear-gradient(135deg,#4A00E0,#8E2DE2);
height:100vh;
display:flex;
justify-content:center;
align-items:center;

}

.container{

width:420px;
padding:35px;
background:rgba(255,255,255,0.12);
backdrop-filter:blur(18px);
border-radius:18px;
box-shadow:0 15px 35px rgba(0,0,0,.3);
text-align:center;
color:white;

}

h1{

margin-bottom:8px;
font-size:30px;

}

p{

margin-bottom:25px;
color:#e5e5e5;

}

input{

width:100%;
padding:12px;
margin:10px 0;
border:none;
outline:none;
border-radius:8px;
font-size:16px;

}

button{

width:100%;
padding:13px;
margin-top:15px;
border:none;
border-radius:8px;
background:#00C9A7;
color:white;
font-size:18px;
cursor:pointer;
transition:.3s;

}

button:hover{

background:#00A98B;
transform:scale(1.03);

}

.result{

margin-top:20px;
padding:15px;
border-radius:8px;
background:rgba(255,255,255,.15);
font-size:22px;
font-weight:bold;

}

.footer{

margin-top:20px;
font-size:13px;
color:#ddd;

}

</style>

</head>

<body>

<div class="container">

<h1>Prediction App</h1>

<p>Naive Bayes Machine Learning Model</p>

<form method="POST">

<input type="number" step="any" name="f1" placeholder="Feature 1" required>

<input type="number" step="any" name="f2" placeholder="Feature 2" required>

<input type="number" step="any" name="f3" placeholder="Feature 3" required>

<input type="number" step="any" name="f4" placeholder="Feature 4" required>

<button type="submit">

Predict

</button>

</form>

{% if prediction %}

<div class="result">

Prediction : {{prediction}}

</div>

{% endif %}

<div class="footer">

Developed using Flask ❤️

</div>

</div>

</body>

</html>

"""

@app.route("/", methods=["GET","POST"])

def home():

    prediction=None

    if request.method=="POST":

        features=[

            float(request.form["f1"]),
            float(request.form["f2"]),
            float(request.form["f3"]),
            float(request.form["f4"])

        ]

        prediction=model.predict([features])[0]

    return render_template_string(HTML,prediction=prediction)

if __name__=="__main__":

    app.run(debug=True)
