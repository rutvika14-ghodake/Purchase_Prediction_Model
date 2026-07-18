from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("naive_model(2).pkl", "rb") as file:
    model = pickle.load(file)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Customer Purchase Prediction</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Segoe UI',sans-serif;
}

body{
height:100vh;
display:flex;
justify-content:center;
align-items:center;
background:linear-gradient(135deg,#3a7bd5,#00d2ff);
}

.container{

width:420px;
background:white;
padding:35px;
border-radius:18px;
box-shadow:0px 12px 35px rgba(0,0,0,.25);

}

h1{

text-align:center;
margin-bottom:25px;
color:#2c3e50;

}

label{

font-weight:600;
display:block;
margin-top:15px;
color:#34495e;

}

input,select{

width:100%;
padding:12px;
margin-top:8px;
border:1px solid #ccc;
border-radius:8px;
font-size:15px;

}

input:focus,
select:focus{

outline:none;
border:1px solid #3498db;

}

button{

width:100%;
margin-top:25px;
padding:14px;
font-size:16px;
font-weight:bold;
border:none;
border-radius:10px;
cursor:pointer;
background:#3498db;
color:white;
transition:.3s;

}

button:hover{

background:#2980b9;

}

.result{

margin-top:25px;
padding:15px;
text-align:center;
border-radius:10px;
font-size:20px;
font-weight:bold;

}

.buy{

background:#d4edda;
color:#155724;

}

.notbuy{

background:#f8d7da;
color:#721c24;

}

.footer{

text-align:center;
margin-top:20px;
font-size:12px;
color:gray;

}

</style>

</head>

<body>

<div class="container">

<h1>Customer Purchase Predictor</h1>

<form method="POST">

<label>Gender</label>

<select name="gender">

<option value="1">Male</option>
<option value="0">Female</option>

</select>

<label>Age</label>

<input
type="number"
name="age"
required
placeholder="Enter Age">

<label>Estimated Salary</label>

<input
type="number"
name="salary"
required
placeholder="Enter Salary">

<button type="submit">
Predict
</button>

</form>

{% if prediction %}

<div class="result {{class_name}}">
{{prediction}}
</div>

{% endif %}

<div class="footer">

Built with Flask • Gaussian Naive Bayes

</div>

</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    class_name = ""

    if request.method == "POST":

        gender = int(request.form["gender"])
        age = int(request.form["age"])
        salary = int(request.form["salary"])

        data = np.array([[gender, age, salary]])

        result = model.predict(data)[0]

        if result == 1:
            prediction = "Customer is likely to PURCHASE"
            class_name = "buy"
        else:
            prediction = "Customer is NOT likely to PURCHASE"
            class_name = "notbuy"

    return render_template_string(
        HTML,
        prediction=prediction,
        class_name=class_name
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
