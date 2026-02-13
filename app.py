# app.py
from flask import Flask, render_template, request, redirect, url_for, make_response, send_file,session
import os
import pandas as pd
import joblib
import sqlite3
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# --- CONFIG ---
app = Flask(__name__)  # no app.secret_key set (per your request)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'Models'
STATIC_FOLDER = 'static'
DATASET_FOLDER = 'Dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# FEATURES / TARGET (your requested fields)
FEATURE_COLUMNS = [
'age','gender','bmi','blood_pressure','cholesterol_level','glucose_level','physical_activity','smoking_status'
,'alcohol_intake','family_history','biomarker_A','biomarker_B','biomarker_C','biomarker_D'
]
TARGET_COLUMN = 'target'


NUMERIC_COLUMNS = ['age','gender','bmi','blood_pressure','cholesterol_level','glucose_level','physical_activity','smoking_status'
,'alcohol_intake','family_history','biomarker_A','biomarker_B','biomarker_C','biomarker_D']

# Globals to store accuracy values (used for graph)
xgb_acc = rf_acc = dec_acc = None

# -------------------------
# Utility: Database helpers
# -------------------------
DB_PATH = 'database.db'

app.secret_key = '123'

# Ensure table exists
def init_db():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS user (
            name TEXT,
            email TEXT,
            mobile TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html')
# ---------- Admin ----------
@app.route('/adminlogin', methods=['GET','POST'])
def adminlogin():
    return render_template('AdminApp/AdminLogin.html')

@app.route('/AdminAction', methods=['POST'])
def AdminAction():
    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']

        if username=='Admin' and password=='Admin':
            return render_template("AdminApp/AdminHome.html")
        else:
            context={'msg':'Login Failed..!!'}
            return render_template("AdminApp/AdminLogin.html",**context)


@app.route('/AdminHome')
def AdminHome():
    return render_template("AdminApp/AdminHome.html")

@app.route('/Upload')
def Upload():
    return render_template("AdminApp/Upload.html")

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

global dataset,filepath
@app.route('/UploadAction', methods=['POST'])
def UploadAction():
    global dataset,filepath
    if 'dataset' not in request.files:
        return "No file part"
    file = request.files['dataset']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    dataset = pd.read_csv(filepath)
    columns = dataset.columns.tolist()
    rows = dataset.head().values.tolist()
    return render_template('AdminApp/ViewDataset.html', columns=columns, rows=rows)

global dataset, X_train, X_test, y_train, y_test

@app.route('/preprocess')
def preprocess():
    global dataset, X_train, X_test, y_train, y_test

    # Load dataset
    dataset = pd.read_csv('Dataset/chronic_disease_dataset.csv')
    dataset.dropna(inplace=True)

    # Feature and target columns
    feature_columns = ['age','gender','bmi','blood_pressure','cholesterol_level','glucose_level','physical_activity','smoking_status'
,'alcohol_intake','family_history','biomarker_A','biomarker_B','biomarker_C','biomarker_D'
]





    # Retain features + target
    selected_columns = feature_columns + ['target']
    dataset = dataset[selected_columns]

    # Prepare X and y
    X = dataset[feature_columns]
    y = dataset['target']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return render_template('AdminApp/SplitStatus.html',
                           total=len(X), train=len(X_train), test=len(X_test))

global svm_acc,smodel
@app.route('/xgboost')
def xgboost():
    global xgb_acc,smodel,X_train, X_test, y_train, y_test

    xgmodel=xgb.XGBClassifier()
    xgmodel.fit(X_train,y_train)
    joblib.dump(xgmodel, "Models/XGModel.joblib")
    pred = xgmodel.predict(X_test)
    xgacc=accuracy_score(y_test, pred)
    pred = xgacc*100
    xgb_acc=float("{:.2f}".format(pred))
    return render_template('AdminApp/AlgorithmStatus.html', msg="Xgboost Model Generated Successfully..!!", Accuracy=str(xgb_acc))



@app.route('/RandomForest')
def RandomForest():
    global rf_acc,rmodel,X_train, X_test, y_train, y_test

    rmodel=RandomForestClassifier()
    rmodel.fit(X_train,y_train)
    joblib.dump(rmodel, "Models/RFModel.joblib")
    pred = rmodel.predict(X_test)
    rfacc=accuracy_score(y_test, pred)
    pred = rfacc*100
    rf_acc=float("{:.2f}".format(pred))
    return render_template('AdminApp/AlgorithmStatus.html', msg="Random Forest Model Generated Successfully..!!", Accuracy=str(rf_acc))


@app.route('/Decision')
def Decision():
    global dec_acc,decmodel,X_train, X_test, y_train, y_test

    decmodel=DecisionTreeClassifier()
    decmodel.fit(X_train,y_train)
    joblib.dump(decmodel, "Models/DecModel.joblib")
    pred = decmodel.predict(X_test)
    decacc=accuracy_score(y_test, pred)
    pred = decacc*100
    dec_acc=float("{:.2f}".format(pred))
    return render_template('AdminApp/AlgorithmStatus.html', msg="Decision Tree Model Generated Successfully..!!", Accuracy=str(dec_acc))


@app.route('/comparison')
def comparison():
    models = ['XGBoost', 'Random Forest', 'Decision Tree']
    accuracies = [xgb_acc, rf_acc, dec_acc]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])

    # Add text labels on top of each bar
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{acc}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

    plt.title('Model Accuracy Comparison', fontsize=16)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)  # Adjust y-axis to match the accuracy range
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('static/model_accuracy.png')  # Make sure folder name is lowercase 'static'
    plt.close()

    return render_template('AdminApp/Grpah.html')
@app.route('/userlogin')
def userlogin():
    return render_template('UserApp/Login.html')

@app.route('/register')
def register():
    return render_template('UserApp/Register.html')

@app.route('/RegAction', methods=['POST'])
def RegAction():
    name = request.form['name']
    email = request.form['email']
    mobile = request.form['mobile']
    username = request.form['username']
    password = request.form['password']

    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username=? OR email=?", (username, email))
    data = cur.fetchone()

    if data is None:
        try:
            cur.execute("INSERT INTO user (name, email, mobile, username, password) VALUES (?, ?, ?, ?, ?)",
                        (name, email, mobile, username, password))
            con.commit()
            msg = "Successfully Registered!"
        except sqlite3.IntegrityError:
            msg = "Username already exists!"
    else:
        msg = "Username or email already exists!"

    con.close()
    return render_template('UserApp/Register.html', msg=msg)

@app.route('/UserAction', methods=['GET','POST'])
def UserAction():
    username = request.form['username']
    password = request.form['password']

    con = sqlite3.connect('database.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM user WHERE username=? AND password=?", (username, password))
    data = cur.fetchone()
    con.close()

    if data is None:
        return render_template('UserApp/Login.html', msg="Login Failed!")
    else:
        session['username'] = data[3]
        return render_template('UserApp/Home.html', username=session['username'])

@app.route('/Detect')
def Detect():
    return render_template('UserApp/Detect.html')

@app.route('/UserHome')
def UserHome():
    return render_template('UserApp/Home.html')


@app.route('/DetectAction', methods=['POST', 'GET'])
def DetectAction():
    if request.method == 'GET':
        return "Please submit the form with POST method."

    # --------------- 1. Collect Inputs ---------------
    age = request.form['age']
    gender = request.form['gender']
    bmi = request.form['bmi']
    blood_pressure = request.form['blood_pressure']
    cholesterol_level = request.form['cholesterol_level']
    glucose_level = request.form['glucose_level']
    physical_activity = request.form['physical_activity']
    smoking_status = request.form['smoking_status']
    alcohol_intake = request.form['alcohol_intake']
    family_history = request.form['family_history']
    biomarker_A = request.form['biomarker_A']
    biomarker_B = request.form['biomarker_B']
    biomarker_C = request.form['biomarker_C']
    biomarker_D = request.form['biomarker_D']

    # --------------- 2. Create DataFrame ---------------
    test = pd.DataFrame([{
        'age': float(age),
        'gender': gender,
        'bmi': float(bmi),
        'blood_pressure': float(blood_pressure),
        'cholesterol_level': float(cholesterol_level),
        'glucose_level': float(glucose_level),
        'physical_activity': physical_activity,
        'smoking_status': smoking_status,
        'alcohol_intake': alcohol_intake,
        'family_history': family_history,
        'biomarker_A': float(biomarker_A),
        'biomarker_B': float(biomarker_B),
        'biomarker_C': float(biomarker_C),
        'biomarker_D': float(biomarker_D)
    }])

    # --------------- 3. Load Model ---------------
    model_path = "Models/DecModel.joblib"
    rf_model = joblib.load(model_path)

    # --------------- 4. Encode Categorical Columns ---------------
    categorical_cols = ['gender', 'physical_activity', 'smoking_status', 'alcohol_intake', 'family_history']
    for col in categorical_cols:
        le = LabelEncoder()
        test[col] = le.fit_transform(test[col])

    # --------------- 5. Predict ---------------
    prediction_num = rf_model.predict(test)[0]
    pred_proba = rf_model.predict_proba(test)[0]

    # --------------- 6. Map Class to Disease Stage ---------------
    risk_map = {
        0: "Healthy",
        1: "At-Risk",
        2: "Early-Stage",
        3: "Chronic",
        4: "Critical"
    }
    prediction_label = risk_map.get(prediction_num, "UNKNOWN")
    prediction_probability = pred_proba[prediction_num]

    print(f"Predicted Output: {prediction_label}, Probability: {prediction_probability:.4f}")

    # --------------- 7. Return to HTML Page ---------------
    return render_template(
        'UserApp/Result.html',
        prediction=prediction_label,
        probability=prediction_probability
    )

@app.route('/about')
def about():
    return render_template('UserApp/about.html')
if __name__ == '__main__':
    app.run(debug=True)
