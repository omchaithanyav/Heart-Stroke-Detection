from flask import Flask, render_template, request
import numpy as np
import tensorflow.keras

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

        gender= float(request.form['gender'])
        age= int(request.form['age'])
        heart_disease= float(request.form['heart_disease'])
        ever_married= int(request.form['ever_married'])
        work_type= float(request.form['work_type'])
        Residence_type= int(request.form['Residence_type'])
        avg_glucose_level= int(request.form['avg_glucose_level'])
        bmi= int(request.form['bmi'])
        smoking_status= int(request.form['smoking_status'])
        hypertension = int(request.form['hypertension'])

        inputs = np.array( [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level,bmi,smoking_status]).reshape(1, -1)

        # prediction
        model = tensorflow.keras.models.load_model('models/best_model.h5')
        pred = model.predict(inputs)
        percent = "{:.2f}".format(pred[0][0]*100)
        if pred>0.3:
                return render_template('result.html', results=f"Yes - {percent} percent chance for Heart Stroke")
        else:
                return render_template('result.html', results=f"No")



if __name__ == '__main__':
    app.run(debug=True, port=5544)