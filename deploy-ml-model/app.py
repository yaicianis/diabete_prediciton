from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = pickle.load(open('diabetes.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Définir les intervalles valides
        intervals = {
            'Pregnancies': (0, 17),
            'Glucose': (0, 199),
            'BloodPressure': (0, 122),
            'SkinThickness': (0, 99),
            'Insulin': (0, 846),
            'BMI': (0, 67.1),
            'DiabetesPedigreeFunction': (0.078, 2.42),
            'Age': (21, 81)
        }

        # Validation des valeurs d'entrée
        if not (intervals['Pregnancies'][0] <= Pregnancies <= intervals['Pregnancies'][1]):
            raise ValueError(f"Le nombre de grossesses doit être entre {intervals['Pregnancies'][0]} et {intervals['Pregnancies'][1]}.")

        if not (intervals['Glucose'][0] <= Glucose <= intervals['Glucose'][1]):
            raise ValueError(f"Le taux de glucose doit être entre {intervals['Glucose'][0]} et {intervals['Glucose'][1]}.")

        if not (intervals['BloodPressure'][0] <= BloodPressure <= intervals['BloodPressure'][1]):
            raise ValueError(f"La pression artérielle doit être entre {intervals['BloodPressure'][0]} et {intervals['BloodPressure'][1]}.")

        if not (intervals['SkinThickness'][0] <= SkinThickness <= intervals['SkinThickness'][1]):
            raise ValueError(f"L'épaisseur de la peau doit être entre {intervals['SkinThickness'][0]} et {intervals['SkinThickness'][1]}.")

        if not (intervals['Insulin'][0] <= Insulin <= intervals['Insulin'][1]):
            raise ValueError(f"Le taux d'insuline doit être entre {intervals['Insulin'][0]} et {intervals['Insulin'][1]}.")

        if not (intervals['BMI'][0] <= BMI <= intervals['BMI'][1]):
            raise ValueError(f"L'IMC doit être entre {intervals['BMI'][0]} et {intervals['BMI'][1]}.")

        if not (intervals['DiabetesPedigreeFunction'][0] <= DiabetesPedigreeFunction <= intervals['DiabetesPedigreeFunction'][1]):
            raise ValueError(f"Le facteur héréditaire du diabète doit être entre {intervals['DiabetesPedigreeFunction'][0]} et {intervals['DiabetesPedigreeFunction'][1]}.")

        if not (intervals['Age'][0] <= Age <= intervals['Age'][1]):
            raise ValueError(f"L'âge doit être entre {intervals['Age'][0]} et {intervals['Age'][1]}.")

        # Créer un tableau avec les données d'entrée
        input_features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Faire une prédiction
        prediction = model.predict(input_features)

        # Interpréter la prédiction
        result = "Diabetique" if prediction[0] == 1 else "Non Diabetique"

        return render_template('index.html', prediction_text=f'Le patient est {result}')

    except ValueError as e:
        return render_template('index.html', prediction_text=str(e))

    except Exception as e:
        # Affiche l'erreur pour le débogage
        print(f'Error: {str(e)}')
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)

