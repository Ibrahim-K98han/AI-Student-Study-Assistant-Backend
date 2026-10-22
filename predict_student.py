import pandas as pd
import joblib

# Saved model load
model = joblib.load("student_model.pkl")

print("Model loaded successfully!")

# New student information
new_student = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 75,
    "previous_mark": 65,
    "assignment_score": 80
}])

# Prediction
predicted_mark = model.predict(new_student)

print("Predicted Final Mark:", round(predicted_mark[0], 2))