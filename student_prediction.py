import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [50, 55, 60, 65, 70, 80, 85, 95],
    "previous_mark": [40, 45, 50, 55, 60, 70, 75, 85],
    "final_mark": [42, 47, 53, 58, 64, 73, 79, 90]
}

df = pd.DataFrame(data)

X = df[
    ["study_hours", "attendance", "previous_mark"]
]

y = df["final_mark"]

print("Features:")
print(X)

print("Target:")
print(y)

print(df)




# Dataset ke training ebong testing data-te vag kori
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# Machine Learning model toiri
model = LinearRegression()

# Model-ke training dewa
model.fit(X_train, y_train)


print("Model Weights:", model.coef_)
print("Model Bias:", model.intercept_)

# Test data diye prediction
predictions = model.predict(X_test)

print("Actual Marks:")
print(y_test.values)

print("Predicted Marks:")
print(predictions)

# Prediction koto ta bhul hocche ta measure kori
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)



new_student = pd.DataFrame([{
    "study_hours": 5,
    "attendance": 75,
    "previous_mark": 65
}])

predicted_mark = model.predict(new_student)

print("New Student Predicted Mark:", round(predicted_mark[0], 2))

