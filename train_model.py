
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: CSV file theke data load
df = pd.read_csv("student_dataset.csv")

print("Dataset Preview:")
print(df.head())

# Step 2: Features ebong Target alada kori
X = df[
    [
        "study_hours",
        "attendance",
        "previous_mark",
        "assignment_score"
    ]
]

y = df["final_mark"]

# Step 3: Training ebong Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 4: Model create
model = LinearRegression()

# Step 5: Model train
model.fit(X_train, y_train)

joblib.dump(model, "student_model.pkl")

print("Model saved successfully!")

# Step 6: Prediction
predictions = model.predict(X_test)

# Step 7: Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nTraining completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# Step 8: Learned weights
print("\nModel Weights:", model.coef_)
print("Model Bias:", model.intercept_)