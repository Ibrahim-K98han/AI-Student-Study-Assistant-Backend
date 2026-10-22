
import pandas as pd
import random

random.seed(42)

students = []

for i in range(100):
    study_hours = random.randint(1, 8)
    attendance = random.randint(40, 100)
    previous_mark = random.randint(30, 95)
    assignment_score = random.randint(30, 100)

    # Sample data-r jonno ekta illustrative formula
    final_mark = (
        study_hours * 3
        + attendance * 0.15
        + previous_mark * 0.35
        + assignment_score * 0.2
    )

    # Mark-ke 0 theke 100-er moddhe rakhi
    final_mark = max(0, min(100, round(final_mark)))

    students.append({
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_mark": previous_mark,
        "assignment_score": assignment_score,
        "final_mark": final_mark
    })

df = pd.DataFrame(students)

df.to_csv("student_dataset.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("Total students:", len(df))