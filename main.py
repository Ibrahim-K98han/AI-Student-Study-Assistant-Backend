import logging
import os
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Check your backend .env file.")

client = genai.Client(api_key=api_key)

app = FastAPI(
    title="Student AI Study Assistant API",
    description="ML mark prediction and multilingual AI study planning",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this for production web clients.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

try:
    model = joblib.load("student_model.pkl")
except Exception as exc:
    raise RuntimeError(
        "Could not load student_model.pkl. Run the server from the folder "
        "containing that file."
    ) from exc


class StudentData(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_mark: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)


SupportedLanguage = Literal["English", "Bangla", "Arabic", "Hindi"]


class StudyPlanRequest(StudentData):
    language: SupportedLanguage = "English"


class StudyPlan(BaseModel):
    summary: str
    study_tips: list[str] = Field(min_length=3, max_length=3)
    weekly_goal: str


def predict_student_mark(data: StudentData) -> float:
    student = pd.DataFrame([{
        "study_hours": data.study_hours,
        "attendance": data.attendance,
        "previous_mark": data.previous_mark,
        "assignment_score": data.assignment_score,
    }])
    prediction = model.predict(student)
    return round(float(prediction[0]), 2)


@app.get("/")
def home():
    return {"message": "Student AI API is running!"}


@app.post("/predict")
def predict(data: StudentData):
    try:
        return {"predicted_mark": predict_student_mark(data)}
    except Exception:
        logger.exception("Mark prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Could not predict the student's mark.",
        )


@app.get("/test-gemini")
def test_gemini():
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Explain AI in one simple sentence.",
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return {"message": response.text}
    except Exception:
        logger.exception("Gemini test request failed")
        raise HTTPException(
            status_code=502,
            detail="Gemini AI service is unavailable. Check the backend terminal.",
        )


@app.post("/study-plan")
def create_study_plan(data: StudyPlanRequest):
    try:
        predicted_mark = predict_student_mark(data)

        language_instructions = {
            "English": "Write all study plan fields in English.",
            "Bangla": "Write all study plan fields in natural Bengali (বাংলা).",
            "Arabic": "Write all study plan fields in Arabic (العربية).",
            "Hindi": "Write all study plan fields in Hindi (हिन्दी).",
        }
        language_instruction = language_instructions[data.language]

        prompt = f"""
You are a helpful student study planner.

Student information:
- Study hours per day: {data.study_hours}
- Attendance: {data.attendance}%
- Previous mark: {data.previous_mark}
- Assignment score: {data.assignment_score}
- Predicted mark: {predicted_mark}

Create a practical study plan based only on this information.
{language_instruction}
Return one short performance summary, exactly three practical study
tips, and one practical weekly goal.
Do not invent personal details.
The predicted mark is an estimate, not a guarantee.
The output must follow the supplied JSON schema.
"""

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": StudyPlan,
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        study_plan = StudyPlan.model_validate_json(response.text)
        return {
            "predicted_mark": predicted_mark,
            "study_plan": study_plan.model_dump(),
            "language": data.language,
        }

    except (ValidationError, ValueError):
        logger.exception("Study plan response validation failed")
        raise HTTPException(
            status_code=502,
            detail="AI returned an invalid study plan. Please try again.",
        )
    except Exception:
        logger.exception("Study plan generation failed")
        raise HTTPException(
            status_code=502,
            detail="Could not generate the study plan. Check the backend terminal.",
        )
