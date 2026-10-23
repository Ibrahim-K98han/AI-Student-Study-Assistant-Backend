import logging
import os
import time
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field, ValidationError

# --------------------------------------------------
# Logging and environment
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your backend .env file."
    )

client = genai.Client(api_key=api_key)

# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Student AI Study Assistant API",
    description="ML mark prediction and multilingual AI study planning",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict for production.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

try:
    model = joblib.load("student_model.pkl")
except Exception as exc:
    raise RuntimeError(
        "Could not load student_model.pkl. Run the server from "
        "the folder containing that file."
    ) from exc

# --------------------------------------------------
# Supported languages
# --------------------------------------------------

SupportedLanguage = Literal["English", "Bangla", "Arabic", "Hindi"]

LANGUAGE_INSTRUCTIONS = {
    "English": """
Write all response text naturally in English.
""",
    "Bangla": """
IMPORTANT: Write ALL response text in natural Bengali (বাংলা),
using Bengali script.

The summary, all three study tips, and weekly goal MUST be
written in Bengali. Do not write English sentences.
Keep only the JSON property names in English.
Use simple, clear Bengali suitable for a student.
""",
    "Arabic": """
Write all response text in natural Arabic using Arabic script.
The summary, all three study tips, and weekly goal must be Arabic.
Keep only the JSON property names in English.
""",
    "Hindi": """
Write all response text in natural Hindi using Devanagari script.
The summary, all three study tips, and weekly goal must be Hindi.
Keep only the JSON property names in English.
""",
}

# --------------------------------------------------
# Request and response models
# --------------------------------------------------

class StudentData(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)
    previous_mark: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)


class StudyPlanRequest(StudentData):
    language: SupportedLanguage = "English"


class StudyPlan(BaseModel):
    summary: str = Field(min_length=1)
    study_tips: list[str] = Field(min_length=3, max_length=3)
    weekly_goal: str = Field(min_length=1)


# --------------------------------------------------
# Gemini retry helper
# --------------------------------------------------

def generate_study_plan_with_retry(**kwargs):
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(**kwargs)

        except errors.APIError as exc:
            retryable_codes = (429, 500, 502, 503, 504)

            if exc.code not in retryable_codes:
                raise

            if attempt == max_attempts - 1:
                raise

            delay = 2 ** (attempt + 1)

            logger.warning(
                "Gemini API error %s. Attempt %s/%s. "
                "Retrying in %s seconds.",
                exc.code,
                attempt + 1,
                max_attempts,
                delay,
            )

            time.sleep(delay)

# --------------------------------------------------
# ML prediction function
# --------------------------------------------------

def predict_student_mark(data: StudentData) -> float:
    student = pd.DataFrame(
        [
            {
                "study_hours": data.study_hours,
                "attendance": data.attendance,
                "previous_mark": data.previous_mark,
                "assignment_score": data.assignment_score,
            }
        ]
    )

    prediction = model.predict(student)

    return round(float(prediction[0]), 2)


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Student AI API is running!"
    }


# --------------------------------------------------
# Predict final mark
# --------------------------------------------------

@app.post("/predict")
def predict(data: StudentData):
    try:
        predicted_mark = predict_student_mark(data)

        return {
            "predicted_mark": predicted_mark
        }

    except Exception:
        logger.exception("Mark prediction failed")

        raise HTTPException(
            status_code=500,
            detail="Could not predict the student's mark.",
        )


# --------------------------------------------------
# Test Gemini connection
# --------------------------------------------------

@app.get("/test-gemini")
def test_gemini():
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Explain AI in one simple sentence.",
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return {
            "message": response.text
        }

    except Exception:
        logger.exception("Gemini test request failed")

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini AI service is unavailable. "
                "Check the backend terminal."
            ),
        )


# --------------------------------------------------
# Generate multilingual AI study plan
# --------------------------------------------------

@app.post("/study-plan")
def create_study_plan(data: StudyPlanRequest):
    try:
        # Step 1: Predict the student's final mark
        predicted_mark = predict_student_mark(data)

        # Step 2: Get selected language instructions
        language_instruction = LANGUAGE_INSTRUCTIONS[data.language]

        # Step 3: Build the Gemini prompt
        prompt = f"""
You are a helpful, multilingual student study planner.

MANDATORY LANGUAGE RULE:
Selected language: {data.language}

{language_instruction}

Every value in the JSON response must follow the selected language.
This includes:
- summary
- all three study_tips
- weekly_goal

Do not mix English sentences into a non-English response.
Do not translate the JSON property names.
Use practical, encouraging advice based only on the supplied data.

STUDENT INFORMATION:
- Study hours per day: {data.study_hours}
- Attendance percentage: {data.attendance}
- Previous mark: {data.previous_mark}
- Assignment score: {data.assignment_score}
- Predicted final mark: {predicted_mark}

TASK:
1. Write one short performance summary.
2. Provide exactly three practical study tips.
3. Write one realistic weekly goal.
4. Do not invent personal details.
5. Explain the predicted mark as an estimate, not a guarantee.
6. Return data matching the supplied JSON schema.

Remember: the selected language is {data.language}.
"""

        # Step 4: Generate structured JSON with retry
        response = generate_study_plan_with_retry(
            model="gemini-3.8-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": StudyPlan,
                "temperature": 0.2,
            },
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        # Step 5: Validate Gemini's JSON response
        study_plan = StudyPlan.model_validate_json(response.text)

        # Step 6: Return the complete result
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
            detail=(
                "Could not generate the study plan. "
                "Check the backend terminal."
            ),
        )