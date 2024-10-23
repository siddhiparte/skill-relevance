from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from relevance import SkillMatcher
import os

app = FastAPI()

# Updated CORS settings for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend-url.vercel.app"  # Update with your Vercel URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SkillMatcher with error handling
try:
    matcher = SkillMatcher('programming_languages.csv')
except Exception as e:
    print(f"Error initializing SkillMatcher: {e}")
    raise

@app.on_event("startup")
async def startup_event():
    try:
        accuracy = matcher.train_classifier()
        print(f"Model trained with accuracy: {accuracy:.2f}")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/api/match-skills")
async def match_skills(skills: List[str] = Body(...)):
    try:
        main_skills = matcher.get_main_skills(skills)
        return {
            "status": "success",
            "matched_skills": list(main_skills),
            "input_skills": skills
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "input_skills": skills
        }