from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from relevance import SkillMatcher

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the SkillMatcher
matcher = SkillMatcher('programming_languages.csv')

@app.on_event("startup")
async def startup_event():
    accuracy = matcher.train_classifier()
    print(f"Model trained with accuracy: {accuracy:.2f}")

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