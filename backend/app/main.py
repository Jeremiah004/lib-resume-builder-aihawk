from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.auth import router as auth
from app.routes.resume import router as resume
from app.routes.user import router as user
from app.core.config import PROJECT_NAME, API_V1_STR

app = FastAPI(
    title=PROJECT_NAME,
    description="API for generating and managing professional resumes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth)
app.include_router(user)
app.include_router(resume)

@app.get("/")
async def root():
    return {"message": "Welcome to Resume Builder API"} 