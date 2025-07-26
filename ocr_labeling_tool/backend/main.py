from fastapi import FastAPI
from routers import upload, segment, label, lines 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(label.router)
app.include_router(upload.router)
app.include_router(segment.router)
app.include_router(lines.router)