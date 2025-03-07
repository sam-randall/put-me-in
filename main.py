

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from fastapi.middleware.cors import CORSMiddleware
from run_optimization import generate_assignment, Game


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing, restrict this in production)
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class Player(BaseModel):
    name: str
    skill_level: float



@app.post("/submit_lineup/")
async def submit_lineup(players: List[Player], initial_lineup: Union[List[str], str] = 'auto'):
    print("Players", players)
    print("Lineup", initial_lineup)
    game = Game(6, 5)

    names = [player.name for player in players]
    skill_level = [player.skill_level for player in players]

    out, status = generate_assignment(names, skill_level, game, initial_lineup)

    out = out.T.to_html()


    return {"vis": out, "optimization_status": status}

# Run the server
# Run this in the terminal: uvicorn main:app --reload