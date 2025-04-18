

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from fastapi.middleware.cors import CORSMiddleware
from run_optimization import generate_assignment, Game

import logging
app = FastAPI()

logger = logging.getLogger('uvicorn.error')

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


class LineupRequest(BaseModel):
    players: List[Player]
    initial_lineup: Union[List[str], str] = 'auto'
    num_periods: int = 6


@app.post("/submit_lineup/")
async def submit_lineup(request: LineupRequest):
    logging.info(request)

    game = Game(request.num_periods, 5)
    names = [player.name for player in request.players]
    skill_level = [player.skill_level for player in request.players]
    out, status = generate_assignment(names, skill_level, game, request.initial_lineup)
    out = out.T.to_html()
    return {"vis": out, "optimization_status": status, 'num_periods': request.num_periods}

# Run the server 
# Run this in the terminal: uvicorn main:app --reload