

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Dict, Optional

from fastapi.middleware.cors import CORSMiddleware
from run_optimization import generate_assignment, Game

import logging

from run_optimization import Constraint
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
    enabled_constraints: Optional[Dict[str, bool]] = None


@app.post("/submit_lineup/")
async def submit_lineup(request: LineupRequest):
    logging.info("Request", request)

    print(request)

    game = Game(request.num_periods, 5)
    names = [player.name for player in request.players]
    skill_level = [player.skill_level for player in request.players]

    # Check if enabled_constraints is None for backward compatibility.
    if request.enabled_constraints is None:
        enabled_constraints = {
            Constraint.MAX_CONSECUTIVE_BENCH_TIME: True,
            Constraint.MAX_CONSECUTIVE_PLAY_TIME: True, 
            Constraint.MIN_PLAY_TIME: True,
            Constraint.MAX_PLAY_TIME: True
        }
    else:
        enabled_constraints = {Constraint(key): items for key, items in request.enabled_constraints.items()} 

    out, status, constraints = generate_assignment(names, skill_level, game, request.initial_lineup, enabled_constraints)
    out = out.T.to_html()


    logging.info("Submit Lineup exiting gracefully")
    return {
        "vis": out,
        "optimization_status": status,
        'constraints': constraints
    }

# Run the server 
# Run this in the terminal: uvicorn main:app --reload