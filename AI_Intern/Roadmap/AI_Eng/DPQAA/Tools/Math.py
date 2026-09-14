import numpy as np

from dataclasses import dataclass
from enum import Enum

class Standout_Category(Enum):
    EXTREME = 2.0
    WELL_ABOVE = 1.5
    GOOD = 1.0

@dataclass
class Stats:
    mean: float=0.0
    std: float=0.0
    top_score: float=0.0
    margin: float=0.0
    ratio: float=0.0
    z_score: float=0.0

def similarity_cosine(vec1: np.ndarray, vec2: np.ndarray) -> float:
    '''Return the cosine similarity between two vectors'''
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (mag1 * mag2))

def retrieve_stat(scores: np.ndarray) -> Stats:
    if len(scores) < 2:
        return Stats()

    scores = np.sort(scores)[::-1]

    top = float(scores[0])
    mean = float(scores.mean())
    std = float(scores.std())

    z_score = (top - mean) / std if std > 0.0 else 0.0

    margin = float(top - scores[1])
    ratio = float(top / max(scores[1], 1e-6))

    return Stats(
        mean=mean,
        std=std,
        top_score=top,
        margin=margin,
        ratio=ratio,
        z_score=z_score,
    )