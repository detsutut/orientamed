import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def dot_progress_bar(score, total_dots=7, absolute=False):
    if absolute:
        filled = "■" + "-•" * int(score) +"-■"
        return f"{filled} {int(score)}"
    else:
        filled_count = round(score * total_dots)
        empty_count = total_dots - filled_count
        filled = "•" * filled_count
        empty = "·" * empty_count
        return f"{filled}{empty} {round(score*100,2)}%"

from pydantic import BaseModel
from typing import List, Optional

class Concept(BaseModel):
    name: str
    id: str
    match_score: float
    semantic_tags: List[str]

class Concepts(BaseModel):
    query: List[Concept]
    answer: List[Concept]

class RetrievedDocuments(BaseModel):
    embeddings: dict
    graphs: dict

class LLMResponse(BaseModel):
    answer: str
    input_tokens_count: int
    output_tokens_count: int
    retrieved_documents: RetrievedDocuments
    concepts: Concepts