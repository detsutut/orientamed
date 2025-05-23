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

class RetrievedDocument(BaseModel):
    id: str
    page_content: str
    metadata: dict
    score: float

class RetrievedDocuments(BaseModel):
    embeddings: dict
    graphs: dict

class Document(BaseModel):
    id: str
    text: str
    metadata: dict

class LLMResponse(BaseModel):
    answer: str
    input_tokens_count: int
    output_tokens_count: int
    retrieved_documents: RetrievedDocuments
    concepts: Concepts