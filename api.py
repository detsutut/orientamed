from fastapi import FastAPI, Query
from typing import Union
from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated
from pydantic import BaseModel, Field
import yaml
from core.rags import Rag
import logging
from boto3 import Session

from core.utils import from_list_to_messages, get_mfa_response


class QueryParams(BaseModel):
    text: str = Field(description="Text to extract concepts from")
    filter_tags: Union[list[str],None] = Field(default=None, description="Text to extract concepts from")
    exclude: bool = Field(default=False, description="Use filter tags as an exclusion list?")
    fuzzy_threshold: int = Field(default=100, description="Fuzzy matching threshold (0-100)")
    use_premium: bool = Field(default=False, description="Use premium translation?")

class GenerateQueryParams(BaseModel):
    user_input: str = Field(description="Query to generate answer")
    history: list[dict] = Field(default=[], description="History as a list of dictionaries with openai-style 'role' and 'content' keys")
    additional_context: Union[str,None] = Field(default=None, description="Additional context to use for the query")
    augment_query: bool = Field(default=False, description="Augmented query")
    retrieve_only: bool = Field(default=False, description="Retrieve only")
    use_graph: bool = Field(default=True, description="Use graph")
    use_embeddings: bool = Field(default=True, description="Use embeddings")

############# SETTINGS ##################
debug = True
with open("reuma_settings.yaml") as stream:
    config = yaml.safe_load(stream)

############# LOGGER ##################
logging.basicConfig(level=logging.INFO if debug else logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app")
logger.setLevel(logging.INFO if debug else logging.INFO)

app = FastAPI(title="RAG",
              contact={
                  "name": "RAG",
                  "url": "https://github.com/detsutut",
                  "email": "buonocore.tms@gmail.com",
              }
              )

mfa_response = get_mfa_response(aws_token)
session = Session(aws_access_key_id=mfa_response['Credentials']['AccessKeyId'],
                  aws_secret_access_key=mfa_response['Credentials']['SecretAccessKey'],
                  aws_session_token=mfa_response['Credentials']['SessionToken'])

RAG = Rag(session=Session(),
          model=config.get("bedrock").get("models").get("model-id"),
          embedder=config.get("bedrock").get("embedder-id"),
          vector_store=config.get("vector-db-path"),
          region=config.get("bedrock").get("region"),
          model_pro=config.get("bedrock").get("models").get("pro-model-id"),
          model_low=config.get("bedrock").get("models").get("low-model-id"),
          promptfile="core/prompts.json")

@app.get("/")
def read_root():
    return {}

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")



@app.get("/check", response_model=dict)
async def check():
    return {
        "status": "ok",
    }

@app.post("/log", response_model=dict)
async def log(aws_token:str):
    mfa_response = get_mfa_response(aws_token)
    return {
        "status": "ok",
    }

@app.post("/generate", response_model=dict)
def generate(query: GenerateQueryParams):
    if not query.user_input:
        return JSONResponse(content={"error": "Please provide a text."}, status_code=400)
    response = RAG.invoke({"query": query.user_input,
                           "history": from_list_to_messages(query.history),
                           "additional_context": query.additional_context,
                           "input_tokens_count": 0,
                           "output_tokens_count": 0,
                           "query_aug": query.augment_query,
                           "retrieve_only": query.retrieve_only,
                           "use_graph": query.use_graph,})
    return {
        "answer": response.get("answer"),
        "input_tokens_count" : response.get("input_tokens_count"),
        "output_tokens_count" : response.get("output_tokens_count"),
        "retrieved_documents" : {
            "embeddings": {
                "docs": response.get("context").get("docs"),
                "scores": response.get("context").get("scores")
            },
            "graphs": {
                "docs": response.get("kg_context").get("docs"),
                "scores": response.get("kg_context").get("scores"),
                "paths": response.get("kg_context").get("paths")
            }
        },
        "concepts": {
            "query": response.get("query_concepts"),
            "answer": response.get("answer_concepts")
        }
    }

