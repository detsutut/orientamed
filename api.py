import os
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from typing import Union
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import yaml
import logging
from logging.handlers import RotatingFileHandler
from boto3 import Session
import uvicorn

############# .ENV ####################

if find_dotenv():
    load_dotenv()
elif find_dotenv("core/secrets.env"):
    load_dotenv("core/secrets.env")
else:
    raise FileNotFoundError("No secrets.env file found.")

############# LOCAL MODULES ####################

from core.rags import Rag
from core.utils import from_list_to_messages, get_mfa_response
from api_utils.login import verify_token, authenticate, create_access_token

############# SETTINGS ##################
with open(os.getenv("API_SETTINGS_PATH")) as stream:
    api_config = yaml.safe_load(stream)
with open(os.getenv("CORE_SETTINGS_PATH")) as stream:
    rag_config = yaml.safe_load(stream)


############# LOGGER ##################
# Create a custom logger
logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG if api_config.get("debug",False) else logging.INFO)  # Capture all logs

# Console handler - prints everything
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if api_config.get("debug",False) else logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
console_handler.setFormatter(console_formatter)

# File handler - saves only ERROR and above
file_handler = RotatingFileHandler("logs.log", maxBytes=api_config.get("logs").get("max-size-mb",1)*1000000, backupCount=api_config.get("logs").get("num-backup",1))
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

############ DATA MODELS ################

class Credentials(BaseModel):
    username: str = Field(description="Username")
    password: str = Field(description="Password")

class GenerateQueryParams(BaseModel):
    user_input: str = Field(description="Query to generate answer")
    history: list[dict] = Field(default=[],
                                description="History as a list of dictionaries with openai-style 'role' and 'content' keys")
    additional_context: Union[str, None] = Field(default=None, description="Additional context to use for the query")
    augment_query: bool = Field(default=False, description="Augmented query")
    retrieve_only: bool = Field(default=False, description="Retrieve only")
    use_graph: bool = Field(default=True, description="Use graph")
    use_embeddings: bool = Field(default=True, description="Use embeddings")

############### API ######################
app = FastAPI(title="OrientaMed", contact={"name": "Tommaso Buonocore", "url": "https://github.com/detsutut", "email": "buonocore.tms@gmail.com"})

logger.info("Initializing RAG model...")
RAG = Rag(session=Session(), vector_store=api_config.get("vector-db-path"))

logger.info("Done.")
@app.get("/")
def read_root():
    return {}


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")


@app.get("/auth/check", response_model=dict)
async def check():
    return {
        "status": "ok",
    }

@app.post("/auth/check", response_model=dict)
async def check(access_token: str):
    payload = verify_token(access_token)
    return {
        "status": "ok",
        "logged": True if payload else None,
    }

@app.post("/auth/login", response_model=dict)
async def log(credentials: Credentials):
    if authenticate(credentials.username, credentials.password):
        token = create_access_token(credentials.username)
        return {
            "status": "ok",
            "access-token": token
        }
    else:
        return JSONResponse(content={
            "status": "error",
            "access-token": None,
            "error": "Invalid username or password"
        }, status_code=401)

@app.post("/auth/init", response_model=dict)
async def init(access_token: str, mfa_token: str):
    payload = verify_token(access_token)
    if payload:
        global RAG
        logger.info("Establishing session with AWS bedrock service...")
        mfa_response = get_mfa_response(mfa_token)
        if mfa_response is None:
            logger.error("Impossible to establish a session with AWS bedrock service. Check your MFA token and try again. If the problem persists, contact the developer.")
            return JSONResponse(content={
                "status": "error",
                "access-token": None,
                "error": "Invalid token"
            }, status_code=401)
        session = Session(aws_access_key_id=mfa_response['Credentials']['AccessKeyId'],
                          aws_secret_access_key=mfa_response['Credentials']['SecretAccessKey'],
                          aws_session_token=mfa_response['Credentials']['SessionToken'])
        RAG = Rag(session=session, vector_store=api_config.get("vector-db-path"))
        logger.info("RAG updated")
        return JSONResponse(content={
            "status": "ok",
        }, status_code=200)
    else:
        return JSONResponse(content={
            "status": "error",
            "error": "Invalid token."
        }, status_code=401)


@app.post("/generate", response_model=dict)
def generate(query: GenerateQueryParams, access_token: str):
    payload = verify_token(access_token)
    if not payload:
        return JSONResponse(content={"error": "Invalid token."}, status_code=401)
    if not query.user_input:
        return JSONResponse(content={"error": "Please provide a text."}, status_code=400)
    try:
        response = RAG.invoke({"query": query.user_input,
                               "history": from_list_to_messages(query.history),
                               "additional_context": query.additional_context,
                               "input_tokens_count": 0,
                               "output_tokens_count": 0,
                               "query_aug": query.augment_query,
                               "retrieve_only": query.retrieve_only,
                               "use_graph": query.use_graph, })
        return JSONResponse(content={
            "answer": response.get("answer"),
            "input_tokens_count": response.get("input_tokens_count"),
            "output_tokens_count": response.get("output_tokens_count"),
            "retrieved_documents": {
                "embeddings": {
                    "docs": [d.model_dump() for d in response.get("context").get("docs")],
                    "scores": response.get("context").get("scores")
                },
                "graphs": {
                    "docs": [d.model_dump() for d in response.get("kg_context").get("docs")],
                    "scores": response.get("kg_context").get("scores"),
                    "paths": response.get("kg_context").get("paths")
                }
            },
            "concepts": {
                "query": response.get("query_concepts"),
                "answer": response.get("answer_concepts")
            }
        }, status_code=200)
    except Exception as e:
        logger.error(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)