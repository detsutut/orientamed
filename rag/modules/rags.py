from typing import Any

from boto3 import Session
from langchain_aws import BedrockLLM, BedrockEmbeddings, InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
import logging
from languagemodel import LanguageModel
from retriever import Retriever
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import textwrap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class rag_prompts:
    query_expansion = ChatPromptTemplate([
        ("system",
         "You are part of an information system that processes users queries. Expand the given query into one query that is similar in meaning but more complete and useful. If there are acronyms or words you are not familiar with, do not try to rephrase them.\nReturn only one version of the question. The query is in Italian. Answer in Italian."),
        ("human", "{question}")])
    rag_standard = ChatPromptTemplate([
        ("system",
         "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nContext: {context} \nAnswer:"),
    ])
    rag_addcontext = ChatPromptTemplate([
        ("system",
         "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nContext: {context} \nAdditional useful information: {additional_context} \nAnswer:"),
    ])
    norag = ChatPromptTemplate([
        ("system",
         "You are an assistant for question-answering tasks. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nAnswer:"),
    ])


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    additional_context: str
    query_aug: bool
    answer: str


class Rag:
    def __init__(self, session: Session,
                 model: BedrockLLM | str,
                 embedder: BedrockEmbeddings | str,
                 vector_store: InMemoryVectorStore | str | None = None,
                 **kwargs):
        self.session = session
        client = session.client("bedrock-runtime", region_name=kwargs.get("region"))
        self.llm = LanguageModel(model, client=client)
        self.retriever = Retriever(embedder, vector_store=vector_store, client=client)
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def retrieve(self, state: State):
        logger.info(f"New retrieval: {state}")
        retrieved_docs = self.retriever.retrieve(state["question"])
        logger.info(f"Retrieved docs: {retrieved_docs}")
        return {"context": retrieved_docs}

    def generate(self, state: State):
        question = state["question"]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        extra_context = state.get("additional_context", None)
        logger.info(f"New query: {state}")
        if state["query_aug"]:
            logger.info(f"Expanding query...")
            messages = rag_prompts.query_expansion.invoke({"question": state["question"]})
            question = self.llm.generate(prompt=messages)
            logger.info(f"Expanded query: {textwrap.shorten(question, width=30)}")
        if extra_context is not None and extra_context != "":
            messages = rag_prompts.rag_addcontext.invoke(
                {"question": question, "context": docs_content, "additional_context": extra_context})
        else:
            messages = rag_prompts.rag_standard.invoke({"question": question, "context": docs_content})
        response = self.llm.generate(prompt=messages)
        return {"answer": response}

    def invoke(self, input: dict[str, Any]):
        return self.graph.invoke(input)
