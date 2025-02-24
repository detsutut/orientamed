from typing import Any, Literal

from boto3 import Session
from langchain_aws import BedrockLLM, BedrockEmbeddings, InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
from languagemodel import LanguageModel
from retriever import Retriever
from langgraph.graph import START, StateGraph, END
from langgraph.types import Command
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
import textwrap

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class rag_prompts:
    query_expansion = ChatPromptTemplate([
        ("system", "You are part of an information system that processes users queries. Expand the given query into one query that is similar in meaning but more complete and useful. If there are acronyms or words you are not familiar with, do not try to rephrase them.\nReturn only one version of the question. The query is in Italian. Answer in Italian."),
        ("human", "{question}")])
    query_expansion_hyde = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Given a question, generate a paragraph that answers the question. If you don't know the answer, try to produce a paragraph anyway. Answer in Italian.\nQuestion: {question} \nParagraph:"),
    ])
    rag_standard = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nContext: {context} \nAnswer:"),
    ])
    rag_addcontext = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nContext: {context} \nAdditional useful information: {additional_context} \nAnswer:"),
    ])
    norag = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Answer in Italian.\nQuestion: {question} \nAnswer:"),
    ])
    history_consolidation = ChatPromptTemplate([
        ("system", "You are part of an information system that processes users queries. Given a chat history and the latest user query which might reference context in the chat history, formulate a standalone query which can be understood without the chat history. Do NOT answer the query, just reformulate it if needed and otherwise return it as is. The query must be in Italian.\n History:"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

def dictlog(d: dict) -> str:
    out_str = "State: {"
    for item in d.items():
        out_str += f"{item[0]}: "
        if type(item[1]) == str:
            out_str += textwrap.shorten(item[1], width=30)
        elif type(item[1]) == list:
            out_str += f"{len(item[1])} elements"
        elif type(item[1]) == bool:
            out_str += f"{item[1]}"
        else:
            out_str += textwrap.shorten(str(item[1]), width=30)
        out_str += ", "
    out_str += "}"
    return out_str

# Define state for application
class State(TypedDict):
    question: str
    history: List[dict]
    context: List[Document]
    additional_context: str
    query_aug: bool
    answer: str


class Rag:
    NORETRIEVE_MSG = "Mi dispiace, non sono riuscito a trovare informazioni rilevanti nelle linee guida."
    NOTALLOWED_MSG = "Mi dispiace, non posso rispondere a questa domanda."

    def __init__(self, session: Session,
                 model: BedrockLLM | str,
                 embedder: BedrockEmbeddings | str,
                 vector_store: InMemoryVectorStore | str | None = None,
                 **kwargs):
        self.session = session
        client = session.client("bedrock-runtime", region_name=kwargs.get("region"))
        self.llm = LanguageModel(model, client=client)
        self.retriever = Retriever(embedder, vector_store=vector_store, client=client)
        graph_builder = StateGraph(State)
        graph_builder.set_entry_point("orchestrator")
        graph_builder.add_node("orchestrator", self.orchestrator)
        graph_builder.add_node("history_consolidator", self.history_consolidator)
        graph_builder.add_node("augmentator", self.augmentator)
        graph_builder.add_node("doc_retriever", self.doc_retriever)
        graph_builder.add_node("generator", self.generator)

        self.graph = graph_builder.compile()

    def doc_retriever(self, state: State) -> Command[Literal["generator", END]]:
        logger.info(f"New retrieval: {state}")
        retrieved_docs = self.retriever.retrieve(state["question"])
        logger.info(f"Retrieved docs: {retrieved_docs}")
        if len(retrieved_docs) == 0:
            return Command(
                update={"context": retrieved_docs,
                        "answer": self.NORETRIEVE_MSG},
                goto= END,
            )
        else:
            return Command(
                update= {"context": retrieved_docs},
                goto="generator",
            )

    def orchestrator(self, state: State) -> Command[Literal["augmentator","doc_retriever","history_consolidator"]]:
        logger.info(f"Dispatching request: {state}")
        previous_user_interactions = [h for h in state["history"] if h["role"]=="user"]
        if len(previous_user_interactions)>0:
            return Command(goto="history_consolidator")
        else:
            return Command(goto="augmentator" if state["query_aug"] else "doc_retriever")

    def history_consolidator(self, state: State) -> Command[Literal["orchestrator"]]:
        logger.info(f"Consolidating previous history...")
        if len(state["history"])>5:
            proximal_history = state["history"][-5:]
        else:
            proximal_history = state["history"]
        for item in proximal_history:
            if item["role"] == "user":
                item["role"] = "human"
            elif item["role"] == "assistant":
                item["role"] = "ai"
        messages = rag_prompts.history_consolidation.invoke({"question": state["question"],
                                                             "history": [(item["role"], item["content"]) for item in proximal_history]})
        logger.info(messages)
        logger.info(proximal_history)
        consolidated_question = self.llm.generate(prompt=messages)
        logger.info(f"Consolidated query: {textwrap.shorten(consolidated_question, width=30)}")
        return Command(
            update= {"question": consolidated_question, "history": []},
            goto="orchestrator",
        )

    def augmentator(self, state:State) -> Command[Literal["doc_retriever"]]:
        logger.info(f"Expanding query...")
        messages = rag_prompts.query_expansion_hyde.invoke({"question": state["question"]})
        augmented_question = self.llm.generate(prompt=messages)
        logger.info(f"Expanded query: {textwrap.shorten(augmented_question, width=30)}")
        return Command(
            update= {"question": augmented_question},
            goto="doc_retriever",
        )

    def generator(self, state: State) -> Command[Literal[END]]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        extra_context = state.get("additional_context", None)
        if extra_context is not None and extra_context != "":
            messages = rag_prompts.rag_addcontext.invoke(
                {"question": state["question"], "context": docs_content, "additional_context": extra_context})
        else:
            messages = rag_prompts.rag_standard.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.generate(prompt=messages)
        return Command(update={"answer": response}, goto=END)

    def invoke(self, input: dict[str, Any]):
        return self.graph.invoke(input)

    def get_image(self):
        return self.graph.get_graph().draw_mermaid_png()