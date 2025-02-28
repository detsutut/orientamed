from typing import Any, Literal

from boto3 import Session
from langchain_aws import BedrockLLM, BedrockEmbeddings, InMemoryVectorStore
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging
from languagemodel import LanguageModel
from retriever import Retriever
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.documents import Document
from langchain_core.messages.human import HumanMessage
from typing_extensions import List, TypedDict
import textwrap

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def messages_to_history_str(messages: list[BaseMessage]) -> str:
    """Convert messages to a history string."""
    string_messages = []
    for message in messages:
        role = message.type
        content = message.content
        string_message = f"{role}: {content}"

        additional_kwargs = message.additional_kwargs
        if additional_kwargs:
            string_message += f"\n{additional_kwargs}"
        string_messages.append(string_message)
    return "\n".join(string_messages)

class RagPrompts:
    query_expansion = ChatPromptTemplate([
        ("system", "You are part of an information system that processes users queries. Expand the "
                   "given query into one query that is similar in meaning but more complete and "
                   "useful. If there are acronyms or words you are not familiar with, do not try to "
                   "rephrase them.\nReturn only one version of the question. The query is in Italian. "
                   "Answer in Italian."),
        ("human", "{question}")])
    query_expansion_hyde = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Given a question, generate a "
                   "paragraph that answers the question. If you don't know the answer, try to produce "
                   "a paragraph anyway. Answer in Italian.\nQuestion: {question} \nParagraph:"),
    ])
    rag_standard = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of "
                   "retrieved context to answer the question. If you don't know the answer, just "
                   "say that you don't know. Use three sentences maximum and keep the answer concise. "
                   "Answer in Italian.\nQuestion: {question} \nContext: {context} \nAnswer:"),
    ])
    rag_addcontext = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of "
                   "retrieved context to answer the question. If you don't know the answer, just say "
                   "that you don't know. Use three sentences maximum and keep the answer concise. "
                   "Answer in Italian.\nQuestion: {question} \nContext: {context} \nAdditional "
                   "useful information: {additional_context} \nAnswer:"),
    ])
    norag = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. If you don't know the answer, "
                   "just say that you don't know. Use three sentences maximum and keep the answer "
                   "concise. Answer in Italian.\nQuestion: {question} \nAnswer:"),
    ])
    history_consolidation = ChatPromptTemplate([
        ("system", ("Given the following conversation between a user and an AI assistant and a follow up "
                    "question from user, rephrase the follow up question to be a standalone question. Ensure "
                    "that the standalone question summarizes the conversation and completes the follow up "
                    "question with all the necessary context. The standalone question must be in Italian.\n"
                    "Chat History:\n{history}\n"
                    "Question: {question}\n"
                    "Standalone question:"))])


# Define state for application
class State(TypedDict):
    question: str
    history: List[BaseMessage]
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

    def orchestrator(self, state: State) -> Command[Literal["augmentator","doc_retriever","history_consolidator"]]:
        logger.info(f"Dispatching request: {state}")
        previous_user_interactions = [message for message in state["history"] if type(message) is HumanMessage]
        if len(previous_user_interactions)>0:
            return Command(goto="history_consolidator")
        else:
            return Command(goto="augmentator" if state["query_aug"] else "doc_retriever")

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

    def history_consolidator(self, state: State) -> Command[Literal["orchestrator"]]:
        logger.info(f"Consolidating previous history...")
        if len(state["history"])>5:
            proximal_history = state["history"][-5:]
        else:
            proximal_history = state["history"]
        messages = RagPrompts.history_consolidation.invoke({"question": state["question"],
                                                             "history": messages_to_history_str(state["history"])})
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
        messages = RagPrompts.query_expansion_hyde.invoke({"question": state["question"]})
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
            messages = RagPrompts.rag_addcontext.invoke(
                {"question": state["question"],
                 "context": docs_content,
                 "additional_context": extra_context})
        else:
            messages = RagPrompts.rag_standard.invoke({"question": state["question"],
                                                       "context": docs_content})
        response = self.llm.generate(prompt=messages)
        return Command(update={"answer": response}, goto=END)

    def invoke(self, input: dict[str, Any]):
        return self.graph.invoke(input)

    def get_image(self):
        return self.graph.get_graph().draw_mermaid_png()