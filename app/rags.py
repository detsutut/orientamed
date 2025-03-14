from typing import Any, Literal, Tuple

from boto3 import Session
from langchain_aws import BedrockLLM, BedrockEmbeddings, InMemoryVectorStore, ChatBedrockConverse
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
import json

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


class Prompts:
    def __init__(self, jsonfile: str):
        with open(jsonfile, 'r') as file:
            data = json.load(file)
            for k, v in data.items():
                self.__setattr__(k, self.__parse__(v))

    def __parse__(self, prompt_dicts: List[dict]):
        return ChatPromptTemplate([(d["role"], d["content"]) for d in prompt_dicts])


# Define state for application
class State(TypedDict):
    question: str
    history: List[BaseMessage]
    context: dict
    additional_context: str = ""
    query_aug: bool
    input_tokens_count: int
    output_tokens_count: int
    answer: str


class Rag:
    NORETRIEVE_MSG = "Mi dispiace, non sono riuscito a trovare informazioni rilevanti nelle linee guida."
    NOTALLOWED_MSG = "Mi dispiace, non posso rispondere a questa domanda."

    def __init__(self, session: Session,
                 model: ChatBedrockConverse | str,
                 embedder: BedrockEmbeddings | str,
                 vector_store: InMemoryVectorStore | str | None = None,
                 **kwargs):
        self.prompts = Prompts(kwargs.get("promptfile", "./prompts.json"))
        self.session = session
        client = session.client("bedrock-runtime", region_name=kwargs.get("region"))
        self.llm = LanguageModel(model, client=client, model_low=kwargs.get("model_low", None),
                                 model_pro=kwargs.get("model_pro", None))
        self.retriever = Retriever(embedder, vector_store=vector_store, client=client)
        graph_builder = StateGraph(State)
        graph_builder.set_entry_point("orchestrator")
        graph_builder.add_node("orchestrator", self.orchestrator)
        graph_builder.add_node("history_consolidator", self.history_consolidator)
        graph_builder.add_node("augmentator", self.augmentator)
        graph_builder.add_node("doc_retriever", self.doc_retriever)
        graph_builder.add_node("generator", self.generator)
        self.graph = graph_builder.compile()

    def generate_norag(self, input: str):
        messages = self.prompts.no_rag.invoke({"question": input})
        response = self.llm.generate(messages=messages)
        return {"answer": response.content,
                "input_tokens_count": response.usage_metadata["input_tokens"],
                "output_tokens_count": response.usage_metadata["output_tokens"]}

    def orchestrator(self, state: State) -> Command[Literal["augmentator", "doc_retriever", "history_consolidator"]]:
        logger.info(f"Dispatching request: {state}")
        previous_user_interactions = [message for message in state["history"] if type(message) is HumanMessage]
        if len(previous_user_interactions) > 0:
            return Command(goto="history_consolidator")
        else:
            return Command(goto="augmentator" if state["query_aug"] else "doc_retriever")

    def doc_retriever(self, state: State) -> Command[Literal["generator", END]]:
        logger.info(f"New retrieval: {state}")
        retrieved_docs, scores = self.retriever.retrive_with_scores(state["question"], n=10, score_threshold=0.6)
        logger.info(f"Retrieved docs: {retrieved_docs}")
        if len(retrieved_docs) == 0:
            return Command(
                update={"context": {"docs": retrieved_docs, "scores": scores},
                        "answer": self.NORETRIEVE_MSG},
                goto=END,
            )
        else:
            return Command(
                update={"context": {"docs": retrieved_docs, "scores": scores}},
                goto="generator",
            )

    def history_consolidator(self, state: State) -> Command[Literal["orchestrator"]]:
        logger.info(f"Consolidating previous history...")
        if len(state["history"]) > 5:
            proximal_history = state["history"][-5:]
        else:
            proximal_history = state["history"]
        messages = self.prompts.history_consolidation.invoke({"question": state["question"],
                                                              "history": messages_to_history_str(
                                                                  state["history"])}).messages
        logger.info(messages)
        logger.info(proximal_history)
        response = self.llm.generate(messages=messages)
        consolidated_question = response.content
        logger.info(f"Consolidated query: {textwrap.shorten(consolidated_question, width=30)}")
        return Command(
            update={"question": consolidated_question,
                    "history": [],
                    "input_tokens_count": state["input_tokens_count"] + response.usage_metadata["input_tokens"],
                    "output_tokens_count": state["output_tokens_count"] + response.usage_metadata["output_tokens"]},
            goto="orchestrator",
        )

    def augmentator(self, state: State) -> Command[Literal["doc_retriever"]]:
        logger.info(f"Expanding query...")
        messages = self.prompts.query_expansion_hyde.invoke({"question": state["question"]}).messages
        response = self.llm.generate(messages=messages)
        augmented_question = response.content
        logger.info(f"Expanded query: {textwrap.shorten(augmented_question, width=30)}")
        return Command(
            update={"question": augmented_question,
                    "input_tokens_count": state["input_tokens_count"] + response.usage_metadata["input_tokens"],
                    "output_tokens_count": state["output_tokens_count"] + response.usage_metadata["output_tokens"]
                    },
            goto="doc_retriever",
        )

    def generator(self, state: State) -> Command[Literal[END]]:
        doc_strings = []
        for i, doc in enumerate(state["context"]["docs"]):
            doc_strings.append(f"Source {i}:\n{doc.page_content}")
        docs_content = "\n".join(doc_strings)
        additional_context = state.get("additional_context", None)
        if type(additional_context) is str and additional_context != "":
            messages = self.prompts.question_with_context_add.invoke(
                {"question": state["question"],
                 "context": docs_content,
                 "additional_context": additional_context}).messages
        else:
            messages = self.prompts.question_with_context_inline_cit.invoke(
                {"question": state["question"], "context": docs_content}).messages
        response = self.llm.generate(messages=messages, level="pro")
        return Command(update={"answer": response.content,
                               "input_tokens_count": state["input_tokens_count"] + response.usage_metadata["input_tokens"],
                               "output_tokens_count": state["output_tokens_count"] + response.usage_metadata["output_tokens"]},
                       goto=END)

    def invoke(self, input: dict[str, Any]):
        return self.graph.invoke(input)

    def get_image(self):
        return self.graph.get_graph().draw_mermaid_png()
