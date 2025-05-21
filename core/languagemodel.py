from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
import logging

logger = logging.getLogger('app.'+__name__)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("langchain_core").setLevel(logging.WARNING)

noSystemPromptModels = [
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
    "cohere.command-text-v14",
    "cohere.command-light-text-v14",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "mistral.mixtral-8x7b-instruct-v0:1"
]

def __instantiateLLM__(model: ChatBedrockConverse | str, client):
    if type(model) is ChatBedrockConverse:
        return model
    else:
        return ChatBedrockConverse(model_id=model, client=client)

class LanguageModel:
    def __init__(self, model: ChatBedrockConverse | str, client=None, model_pro: ChatBedrockConverse | str | None = None, model_low: ChatBedrockConverse | str | None = None,):
        self.llm = __instantiateLLM__(model, client)
        self.llm_pro = __instantiateLLM__(model_pro, client) if model_pro is not None else __instantiateLLM__(model, client)
        self.llm_low = __instantiateLLM__(model_low, client) if model_low is not None else __instantiateLLM__(model, client)

    def __sanitize_msgs__(self, messages: list[BaseMessage]):
        i = 0
        while i < len(messages):
            if type(messages[i]) is SystemMessage:
                messages[i]=HumanMessage(messages[i].content)
                messages.insert(i + 1, AIMessage("Okay."))
            i += 1
        return messages

    def generate(self, messages: list[BaseMessage], level: Literal["standard","pro","low"]="standard",**kwargs)->AIMessage:
        llm = self.llm_pro if level == "pro" else self.llm_low if level == "low" else self.llm
        allowed_keys = ["temperature","max_tokens"]
        llm.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        if llm.model_id in noSystemPromptModels:
            generated_message = llm.invoke(self.__sanitize_msgs__(messages))
        else:
            generated_message = llm.invoke(messages)
        return generated_message