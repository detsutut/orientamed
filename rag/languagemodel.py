from langchain_aws import BedrockLLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageModel:
    def __init__(self, model: BedrockLLM | str, client=None):
        if type(model) is BedrockLLM:
            self.llm = model
        else:
            self.llm = BedrockLLM(
                model_id=model,
                client=client,
            )

    def generate(self, prompt: str):
        return self.llm.invoke(prompt)
