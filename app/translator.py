from typing import Literal
from deep_translator import GoogleTranslator
import googletrans
from dotenv import dotenv_values
import asyncio
import lara_sdk
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeeplTranslator:
    def __init__(self, api_key:str, target='en'):
        self.api_key = api_key
        self.target = target

    def translate(self, text, target=None):
        url = "https://api-free.deepl.com/v2/translate"
        params = {'text': text, 'target_lang': self.target if target is None else target, 'auth_key': self.api_key}
        response = requests.post(url, data=params)
        if response.status_code == 200:
            translated_text = response.json()['translations'][0]['text']
            return translated_text
        else:
            print("Error:", response.status_code, response.text)
            return None

class GoogleCloudTranslator:
    def __init__(self, api_key:str, target='en'):
        self.api_key = api_key
        self.target = target

    def translate(self, text, target=None):
        url = "https://translation.googleapis.com/language/translate/v2"
        params = {'q': text, 'target': self.target if target is None else target, 'key': self.api_key}
        response = requests.post(url, data=params)
        if response.status_code == 200:
            translated_text = response.json()['data']['translations'][0]['translatedText']
            return translated_text
        else:
            print("Error:", response.status_code, response.text)
            return None

class Translators:
    def __init__(self, secrets_path="trans_secrets.env"):
        self.free_translator = (GoogleTranslator(source='auto', target='en'),{'target':'en','type':'google_translate'})
        self.premium_translators = []
        self.__init_premium_translators__(secrets_path)

    def __init_premium_translators__(self, secrets_path):
        if dotenv_values(secrets_path).get("DEEPL_KEYS"):
            deepl_keys = json.loads(dotenv_values(secrets_path).get("DEEPL_KEYS"))
            for deepl_key in deepl_keys:
                self.premium_translators.append((DeeplTranslator(api_key=deepl_key, target='en'), {'target':'en','type':'deepl'}))
        if dotenv_values(secrets_path).get("LARA_KEY_ID") and dotenv_values(secrets_path).get("LARA_KEY_SECRET"):
            lara_credentials = lara_sdk.Credentials(access_key_id=dotenv_values(secrets_path).get("LARA_KEY_ID"),
                                           access_key_secret=dotenv_values(secrets_path).get("LARA_KEY_SECRET"))
            self.premium_translators.append((lara_sdk.Translator(lara_credentials),{'target':'en-US','type':'lana'}))
        if dotenv_values(secrets_path).get("CLOUD_TRANSLATE_KEY"):
            self.premium_translators.append((GoogleCloudTranslator(api_key=dotenv_values(secrets_path).get("CLOUD_TRANSLATE_KEY"), target='en'), {'target':'en','type':'cloud_translate'}))

    async def __detect_languages__(self, text: str):
        async with googletrans.Translator(raise_exception=True) as translator:
            return await translator.detect(text)

    def detect_language(self, text: str):
        language = asyncio.run(self.__detect_languages__(text))
        return language.lang

    def __premium_translation_loop__(self, text: str):
        for translator, kwargs in self.premium_translators:
            try:
                result = translator.translate(text)
            except Exception as e:
                print(e)
            else:
                logger.info(f"Translated with premium api \'{kwargs['type']}\': {result}")
                if kwargs.get('type') == 'lana':
                    return result.translation
                else:
                    return result
        return None

    def translate(self, text: str, use_premium: Literal["never","last","first"] = "never"):
        if use_premium == "never" or use_premium == "last":
            try:
                translator, kwargs = self.free_translator
                translated_text = translator.translate(text)
            except Exception as e:
                logger.error(e)
                if use_premium == "never":
                    return None
            else:
                logger.info(f"Translated with free api: {translated_text}")
                return translated_text
        if use_premium == "first":
            return self.__premium_translation_loop__(text)
        else:
            #you are not supposed to enter here
            return None

if __name__ == "__main__":
    t = Translators()
    print(t.translate('ciao, mondo!', use_premium="first"))