import logging
import boto3
from dotenv import dotenv_values
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

AWS_SECRETS = "aws_secrets.env"
GRADIO_SECRETS = "gradio_secrets.env"

def get_admin_username():
    return dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR")

def from_list_to_messages(chat:list[dict]):
    template = ChatPromptTemplate([MessagesPlaceholder("history")]).invoke({"history":[(message["role"],message["content"]) for message in chat]})
    return template.to_messages()

def token_auth(username: str, password: str):
    logger.info(f"Login attempt from user '{username}'")
    # ADMIN LOGIN
    if username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR"):
        return get_mfa_response(str(password)) is not None
    # OTHER USERS LOGIN
    else:
        for user,pwd in zip(json.loads(dotenv_values(GRADIO_SECRETS).get("GRADIO_USRS")), json.loads(dotenv_values(GRADIO_SECRETS).get("GRADIO_PWDS"))):
            check_user = (username == user)
            check_password = (password == pwd)
            if check_user and check_password:
                return True
        return False
    return False

def get_mfa_response(mfa_token, duration: int = 900):
    logger.debug("Checking MFA token...")
    if len(mfa_token) != 6:
        return None
    try:
        sts_client = boto3.client('sts',
                            aws_access_key_id=dotenv_values(AWS_SECRETS).get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=dotenv_values(AWS_SECRETS).get("AWS_SECRET_ACCESS_KEY"))
        response = sts_client.get_session_token(DurationSeconds=duration,
                                                SerialNumber=dotenv_values(AWS_SECRETS).get("AWS_ARN_MFA_DEVICE"),
                                                TokenCode=mfa_token)
        return response
    except Exception as e:
        logger.error(str(e))
        return None

def dot_progress_bar(score, total_dots=7):
    filled_count = round(score * total_dots)
    empty_count = total_dots - filled_count
    filled = "•" * filled_count
    empty = "·" * empty_count
    return f"{filled}{empty} {round(score*100,2)}%"
