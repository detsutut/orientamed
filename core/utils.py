import logging
import os
import boto3
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger('app.'+__name__)

def from_list_to_messages(chat:list[dict]):
    template = ChatPromptTemplate([MessagesPlaceholder("history")]).invoke({"history":[(message["role"],message["content"]) for message in chat]})
    return template.to_messages()

def get_mfa_response(mfa_token, duration: int = 900):
    logger.debug("Checking MFA token...")
    if len(mfa_token) != 6:
        return None
    try:
        sts_client = boto3.client('sts',
                            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"))
        response = sts_client.get_session_token(DurationSeconds=duration,
                                                SerialNumber=os.environ.get("AWS_ARN_MFA_DEVICE"),
                                                TokenCode=mfa_token)
        return response
    except Exception as exc:
        logger.error(str(exc))
        return None