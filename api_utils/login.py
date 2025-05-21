import logging
import jwt
from datetime import timedelta, datetime, timezone
import os
import yaml

logger = logging.getLogger('app.'+__name__)

with open(os.getenv("API_SETTINGS_PATH")) as stream:
    api_config = yaml.safe_load(stream)

def authenticate(username: str, password: str):
    logger.debug(f"Authenticating user {username}...")
    return password == os.environ.get("USR_"+username,"")

def create_access_token(username: str, expires_delta: timedelta | None = None):
    logger.info(f"Creating new access token for user {username}...")
    to_encode = {"usr": username}
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=api_config.get("access-token-expire-minutes",15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.environ.get("SECRET_KEY"), algorithm=api_config.get("algorithm",'HS256'))
    return encoded_jwt

def verify_token(token: str):
    logger.debug(f"Veryfing token {token}...")
    try:
        payload = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=[api_config.get("algorithm",'HS256')])
        return payload.get("usr")
    except jwt.ExpiredSignatureError as e:
        logger.error(e)
        return None
    except jwt.DecodeError as e:
        logger.error(e)
        return None
    except Exception as e:
        logger.error(e)
        return None