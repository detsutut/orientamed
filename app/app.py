import json
from datetime import datetime
from pathlib import Path
import csv
import gradio as gr
from bedrock_inference.bedrock import aws_login_mfa
import os
import logging
from rags import Rag, RagPrompts
from dotenv import dotenv_values
import yaml
import random
from boto3 import Session
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import argparse

# Define the parser
parser = argparse.ArgumentParser()
parser.add_argument('--settings', action="store", dest='settings_file', default='settings.yaml')
args = parser.parse_args()


# GLOBAL VARIABLES: SHARED BETWEEN USERS AND SESSIONS
logger = logging.getLogger(__name__)

with open(args.settings_file) as stream:
    config = yaml.safe_load(stream)

wd = os.path.abspath(os.path.dirname(args.settings_file))
os.chdir(wd)


AWS_SECRETS = config.get("bedrock").get("secrets-path")
GRADIO_SECRETS = config.get("gradio").get("secrets-path")

logger.info("Initializing RAG...")
rag = Rag(session=Session(),
          model=config.get("bedrock").get("model-id"),
          embedder=config.get("bedrock").get("embedder-id"),
          vector_store=config.get("vector-db-path"),
          region=config.get("bedrock").get("region"))


def start_mfa_session(mfa_token, duration: int = 900):
    logger.info("Checking MFA token...")
    try:
        session = aws_login_mfa(arn=dotenv_values(AWS_SECRETS).get("AWS_ARN_MFA_DEVICE"),
                                aws_access_key_id=dotenv_values(AWS_SECRETS).get("AWS_ACCESS_KEY_ID"),
                                aws_secret_access_key=dotenv_values(AWS_SECRETS).get("AWS_SECRET_ACCESS_KEY"),
                                token=mfa_token,
                                duration=duration)
        return session
    except Exception as e:
        logger.error(str(e))
        return None


def token_auth(username: str, password: str):
    logger.info(f"Login attempt from user '{username}'")
    # ADMIN LOGIN
    if username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR") and len(password) == 6:
        session = start_mfa_session(str(password))
        return type(session) == Session
    # OTHER USER LOGIN
    else:
        check_user = username == dotenv_values(GRADIO_SECRETS).get("GRADIO_TESTUSR")
        check_password = password == dotenv_values(GRADIO_SECRETS).get("GRADIO_TESTPWD")
        return check_user and check_password


def update_rag(mfa_token, model_id=""):
    global rag
    session = start_mfa_session(str(mfa_token))
    if type(session) is Session:
        if model_id=="":
            model_id = config.get("bedrock").get("model-id")
        logger.info("Trying to update rag...")
        try:
            rag_attempt = Rag(session=session,
                            model=model_id,
                            embedder=config.get("bedrock").get("embedder-id"),
                            vector_store=config.get("vector-db-path"),
                            region=config.get("bedrock").get("region"))
        except Exception as e:
            logger.error("update failed")
            logger.error(str(e))
        rag = rag_attempt
        logger.info("Rag updated")
        return True
    else:
        return False


def upload_file(filepath: str):
    rag.retriever.upload_file(filepath)

def from_list_to_messages(chat:list[dict]):
    template = ChatPromptTemplate([MessagesPlaceholder("history")]).invoke({"history":[(message["role"],message["content"]) for message in chat]})
    return template.to_messages()

def reply(message, history, enable_rag, additional_context, query_aug, request: gr.Request):
    if rag is None:
        logger.error("LLM not configured")
        gr.Error("Error: LLM not configured")
    else:
        try:
            if enable_rag:
                response = rag.invoke({"question": message,
                                       "history": from_list_to_messages(history),
                                       "additional_context": additional_context,
                                       "query_aug": query_aug})
                answer = response["answer"]
                sources = list(set([os.path.basename(context.metadata.get("source", "")) for context in response["context"]]))
                return [gr.ChatMessage(role="assistant", content=answer),
                        gr.ChatMessage(role="assistant", content="[" + ", ".join(sources) + "]",
                                    metadata={"title": "📖 Sorgenti Utilizzate"})]
            else:
                messages = RagPrompts.norag.invoke({"question": message})
                answer = rag.llm.generate(messages)
                return gr.ChatMessage(role="assistant", content=answer)
        except Exception as e:
            logger.error(str(e))
            gr.Error("Error: " + str(e))


def onload(request: gr.Request):
    logging_info = {
        "username":request.username,
        "ip":request.client.host,
        "headers":request.headers,
        "session_hash":request.session_hash,
        "query_params":dict(request.query_params)
    }
    logger.info(f"Login details: {logging_info}")
    return request.username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR")


def toggle_interactivity(is_admin):
    logger.info("Updating admin functionalities")
    return gr.UploadButton(file_count="single", interactive=is_admin)

custom_theme = gr.themes.Ocean().set(body_background_fill="linear-gradient(to right top, #f2f2f2, #f1f1f4, #f0f1f5, #eff0f7, #edf0f9, #ebf1fb, #e9f3fd, #e6f4ff, #e4f7ff, #e2faff, #e2fdff, #e3fffd)")


with gr.Blocks(title="OrientaMed", theme=custom_theme, css="footer {visibility: hidden}") as demo:
    gr.Markdown(f"<center><h1><img src='gradio_api/file={config.get('gradio').get('logo-img')}' style='height:1.2em; display:inline-block;'> OrientaMed - Demo</h1></center>")
    admin_state = gr.State(False)
    with gr.Tab("Chat"):
        history = [{"role": "assistant", "content": random.choice(config.get('gradio').get('greeting-messages'))}]
        chatbot = gr.Chatbot(history, type="messages", show_copy_button=True, layout="panel",
                             avatar_images=(None, config.get('gradio').get("avatar-img")))
        interface = gr.ChatInterface(fn=reply, type="messages",
                                     chatbot=chatbot,
                                     flagging_mode="manual",
                                     flagging_options=config.get('gradio').get('flagging-options'),
                                     flagging_dir="./logs",
                                     save_history=True,
                                     analytics_enabled = False,
                                     examples=[[e] for e in config.get('gradio').get('examples')],
                                     additional_inputs=[gr.Checkbox(label="Usa Knowledge Base", value=True, render=False),
                                                        gr.Checkbox(label="Usa Query Augmentation", value=False, render=False),
                                                        gr.Textbox(label="Altre Info", placeholder="Inserisci qui altre informazioni utili", render=False)],
                                     additional_inputs_accordion=gr.Accordion("Opzioni",open=False),
                                     )

        # Workaround to take into account username and ip address in logs
        # for some reason, overriding the like callback causes two consecutive calls at few ms of distance
        # we set a Session state flag to suppress the second call
        # Session states are not shared between sessions, therefore there should be no concurrency issue
        double_log_flag = gr.State(True)
        def manual_logger(data: gr.LikeData, messages: list, double_log_flag, request: gr.Request):
            if double_log_flag:
                log_filepath = "./logs/log_"+request.username.replace(".","_").replace("/","_") +"_"+ request.client.host.replace(".", "_") + ".csv"
                is_new = not Path(log_filepath).exists()
                csv_data = [json.dumps(messages), data.value, data.index, data.liked, request.client.host, request.username, str(datetime.now())]
                with open(log_filepath, "a", encoding="utf-8", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if is_new:
                        writer.writerow(["conversation", "message", "index", "flag", "host",  "username", "timestamp"])
                    writer.writerow(gr.utils.sanitize_list_for_csv(csv_data))
            return not double_log_flag
        interface.chatbot.like(manual_logger, [chatbot, double_log_flag], double_log_flag)

    with gr.Tab("Settings"):
        with gr.Group():
            gr.FileExplorer(label="Knowledge Base",
                            root_dir=config.get('kb-folder'),
                            glob=config.get('globs')[0],
                            interactive=False)
            upload_button = gr.UploadButton(file_count="single", interactive=admin_state.value)
        with gr.Group():
            mfa_input = gr.Textbox(label="AWS MFA token", placeholder="123456")
            model_input = gr.Textbox(label="Bedrock Model ID", placeholder="")
            btn = gr.Button("Confirm")
    gr.HTML("<br><div style='display:flex; justify-content:center; align-items:center'><img src='gradio_api/file=./assets/u.png' style='width:7%; min-width : 100px;'><img src='gradio_api/file=./assets/d.png' style='width:7%; padding-left:1%; padding-right:1%; min-width : 100px;'><img src='gradio_api/file=./assets/b.png' style='width:7%; min-width : 100px;'></div>", elem_id="footer")
    upload_button.upload(upload_file, upload_button, None)
    btn.click(fn=update_rag, inputs=[mfa_input, model_input], outputs=admin_state)
    admin_state.change(toggle_interactivity, inputs=admin_state, outputs=upload_button)
    demo.load(onload, inputs=None, outputs=admin_state)
demo.launch(server_name="0.0.0.0",
            auth=token_auth,
            pwa=True,
            favicon_path=config.get('gradio').get('logo-img'),
            allowed_paths=['./assets'])