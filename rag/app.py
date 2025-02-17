import gradio as gr
from bedrock_inference.bedrock import aws_login_mfa
from gradio import ChatMessage
import os
import logging
from rags import Rag, rag_prompts
from dotenv import dotenv_values
import yaml
import random

# GLOBAL VARIABLES: SHARED BETWEEN USERS AND SESSIONS
logger = logging.getLogger(__name__)

with open("./reuma_settings.yaml") as stream:
    config = yaml.safe_load(stream)

os.chdir(config.get("working-dir"))

AWS_SECRETS = config.get("bedrock").get("secrets-path")
GRADIO_SECRETS = config.get("gradio").get("secrets-path")

rag = None


def token_auth(username: str, password: str):
    # ADMIN LOGIN
    if username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR") and len(password) == 6:
        return gradio_init(str(password))
    # OTHER USER LOGIN
    else:
        return username == dotenv_values(GRADIO_SECRETS).get("GRADIO_TESTUSR") and password == dotenv_values(
            GRADIO_SECRETS).get("GRADIO_TESTPWD")


def rag_init(mfa_token: str, model_id="") -> Rag | None:
    logger.info("Logging in AWS...")
    if model_id == "":
        model_id = config.get("bedrock").get("model-id")
    try:
        session = aws_login_mfa(arn=dotenv_values(AWS_SECRETS).get("AWS_ARN_MFA_DEVICE"),
                                aws_access_key_id=dotenv_values(AWS_SECRETS).get("AWS_ACCESS_KEY_ID"),
                                aws_secret_access_key=dotenv_values(AWS_SECRETS).get("AWS_SECRET_ACCESS_KEY"),
                                token=mfa_token,
                                duration=18000)
        rag_attempt = Rag(session=session,
                          model=model_id,
                          embedder=config.get("bedrock").get("embedder-id"),
                          vector_store=config.get("vector-db-path"),
                          region=config.get("bedrock").get("region"))
    except Exception as e:
        logger.error("Login failed")
        logger.error(str(e))
        return None
    logger.info("Login successful")
    return rag_attempt


def gradio_init(mfa_token, model_id=""):
    global rag
    r = rag_init(mfa_token, model_id)
    if r is not None:
        rag = r
        gr.Info("Login successful")
        return True
    else:
        gr.Error("Login failed")
        return False


def gradio_init_upload(mfa_token, model_id):
    admin_flag = gradio_init(mfa_token, model_id)
    return admin_flag, gr.UploadButton(file_count="single", interactive=admin_flag)


def upload_file(filepath: str):
    rag.retriever.upload_file(filepath)


def reply(message, history, enable_rag, additional_context, query_aug):
    if enable_rag:
        response = rag.invoke({"question": message, "additional_context": additional_context, "query_aug": query_aug})
        answer = response["answer"]
        sources = list(set([os.path.basename(context.metadata.get("source", "")) for context in response["context"]]))
        return [ChatMessage(role="assistant", content=answer),
                ChatMessage(role="assistant", content="[" + ", ".join(sources) + "]",
                            metadata={"title": "📖 Sorgenti Utilizzate"})]
    else:
        messages = rag_prompts.norag.invoke({"question": message})
        answer = rag.llm.generate(messages)
        return ChatMessage(role="assistant", content=answer)


def update_state(request: gr.Request):
    return request.username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR")


def toggle_interactivity(is_admin, *components):
    print("updating functionalities")
    print(is_admin)
    print(components)
    """Update the 'interactive' property for all given components."""
    return [comp.update(interactive=is_admin) for comp in components]


with gr.Blocks(theme="earneleh/paris") as demo:
    gr.Markdown("<center><h1>Assistente Reuma Triage Demo</h1></center>")
    admin_state = gr.State(False)
    demo.load(update_state, None, admin_state)
    with gr.Tab("Chat"):
        history = [{"role": "assistant", "content": random.choice(config.get('gradio').get('greeting-messages'))}]
        chatbot = gr.Chatbot(history, type="messages", show_copy_button=True,
                             avatar_images=(None, config.get("avatar-img")))
        with gr.Row():
            with gr.Column(scale=1):
                toggle_rag = gr.Checkbox(label="Usa Knowledge Base", value=True)
                query_aug = gr.Checkbox(label="Usa Query Augmentation", value=False)
            with gr.Column(scale=1):
                additional_context = gr.Textbox(label="Altre Info",
                                                placeholder="Inserisci qui altre informazioni utili")
        gr.ChatInterface(fn=reply, type="messages",
                         chatbot=chatbot,
                         flagging_mode="manual",
                         flagging_options=config.get('gradio').get('flagging_options'),
                         flagging_dir=config.get('working-dir'),
                         save_history=True,
                         examples=[[e] for e in config.get('gradio').get('examples')],
                         additional_inputs=[toggle_rag, additional_context, query_aug])
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
    upload_button.upload(upload_file, upload_button, None)
    btn.click(fn=gradio_init_upload, inputs=[mfa_input, model_input], outputs=[admin_state, upload_button])
    admin_state.change(toggle_interactivity, inputs=[admin_state, upload_button], outputs=[upload_button])
demo.launch(share=False, auth=token_auth, pwa=True)
