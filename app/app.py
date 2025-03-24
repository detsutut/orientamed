import json
from pathlib import Path
import csv
import boto3
import gradio as gr
import os
import logging
from rags import Rag
from dotenv import dotenv_values
import yaml
import random
from boto3 import Session
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import textwrap
import re
import io
from PIL import Image
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from gradio_modal import Modal
import dayplot as dp

# Define the parser
parser = argparse.ArgumentParser()
parser.add_argument('--settings', action="store", dest='settings_file', default='settings.yaml')
parser.add_argument('--sslcert', action="store", dest='ssl_certfile', default=None)
parser.add_argument('--sslkey', action="store", dest='ssl_keyfile', default=None)
parser.add_argument('--debug', action="store", dest='debug', default=False, type=bool)
parser.add_argument('--local', action="store", dest='local', default=False, type=bool)
args = parser.parse_args()


# GLOBAL VARIABLES: SHARED BETWEEN USERS AND SESSIONS
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

with open(args.settings_file) as stream:
    config = yaml.safe_load(stream)

wd = os.path.abspath(os.path.dirname(args.settings_file))
os.chdir(wd)


AWS_SECRETS = config.get("bedrock").get("secrets-path")
GRADIO_SECRETS = config.get("gradio").get("secrets-path")

logger.debug("Initializing RAG...")
rag = Rag(session=Session(),
          model=config.get("bedrock").get("models").get("model-id"),
          embedder=config.get("bedrock").get("embedder-id"),
          vector_store=config.get("vector-db-path"),
          region=config.get("bedrock").get("region"),
          model_pro=config.get("bedrock").get("models").get("pro-model-id"),
          model_low=config.get("bedrock").get("models").get("low-model-id"))

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


def update_rag(mfa_token, use_mfa_session=args.local):
    global rag
    logger.debug("Trying to update rag...")
    mfa_response = get_mfa_response(mfa_token)
    if mfa_response is not None:
        try:
            if use_mfa_session:
                session = boto3.Session(aws_access_key_id=mfa_response['Credentials']['AccessKeyId'],
                                        aws_secret_access_key=mfa_response['Credentials']['SecretAccessKey'],
                                        aws_session_token=mfa_response['Credentials']['SessionToken'])
            else:
                session = Session()
            rag_attempt = Rag(session=session,
                            model=config.get("bedrock").get("models").get("model-id"),
                            embedder=config.get("bedrock").get("embedder-id"),
                            vector_store=config.get("vector-db-path"),
                            region=config.get("bedrock").get("region"),
                            model_pro=config.get("bedrock").get("models").get("pro-model-id"),
                            model_low=config.get("bedrock").get("models").get("low-model-id"))
            rag = rag_attempt
            logger.debug("Rag updated")
            return True, ""
        except Exception as e:
            logger.error("update failed")
            logger.error(str(e))
            return False, ""
    else:
        return False, ""


def upload_file(filepath: str):
    rag.retriever.upload_file(filepath)

def from_list_to_messages(chat:list[dict]):
    template = ChatPromptTemplate([MessagesPlaceholder("history")]).invoke({"history":[(message["role"],message["content"]) for message in chat]})
    return template.to_messages()

LOG_STAT_FILE = "logs/token_usage.json"

def log_token_usage(ip_address: str, input_tokens: int, output_tokens: int):
    """Logs the input and output token usage for a given IP address with timestamps."""
    os.makedirs(os.path.dirname(LOG_STAT_FILE), exist_ok=True)

    if os.path.exists(LOG_STAT_FILE):
        with open(LOG_STAT_FILE, "r") as file:
            data = json.load(file)
    else:
        data = {}

    # Ensure IP has an entry
    if ip_address not in data:
        data[ip_address] = {"input_tokens": [], "output_tokens": []}

    # Append new token usage
    now = datetime.now().isoformat()
    data[ip_address]["input_tokens"].append((input_tokens, now))
    data[ip_address]["output_tokens"].append((output_tokens, now))

    # Save back to file
    with open(LOG_STAT_FILE, "w") as file:
        json.dump(data, file, indent=4)

def get_usage_stats():
    """Computes total users, total input/output tokens, averages, and cumulative daily token usage."""
    if not os.path.exists(LOG_STAT_FILE):
        return {
            "total_users": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_input_tokens_per_user_per_day": 0,
            "avg_output_tokens_per_user_per_day": 0,
            "cumulative_tokens_per_day": [],
            "cumulative_input_tokens_per_day": [],
            "cumulative_output_tokens_per_day": []
        }

    with open(LOG_STAT_FILE, "r") as file:
        data = json.load(file)

    total_users = len(data)
    total_input_tokens = 0
    total_output_tokens = 0
    daily_totals = defaultdict(lambda: [0, 0])  # {date: [input_tokens, output_tokens]}

    for usage in data.values():
        for tokens, timestamp in usage["input_tokens"]:
            date = datetime.fromisoformat(timestamp).date()
            total_input_tokens += tokens
            daily_totals[date][0] += tokens

        for tokens, timestamp in usage["output_tokens"]:
            date = datetime.fromisoformat(timestamp).date()
            total_output_tokens += tokens
            daily_totals[date][1] += tokens

    # Compute averages
    active_days = len(daily_totals)
    avg_input_tokens_per_user_per_day = total_input_tokens / (total_users * active_days) if total_users > 0 and active_days > 0 else 0
    avg_output_tokens_per_user_per_day = total_output_tokens / (total_users * active_days) if total_users > 0 and active_days > 0 else 0

    # Cumulative token count per day
    sorted_dates = sorted(daily_totals.keys())
    cumulative_input = 0
    cumulative_output = 0
    cumulative_tokens_per_day = []
    cumulative_input_tokens_per_day = []
    cumulative_output_tokens_per_day = []

    for date in sorted_dates:
        cumulative_input += daily_totals[date][0]
        cumulative_output += daily_totals[date][1]
        cumulative_tokens_per_day.append((cumulative_input + cumulative_output, date.isoformat()))
        cumulative_input_tokens_per_day.append((cumulative_input, date.isoformat()))
        cumulative_output_tokens_per_day.append((cumulative_output, date.isoformat()))

    return {
        "total_users": total_users,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_tokens_per_user_per_day": round(avg_input_tokens_per_user_per_day),
        "avg_output_tokens_per_user_per_day": round(avg_output_tokens_per_user_per_day),
        "cumulative_tokens_per_day": cumulative_tokens_per_day,
        "cumulative_input_tokens_per_day": cumulative_input_tokens_per_day,
        "cumulative_output_tokens_per_day": cumulative_output_tokens_per_day
    }

import plotly.graph_objects as go

def plot_cumulative_tokens():
    """Plots cumulative token usage over time with stacked bars for input and output tokens and a line for total cumulative tokens using Plotly."""
    stats = get_usage_stats()
    if not stats["cumulative_tokens_per_day"]:
        logger.warning("No data to plot.")
        return

    dates = [datetime.fromisoformat(d) for _, d in stats["cumulative_tokens_per_day"]]
    input_tokens = [t for t, _ in stats["cumulative_input_tokens_per_day"]]
    output_tokens = [t for t, _ in stats["cumulative_output_tokens_per_day"]]
    total_tokens = [t for t, _ in stats["cumulative_tokens_per_day"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(x=dates, y=input_tokens, name="Input Tokens", marker_color='royalblue',width=1000*3600*24*0.5))
    fig.add_trace(go.Bar(x=dates, y=output_tokens, name="Output Tokens", marker_color='lightsalmon',width=1000*3600*24*0.5))
    fig.add_trace(go.Scatter(x=dates, y=total_tokens, mode='lines+markers', name="Total", line=dict(color='darkslategrey')))

    fig.update_layout(
        title="Cumulative Token Usage Over Time",
        xaxis_title="Date",
        yaxis_title="Tokens [tokens/dd]",
        barmode='stack',
        xaxis=dict(tickangle=-45),
        legend_title="Legend",
        template="plotly_white"
    )

    return fig

import matplotlib.pyplot as plt
def plot_cumulative_tokens_heatmap():
    """Plots cumulative token usage over time with stacked bars for input and output tokens and a line for total cumulative tokens using Plotly."""
    stats = get_usage_stats()
    if not stats["cumulative_tokens_per_day"]:
        logger.warning("No data to plot.")
        return

    dates = [datetime.fromisoformat(d) for _, d in stats["cumulative_tokens_per_day"]]
    total_tokens = [t for t, _ in stats["cumulative_tokens_per_day"]]
    fig, ax = plt.subplots(figsize=(15, 6))
    dp.calendar(
        dates,
        total_tokens,
        start_date=datetime.now() - timedelta(days=365),
        end_date=datetime.now(),
        ax=ax,
    )

    return fig


LOG_FILE = "logs/usage_log.json"

#20K = approx 20 cents with most expensive models
def check_ban(ip_address: str, max_tokens: int = 20000) -> bool:
    tokens_consumed, last_access, banned = read_usage_log(ip_address)
    timediff = datetime.now() - last_access
    if tokens_consumed>max_tokens:
        # if you exceeded the tokens quota and less than 24 hours have passed since last call
        # then you are banned
        if timediff.total_seconds()<(24*60*60):
            update_usage_log(ip_address, 0, True)
            return True
        # if you exceeded the tokens quota but your last call was more than 24 hours ago, the ban ends
        else:
            update_usage_log(ip_address, 0, False)
            return False
    # if you did not exceed the tokens quota, you are always good to go
    else:
        return False

def read_usage_log(ip_address: str) -> (int, datetime, bool):
    if not os.path.exists(LOG_FILE):
        return 0, datetime.now(), False
    with open(LOG_FILE, "r") as file:
        data = json.load(file)
    if ip_address in data:
        entry = data[ip_address]
        return entry["tokens_count"], datetime.fromisoformat(entry["last_call"]), entry["banned_flag"]
    return 0, datetime.now(), False


def update_usage_log(ip_address: str, tokens_consumed: int, banned: bool):
    """Updates the usage log, modifying the entry for the given IP address."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as file:
            data = json.load(file)
    else:
        data = {}

    was_banned = data.get(ip_address, {}).get("banned_flag", False)
    tokens_count = data.get(ip_address, {}).get("tokens_count", 0) + tokens_consumed

    if was_banned and not banned:
        tokens_count = 0  # Reset tokens if ban is lifted

    data[ip_address] = {
        "tokens_count": tokens_count,
        "last_call": datetime.now().isoformat(),
        "banned_flag": banned
    }

    with open(LOG_FILE, "w") as file:
        json.dump(data, file, indent=4)

def dot_progress_bar(score, total_dots=7):
    filled_count = round(score * total_dots)
    empty_count = total_dots - filled_count
    filled = "•" * filled_count
    empty = "·" * empty_count
    return f"{filled}{empty} {round(score*100,2)}%"

def reply(message, history, is_admin, enable_rag, additional_context, query_aug, request: gr.Request):
    if rag is None:
        logger.error("LLM not configured")
        gr.Error("Error: LLM not configured")
    else:
        is_banned = check_ban(request.client.host) if not is_admin else False
        if is_banned:
            logger.error("exceeded daily usage limit!")
            gr.Error("Error: exceeded daily usage limit")
            return [gr.ChatMessage(role="assistant", content="Sembra che tu abbia esaurito la tua quota giornaliera. Riprova più tardi.")]
        try:
            if enable_rag:
                response = rag.invoke({"question": message,
                                       "history": from_list_to_messages(history),
                                       "additional_context": additional_context,
                                       "input_tokens_count":0,
                                       "output_tokens_count":0,
                                       "query_aug": query_aug})
                answer = response["answer"]
                input_tokens_count = response["input_tokens_count"]
                output_tokens_count = response["output_tokens_count"]
                update_usage_log(request.client.host, input_tokens_count+output_tokens_count*4, False)
                log_token_usage(request.client.host, input_tokens_count, output_tokens_count)
                answer = re.sub(r"(\[[\d,\s]*\])",r"<sup>\1</sup>",answer)
                citations = {}
                citations_str = ""
                retrieved_documents = response["context"]["docs"]
                retrieved_scores = response["context"]["scores"]
                for i, document in enumerate(retrieved_documents):
                    source = os.path.basename(document.metadata.get("source", ""))
                    content = document.page_content
                    doc_string = f"[{i}] **{source}** - *\"{textwrap.shorten(content,500)}\"* (Confidenza: {dot_progress_bar(retrieved_scores[i])})"
                    citations.update({i: {"source":source, "content":content}})
                    citations_str += ("- "+doc_string+"\n")
                return [gr.ChatMessage(role="assistant", content=answer),
                        gr.ChatMessage(role="assistant", content=citations_str,
                                    metadata={"title": "📖 Linee guida correlate"})]
            else:
                response = rag.generate_norag(message)
                answer = response["answer"]
                return gr.ChatMessage(role="assistant", content=answer)
        except Exception as e:
            logger.error(str(e))
            gr.Error("Error: " + str(e))


def onload(disclaimer_seen:bool, request: gr.Request):
    logging_info = {
        "username":request.username,
        "ip":request.client.host,
        "headers":request.headers,
        "session_hash":request.session_hash,
        "query_params":dict(request.query_params)
    }
    logger.debug(f"Login details: {logging_info}")
    admin_priviledges = request.username == dotenv_values(GRADIO_SECRETS).get("GRADIO_ADMNUSR")
    if not disclaimer_seen:
        modal_visible = True
        disclaimer_seen = True
    else:
        modal_visible = False
    return [admin_priviledges,
            Modal(visible=modal_visible),
            disclaimer_seen,
            gr.Checkbox(interactive=admin_priviledges),
            gr.Checkbox(interactive=admin_priviledges)]


def toggle_interactivity(is_admin):
    logger.debug("Updating admin functionalities")
    return [gr.UploadButton(file_count="single", interactive=is_admin),
            gr.Tab("Stats", visible=is_admin),
            gr.Checkbox(interactive=is_admin),
            gr.Checkbox(interactive=is_admin)
            ]

def update_stats():
    stats_plot = gr.Plot(plot_cumulative_tokens())
    stats = get_usage_stats()
    return [stats_plot, stats['total_users'], stats['avg_input_tokens_per_user_per_day'], stats['avg_output_tokens_per_user_per_day'], round(stats['avg_input_tokens_per_user_per_day']/stats['avg_output_tokens_per_user_per_day'],2)]

custom_theme = gr.themes.Ocean().set(body_background_fill="linear-gradient(to right top, #f2f2f2, #f1f1f4, #f0f1f5, #eff0f7, #edf0f9, #ebf1fb, #e9f3fd, #e6f4ff, #e4f7ff, #e2faff, #e2fdff, #e3fffd)")

def _export(history):
    with open('logs/chat_history.txt', 'w') as f:
        f.write(str(history))
    return 'logs/chat_history.txt'

with gr.Blocks(title="OrientaMed", theme=custom_theme, css="footer {visibility: hidden}", head="""<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Funnel+Display&display=swap" rel="stylesheet">""") as demo:
    with Modal(visible=False) as modal:
        gr.Markdown("""### ⚠️ Disclaimer / Avviso  

**English:**  
By using this application, you acknowledge that you must not enter any patient-sensitive or personally identifiable health information. Data entered may be shared with third-party services, and any disclosure of patient information is your own responsibility.<br/>This platform is not designed to store or process protected health data.  

Conversations made using this AI tool are not intended to replace the opinion of a medical expert and should always be critically evaluated by a qualified professional. The app is intended to be used as a reference by general practitioners and should not be relied upon as the sole source for medical decision-making.

**Italiano:**  
Utilizzando questa applicazione, riconosci di non dover inserire informazioni sensibili sui pazienti o dati sanitari personali identificabili. I dati inseriti possono essere condivisi con servizi di terze parti e qualsiasi divulgazione di informazioni sui pazienti è sotto la tua responsabilità.<br/>Questa piattaforma non è progettata per archiviare o elaborare dati sanitari protetti.

Le conversazioni effettuate utilizzando questo strumento di intelligenza artificiale non sono destinate a sostituire il parere di un esperto medico e dovrebbero essere sempre valutate criticamente da un professionista qualificato. L'app è destinata ad essere utilizzata come riferimento dai medici di base e non deve essere considerata come l'unica fonte per le decisioni mediche.  
""")
    gr.Markdown(f"<center><h1 style=\042font-size:2.7em; font-family: 'Funnel Display', sans-serif;\042><sup style='color: #61e9b7; font-size:1em;'>+</sup>OrientaMed<span style='color: #61e9b7; font-size:1em;'> .</span></h1></center>")
    admin_state = gr.State(False)
    disclaimer_seen = gr.BrowserState(False)
    kb = gr.Checkbox(label="Usa Knowledge Base", value=True, render=False)
    qa = gr.Checkbox(label="Usa Query Augmentation", value=False, render=False)
    with gr.Tab("Chat"):
        history = [{"role": "assistant", "content": random.choice(config.get('gradio').get('greeting-messages'))}]
        chatbot = gr.Chatbot(history, type="messages", show_copy_button=True, layout="panel", resizable=True,
                             avatar_images=(None, config.get('gradio').get("avatar-img")))
        interface = gr.ChatInterface(fn=reply, type="messages",
                                     chatbot=chatbot,
                                     flagging_mode="manual",
                                     flagging_options=config.get('gradio').get('flagging-options'),
                                     flagging_dir="./logs",
                                     save_history=True,
                                     analytics_enabled = False,
                                     examples=[[e] for e in config.get('gradio').get('examples')],
                                     additional_inputs=[admin_state,
                                                        kb,
                                                        qa,
                                                        gr.Textbox(label="Procedure interne, protocolli, anamnesi da affiancare alle linee guida",
                                                                   placeholder="Inserisci qui eventuali procedure interne, protocolli o informazioni aggiuntive riguardanti il paziente. Queste informazioni verranno affiancate alle linee guida nell'elaborazione della risposta.",
                                                                   lines=2,
                                                                   render=False)],
                                     additional_inputs_accordion="Opzioni",
                                     )
        download_btn = gr.Button("Scarica la conversazione", variant='secondary')
        download_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_btn_hidden")
        download_btn.click(fn=_export, inputs=chatbot, outputs=[download_btn_hidden]).then(fn=None, inputs=None,
                                                                                        outputs=None,
                                                                                        js="() => document.querySelector('#download_btn_hidden').click()")


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

    with gr.Tab("Settings") as settings:
        with gr.Group():
            gr.FileExplorer(label="Knowledge Base",
                            root_dir=config.get('kb-folder'),
                            glob=config.get('globs')[0],
                            interactive=False)
            upload_button = gr.UploadButton(file_count="single", interactive=admin_state.value)
        with gr.Group():
            mfa_input = gr.Textbox(label="AWS MFA token", placeholder="123456")
            btn = gr.Button("Confirm")
    with gr.Tab("Stats", visible=False) as stats_tab:
        with gr.Group():
            stats = get_usage_stats()
            with gr.Row():
                stats_users = gr.Textbox(label="Total users", value=f"{stats['total_users']}", interactive=False)
                stats_input = gr.Textbox(label="Average user input [tokens/dd]", value=f"{stats['avg_input_tokens_per_user_per_day']}", interactive=False)
                stats_output = gr.Textbox(label="Average user output [tokens/dd]", value=f"{stats['avg_output_tokens_per_user_per_day']}", interactive=False)
                stats_ratio = gr.Textbox(label="Input/Output ratio", value=f"{round(stats['avg_input_tokens_per_user_per_day']/stats['avg_output_tokens_per_user_per_day'],2)}", interactive=False)
            stats_plot = gr.Plot(plot_cumulative_tokens())
            stats_heat = gr.Plot(plot_cumulative_tokens_heatmap())
        with gr.Group():
            gr.Image(label="Workflow schema", value=Image.open(io.BytesIO(rag.get_image())))

    gr.HTML("<br><div style='display:flex; justify-content:center; align-items:center'><img src='gradio_api/file=./assets/u.png' style='width:7%; min-width : 100px;'><img src='gradio_api/file=./assets/d.png' style='width:7%; padding-left:1%; padding-right:1%; min-width : 100px;'><img src='gradio_api/file=./assets/b.png' style='width:7%; min-width : 100px;'></div><br><div style='display:flex; justify-content:center; align-items:center'><small>© 2024 - 2025 | BMI Lab 'Mario Stefanelli' | DHEAL-COM | <a href='https://github.com/detsutut/dheal-com-rag-demo'>GitHub</a> </small></div>", elem_id="footer")
    upload_button.upload(upload_file, upload_button, None)
    mfa_input.submit(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    btn.click(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    admin_state.change(toggle_interactivity, inputs=admin_state, outputs=[upload_button,stats_tab,kb,qa])
    stats_tab.select(update_stats, inputs=None, outputs=[stats_plot, stats_users, stats_input, stats_output, stats_ratio] )
    demo.load(onload, inputs=disclaimer_seen, outputs=[admin_state,modal,disclaimer_seen,kb,qa])
demo.launch(server_name="0.0.0.0",
            server_port=7860,
            auth=token_auth,
            ssl_keyfile = args.ssl_keyfile,
            ssl_certfile = args.ssl_certfile,
            ssl_verify = False,
            pwa=True,
            favicon_path=config.get('gradio').get('logo-img'),
            allowed_paths=['./assets'])