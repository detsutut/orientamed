import json
from pathlib import Path
import csv
import boto3
import gradio as gr
import os
import logging
import yaml
import random
from boto3 import Session
import textwrap
import re
import io
from PIL import Image
import argparse
from datetime import datetime
from gradio_modal import Modal

############# CLI ARGUMENTS ##################
parser = argparse.ArgumentParser()
parser.add_argument('--settings', action="store", dest='settings_file', default='settings.yaml')
parser.add_argument('--sslcert', action="store", dest='ssl_certfile', default=None)
parser.add_argument('--sslkey', action="store", dest='ssl_keyfile', default=None)
parser.add_argument('--debug', action="store", dest='debug', default=False, type=bool)
parser.add_argument('--local', action="store", dest='local', default=False, type=bool)
args = parser.parse_args()

############# LOGGER ##################
logging.basicConfig(level=logging.INFO if args.debug else logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("app")
logger.setLevel(logging.INFO if args.debug else logging.INFO)

############# LOAD CONFIGS ##################
with open(args.settings_file) as stream:
    config = yaml.safe_load(stream)

with open("gui_settings.yaml") as stream:
    gui_config = yaml.safe_load(stream)

############# CHANGE DIRECTORY ##################
wd = os.path.abspath(os.path.dirname(args.settings_file))
os.chdir(wd)

############# LOCAL IMPORTS ##################
from app_logging import get_usage_stats, log_token_usage, read_usage_log, plot_daily_tokens_heatmap, update_usage_log, plot_cumulative_tokens, export_history, get_eval_stats_plot
from app_utils import get_mfa_response, token_auth, dot_progress_bar, get_admin_username, from_list_to_messages
from rags import Rag

############# GLOBAL VARIABLES ##################
LOG_STAT_FILE = "logs/token_usage.json"
LOG_FILE = "logs/usage_log.json"
LOG_EVAL_FILE = "logs/evaluations.jsonl"
LOG_CHAT_HISTORY = "logs/chat_history.txt"
CUSTOM_THEME = gr.themes.Ocean().set(body_background_fill="linear-gradient(to right top, #f2f2f2, #f1f1f4, #f0f1f5, #eff0f7, #edf0f9, #ebf1fb, #e9f3fd, #e6f4ff, #e4f7ff, #e2faff, #e2fdff, #e3fffd)")
RAG = Rag(session=Session(),
          model=config.get("bedrock").get("models").get("model-id"),
          embedder=config.get("bedrock").get("embedder-id"),
          vector_store=config.get("vector-db-path"),
          region=config.get("bedrock").get("region"),
          model_pro=config.get("bedrock").get("models").get("pro-model-id"),
          model_low=config.get("bedrock").get("models").get("low-model-id"))


def update_rag(mfa_token, use_mfa_session=args.local):
    global RAG
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
            RAG = rag_attempt
            logger.debug("Rag updated")
            return True, ""
        except Exception as e:
            logger.error("update failed")
            logger.error(str(e))
            return False, ""
    else:
        return False, ""

def upload_file(filepath: str):
    global RAG
    RAG.retriever.upload_file(filepath)

#20K = approx 20 cents with most expensive models
def check_ban(ip_address: str, max_tokens: int = 20000) -> bool:
    tokens_consumed, last_access, banned = read_usage_log(ip_address)
    timediff = datetime.now() - last_access
    if tokens_consumed>max_tokens:
        # if tokens quota exceeded less than 24 hours ago --> you are banned
        if timediff.total_seconds()<(24*60*60):
            update_usage_log(ip_address, 0, True)
            return True
        # if tokens quota exceeded less than 24 hours ago --> you ban ends
        else:
            update_usage_log(ip_address, 0, False)
            return False
    else:
        return False

def reply(message, history, is_admin, enable_rag, enable_rag_graph, query_aug, additional_context, request: gr.Request):
    global RAG
    admin_or_test = is_admin or request.username=="test"
    is_banned = check_ban(request.client.host) if not admin_or_test else False #don't check if admin or testing
    if is_banned:
        logger.error("exceeded daily usage limit!")
        gr.Error("Error: exceeded daily usage limit")
        return [gr.ChatMessage(role="assistant", content="Sembra che tu abbia esaurito la tua quota giornaliera. Riprova più tardi.")]
    try:
        if enable_rag:
            response = RAG.invoke({"query": message,
                                   "history": from_list_to_messages(history),
                                   "additional_context": additional_context,
                                   "input_tokens_count":0,
                                   "output_tokens_count":0,
                                   "query_aug": query_aug,
                                   "use_graph": enable_rag_graph})
            answer = response["answer"]
            input_tokens_count = response["input_tokens_count"]
            output_tokens_count = response["output_tokens_count"]
            update_usage_log(request.client.host, input_tokens_count+output_tokens_count*4, False)
            log_token_usage(request.client.host, input_tokens_count, output_tokens_count)
            answer = re.sub(r"(\[[\d,\s]*\])",r"<sup>\1</sup>",answer)
            ###
            if enable_rag_graph:
                concepts_str = ""
                retrieved_concepts = response["query_concepts"]+response["answer_concepts"]
                for concept in retrieved_concepts:
                    concept_string = f"**{concept['name'].upper()}**: {concept['semantic_tags']} ({dot_progress_bar(concept['match_score']/100)})"
                    concepts_str += ("- "+concept_string+"\n")
            ###
            citations = {}
            citations_str = ""
            retrieved_documents = response["context"]["docs"]
            retrieved_scores = response["context"]["scores"]
            for i, document in enumerate(retrieved_documents):
                source = os.path.basename(document.metadata.get("source", ""))
                content = document.page_content
                doc_string = f"[{i+1}] **{source}** - *\"{textwrap.shorten(content,500)}\"* (Confidenza: {dot_progress_bar(retrieved_scores[i])})"
                citations.update({i: {"source":source, "content":content}})
                citations_str += ("- "+doc_string+"\n")
            ###
            if enable_rag_graph:
                kg_citations_str = ""
                retrieved_documents = response["kg_context"]["docs"]
                retrieved_scores = response["kg_context"]["scores"]
                retrieved_paths = response["kg_context"]["paths"]
                for i, document in enumerate(retrieved_documents):
                    source = os.path.basename(document.metadata.get("source", ""))
                    content = document.page_content
                    doc_string = f"[KG{i+1}] **{source}** - *\"{textwrap.shorten(content,100)}\"* - (Distanza: {dot_progress_bar(retrieved_scores[i], absolute=True)}) \n{retrieved_paths[i]}"
                    kg_citations_str += ("- "+doc_string+"\n")
                return [gr.ChatMessage(role="assistant", content=answer),
                        gr.ChatMessage(role="assistant", content=concepts_str,
                                       metadata={"title": "🌐 SNOMED Concepts"}),
                        gr.ChatMessage(role="assistant", content=kg_citations_str,
                                       metadata={"title": "🌐 Percorsi collegati"}),
                        gr.ChatMessage(role="assistant", content=citations_str,
                                    metadata={"title": "📖 Linee guida correlate"})]
            else:
                return [gr.ChatMessage(role="assistant", content=answer),
                        gr.ChatMessage(role="assistant", content=citations_str,
                                    metadata={"title": "📖 Linee guida correlate"})]
        else:
            response = RAG.generate_norag(message)
            answer = response["answer"]
            return gr.ChatMessage(role="assistant", content=answer)
    except Exception as e:
        logger.error(str(e))
        gr.Error("Error: " + str(e))

def usereval(*args):
    global eval_components
    session = args[-1]
    data = {"ip": session["ip"],
            "username": session["username"],
            "session_hash": session["session_hash"],
            "timestamp": str(datetime.now()),
            "liked": args[-3],
            "evaluation": dict(zip([c[0] for c in eval_components], args[:-3])),
            "conversation": json.dumps(args[-2])}
    with open("logs/evaluations.jsonl", "a") as file:
        file.write(json.dumps(data) + '\n')
    return [None] * len(args[:-3]) + [Modal(visible=False)]

def onload(disclaimer_seen:bool, request: gr.Request):
    logging_info = {
        "username":request.username,
        "ip":request.client.host,
        "headers":request.headers,
        "session_hash":request.session_hash,
        "query_params":dict(request.query_params)
    }
    logger.debug(f"Login details: {logging_info}")
    admin_priviledges = request.username == get_admin_username()
    if not disclaimer_seen:
        modal_visible = True
        disclaimer_seen = True
    else:
        modal_visible = False
    return [admin_priviledges,
            Modal(visible=modal_visible),
            disclaimer_seen,
            gr.Checkbox(interactive=admin_priviledges),
            gr.Checkbox(interactive=admin_priviledges),
            logging_info]


def toggle_interactivity(is_admin):
    logger.debug("Updating admin functionalities")
    return [gr.UploadButton(file_count="single", interactive=is_admin),
            gr.Tab("Admin Panel", visible=is_admin),
            gr.Checkbox(interactive=is_admin),
            gr.Checkbox(interactive=is_admin)
            ]

def update_stats():
    stats = get_usage_stats()
    return [gr.Plot(plot_cumulative_tokens()), gr.Plot(get_eval_stats_plot()), stats['total_users'], stats['avg_input_tokens_per_user_per_day'], stats['avg_output_tokens_per_user_per_day'], round(stats['avg_input_tokens_per_user_per_day']/stats['avg_output_tokens_per_user_per_day'],2)]

with gr.Blocks(title=gui_config.get("app_title"), js="function anything() {document.getElementById('options').style.display='none';}", theme=CUSTOM_THEME, css_paths="app.css", head_paths="app_head.html") as demo:
    with Modal(visible=False) as modal:
        gr.Markdown(gui_config.get("disclaimer"))
    gr.Markdown(gui_config.get("app_logo_html"))
    admin_state = gr.State(False)
    disclaimer_seen = gr.BrowserState(False)
    kb = gr.Checkbox(label="Usa Knowledge Base", value=True, render=False)
    graph = gr.Checkbox(label="Usa Knowledge Graphs", value=True, render=False)
    qa = gr.Checkbox(label="Usa Query Augmentation", value=False, render=False)
    session_state =gr.State()

    with Modal(visible=False) as evalmodal:
        like_dislike_state = gr.State("")
        with gr.Row():
            ec_main_text = gr.Textbox(label="Message", info="Text under evaluation", lines=2, interactive=False, elem_id="eval_main_text")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Medical Accuracy", open=True, elem_id="eval1"):
                    ma1=gr.Radio(choices=[1, 2, 3, 4, 5], interactive=True, value=None, label="Question Comprehension", info="1 = Misunderstood, 5 = Understood")
                    ma2=gr.Radio(choices=[1, 2, 3, 4, 5], interactive=True, value=None, label="Logical Reasoning", info="1 = Illogical, 5 = Logical")
                    ma3=gr.Radio(choices=[1, 2, 3, 4, 5], interactive=True, value=None, label="Alignment with Clinical Guidelines", info="1 = Not Aligned, 5 = Aligned")
                    ma4=gr.Radio(choices=[1, 2, 3, 4, 5], interactive=True, value=None, label="Completeness", info="1 = Incomplete, 5 = Complete")
                with gr.Accordion("Safety", open=False, elem_id="eval2"):
                    sa1=gr.Slider(minimum=1, maximum=5, step=1, value=-1, show_reset_button=False, interactive=True, label="Possibility of Harm", info="1 = Low, 5 = High")
                    sa2=gr.Slider(minimum=1, maximum=5, step=1, value=-1, show_reset_button=False, interactive=True, label="Extent of Possible harm", info="1 = No Harm, 5 = Severe Harm")
                with gr.Accordion("Communication", open=False, elem_id="eval3"):
                    co1=gr.Slider(minimum=1, maximum=5, step=1, value=-1, show_reset_button=False, interactive=True, label="Tone", info="1 = Inappropriate, 5 = Appropriate")
                    co2=gr.Slider(minimum=1, maximum=5, step=1, value=-1, show_reset_button=False, interactive=True, label="Coherence", info="1 = Incoherent, 5 = Coherent")
                    co3=gr.Slider(minimum=1, maximum=5, step=1, value=-1, show_reset_button=False, interactive=True, label="Helpfulness", info="1 = Unhelpful, 5 = Helpful")
            with gr.Column(scale=2):
                with gr.Accordion("Citations",open=False) as ec_cit_accordion:
                    ec_citations = gr.Textbox(label="Citations", placeholder="No citations", lines=10, show_label=False, interactive=True, elem_id="citations_eval")
                cm = gr.Textbox(label="Comments", info="Write your comments here. What could be improved? How?", interactive=True, lines=5)
                tb = gr.Textbox(label="Your answer", info="How would you answer instead? You can copy and paste the original text here to add or edit info or completely rewrite it from scratch.", interactive=True, lines=5)
                submiteval_btn = gr.Button("SUBMIT", variant="primary", elem_id="eval_button_submit")

    eval_components = [("question_comprehension",ma1), ("logical",ma2), ("guidelines_aligment",ma3),
                       ("completeness",ma4),("harm",sa1),("harm_extent", sa2), ("tone",co1),
                       ("coherence",co2),("helpfulness",co3),("main_text",ec_main_text),("citations",ec_citations),("comments",cm),("answer",tb)]

    evalmodal.blur(lambda: [None]*len(eval_components), outputs=[t[1] for t in eval_components])

    with gr.Tab("Chat"):
        history = [{"role": "assistant", "content": random.choice(config.get('gradio').get('greeting-messages'))}]
        chatbot = gr.Chatbot(history, type="messages", show_copy_button=False, show_label=False, layout="panel", resizable=True,
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
                                                        graph,
                                                        qa,
                                                        gr.Textbox(label="Procedure interne, protocolli, anamnesi da affiancare alle linee guida",
                                                                   info="Queste informazioni verranno affiancate alle linee guida nell'elaborazione della risposta e citate con il numero [0].",
                                                                   placeholder="Inserisci qui eventuali procedure interne, protocolli o informazioni aggiuntive riguardanti il paziente.",
                                                                   lines=4,
                                                                   render=False)],
                                     additional_inputs_accordion=gr.Accordion(label="Opzioni", open=False, elem_id="options"),
                                     )
        download_btn = gr.Button("Scarica la conversazione", variant='secondary')
        download_btn_hidden = gr.DownloadButton(visible=False, elem_id="download_btn_hidden")
        download_btn.click(fn=export_history, inputs=chatbot, outputs=[download_btn_hidden]).then(fn=None, inputs=None,
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

        def open_modal(data: gr.LikeData):
            if data.liked=="":
                return ["","","",gr.Accordion(open=False),Modal(visible=False)]
            if len(data.value)>1:
                citations = "\n".join(data.value[1:])
            else:
                citations = ""
            return [str(data.liked), gr.Textbox(value=data.value[0]), gr.Textbox(value=citations), gr.Accordion(open=False), Modal(visible=True)]

        interface.chatbot.like(open_modal, outputs=[like_dislike_state, ec_main_text, ec_citations, ec_cit_accordion, evalmodal])
        submiteval_btn.click(usereval, [t[1] for t in eval_components]+[like_dislike_state,chatbot,session_state], [t[1] for t in eval_components]+[evalmodal])

    with gr.Tab("Settings") as settings:
        with gr.Group():
            gr.FileExplorer(label="Knowledge Base",
                            root_dir=config.get('kb-folder'),
                            glob=config.get('globs')[0],
                            interactive=False)
            upload_button = gr.UploadButton(file_count="single", interactive=admin_state.value)
        with gr.Group():
            mfa_input = gr.Textbox(label="AWS MFA token", placeholder="123456", type="password")
            btn = gr.Button("Confirm")
    with gr.Tab("Admin Panel", visible=False) as stats_tab:
        with gr.Group():
            stats = get_usage_stats()
            with gr.Row():
                stats_users = gr.Textbox(label="Total users", value=f"{stats['total_users']}", interactive=False)
                stats_input = gr.Textbox(label="Average user input [tokens/dd]", value=f"{stats['avg_input_tokens_per_user_per_day']}", interactive=False)
                stats_output = gr.Textbox(label="Average user output [tokens/dd]", value=f"{stats['avg_output_tokens_per_user_per_day']}", interactive=False)
                stats_ratio = gr.Textbox(label="Input/Output ratio", value=f"{round(stats['avg_input_tokens_per_user_per_day']/stats['avg_output_tokens_per_user_per_day'],2)}", interactive=False)
            with gr.Row():
                stats_plot = gr.Plot(plot_cumulative_tokens())
                eval_plot = gr.Plot(get_eval_stats_plot())
            stats_heat = gr.Plot(plot_daily_tokens_heatmap())
            with gr.Row():
                usage_log_btn = gr.DownloadButton("Usage Log Download", value="logs/usage_log.json")
                evaluation_log_btn = gr.DownloadButton("Evaluations Download", value="logs/evaluations.jsonl")
        with gr.Group():
            gr.Image(label="Workflow schema", value=Image.open(io.BytesIO(RAG.get_image())))

    gr.HTML("<br><div style='display:flex; justify-content:center; align-items:center'><img src='gradio_api/file=./assets/u.png' style='width:7%; min-width : 100px;'><img src='gradio_api/file=./assets/d.png' style='width:7%; padding-left:1%; padding-right:1%; min-width : 100px;'><img src='gradio_api/file=./assets/b.png' style='width:7%; min-width : 100px;'></div><br><div style='display:flex; justify-content:center; align-items:center'><small>© 2024 - 2025 | BMI Lab 'Mario Stefanelli' | DHEAL-COM | <a href='https://github.com/detsutut/dheal-com-rag-demo'>GitHub</a> </small></div>", elem_id="footer")
    upload_button.upload(upload_file, upload_button, None)
    mfa_input.submit(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    btn.click(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    admin_state.change(toggle_interactivity, inputs=admin_state, outputs=[upload_button,stats_tab,kb,qa])
    stats_tab.select(update_stats, inputs=None, outputs=[stats_plot, eval_plot, stats_users, stats_input, stats_output, stats_ratio] )
    demo.load(onload, inputs=disclaimer_seen, outputs=[admin_state,modal,disclaimer_seen,kb,qa,session_state])

demo.launch(server_name="0.0.0.0",
            server_port=7860,
            auth=token_auth,
            ssl_keyfile = args.ssl_keyfile,
            ssl_certfile = args.ssl_certfile,
            ssl_verify = False,
            pwa=True,
            favicon_path=config.get('gradio').get('logo-img'),
            allowed_paths=['./assets'])