import json
from pathlib import Path
import csv
import requests
import gradio as gr
import os
import logging
import yaml
import random
import textwrap
import re
import io
from PIL import Image
import argparse
from datetime import datetime
from gradio_modal import Modal
import traceback
import pandas as pd

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
from app_utils import dot_progress_bar

############# GLOBAL VARIABLES ##################
LOG_STAT_FILE = "logs/token_usage.json"
LOG_FILE = "logs/usage_log.json"
LOG_EVAL_FILE = "logs/evaluations.jsonl"
LOG_CHAT_HISTORY = "logs/chat_history.txt"
CUSTOM_THEME = gr.themes.Ocean().set(body_background_fill="linear-gradient(to right top, #f2f2f2, #f1f1f4, #f0f1f5, #eff0f7, #edf0f9, #ebf1fb, #e9f3fd, #e6f4ff, #e4f7ff, #e2faff, #e2fdff, #e3fffd)")

############# RESPONSE DATA MODEL ###############

from pydantic import BaseModel
from typing import List, Optional

class Concept(BaseModel):
    name: str
    id: str
    match_score: float
    semantic_tags: List[str]

class Concepts(BaseModel):
    query: List[Concept]
    answer: List[Concept]

class RetrievedDocuments(BaseModel):
    embeddings: dict
    graphs: dict

class LLMResponse(BaseModel):
    answer: str
    input_tokens_count: int
    output_tokens_count: int
    retrieved_documents: RetrievedDocuments
    concepts: Concepts


def upload_file(filepath: str):
    global RAG
    #RAG.retriever.upload_file(filepath)

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



def reply(message, history, is_admin, access_token, enable_rag, enable_rag_graph, query_aug, retrieve_only, additional_context, request: gr.Request):
    admin_or_test = is_admin or request.username=="test" or args.debug
    is_banned = check_ban(request.client.host) if not admin_or_test else False #don't check if admin or testing
    if is_banned:
        logger.error("exceeded daily usage limit!")
        gr.Error("Error: exceeded daily usage limit")
        return [gr.ChatMessage(role="assistant", content="Sembra che tu abbia esaurito la tua quota giornaliera. Riprova più tardi.")]
    try:
        if enable_rag:
            url = "https://dheal-com.unipv.it:7861/generate"
            data = {'user_input': message,
                      'history': history,
                      "additional_context": additional_context,
                      "augment_query": query_aug,
                      "use_graph": enable_rag_graph,
                      "retrieve_only": retrieve_only,
                      "use_embeddings": enable_rag
            }
            params = {'access_token': access_token}
            res = requests.post(url,
                                     data=json.dumps(data),
                                     params=params,
                                     headers={"ContentType": "application/json"})
            if res.status_code != 200:
                return gr.ChatMessage(role="assistant", content="Unexpected error.")
            else:
                try:
                    response = LLMResponse.model_validate(res.json())
                except Exception as e:
                    return gr.ChatMessage(role="assistant", content="Unexpected error.")
            update_usage_log(request.client.host, response.input_tokens_count+response.output_tokens_count*4, False)
            log_token_usage(request.client.host, response.input_tokens_count, response.output_tokens_count)
            retrieved_documents = response.retrieved_documents.embeddings["docs"]
            retrieved_documents_kg = response.retrieved_documents.graphs["docs"]
            def replace_citations(match):
                raw_refs = [ref.strip() for ref in match.group(1).split(',')]
                formatted_strings=[]
                for ref in raw_refs:
                    pos = int(re.findall(r"\d+", ref)[0])-1
                    if re.fullmatch(r"\d+", ref):
                        document = retrieved_documents[pos]
                        source = os.path.basename(document.get("metadata").get("source", "???"))
                        title = os.path.basename(document.get("metadata").get("title", "???"))
                        formatted_strings.append(f"<span class='tooltip'>{pos+1}<span class='tooltip-text tooltip-cit'>{title} - {source}</span></span>")
                    elif re.fullmatch(r"KG\d+", ref):
                        document = retrieved_documents_kg[pos]
                        source = os.path.basename(document.get("metadata").get("source", "???"))
                        title = os.path.basename(document.get("metadata").get("title", "???"))
                        formatted_strings.append(f"<span class='tooltip'>KG{pos+1}<span class='tooltip-text tooltip-cit-kg'>{title} - {source}</span></span>")
                return f"<sup id='cit'><span>[{','.join(formatted_strings)}]</span></sup>"
            pattern = r"\[((?:\s*(?:\d+|KG\d+)\s*,?)+)\]"
            answer = re.sub(pattern, replace_citations, response.answer)
            ###
            if enable_rag_graph:
                concepts_str = "<strong>QUERY</strong>\n<div class='concept_container'>"
                for concept in response.concepts.query:
                    concept_string = f"<div class='concept tooltip' id='cquery'>{concept.name.upper()} <span class='tooltip-text'>ID: {concept.id}, Match: {int(concept.match_score*100)}%</span></div>"
                    concepts_str += concept_string
                concepts_str += "</div>\n<strong>ANSWER</strong>\n<div class='concept_container'>"
                for concept in response.concepts.answer:
                    concept_string = f"<div class='concept tooltip' id='canswer'>{concept.name.upper()} <span class='tooltip-text'>ID: {concept.id}, Match: {int(concept.match_score*100)}%</span></div>"
                    concepts_str += concept_string
                concepts_str += "</div>"
            ###
            citations = {}
            citations_str = ""
            retrieved_documents = response.retrieved_documents.embeddings["docs"]
            retrieved_scores = response.retrieved_documents.embeddings["scores"]
            for i, document in enumerate(retrieved_documents):
                source = os.path.basename(document.get("metadata").get("source", "???"))
                title = os.path.basename(document.get("metadata").get("title", "???"))
                content = document.get("page_content")
                doc_string = f"[{i+1}] **{title.strip()}** , **{source.strip()}** - *\"{textwrap.shorten(content,500)}\"* (Similarità: {dot_progress_bar(retrieved_scores[i])})"
                citations.update({i: {"source":source, "content":content}})
                citations_str += ("- "+doc_string+"\n")
            citations_str += ("\nIDS: "+str([d.get("metadata").get("doc_id", "???") for d in retrieved_documents]))
            ###
            if enable_rag_graph:
                kg_citations_str = ""
                retrieved_documents = response.retrieved_documents.graphs["docs"]
                retrieved_scores = response.retrieved_documents.graphs["scores"]
                retrieved_paths = response.retrieved_documents.graphs["paths"]
                for i, document in enumerate(retrieved_documents):
                    source = os.path.basename(document.get("metadata").get("source", "???"))
                    title = os.path.basename(document.get("metadata").get("title", "???"))
                    content = document.get("page_content")
                    doc_string = f"[KG{i+1}] **{title.strip()}** , **{source.strip()}** - *\"{textwrap.shorten(content,100)}\"* - (Distanza: {dot_progress_bar(retrieved_scores[i], absolute=True)}) \n{retrieved_paths[i]}"
                    kg_citations_str += ("- "+doc_string+"\n")
                kg_citations_str += ("\nIDS: "+str([d.get("id") for d in retrieved_documents]))
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
            response = {} #RAG.generate_norag(message)
            answer = response["answer"]
            return gr.ChatMessage(role="assistant", content=answer)
    except Exception as e:
        logger.error(traceback.format_exc())
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
    is_admin = False
    admin_priviledges = is_admin or args.debug
    if not disclaimer_seen:
        modal_visible = True
        disclaimer_seen = True
    else:
        modal_visible = False
    return [admin_priviledges,
            Modal(visible=modal_visible),
            disclaimer_seen,
            gr.Checkbox(interactive=True), #debug
            gr.Checkbox(interactive=admin_priviledges),
            logging_info]


def toggle_interactivity(is_admin):
    logger.debug("Updating admin functionalities")
    return [gr.UploadButton(file_count="single", interactive=is_admin),
            gr.Tab("Admin Panel", visible=is_admin),
            gr.Checkbox(interactive=True), #debug
            gr.Checkbox(interactive=is_admin)
            ]

def update_stats():
    stats = get_usage_stats()
    return [gr.Plot(plot_cumulative_tokens()), gr.Plot(get_eval_stats_plot()), stats['total_users'], stats['avg_input_tokens_per_user_per_day'], stats['avg_output_tokens_per_user_per_day'], round(stats['avg_input_tokens_per_user_per_day']/stats['avg_output_tokens_per_user_per_day'],2)]

with gr.Blocks(title=gui_config.get("app_title"), js="function anything() {document.getElementById('options').style.display='none';}", theme=CUSTOM_THEME, css_paths="app.css", head_paths="app_head.html") as demo:
    with Modal(visible=False) as disclaimer_modal:
        gr.Markdown(gui_config.get("disclaimer"))
    gr.Markdown(gui_config.get("app_logo_html"))
    admin_state = gr.State(False)
    disclaimer_seen = gr.BrowserState(False)
    access_token = gr.State(None)
    kb = gr.Checkbox(label="Usa Knowledge Base", value=True, render=False)
    graph = gr.Checkbox(label="Usa Knowledge Graphs", value=True, render=False)
    qa = gr.Checkbox(label="Usa Query Augmentation", value=False, render=False)
    ro = gr.Checkbox(label="Usa solo come recupero fonti", value=False, render=False)
    session_state =gr.State()

    with Modal(visible=True) as loginmodal:
        user = gr.Text(label="Username")
        pw = gr.Text(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_result = gr.Markdown()

    def login(user,pw):
        response =requests.post("https://dheal-com.unipv.it:7861/auth/login",
                      data=json.dumps({"username":user,"password":pw}),
                      headers={"ContentType": "application/json"})
        if response.status_code == 200:
            token = response.json()["access-token"]
            return f"Login successful", token, Modal(visible=False), Modal(visible=True)
        else:
            return "Invalid credentials", None, Modal(visible=True), Modal(visible=False)

    login_btn.click(login,inputs=[user, pw],outputs=[login_result, access_token, loginmodal, disclaimer_modal])
    pw.submit(login, inputs=[user, pw], outputs=[login_result, access_token, loginmodal, disclaimer_modal])

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
                                     additional_inputs=[admin_state, access_token,
                                                        kb,
                                                        graph,
                                                        qa,
                                                        ro,
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
        # Session states are not shared between sessions, therefore, there should be no concurrency issue
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
    gr.HTML("<br><div style='display:flex; justify-content:center; align-items:center'><img src='gradio_api/file=./assets/u.png' style='width:7%; min-width : 100px;'><img src='gradio_api/file=./assets/d.png' style='width:7%; padding-left:1%; padding-right:1%; min-width : 100px;'><img src='gradio_api/file=./assets/b.png' style='width:7%; min-width : 100px;'></div><br><div style='display:flex; justify-content:center; align-items:center'><small>© 2024 - 2025 | BMI Lab 'Mario Stefanelli' | DHEAL-COM | <a href='https://github.com/detsutut/dheal-com-rag-demo'>GitHub</a> </small></div>", elem_id="footer")
    upload_button.upload(upload_file, upload_button, None)
    #mfa_input.submit(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    #btn.click(fn=update_rag, inputs=[mfa_input], outputs=[admin_state,mfa_input])
    #admin_state.change(toggle_interactivity, inputs=admin_state, outputs=[upload_button,stats_tab,kb,qa])
    #stats_tab.select(update_stats, inputs=None, outputs=[stats_plot, eval_plot, stats_users, stats_input, stats_output, stats_ratio] )
    demo.load(onload, inputs=disclaimer_seen, outputs=[admin_state,disclaimer_modal,disclaimer_seen,kb,qa,session_state])

demo.launch(server_name="0.0.0.0",
            server_port=7860,
            auth=None,
            ssl_keyfile = args.ssl_keyfile,
            ssl_certfile = args.ssl_certfile,
            ssl_verify = False,
            pwa=True,
            favicon_path=config.get('gradio').get('logo-img'),
            allowed_paths=['./assets'])