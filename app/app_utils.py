import logging
import re
import os
import textwrap
from typing import List
from data_models import RetrievedDocument, Concept

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def parse_references(retrieved_documents: List[RetrievedDocument]=[], graph:bool=False, wrap:int=500):
    references_str = ""
    prefix = "KG" if graph else ""
    for i, document in enumerate(retrieved_documents):
        source = os.path.basename(document.metadata.get("source", "???"))
        title = os.path.basename(document.metadata.get("title", "???"))
        doc_string = f"[{prefix}{i + 1}] **{title.strip()}** , **{source.strip()}** - *\"{textwrap.shorten(document.page_content, wrap)}\"* (Similarità: {dot_progress_bar(document.score, absolute=graph)})"
        if graph:
            doc_string += ("\n"+document.metadata.get("path", "???"))
        references_str += ("- " + doc_string + "\n")
    references_str += ("\nIDS: " + str([d.get("metadata").get("doc_id", "???") for d in retrieved_documents]))
    return references_str

def parse_concepts(title="CONCEPTS", concepts: List[Concept]=[], type="query"):
    concepts_str = f"<strong>{title}</strong>\n<div class='concept_container'>"
    div_id = "cquery" if type=="query" else "canswer"
    for concept in concepts:
        concept_string = f"<div class='concept tooltip' id='{div_id}'>{concept.name.upper()} <span class='tooltip-text'>ID: {concept.id}, Match: {int(concept.match_score * 100)}%</span></div>"
        concepts_str += concept_string
    concepts_str += "</div>"
    return concepts_str

def parse_citations(text:str, retrieved_documents: List[RetrievedDocument]=[], retrieved_documents_kg: List[RetrievedDocument]=[]):
    kg_prefix = "KG"
    def replace_citations(match):
        raw_refs = [ref.strip() for ref in match.group(1).split(',')]
        formatted_strings = []
        for ref in raw_refs:
            pos = int(re.findall(r"\d+", ref)[0]) - 1
            if re.fullmatch(r"\d+", ref):
                document = retrieved_documents[pos]
                source = os.path.basename(document.get("metadata").get("source", "???"))
                title = os.path.basename(document.get("metadata").get("title", "???"))
                formatted_strings.append(f"<span class='tooltip'>{pos + 1}<span class='tooltip-text tooltip-cit'>{title} - {source}</span></span>")
            elif re.fullmatch(kg_prefix+r"\d+", ref):
                document = retrieved_documents_kg[pos]
                source = os.path.basename(document.get("metadata").get("source", "???"))
                title = os.path.basename(document.get("metadata").get("title", "???"))
                formatted_strings.append(
                    f"<span class='tooltip'>{kg_prefix}{pos + 1}<span class='tooltip-text tooltip-cit-kg'>{title} - {source}</span></span>")
        return f"<sup id='cit'><span>[{','.join(formatted_strings)}]</span></sup>"
    pattern = r"\[((?:\s*(?:\d+|"+kg_prefix+r"\d+)\s*,?)+)\]"
    return re.sub(pattern, replace_citations, text)

def dot_progress_bar(score, total_dots=7, absolute=False):
    if absolute:
        filled = "■" + "-•" * int(score) +"-■"
        return f"{filled} {int(score)}"
    else:
        filled_count = round(score * total_dots)
        empty_count = total_dots - filled_count
        filled = "•" * filled_count
        empty = "·" * empty_count
        return f"{filled}{empty} {round(score*100,2)}%"

