import json
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz
from flashtext import KeywordProcessor
import re
from typing import List



class SNOMEDMapper:
    def __init__(self, csv_path):
        """
        Load SNOMED terms from a CSV file with 'FSN' and 'ID' columns.
        """
        df = pd.read_csv(csv_path)

        # Ensure correct column names
        if 'FSN' not in df.columns or 'ID' not in df.columns:
            raise ValueError("CSV must have 'FSN' and 'ID' columns.")

        # Convert to dictionary {FSN: ID} (all lowercase for case-insensitive matching)
        snomed_dict = defaultdict(list)
        for _, row in df.iterrows():
            term, semantic_tags = self.__preprocess_FSN__(row["FSN"].strip().lower())
            snomed_dict[term].append({"concept_id": str(row["ID"]).strip(),
                                      "semantic_tags": semantic_tags})
        self.snomed_dict = snomed_dict
        self.keyword_processor = KeywordProcessor()

        # Add terms for fast exact matching
        for term in self.snomed_dict:
            self.keyword_processor.add_keyword(term)

    def __preprocess_FSN__(self, fsn:str):
        semantic_tags = [tag.strip() for tag in re.findall("\((.*?)\)", fsn)]
        term = re.sub("\(.*\)","", fsn).strip()
        return term, semantic_tags



    def exact_match(self, text):
        """Find exact SNOMED term matches in the text."""
        found_terms = self.keyword_processor.extract_keywords(text.lower())
        matches = []
        for term in found_terms:
            concepts = self.snomed_dict[term]
            for concept in concepts:
                matches.append({"name":term, "id": concept["concept_id"], "semantic_tags": concept["semantic_tags"], "match_score": 100})
        return matches

    def fuzzy_match(self, text, threshold=85):
        """Find fuzzy SNOMED term matches in the text using RapidFuzz."""
        text = text.lower()
        matches = []
        for term in self.snomed_dict:
            score = fuzz.partial_ratio(term, text)
            if score >= threshold:
                concepts = self.snomed_dict[term]
                for concept in concepts:
                    matches.append({"name":term, "id": concept["concept_id"], "semantic_tags": concept["semantic_tags"], "match_score": score})
        return sorted(matches, key=lambda x: -x["match_score"])  # Sort by best match

    def map_text_to_snomed(self, text, fuzzy_threshold=100, filter_tags=[], exclude=False, unique=False):
        """
        Maps input text to SNOMED terms using:
        1. Exact matching (preferred if found)
        2. Fuzzy matching (if no exact matches)
        """
        matches = self.exact_match(text) if fuzzy_threshold==100 else self.fuzzy_match(text, fuzzy_threshold)
        #shortcut to avoid useless loop when filtering is not required
        if filter_tags:
            filtered_matches = []
            for match in matches:
                filter_tags_detected = any([tag in filter_tags for tag in match["semantic_tags"]])
                if (filter_tags_detected and not exclude) or (not filter_tags_detected and exclude):
                    filtered_matches.append(match)
            matches = filtered_matches
        if unique:
            dict_strings = [json.dumps(d, sort_keys=True) for d in matches]
            unique_matches = [json.loads(x) for x in list(set(dict_strings))]
            matches = unique_matches
        return matches


# Example Usage
csv_path = "data/reuma_dict.csv"  # Replace with your actual CSV file path
mapper = SNOMEDMapper(csv_path)
