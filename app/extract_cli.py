import os

from translator import Translators
from concept_mapper import SNOMEDMapper
import argparse
import pandas as pd

############# TOOLS ##################
mapper = SNOMEDMapper("dictionaries/reuma_dict_extended.csv")
translators = Translators(secrets_path="secrets.env")

def extract(text:str,
            fuzzy_threshold:int=100,
            filter_tags: list[str]=[],
            exclude: bool=False,
            use_premium:bool=False)->pd.DataFrame:
    language = translators.detect_language(text)
    if language != "en":
        translated_text = translators.translate(text, use_premium="first" if use_premium else "never")
    else:
        translated_text = text
    concepts = mapper.map_text_to_snomed(translated_text, fuzzy_threshold=fuzzy_threshold / 100,
                                         filter_tags=filter_tags,
                                         exclude=exclude,
                                         unique=True)
    if not concepts:
        return None
    else:
        df = pd.DataFrame(concepts)
        merged_df = (
            df.groupby(['id', 'name', 'match_score'], as_index=False)
            .agg({'semantic_tags': lambda tags: sum(tags, [])})
        )
        return merged_df

if __name__ == "__main__":
    ############# CLI ARGUMENTS ##################
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action="store", dest='text', default='', type=str)
    parser.add_argument('--filter_tags', action="store", dest='filter_tags', default=[], type=list[str])
    parser.add_argument('--exclude', action="store", dest='exclude', default=False, type=bool)
    parser.add_argument('--overlap', action="store", dest='fuzzy_threshold', default=100, type=int)
    parser.add_argument('--premium', action="store", dest='use_premium', default=False, type=bool)
    args = parser.parse_args()

    os.chdir(os.path.dirname(__file__))
    print(args.__dict__)
    results_dataframe = extract(**args.__dict__)
    print(results_dataframe.to_dict())