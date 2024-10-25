import bz2
from collections import Counter
import gzip
from math import log
import re
from pathlib import Path

from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer


import numpy as np
import pandas as pd
import spacy
import textstat

import saffine.multi_detrending as md
import neurokit2 as nk

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# import things for feature run
from utils_feature_extraction import *

import json
from pathlib import Path

from lexical_diversity import lex_div as ld

import neurokit2 as nk
from nltk.tokenize import sent_tokenize, word_tokenize

import nltk
import pandas as pd
import spacy
from tqdm import tqdm

# and SA
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

def check_args(args):
    """
    checks whether the arguments provided are compatible / are supported by the pipe
    """
    # checking that syuzhet is not trying to be run on ucloud
    if args.ucloud == True and "syuzhet" in args.sentiment_method:
        return "you cannot do syuzhet on ucloud"
    # checking that vader and syuzhet are not being used with danish books
    elif args.lang == "danish" and args.sentiment_method != "afinn":
        return f"you cannot do {args.sentiment_method} in {args.lang}"
    else:
        return None


def get_nlp(lang: str):
    """
    checks if the spacy model is loaded, errors if not
    """
    if lang == "english":
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "en_core_web_sm not downloaded, run python3 -m spacy download en_core_web_sm"
            ) from e

    elif lang == "danish":
        try:
            nlp = spacy.load("da_core_news_sm")

        except OSError as e:
            raise OSError(
                "da_core_news_sm not downloaded, run python3 -m spacy download da_core_news_sm"
            ) from e

    return nlp


def extract_text(filename: str) -> str:
    """
    read the text from given filename
    """
    with open(filename, "r") as f:
        text = f.read()
        return text


def avg_wordlen(words: list[str]) -> float:
    """
    calculates average wordlength from a list of words
    """
    len_all_words = [len(word) for word in words]
    avg_word_length = sum(len_all_words) / len(words)
    return avg_word_length


def avg_sentlen(sents: list[str]) -> float:
    """
    calculates average sentence length from a list of sentences
    """
    avg_sentlen = sum([len(sent) for sent in sents]) / len(sents)
    return avg_sentlen


def compressrat(sents: list[str]):
    """
    Calculates the GZIP compress ratio and BZIP compress ratio for the first 1500 sentences in a list of sentences
    """
    # skipping the first that are often title etc
    selection = sents[2:1502]
    asstring = " ".join(selection)  # making it a long string
    encoded = asstring.encode()  # encoding for the compression

    # GZIP
    g_compr = gzip.compress(encoded, compresslevel=9)
    gzipr = len(encoded) / len(g_compr)

    # BZIP
    b_compr = bz2.compress(encoded, compresslevel=9)
    bzipr = len(encoded) / len(b_compr)

    return gzipr, bzipr




def cleaner(text: str, lower=False) -> str:
    text = re.sub(r"[0-9]", "", text)
    text = re.sub(r'[,.;:"?!*()\']', "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)

    if lower:
        text = text.lower()
    return text


def text_entropy(text: str, language: str, base=2, asprob=True, clean=True):
    if clean:
        text = cleaner(text)

    words = word_tokenize(text, language=language)
    total_len = len(words) - 1
    bigram_transform_prob = Counter()
    word_transform_prob = Counter()

    # Loop through each word in the cleaned text and calculate the probability of each bigram
    for i, word in enumerate(words):
        if i == 0:
            word_transform_prob[word] += 1

            # very first word gets assigned as first pre
            pre = word
            continue

        word_transform_prob[word] += 1
        bigram_transform_prob[(pre, word)] += 1
        pre = word

    # return transformation probability if asprob is set to true
    if asprob:
        return word_transform_prob, bigram_transform_prob
    # if not, calculate the entropy and return that
    if not asprob:
        log_n = log(total_len, base)

        bigram_entropy = cal_entropy(base, log_n, bigram_transform_prob)
        word_entropy = cal_entropy(base, log_n, word_transform_prob)

        return bigram_entropy / total_len, word_entropy / total_len


def cal_entropy(base, log_n, transform_prob):
    entropy = sum([-x * (log(x, base) - log_n) for x in transform_prob.values()])
    return entropy


def text_readability(text: str):
    flesch_grade = textstat.flesch_kincaid_grade(text)
    flesch_ease = textstat.flesch_reading_ease(text)
    smog = textstat.smog_index(text)
    ari = textstat.automated_readability_index(text)
    dale_chall_new = textstat.dale_chall_readability_score_v2(text)

    return flesch_grade, flesch_ease, smog, ari, dale_chall_new


def get_spacy_attributes(token):
    # Save all token attributes in a list
    token_attributes = [
        token.i,
        token.text,
        token.lemma_,
        token.is_punct,
        token.is_stop,
        token.morph,
        token.pos_,
        token.tag_,
        token.dep_,
        token.head,
        token.head.i,
        token.ent_type_,
    ]

    return token_attributes


def create_spacy_df(doc_attributes: list) -> pd.DataFrame:
    df_attributes = pd.DataFrame(
        doc_attributes,
        columns=[
            "token_i",
            "token_text",
            "token_lemma_",
            "token_is_punct",
            "token_is_stop",
            "token_morph",
            "token_pos_",
            "token_tag_",
            "token_dep_",
            "token_head",
            "token_head_i",
            "token_ent_type_",
        ],
    )
    return df_attributes


def filter_spacy_df(df: pd.DataFrame) -> pd.DataFrame:
    spacy_pos = ["NOUN", "VERB", "ADJ", "INTJ"]

    filtered_df = df.loc[
        (df["token_is_punct"] == False)
        & (df["token_is_stop"] == False)
        & (df["token_pos_"].isin(spacy_pos))
    ]

    filtered_df["token_roget_pos_"] = filtered_df["token_pos_"].map(
        {"NOUN": "N", "VERB": "V", "ADJ": "ADJ", "INTJ": "INT"}
    )
    return filtered_df


def save_spacy_df(spacy_df, filename, out_dir) -> None:
    Path(f"{out_dir}/spacy_books/").mkdir(exist_ok=True)
    spacy_df.to_csv(f"{out_dir}/spacy_books/{filename.stem}_spacy.csv")
    #spacy_df.to_csv(f"{out_dir}/spacy_books/{filename}_spacy.csv")



def get_token_categories(df: pd.DataFrame) -> str:
    token_categories = df.apply(
        lambda row: roget.categories(str(row["token_lemma_"]), row["token_roget_pos_"]),
        axis=1,
    ).to_string()

    return token_categories


def make_dico(lexicon: list) -> dict:
    tabs = [line.split("\t") for line in lexicon]

    words = [word[0] for word in tabs if len(tabs) > 1]
    counts = [word[1:] for word in tabs if len(tabs) > 1]

    dico = {}
    for i, word in enumerate(words):
        dico[word] = counts[i]

    return dico

# For NDD (Jockers' version)


def calculate_dependency_distances(df, full_stop_indices):

    dependency_distances = []
    normalized_dependency_distances = []
    start_idx = 0

    for stop_idx in full_stop_indices:
        # Extract each sentence based on full stops
        sentence_df = df.loc[start_idx:stop_idx].copy()
        sentence_df_filtered = sentence_df[~sentence_df['token_is_punct'] & (sentence_df['token_pos_'] != 'SPACE')].copy()

        if not sentence_df_filtered.empty:
            # Find the root by using 'ROOT' in 'token_dep_' column
            root_token_row = sentence_df_filtered[sentence_df_filtered['token_dep_'] == 'ROOT']

            if not root_token_row.empty:

                root_idx = root_token_row['token_i'].iloc[0]

                # Calculating the root distance relative to the start of the sentence
                sentence_start_idx = sentence_df_filtered['token_i'].min()

                root_distance = root_idx - sentence_start_idx  # Adjusted root distance

                # Calculate MDD for the sentence
                sentence_df_filtered['dependency_distance'] = np.abs(sentence_df_filtered['token_i'] - sentence_df_filtered['token_head_i'])
                mdd = sentence_df_filtered['dependency_distance'].mean()

                dependency_distances.append(mdd)

                # Calculate sentence length excluding punctuation
                sentence_length = len(sentence_df_filtered)

                # Calculate NDD using the formula, avoiding division by zero or negative numbers
                if mdd > 0 and sentence_length > 0 and root_distance >= 0:
                    root_sentence_product = (root_distance + 1) * sentence_length  # +1 to avoid zero distance issue
                    if root_sentence_product > 0:
                        ndd = abs(np.log(mdd / np.sqrt(root_sentence_product)))
                        normalized_dependency_distances.append(ndd)

        # Move to the next sentence
        start_idx = stop_idx + 1

    # Calculate average NDD across all sentences
    average_ndd = np.mean(normalized_dependency_distances) if normalized_dependency_distances else None
    std_ndd = np.std(normalized_dependency_distances) if normalized_dependency_distances else None
    #print('Average NDD:', average_ndd)

    #print('Average MDD:', np.mean(dependency_distances))

    return average_ndd, std_ndd, np.mean(dependency_distances), np.std(dependency_distances)



def integrate(x: list[float]) -> np.matrix:
    return np.mat(np.cumsum(x) - np.mean(x))


def get_hurst(arc: list[float]):
    y = integrate(arc)
    uneven = y.shape[1] % 2
    if uneven:
        y = y[0, :-1]

    step_size = 1
    q = 3
    order = 1
    xy = md.multi_detrending(y, step_size, q, order)

    x = np.squeeze(np.asarray(xy[0]))
    y = np.squeeze(np.asarray(xy[1]))

    hurst = round(np.polyfit(x, y, 1)[0], 2)
    return hurst


def calculate_approx_entropy_sliding(arc, window_size=1000, step_size=500, dimension=2, tolerance='sd'):
    """
    Calculates the average Approximate Entropy (ApEn) for a given arc using a sliding window approach.
    
    Parameters:
        arc (list or array): The input data sequence (arc) for which ApEn will be calculated.
        window_size (int): Size of the sliding window. Default is 1000.
        step_size (int): Step size for moving the sliding window. Default is 500.
        dimension (int): Dimension parameter for ApEn. Default is 2.
        tolerance (float or str): Tolerance parameter for ApEn ('sd' or a numeric value). Default is 'sd'.
    
    """
    approx_entropy_dic_sliding = {}
    windows_per_arc = []
    
    # Init list for getting means of ApEn values
    app_ent_values = []
    
    # Iterate over the arc with a sliding window
    for j in range(0, len(arc) - window_size + 1, step_size):
        arc_window = arc[j:j + window_size]  # Extract window of size `window_size`
        
        # Measure ApEn for window
        app_ent, _ = nk.entropy_approximate(arc_window, dimension=dimension, tolerance=tolerance)

        # Append ApEn value to list
        app_ent_values.append(app_ent)
    
    # Get average ApEn
    if app_ent_values:
        avg_app_ent = np.mean(app_ent_values)
    else:
        avg_app_ent = np.nan  # Handle case where no windows
    
    return avg_app_ent