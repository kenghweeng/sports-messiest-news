from bs4 import BeautifulSoup
import csv
from collections import defaultdict

import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register

import numpy as np
import os
import pandas as pd
import re
import requests
import sys
import time
import torch
from tqdm import tqdm
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.request import urlopen
import validators

from sentence_transformers import SentenceTransformer
from summarizer.sbert import SBertSummarizer
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

# Summarization:
def get_summaries(articles_df, n_workers, model, summary_pickle_path):
    """Purpose of this function is just to get textual summaries for all articles in articles_df
    Args:
        articles_df (pd.DataFrame): dataframe containing all relevant articles relating to our required players.
        n_workers (int): number of cores our device/server has to parallelize the work.
        model (Pytorch model): a summarizer model based on PyTorch 
        summary_pickle_path (str): a string denoting the filepath to save the summarized dataframe

    Returns:
        a pd.DataFrame, essentially articles_df with an additional column of summaries for each article.
    """
    
    # helper to get summary for each article in article_df
    def get_summary(text, model):
        len_sent = model.calculate_optimal_k(text, k_max=5) # allow max of 5 sentences
        return model(text, num_sentences=len_sent)

    ddf = dd.from_pandas(articles_df, npartitions=n_workers)
    ddf["summary"] = ddf.apply(lambda row: get_summary(row["article_text"], model), axis=1, meta='object')
    with pbar:
        articles_df = ddf.compute()
    
    # Summarization may not have went well, in this case, we use the article snippet!
    articles_df.loc[articles_df["summary"] == '', "summary"] = articles_df.loc[articles_df["summary"] == '', "article_snippet"]
    articles_df.loc[articles_df["summary"].isna(), "summary"] = articles_df.loc[articles_df["summary"].isna(), "article_snippet"] 
    articles_df.to_pickle(summary_pickle_path)
    
    print("We have generated summaries for our articles!")
    return articles_df

# Sentiments for our articles dataframe:
def get_sentiments(articles_df, n_workers, model, sentiments_pickle_path):
    # helper to get sentiment scores for each article
    def get_sentiment(text, model):
        sentiments_dict = model(text)[0]
        label, score = sentiments_dict['label'], sentiments_dict['score']
        if label == 'NEGATIVE':
            return -1 * score
        return score
    ddf = dd.from_pandas(articles_df, npartitions=n_workers)
    ddf["sentiment_score"] = ddf.apply(lambda row: get_sentiment(row["summary"], model), axis=1, meta='object')
    with pbar:
        articles_df = ddf.compute()
        
    articles_df.to_pickle(sentiments_pickle_path)
    print("We have generated sentiments for our articles!")
    return articles_df

# Entities for our articles dataframe:
def get_entities(articles_df, n_workers, model, ner_pickle_path):
    # helper to get entities and their counts for each article
    def get_ner(text, model):
        res = defaultdict(set)
        try:
            ner_result = model(text)
            pairs = map(lambda entity_dict: (entity_dict['entity_group'], entity_dict['word']), ner_result)
            for ent_label, ent_name in pairs:
                if ent_label in ('ORG', 'PER'):
                    res[ent_label].add(ent_name)
        except Exception as e:
            print(e)
            pass
        return res
    
    ddf = dd.from_pandas(articles_df, npartitions=n_workers)
    ddf["entities"] = ddf.apply(lambda row: get_ner(row["summary"], model), axis=1, meta='object')
    with pbar:
        articles_df = ddf.compute()
        
    articles_df.to_pickle(ner_pickle_path)
    print("We have generated entities for our articles!")
    
    # From the entities object, we also want to count the number of distint entities.
    def get_counts(entity_dict, tag):
        res = entity_dict.get(tag, 0)
        return 0 if res == 0 else len(res)
    
    # We count the number of distinct people inside the article
    articles_df["people_count"] = articles_df["entities"].apply(get_counts, tag="PER")
    # We count the number of distinct organizations/teams/clubs inside the article
    articles_df["org_count"] = articles_df["entities"].apply(get_counts, tag="ORG")
    print("We have also generated the counts for PEOPLE and ORGANIZATIONS.")
    
    return articles_df

# Document embeddings for our article summaries
def get_embeddings(articles_df, model, embed_pickle_path):
    # helper to get embeddings for each summarized article:
    def encode_document(text, model):
        embeddings = None
        try:
            embeddings = model.encode(text)
        except:
            pass
        return embeddings
    
    articles_df["article_embeddings"] = articles_df["summary"].progress_apply(encode_document, model=model)
    
    # Doing some tidy-up of work done in the NLP pipeline
    articles_df.drop(["article_source", "name_found"], axis=1, inplace=True)
    articles_df["article_date"] = pd.to_datetime(articles_df["article_date"])
    articles_df["season"] = articles_df["article_date"].dt.year - 1
    articles_df = articles_df.sort_values(["player_id", "article_date"]).groupby("player_id").head(100) # choose 100 to get all rows basically.
    articles_df.to_pickle(embed_pickle_path)
    
    return articles_df

    