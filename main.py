import ast
import argparse
import configparser
import os
import pandas as pd
import sys
import time
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# NER pipeline:
from sentence_transformers import SentenceTransformer
from summarizer.sbert import SBertSummarizer
from transformers import pipeline

# Our own modules:
from players import get_players_per_year, prepare_league_players
from players_values import get_players_values
from players_stats import get_players_stats
from team_stats import get_team_stats
from players_articles import get_all_forum_transfers, get_news_articles
from nlp_pipeline import get_summaries, get_sentiments, get_entities, get_embeddings
from utils import *

# Write combine dataframe logics here.
def enrich_names(articles_df, names_df, articles_names_csv_path):
    names_df.drop_duplicates(inplace=True) # drop duplicate name/ID pairs
    
    articles_df_names = articles_df.merge(names_df, on='player_id', how='inner')
    articles_df_names = articles_df_names[~articles_df_names['article_text'].isna()]
    # we try to find names of player in question in the article text
    articles_df_names["name_found"] = articles_df_names.progress_apply(lambda row: any(name in row["article_text"].upper() \
                                                                                       for name in row["player_name"].upper().split()), axis=1)
    # if we can't find it, we make use of the forum quoted snippet as the article instead.
    articles_df_names.loc[articles_df_names["name_found"] == False, "article_text"] = articles_df_names.loc[articles_df_names["name_found"] == False, "article_snippet"]
    articles_df_names.to_csv(articles_names_csv_path, index=False)
    return articles_df_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=argparse.FileType('r'), help='config file', default='config.ini')
    parser.add_argument('-o', dest='output', type=argparse.FileType('w'), help='output file', default=sys.stdout)
    args = parser.parse_args()
    args.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"}
    
    if args.config:
        config = configparser.ConfigParser()
        config.read_file(args.config)
        args.startyear = int(config['PLAYERS']['STARTYEAR'])
        args.endyear = int(config['PLAYERS']['ENDYEAR'])
        args.pages = int(config['PLAYERS']['PAGES'])
        args.leagues = ast.literal_eval(config.get('PLAYERS', 'LEAGUES'))
        args.leaguelinks = ast.literal_eval(config.get('PLAYERS', 'LEAGUE_LINKS'))
        args.n_workers = int(config['DEVICE']['N_WORKERS'])
        args.gpus = ast.literal_eval(config.get('DEVICE', 'GPU'))
    
    if not os.path.exists('data'):
        print('Generating data folder')
        os.makedirs('data/')
    
    # We first get players from start to end years, and belonging to major leagues only.
    player_csv_path = f'data/majorleagues_{args.startyear}{args.endyear}.csv'
    if not os.path.exists(player_csv_path):
        print("Collecting players")
        player_df = pd.concat([get_players_per_year(year, args.pages, args.headers) for year in range(args.startyear, args.endyear+1)], ignore_index=True)
        player_df = prepare_league_players(player_df, player_csv_path, args.leagues)
    else:
        print('We already have the required players information.')
        player_df = pd.read_csv(player_csv_path)
    
    # Getting market values for the major league players.
    marketvalue_csv_path = f'data/majorleagues_{args.startyear}{args.endyear}_values.csv'
    if not os.path.exists(marketvalue_csv_path):
        values_df = get_players_values(player_df, marketvalue_csv_path)
    else:
        print('We already have market values for the required players.')
        values_df = pd.read_csv(marketvalue_csv_path)
        
    # Getting historical league match stats for major league players from specified start year to end year.
    playerstats_csv_path = f'data/majorleagues_{args.startyear-1}{args.endyear}_playerstats.csv'
    if not os.path.exists(playerstats_csv_path):
        playerstats_df = get_players_stats(player_df, playerstats_csv_path, args.startyear-1, args.endyear, args.headers)
    else:
        print('We already have match statistics for the required players.')
        playerstats_df = pd.read_csv(playerstats_csv_path)
    
    # Getting historical team-level stats for major leagues from specified start year to end year.
    teamstats_csv_path = f'data/majorleagues_{args.startyear-1}{args.endyear}_teamstats.csv'
    if not os.path.exists(teamstats_csv_path):
        teamstats_df = get_team_stats(args.leaguelinks, teamstats_csv_path, args.startyear-1, args.endyear, args.headers)
    else:
        print('We already have team statistics for the required teams.')
        teamstats_df = pd.read_csv(teamstats_csv_path)
        
    # Getting forum links for major league players discussing transfer news starting from a specified start year
    forum_links_pickle_path = f'data/majorleagues_{args.startyear}{args.endyear}_forumlinks.pkl'
    if not os.path.exists(forum_links_pickle_path):
        forum_df = get_all_forum_transfers(player_df, forum_links_pickle_path, args.startyear, args.headers)
    else:
        print('We already have all of our forum links for the required players.')
        forum_df = pd.read_pickle(forum_links_pickle_path)
    print(forum_df.shape)
    
    # Getting all English articles from the required players.
    articles_csv_path = f'data/majorleagues_{args.startyear}{args.endyear}_articles.csv'
    if not os.path.exists(articles_csv_path):
        articles_df = get_news_articles(forum_df, articles_csv_path, args.headers)
    else:
        print('We already have all of the relevant English articles for the required players.')
        articles_df = pd.read_csv(articles_csv_path)
        
    # currently articles_df does not have player names, only player IDs,
    # we use our player_df and perform a merge to enrich the articles_df with names
    articles_names_csv_path = f'data/majorleagues_{args.startyear}{args.endyear}_articles_names.csv'
    if not os.path.exists(articles_names_csv_path):
        # enrich_names is the function in this main.py file
        articles_df = enrich_names(articles_df, player_df[["player_name", "player_id"]], articles_names_csv_path)
    else:
        print('We already have the articles together with the player names related to the articles.')
        articles_df = pd.read_csv(articles_names_csv_path)
        
    # Check GPU here, and caveat about speed.
    if torch.cuda.is_available():
        gpu_string = ",".join(map(str, args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_string
        print(f"We are using GPU {gpu_string} for the NLP pipeline!")
        torch.cuda.empty_cache()
    else:
        print("We are CPU for the NLP pipeline, which can be very slow. Consider using a GPU server.")
            
    # We now proceed with the NER pipeline, introducing models used for the NER pipeline:
    # Summarization -> Sentiments of summary -> NER of summary -> BERT-embeddings of summary
    
    # Generating summaries:
    summary_pickle_path = f'data/majorleagues_{args.startyear}{args.endyear}_summaries.pkl'
    if not os.path.exists(summary_pickle_path):
        summarizer_model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        articles_df = get_summaries(articles_df, args.n_workers, summarizer_model, summary_pickle_path)
    else:
        print('We already have the article summaries for the required players.')
        articles_df = pd.read_pickle(summary_pickle_path)
        
    # Generating sentiments:
    sentiments_pickle_path = f'data/majorleagues_{args.startyear}{args.endyear}_sentiments.pkl'
    if not os.path.exists(sentiments_pickle_path):
        sentiment_pipe = pipeline("sentiment-analysis", truncation=True)
        articles_df = get_sentiments(articles_df, args.n_workers, sentiment_pipe, sentiments_pickle_path)
    else:
        print('We already have the sentiments for the required articles.')
        articles_df = pd.read_pickle(sentiments_pickle_path)
    
    # Generating entities:
    ner_pickle_path = f'data/majorleagues_{args.startyear}{args.endyear}_entities.pkl'
    if not os.path.exists(ner_pickle_path):
        ner_pipe = pipeline("ner", aggregation_strategy='max')
        articles_df = get_entities(articles_df, args.n_workers, ner_pipe, ner_pickle_path)
    else:
        print('We already have the entities for the required articles.')
        articles_df = pd.read_pickle(ner_pickle_path)
    
    # Generating embeddings:
    embed_pickle_path = f'data/majorleagues_{args.startyear}{args.endyear}_embed.pkl'
    if not os.path.exists(embed_pickle_path):
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        articles_df = get_embeddings(articles_df, embeddings_model, embed_pickle_path)
    else:
        print('We already have generated the embeddings for the articles.')
        articles_df = pd.read_pickle(embed_pickle_path)
    
    
    # Combine all relevant dataframes to form final df
    
    # ablation using both sets of information:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7826718/pdf/entropy-23-00090.pdf
    # pseudo-algo, difficulties, CSS, resolutions.
    # Understanding how to use silas for ablation studies
    # wordclouds, visualizations, REFCV, correlation matrix.
    
    # IF GOT TIME:
    # talk about inference, how to predict for new teams.
    
    