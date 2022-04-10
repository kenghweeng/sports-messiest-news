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
from src.players import get_players_per_year, prepare_league_players
from src.players_values import get_players_values
from src.players_stats import get_players_stats
from src.team_stats import get_team_stats
from src.players_articles import get_all_forum_transfers, get_news_articles
from src.nlp_pipeline import get_summaries, get_sentiments, get_entities, get_embeddings
from src.utils import *

# Helper used in the main logic below.
def enrich_names(articles_df, names_df, articles_names_csv_path):
    """This function is used to reconcile player names in names_df with player IDs with articles collected.

    Args:
        articles_df (pd.DataFrame): dataframe collecting features relating to articles, and contains player IDs only
        names_df (pd.DataFrame): dataframe relating to player IDs and names
        articles_names_csv_path (str): file path to save merged dataframe to.

    Returns:
        _type_: _description_
    """
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

# Important function used to combine all sources of info into a unified dataframe!
def combine_all_dataframes(combined_pkl_path, articles_df, player_df, playerstats_df, values_df, teamstats_df):
    """The purpose of this function is to basically merge all relevant information into a single source of data
    
    Args:
        combined_pkl_path (str): where to pickle the unified dataframe to.
        articles_df (pd.DataFrame): dataframe containing information on transfer news-articles
        player_df (pd.DataFrame): dataframe containing information on player metadata
        values_df (pd.DataFrame): dataframe containing information on player market values
        playerstats_df (pd.DataFrame): dataframe containing information on player historical league statistics
        teamstats_df (pd.DataFrame): dataframe containing information on club statistics in speciifed leagues

    Returns:
        pd.DataFrame: merged dataframe containing all of the goodness as above
    """
    # Merging articles with player_metadata:
    player_df = player_df[["player_id", "player_height", "player_foot", "position"]]
    player_df.drop_duplicates(subset=['player_id'], inplace=True) 
    articles_df = articles_df.merge(player_df, on=["player_id"], how='left') # contains player's metadata

    # Merging articles with player statistics:
    playerstats_df.drop_duplicates(subset=["player_name", "season"], inplace=True)
    articles_df = articles_df.merge(playerstats_df, on=["player_name", "season"], how='left') # we now have enriched with player stats

    # Merging articles with team statistics:
    # each article has 2 teams involved, we first merge with the source club.
    articles_df["team_id"] = articles_df["club_from"].str.split('/').str[-1].astype('int')
    team_id_pattern = r"(.*?)/verein/(\d+)/.*" # get team ID
    teamstats_df["team_id"] = teamstats_df["team_url"].str.extract(team_id_pattern, flags=re.I, expand=False)[1].astype('int')
    # teamstats_df["team_id"] = teamstats_df["team_url"].str.split('/').str[-1].astype('int')
    articles_df = articles_df.merge(teamstats_df, on=["season", "team_id"], how='left')

    # we now merge with the destination club's statistics
    articles_df["team_to_id"] = articles_df["club_to"].str.split('/').str[-1].astype('int')
    teamstats_df.rename(columns={"team_id": "team_to_id"}, inplace=True)
    articles_df = articles_df.merge(teamstats_df, on=["season", "team_to_id"], how='left', suffixes=('_src', '_dst'))
    print(f'There are {articles_df[articles_df["team_rank_src"].isna() | articles_df["team_rank_dst"].isna()].shape[0]} articles which did not involve major leagues, and we will remove these.')
    articles_df = articles_df.dropna(subset=['team_rank_src', 'team_rank_dst'])

    # we finally merge with the market values of players, before the merge we need to make sure both dataframes are sorted based on datetime.
    values_df["date"] = pd.to_datetime(values_df["date"])
    # sort the dates of values ascendingly within each player group
    values_df = values_df.sort_values(["player_name", "date"]).groupby("player_name").head(100) # sort the dates of values ascendingly within each player group
    articles_df = articles_df.sort_values(by=["player_name", "article_date"]).groupby("player_name").head(100)
    articles_df["article_date"] = pd.to_datetime(articles_df["article_date"])

    combined = pd.DataFrame()

    for player_name, player_group in articles_df.groupby("player_name"):
        player_market_vals = values_df[values_df["player_name"] == player_name]
        res_df = pd.merge_asof(player_group, player_market_vals, left_on='article_date', right_on='date', direction='backward')
        combined = combined.append(res_df, ignore_index=True)

    combined.dropna(subset=["market_value"], inplace=True)
    combined["market_value"] = combined["market_value"].apply(format_currency)
    combined = combined.reset_index(drop=True)
    embeddings = list(combined["article_embeddings"].values)
    combined.drop("article_embeddings", axis=1, inplace=True)
    embeddings = pd.DataFrame(embeddings, columns=[f'bert_embedding_{i}' for i in range(len(embeddings[0]))])
    combined = pd.concat([combined, embeddings], axis=1)
    combined["label"] = combined["transfer_str"].replace({"Transfer failed": 0, "Done deal": 1})
    
    # Removing columns which have duplicate info
    combined = combined.drop([
            'transfer_str',
            'player_id',
            'club_from',
            'club_to',
            'article_link',
            'article_snippet',
            'article_text',
            'player_name_x',
            'entities',
            'team_id',
            'team_manager_src',
            'team_url_src',
            'team_to_id',
            'team_manager_dst',
            'team_url_dst',
            'player_name_y',
            'date',
            'player_club'], axis=1)
    
    combined["season"] = combined["season"].astype("object")
    # Randomly shuffle the rows
    combined = combined.sample(frac=1)
    combined = combined.reset_index(drop=True)
    
    combined.to_pickle(combined_pkl_path)
    print(f'We have pickled the combined dataframe at {combined_pkl_path}!')
    return combined

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
    marketvalue_csv_path = f'data/majorleagues_{args.startyear-1}{args.endyear}_values.csv'
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
        print("We are using CPU for the NLP pipeline, which can be very slow. Consider using a GPU server.")
            
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
    
    # Final step: creating the soccer transfer-news dataset!
    combined_pkl_path = f'data/majorleagues_{args.startyear}{args.endyear}_combined.pkl'
    if not os.path.exists(combined_pkl_path):
        combined = combine_all_dataframes(combined_pkl_path, articles_df, player_df, playerstats_df, \
                                             values_df, teamstats_df)
    else:
        print('We have previously generated the combined soccer transfer-news dataset!')
        combined = pd.read_pickle(combined_pkl_path)
    
    print(f'We have a total of {combined.shape[0]} articles and a total of {combined.shape[1]} enriched columns in our transfer-news dataset.')
    