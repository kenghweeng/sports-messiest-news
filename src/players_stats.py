from bs4 import BeautifulSoup
import csv
from collections import defaultdict

import numpy as np
import os
import pandas as pd
import re
import requests
import sys
import time
from tqdm import tqdm
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.request import urlopen
import validators

import warnings
warnings.filterwarnings("ignore")

from src.utils import format_currency, format_text

###############################################
# Getting Player Historical Match Statistics:
# Example URL:
# https://www.transfermarkt.com/jack-grealish/leistungsdaten/spieler/203460/?saison=2020

# We can replace the last 4 digits representing a particular year's player statistics with any other desired year
###############################################

def get_players_stats(player_df, playerstats_csv_path, startyear, endyear, headers):
    """The purpose of this function is to get match statistics (i.e goals scored, minutes played, etc) for all players in player_df 
    across all years spanning from the startyear to endyear variables.

    Args:
        player_df (pandas.DataFrame): dataframe containing players that we want to collect match statistics for.
        playerstats_csv_path (CSV filepath): where we want to save our resultant player statistics CSV.
        startyear (int): year when we want to begin collecting match statistics from, PER PLAYER.
        endyear (int): the last possible year when we want to stop collecting match statistics from, PER PLAYER.
        headers (JSON): basically the web requests header for our scraper to function.

    Returns:
        a pandas.DataFrame: a dataframe that contains all season statistics for all major league players across specified years.
    """
    # Below is the helper function to get the season match statistics PER PLAYER in the player_df.
    def get_stats_per_player(stats_url, year):
        stats_year_url = urlparse(stats_url)._replace(query=f"saison={year}").geturl()
        stats_html = requests.get(stats_year_url, headers=headers)
        stats_soup = BeautifulSoup(stats_html.content)
        
        # We make sure that all the statistics are stripped or cleaned to integers with the below helper:
        def int_string(string):
            string = re.sub(r"[\-\.\']", "", string)
            if not string:
                return 0
            else:
                return int(string)
        
        # Main logic of getting stats for each player here:
        try:
            res = stats_soup.select('.responsive-table > .grid-view > .items > tfoot')[0].find_all('td')[2:]
            res = list(map(int_string, [x.text for x in res]))
            games, goals_scored, goals_concede, clean_sheet, assists, yellow, second_yellow, red, mins = [0] * 9
            if len(res) == 7:
                games, goals_scored, assists, yellow, second_yellow, red, mins = res
            else:
                games, goals_scored, yellow, second_yellow, red, goals_concede, clean_sheet, mins = res
        except:
            res = None
            
        return [games, goals_scored, goals_concede, clean_sheet, assists, yellow, second_yellow, red, mins] if res else [0] * 9
        
    # After having a helper to get stats for each player, we iterate over the actual players dataframe.
    players_stats = []

    for name, stats_url in tqdm(player_df[["player_name", "player_stats_url"]].values):
        for year in range(startyear, endyear+1):
            res = get_stats_per_player(stats_url, year)
            players_stats.append([name, year] + res)
    
    playerstats_df = pd.DataFrame(players_stats, columns=['player_name', 'season', 'games', 'goals_scored', 'goals_concede', 'clean_sheet', 'assists', 'yellow', 'second_yellow', 'red', 'mins'])
    playerstats_df.to_csv(playerstats_csv_path, index=False)
    return playerstats_df
