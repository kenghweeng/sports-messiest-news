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

###########################################
# Getting Player High Level Information:
# Example URL: 
# https://www.transfermarkt.com/statistik/saisontransfers?saison-id=2021
# and then going to the "Detailed" tab.
###########################################

def get_players_per_year(year, pages, headers):
    """The purpose of this function is to obtain candidate players to seed for more transfer news, where the players
       are curated from transfermkt top-transfers section as given in the Example URL.

    Args:
        year (int): year to collect for players
        pages (int): number of pages to scrape from the website for that year
        headers (JSON): basically the web requests header for our scraper to function.

    Returns:
        a pandas.DataFrame: a dataframe that contains high level information: (age, name, position,
                            relevant clubs) for our candidate players for a given year.
    """
    
    print(f"We are collecting players for the year {year}")
    players_list = []
    for page in tqdm(range(1, pages+1)):
        url = f'https://www.transfermarkt.com/transfers/saisontransfers/statistik/top/ajax/yw0/saison_id/{year}/transferfenster/alle/land_id//ausrichtung//spielerposition_id//altersklasse//leihe//plus/1/galerie/0/page/{page}'
        html = requests.get(url, headers=headers)
        soup = BeautifulSoup(html.content, 'lxml')
        soup = soup.select('.responsive-table > .grid-view > .items > tbody')[0] # choosing the table to grab values from

        for cells in soup.find_all(class_=re.compile("^(even|odd)$")):
            try:
                td_tags = cells.find_all('td')
                fee = td_tags[-1].text
                
                # getting destination club info
                country_to = td_tags[15].img['alt']
                league_to = td_tags[15].a.text if cells.find_all('td')[15].a is not None else 'Without League'
                club_to = td_tags[13].img['alt']
                club_to_url = urljoin(url, td_tags[13].a['href'])
                
                # getting source club info
                country_from = td_tags[11].img['title']
                league_from = td_tags[11].a.text if td_tags[11].a is not None else 'Without League'
                club_from = td_tags[9].img['alt']
                club_from_url = urljoin(url, td_tags[9].a['href'])
                
                # getting more player-specific metadata
                position = td_tags[4].text
                player_url = urljoin(url, td_tags[3].a['href'])
                player_html = requests.get(player_url, headers=headers).content
                player_soup = BeautifulSoup(player_html)
                player_info = [ele.get_text(strip=True) for ele in player_soup.find_all(class_="info-table__content--regular")]
                player_data = player_soup.find_all(class_="info-table__content--bold")
                
                birthday, height, foot = player_data[player_info.index("Date of birth:")].get_text(strip=True), player_data[player_info.index("Height:")].get_text(strip=True).replace(',', '')[:3], player_data[player_info.index("Foot:")].get_text(strip=True)

                player = {
                    'player_name': td_tags[1].select('td > img')[0]['title'],
                    'player_url': player_url,
                    'player_birthday': birthday,
                    'player_height': height,
                    'player_foot': foot,
                    'position': position,
                    'country_from': country_from,
                    'league_from': format_text(league_from),
                    'club_from': club_from,
                    'club_from_url': club_from_url,
                    'country_to': country_to,
                    'league_to': format_text(league_to),
                    'club_to': club_to,
                    'club_to_url': club_to_url,
                    'fee': format_currency(fee),
                    'year': year
                }
                
                players_list.append(player)
                
            except Exception as e:
                # print(e)
                pass
        
    return pd.DataFrame(players_list)

def prepare_league_players(player_df, player_csv_path, leagues_lst):
    """Purpose of this function is to preprocess the players collected to only be the ones belonging
       in the list of desired leagues. Can set this in the config file. 
       
       Also do some preprocessing on fields to prepare for downstream collection of other info.

    Args:
        player_df (pandas.DataFrame): dataframe containing all players in a given time period
        leagues_lst (List): list of leagues (String) which we are interested in.
        
    Returns:
        a pandas.DataFrame: a dataframe which is filtered based on desired leagues
                            and is ready for downstream collection for other relevant features.
    """
    # filter by list of leagues
    player_df = player_df.loc[(player_df["league_from"].isin(leagues_lst)) & (player_df["league_to"].isin(leagues_lst))]
    
    # we edit the following URL fields so that we can scrape the correct links later on.
    player_df["player_mv_url"] = player_df["player_url"].str.replace('/profil/', '/marktwertverlauf/')
    player_df["player_stats_url"] = player_df["player_url"].str.replace('/profil/', '/leistungsdaten/')
    
    # we also extract the player IDs to be used as primary keys later.
    id_pattern = r"(.*?)/(\d+)"
    player_df["player_id"] = player_df["player_url"].str.extract(id_pattern, flags=re.I, expand=False)[1]
    player_df.to_csv(player_csv_path, index=False)  # save to CSV for future use.
    
    return player_df
