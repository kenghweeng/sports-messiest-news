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

from utils import format_currency, format_text

###############################################
# Getting Team-Level Statistics
# Example URL - Premier League:
# https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1?saison_id=2019

# We can replace the last 4 digits representing a particular year's league statistics with any other desired year
###############################################

def get_team_stats(league_links, teamstats_csv_path, startyear, endyear, headers):
    """The purpose of this function is to get team-level statistics (i.e wins, losses, expenditures) for each team belonging to
    all leagues given in the league_links list, across all years spanning from the startyear to endyear variables.

    Args:
        league_links (List): a list containing URL links for leagues that we want to collect team statistics for.
        teamstats_csv_path (CSV filepath): where we want to save our resultant team-level statistics CSV.
        startyear (int): year when we want to begin collecting team statistics from,  PER LEAGUE.
        endyear (int): the last possible year when we want to stop collecting team statistics from, PER LEAGUE.
        headers (JSON): basically the web requests header for our scraper to function.

    Returns:
        a pandas.DataFrame: a dataframe that contains team statistics for clubs belonging to input leagues across the specified years.
    """
    # Below is the helper function to get every team's statistics PER league in the league_links list, in a given year
    def get_stats_per_team(league_url, year):
        teams_list = []
        year_url = urlparse(league_url)._replace(query=f"saison_id={year}").geturl()
        year_html = requests.get(year_url, headers=headers)
        year_soup = BeautifulSoup(year_html.content)
        # we look at the team ranking belonging to a specific league and specific year.
        year_ranking_soup = year_soup.select('.tab-print > .box > .grid-view > .items > tbody')[0].find_all('td')
        
        # we construct a key-value pair dictionary, where the key is the team's name and value is the team's league rank.
        rank_dict = {}
    
        for i in range(0,len(year_ranking_soup),6): # we have a step size of 6 here because if you look at the example URL, 6 columns in table.
            team_rank, team_url, _, team_matches, team_pd, team_pts = year_ranking_soup[i:i+6]
            team_url = team_url.find('a')['href']
            team_url = re.sub(r'/spielplan/', r'/startseite/', team_url)

            team_rank = int(team_rank.get_text(strip=True))
            team_league_matches = int(team_matches.get_text(strip=True))
            team_league_pd = int(team_pd.get_text(strip=True))
            team_league_pts = int(team_pts.get_text(strip=True))
            
            rank_dict[team_url] = [team_rank, team_league_matches, team_league_pd, team_league_pts]
            
        year_soup = year_soup.select('.responsive-table > .grid-view > .items > tbody')[0]
        for _, cells in tqdm(enumerate(year_soup.find_all(class_=re.compile("^(even|odd)$")))):
            td_tags = cells.find_all("td")
            team_url = td_tags[1].find('a')['href']
            team_rank, team_league_matches, team_league_pd, team_league_pts = rank_dict[team_url]
            team_url = urljoin(league_url, team_url)
            
            # we go into each specific team's URL page to get more specific team information:
            # such as wins, losses, average age of players, etc.
            
            # example team URL page: 
            # https://www.transfermarkt.com/manchester-city/startseite/verein/281/saison_id/2019
            team_html = requests.get(team_url, headers=headers)
            team_soup = BeautifulSoup(team_html.content, "lxml")

            team_manager = team_soup.select('.container-hauptinfo')[0].get_text(strip=True)
            team_transfer_income = format_currency(team_soup.select('.transfer-record__total--positive')[0].get_text(strip=True))
            team_transfer_expense = format_currency(team_soup.select('.transfer-record__total--negative')[0].get_text(strip=True))
            team_transfer_net = team_transfer_income - team_transfer_expense

            team_name = td_tags[1].get_text(strip=True)
            team_avg_age = float(td_tags[3].get_text(strip=True))
            team_foreigners = int(td_tags[4].get_text(strip=True))
            team_avg_val = td_tags[5].get_text(strip=True)
            team_total_val = td_tags[6].get_text(strip=True)

            team = {
                    'team_name': team_name,
                    'team_url': team_url,
                    'team_manager': team_manager,
                    'team_league_matches': team_league_matches,
                    'team_league_pd': team_league_pd,
                    'team_league_pts': team_league_pts,
                    'team_avg_age': team_avg_age,
                    'team_foreigners': team_foreigners,
                    'team_avg_val': format_currency(team_avg_val),
                    'team_total_val': format_currency(team_total_val),
                    'team_transfer_income': team_transfer_income,
                    'team_transfer_expense': team_transfer_expense,
                    'team_transfer_net': team_transfer_net,
                    'team_rank': team_rank,
                    'season': year
                    }
            teams_list.append(team)

        return teams_list
    
    # After writing the helper function to get specific team-level information for a league,
    # proceed to do so for all leagues.
    
    teamstats_df = []
    for league in tqdm(league_links):
        print(f"We are collecting for the league: {league}")
        for year in tqdm(range(startyear, endyear+1)):
            print(f"We are collecting for the above league in the year {year}")
            # calling helper per league per year
            teamstats_df.extend(get_stats_per_team(league, year))
    
    teamstats_df = pd.DataFrame(teamstats_df)
    teamstats_df.to_csv(teamstats_csv_path, index=False)
    return teamstats_df
