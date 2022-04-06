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

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

from src.utils import format_currency, format_text

###########################################
# Getting Player Historical Market Values:
# Example URL:
# https://www.transfermarkt.com/jack-grealish/marktwertverlauf/spieler/203460
###########################################

def get_players_values(player_df, csv_path):
    """This function is used to collect historical market values for all players in the player_df, 
    we need Selenium to navigate the Javascript-enabled client-side information.

    Args:
        player_df (pandas.DataFrame): dataframe containing players that we want to collect market values for.
        csv_path (CSV filepath): where we want to save our resultant market-value CSV.

    Returns:
        a pandas.DataFrame: a dataframe that contains all historic market values for all major league players. 
    """
    # Below is the helper function to get market values PER PLAYER in the player_df.
    def get_values_per_player(name, link):
        values_lst = [] # eventually: a list of lists for containing player's values at certain dates.
        
        # can comment bottom 2 lines if you want to see the scraper live on a Chrome browser.
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(link)
        driver.maximize_window()
        time.sleep(3) # spacing out requests to prevent throttling of Selenium scraper by the transfermkt server.
        
        def get_text_after_colon(string):   
            # to be used later inside the outer function.
            return re.search(r"[\w\s]*:(.+)", string, re.IGNORECASE).group(1).strip()
        
        # Main logic of getting market values for each player here:
        try:
            # switch to the "Accept Cookies" form found in an Javascript iframe, and close the window.
            WebDriverWait(driver,10).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH,'//iframe[@id="sp_message_iframe_575843"]')))
            WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.XPATH,"//button[contains(@title,'ACCEPT ALL')]"))).click()
            
            # go to actual website after closing.
            driver.switch_to.default_content()
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            # go to CSS element containing the chart of historical market values for player.
            WebDriverWait(driver,5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#highcharts-0 > svg > g.highcharts-series-group > g.highcharts-markers.highcharts-tracker > image")))
            elements = driver.find_elements(by=By.CSS_SELECTOR, value="#highcharts-0 > svg > g.highcharts-series-group > g.highcharts-markers.highcharts-tracker > image")
            a = ActionChains(driver)

            for ele in elements:
                # a is the ActionChains allowing for hovering around different elements in the market values chart
                a.move_to_element(ele).perform()
                textbox = driver.find_element(by=By.CSS_SELECTOR, value="#highcharts-0 > div > span").text
                if textbox:
                    res = [name] + list(map(get_text_after_colon, ("Date :" + textbox).split("\n")))
                    if res not in values_lst:
                        values_lst.append(res)
        except:
            pass
        
        finally:
            driver.quit()
            return values_lst
    
    # After writing the helper, we can use it on each player while iterating over the player dataframe.
    player_links = player_df[["player_name", "player_mv_url"]]
    market_values = []
    
    # use tqdm wrapper to see progress of scraper
    for name, value_link in tqdm(player_links.values):
        res = get_values_per_player(name, value_link)
        market_values.extend(res)
    
    # deduplicating any unncessary info and giving named columns:
    values_df = pd.DataFrame(market_values, columns=["player_name", "date", "market_value", "player_club", "age"])
    values_df.to_csv(csv_path, index=False)
    return values_df
