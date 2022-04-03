from bs4 import BeautifulSoup
import csv
from collections import defaultdict

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

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
tqdm.pandas()

from utils import format_currency, format_text

#################################################################################################################
# Getting all news articles links:
# This is a 2-step approach:

# Step 1: We first proceed to collect all forum posts for all of our major league players.
# Hierachy: Each forum URL -> multiple forum threads -> each forum thread has multiple pages with multiple posts in each page.
#################################################################################################################

# An example forum URL would be as below, and determined by 2 possible "forumid" values (local and international forums), and unique playerIDs
# "https://www.transfermarkt.com/geruechtekueche/detail/forum/{forumid}/gk_spieler_id/{playerid}".format(forumid=180, playerid=203460)

# An example forum THREAD URL would be as below:
# Note that there is a thread_id parameter in the URL.
# https://www.transfermarkt.com/jack-grealish-to-manchester-city-/thread/forum/180/thread_id/13116/

# An example forum POST URL would be as below:
# https://www.transfermarkt.com/jack-grealish-to-manchester-city-/thread/forum/180/thread_id/13116/post_id/24424

# Step 1: We follow the hierarchy: forum URL -> forum thread URL -> all forum page URLs
def get_all_forum_transfers(player_df, forum_links_pickle_path, startyear, headers):
    forum_links = []
    
    # COLLECTING FORUM URLS
    def get_forum_urls(player_id):
        forum_ids = [180,343]
        for forum_id in forum_ids:
            player_forum_link = f"https://www.transfermarkt.com/geruechtekueche/detail/forum/{forum_id}/gk_spieler_id/{player_id}"
            forum_links.append([player_id, player_forum_link]) # we represent each player with it's unique ID and forum link
            
    print("Collecting forum thread links for all players in the input player_df")
    player_df["player_id"].progress_apply(get_forum_urls)
    
    # COLLECTING FORUM THREAD URLS
    def get_forum_threads(news_links, startyear):
        news_data = [["player_id", "transfer_date", "transfer_str", "club_from", "club_to", "forum_link"]]
        
        # We collect all threads for the forum URLs:
        for player_id, news_link in tqdm(news_links):
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36"}
                html = requests.get(news_link, headers=headers)
                soup = BeautifulSoup(html.content, "lxml")
                forum_soup = soup.select('.threaduebersicht-threads')[-1]
                
                for thread in forum_soup.find_all(class_='thread'):
                    single_news = [player_id]
                    news_date = thread.find(class_='post-datum').text.split('-')[0].strip()
                    # we basically stop collecting transfer news threads which were discussing
                    # transfer news before our "earliest" year of interest.
                    if news_date[-4:] < str(startyear):
                        break

                    single_news.append(news_date)
                    # found transfer ground truth label
                    single_news.append(thread.find(class_='wertungskasten')["title"].strip())
                    # found origin and dest clubs
                    single_news.extend([urljoin(news_link, element.find('a')['href']) for element in thread.find_all(class_='gk-wappen')])
                    # found link to get news from
                    single_news.append(urljoin(news_link, thread.find(class_='wechsel-verein-name').find('a')['href']))
                    news_data.append(single_news)
            except:
                continue
        return news_data
    
    player_forum_threads = get_forum_threads(forum_links, startyear)
    forum_thread_df = pd.DataFrame(player_forum_threads[1:], columns = player_forum_threads[0])
    
    # We note that the existing thread links do not provide the number of pages in each thread in their URL strings.
    # We can identify the number of pages with the following helper, which uses the wonderful urlparse library.
    def clean_link(link, headers):
        try:
            new_link = urldefrag(requests.get(link, headers=headers).url).url
        except:
            new_link = ''
        return new_link
    
    print("We are now generating forum thread links with page counts in their links!")
    forum_thread_df["clean_link"] = forum_thread_df["forum_link"].progress_apply(clean_link, headers=headers)
    
    # We then remove some of the thread links, which are either:
    # 1. defunct or expired link
    forum_thread_df = forum_thread_df[forum_thread_df["clean_link"] != '']
    # 2. we want to collect articles which have either confirmed transfers or known failed transfers.
    forum_thread_df = forum_thread_df.loc[forum_thread_df["transfer_str"].isin(["Transfer failed", "Done deal"])]
    # 3. some of the page counts are abnormally large, this may be an error on their frontend values.
    # we will remove those.
    counts = forum_thread_df["clean_link"].str.split("/").str[-1].astype('int')
    forum_thread_df = forum_thread_df.loc[counts <= 10]
    
    # COLLECT FORUM POSTS URLS
    # The purpose of this helper is just to generate a list of all pages URLs for the thread link.
    def get_forum_pages(thread_link):
        all_pages = []
        link_elements = thread_link.split('/')
        page_num = int(link_elements[-1])
        link_elements.pop(-1)

        while page_num >= 1:
            all_pages.append('/'.join(link_elements + [str(page_num)]))
            page_num -= 1
            
        return all_pages
    
    # we call the helper here.
    forum_thread_df["all_links"] = forum_thread_df.progress_apply(lambda row: get_forum_pages(row["clean_link"]), axis=1)
    forum_thread_df.to_pickle(forum_links_pickle_path)
    return forum_thread_df

#################################################################################################################
# Getting all news articles text:
# This is a 2-step approach:

# Step 2: After getting all forum posts links, we can proceed to scrape for each quoted
# news article and get the relevant article text!

# Note that we do use some heuristics to select for articles:
# 1. We only keep English articles.
# 2. We only keep those articles which has mentioned the player in question.
#################################################################################################################

def get_news_articles(forum_thread_df, articles_csv_path, headers):
    news_article_links = [["player_id", "transfer_str", "club_from", "club_to", "article_link", "article_snippet", "article_date"]]
    
    # below helper is used to collect the actual article URL from the forum post URLs:
    # row is referring to each row in the forum_thread_df containing all the forum post URLs
    def get_all_news_sources(row, headers):
        player_id, _, transfer_str, club_from, club_to, *_, all_links = row
        player_info = [player_id, transfer_str, club_from, club_to]
        possible_dates = []
        possible_news = []
        possible_snips = []
        
        for page_link in all_links:
            page_html = requests.get(page_link, headers=headers)
            page_soup = BeautifulSoup(page_html.content, "lxml")
            page_posts = page_soup.select(".box-border-top")
            
            def find_post_metadata(soup):
                while not soup.find(class_="post-header-datum"):
                    soup = soup.parent

                return format_text(soup.find(class_="content").text), re.search(r'(.*?) -(.*)', soup.find(class_="post-header-datum").text.strip(), re.IGNORECASE).group(1)
            
            news_snips = []
            news_dates = []
            
            # In each forum post, we can collect the date of the news article, and some short snippets of the article quoted in forum.
            for element in page_posts:
                text_snips, text_date = find_post_metadata(element)
                news_snips.append(text_snips)
                news_dates.append(text_date)
                
            # We collect the actual article URL here.
            news_articles = [div.find("a")["href"] for div in page_posts]

            for i, news in enumerate(news_articles):
                if news not in possible_news:
                    possible_news.append(news)
                    possible_dates.append(news_dates[i])
                    possible_snips.append(news_snips[i])

        final_news = list(zip(possible_news, possible_snips, possible_dates))
        news_article_links.extend(list(map(lambda row: player_info + list(row), final_news)))
        
    # We call the helper to get all related article URLs
    forum_thread_df.progress_apply(lambda row: get_all_news_sources(row, headers), axis=1)
    articles_df = pd.DataFrame(news_article_links[1:], columns=news_article_links[0])
    # Find out the distribution of news sources: BBC, skysports, etc.
    articles_df["article_source"] = articles_df["article_link"].apply(lambda row: urlparse(row).netloc)
    
    def detect_language(snippet):
        if validators.url(snippet): # check whether snippet was actually legitimate.
            return "URL"
        else:
            return detect(snippet)
        
    print("We are getting the languages of our article URLs, and keeping only those in English.")
    articles_df["article_lang"] = articles_df["article_snippet"].progress_apply(detect_language)
    articles_df = articles_df[articles_df["article_lang"] == 'en']
    articles_df.drop("article_lang", axis=1, inplace=True) # no more lang column needed
    
    def get_website_text(url, headers):
        try:
            html = requests.get(url, headers)
            soup = BeautifulSoup(html.content, features="html.parser")
            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.encode("ascii", "ignore").decode().strip().strip('’') for line in lines for phrase in line.split("  "))
            # drop blank lines
            def is_legitimate_chunk(chunk):
                if len(chunk.split(" ")) < 15:
                    return False
                elif 'SHARE' in chunk.upper():
                    return False
                elif 'SUBSCRIBE' in chunk.upper():
                    return False
                return True
            
            text = " ".join(chunk for chunk in chunks if is_legitimate_chunk(chunk))
            return text
        except:
            return ""
    # We get the text from the source article URLs.
    articles_df["article_text"] = articles_df.progress_apply(lambda row: get_website_text(row["article_link"], headers), axis=1)
    print(f"We have collected all news articles relevant to our major league players.")
    articles_df.to_csv(articles_csv_path, index=False)