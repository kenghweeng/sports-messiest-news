import ast
import argparse
from bs4 import BeautifulSoup
import configparser
import csv
from collections import defaultdict

import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

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

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

from sentence_transformers import SentenceTransformer
from summarizer.sbert import SBertSummarizer
from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

# Pandas settings:
tqdm.pandas()
pd.options.display.max_rows = 50
pd.options.display.max_columns = 100

# args.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36"}