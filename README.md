An interesting foray into misinformation classification for soccer transfers in major European leagues.
This repository uses `poetry` for dependencies management.

This work has two main contributions:
* 1. Scalable web scraper for collection of transfer news-articles & relevant metadata from transfermarkt.com as well as running the NLP enrichment pipeline on the articles. Activate the poetry environment and run the `main.py` script with an input config file: `config.ini`. Note that the `config.ini` file provides the user with customisation on what years to scrape for, what teams to scrape for, and 
the number of workers to run multi-threading scraping jobs for.

  The command for the end-to-end dataset generation would be: `poetry run main.py --config config.ini`

* 2. Machine-learning work for misinformation classification on articles collected from Point 1.
If you did not run Point 1, we have already provided the generated dataset as a `pickle` file: at [data/majorleagues_20192021_combined.pkl](https://github.com/kenghweeng/sports-messiest-news/blob/main/data/majorleagues_20192021_combined.pkl)

You can then read the [soccer_eda.ipynb](https://github.com/kenghweeng/sports-messiest-news/blob/main/soccer_eda.ipynb)
 notebook for a comprehensive overview on ML experiments done.
