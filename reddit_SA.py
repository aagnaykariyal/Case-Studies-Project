import praw  # Reddit crawler
import pandas as pd
import numpy as np

import datetime as dt
from pprint import pprint
from itertools import chain

from praw.models import MoreComments

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer  # tokenize words

# import matplotlib.pyplot as plt
# %matplotlib inline
# plt.rcParams["figure.figsize"] = (10, 8) # default plot size
# import seaborn as sns
# sns.set(style='whitegrid', palette='Dark2')

from wordcloud import WordCloud
import emoji
import re
# import en_core_web_sm
import spacy

# ------------- End of Libraries ---------------#

# -------------- Reddit API Connection ----------------#

reddit = praw.Reddit(
    client_id="zw8uC_emsN_GTyOai4v6OQ",
    client_secret="XJf-l9V2TNGaF9fn6486C0OOEpRplA",
    user_agent="ua"
)

# -------------- Main Code ----------------#

potential_subreddits = ["pharma", "RaceAndIntelligence", "MixedRaceAndProud", "BadPharma",
                        "genderedracism", "mixedrace", "AccidentalRacism", "racism", "BlackRacism", "Conservative",
                        "CitizensJournal", "AMillionLittleThings"]


words = ["Race", "Racist", "Black People", "Racism"]

data = {}

for sreddit in potential_subreddits:
    subreddit = reddit.subreddit(sreddit)
    for submission in subreddit.hot(limit=50):  # 6 posts and 50 comments
        # print(submission.title)
        # print('Submission ID: ', submission.id, '\n')
        for word in words:
            if word.lower() in submission.title.lower():
                data[submission.id] = submission.title
            else:
                None

sub_id = list(data.keys())
posts = list(data.values())

print(sub_id)
All_Comments = []

for sid in sub_id:
    post1 = reddit.submission(id=sid)

    post1.comments.replace_more(limit=None)
    for comments in post1.comments.list():
        All_Comments.append(comments.body)

print(All_Comments)
print('Total comments scraped = ', (len(All_Comments)))
