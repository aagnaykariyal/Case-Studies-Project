import praw  # Reddit crawler
import pandas as pd
import numpy as np
import seaborn as sns
import colorsys
from matplotlib import colors
import emoji
import re
# import en_core_web_sm
import spacy


import datetime as dt
from pprint import pprint
from itertools import chain

from praw.models import MoreComments

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer  # tokenize words

# ------ Downloading datasets ------#
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8) # default plot size
sns.set(style='whitegrid', palette='Dark2')

from wordcloud import WordCloud, ImageColorGenerator

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

All_Comments = []

for sid in sub_id:
    post1 = reddit.submission(id=sid)

    post1.comments.replace_more(limit=None)
    for comments in post1.comments.list():
        All_Comments.append(comments.body)

print('Total posts scraped = ', len(posts))
print('Total comments scraped = ', (len(All_Comments)))

sid = SentimentIntensityAnalyzer()

main_data = []
for data in All_Comments:
    main_data.append(data)
for data in posts:
    main_data.append(data)

sa = []
for dat in main_data:
    sc = sid.polarity_scores(dat)
    sa.append(sc)

print('The amount of responses = ', len(sa))
pprint(sa[:3])

sentiment_df = pd.DataFrame.from_records(sa)
sentiment_df['title'] = main_data
pprint(sentiment_df)

THRESHOLD = 0.2
conditions = [
    (sentiment_df['compound'] <= -THRESHOLD),
    ((sentiment_df['compound'] > -THRESHOLD) & (sentiment_df['compound'] < THRESHOLD)),
    (sentiment_df['compound'] >= THRESHOLD),
]

values = ['neg', 'neu', 'pos']
sentiment_df['label'] = np.select(conditions, values)
# nums = np.select(conditions, values)
# pprint(sentiment_df)

count = sentiment_df.label.value_counts()
pprint(count)


# sns.histplot(sentiment_df.label)

# ------ This code was used to check if the sentiment was accurate ------#

#
#
# def news_title_output(df, label):
#     res = df[df['label'] == label].title.values
#     print(f'{"=" * 20}')
#     print("\n".join(title for title in res))
#
#
# sent_sub = sentiment_df.groupby('label').sample(n=5, random_state=7)
#
# print("POSITIVE")
# news_title_output(sent_sub, "pos")
#
# print("NEGATIVE")
# news_title_output(sent_sub, "neg")
#
# print("NEUTRAL")
# news_title_output(sent_sub, "neu")

# ---------- To create a word cloud ---------#
# ---------- Tokenization ----------#

stop_words = stopwords.words('english')
print(stop_words)


def custom_tokenize(text):
    text = text.replace("'", "").replace("-", "").lower()  # removes single quotes and dashes

    # Splits words
    tk = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tk.tokenize(text)

    # removes stop words
    wrds = [w for w in tokens if not w in stop_words]
    return wrds


def tokens_2_wrds(df, label):
    titles = df[df['label'] == label].title
    tokens = titles.apply(custom_tokenize)
    wrds = list(chain.from_iterable(tokens))
    return wrds


pos_wrds = tokens_2_wrds(sentiment_df, 'pos')
neg_wrds = tokens_2_wrds(sentiment_df, 'neg')

pos_freq = nltk.FreqDist(pos_wrds)
neg_freq = nltk.FreqDist(neg_wrds)
# pprint(pos_freq.most_common(20))
# pprint(neg_freq.most_common(20))


def color_gen(col):
    color1 = col
    r, g, b = colors.to_rgb(color1)        # red, green, blue
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return 'hsl(' + str(h*360) + ', 100%%, %d%%)'


hsl_val = color_gen('xkcd:blood red')
hsl_val2 = color_gen('xkcd:navy blue')


def hsl_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return hsl_val % np.random.randint(0, 100)


def hsl_color_func2(word, font_size, position, orientation, random_state=None, **kwargs):
    return hsl_val2 % np.random.randint(0, 100)


wc_pos = WordCloud(
    background_color='Black',
    max_words=100,
    color_func=hsl_color_func,
    stopwords=stop_words            # Removing common words
)

wc_neg = WordCloud(
    background_color='Black',
    max_words=100,
    color_func=hsl_color_func2,
    stopwords=stop_words            # Removing common words
)

unique_string = (" ").join(pos_wrds)
unique_string2 = (" ").join(neg_wrds)

wc_pos.generate(unique_string)
wc_pos.to_file('pos_output.png')

wc_neg.generate(unique_string2)
wc_neg.to_file('neg_output.png')
# sentiment_df.to_csv('sentiment_analysis_data.csv')
