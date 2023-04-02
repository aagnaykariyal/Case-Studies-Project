import praw                         # Reddit crawler
import pandas as pd
import numpy as np
import seaborn as sns
import colorsys
import nltk
from matplotlib import colors
from pprint import pprint
from itertools import chain
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# ------ Downloading datasets ------#
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('stopwords')

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)  # default plot size
sns.set(style='whitegrid', palette='Dark2')

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


words = ["Race", "Racist", "Black People", "Racism"]  # Words to filter

data = {}

for sreddit in potential_subreddits:
    subreddit = reddit.subreddit(sreddit)
    for submission in subreddit.hot(limit=50):
        for word in words:
            if word.lower() in submission.title.lower():  # Comparing data with our dataset
                data[submission.id] = submission.title    # Inserting posts and their id to a dict

sub_id = list(data.keys())  # Submission ID of each post
posts = list(data.values())  # List of all posts

All_Comments = []

for sid in sub_id:  # Loop to iterate through the comments of all posts
    post1 = reddit.submission(id=sid)

    post1.comments.replace_more(limit=None)
    for comments in post1.comments.list():
        All_Comments.append(comments.body)

print('Total posts scraped = ', len(posts))
print('Total comments scraped = ', (len(All_Comments)))

sid = SentimentIntensityAnalyzer()  # Assigning Vader Sentiment Analyzer

main_data = []  # Creating a List to accomodate both posts and comments
for data in All_Comments:
    main_data.append(data)
for data in posts:
    main_data.append(data)

sa = []  # Creating a List to accomodate the polarity scores of the posts and comments
for dat in main_data:
    sc = sid.polarity_scores(dat)
    sa.append(sc)

print('The amount of responses = ', len(sa))
pprint(sa[:3])

sentiment_df = pd.DataFrame.from_records(sa)  # Creating a DataFrame from the polarity data
sentiment_df['title'] = main_data  # Inserting posts and comments in a column called title

THRESHOLD = 0.2  # Threshold to help assign values
conditions = [  # Conditions to help determine values
    (sentiment_df['compound'] <= -THRESHOLD),
    ((sentiment_df['compound'] > -THRESHOLD) & (sentiment_df['compound'] < THRESHOLD)),
    (sentiment_df['compound'] >= THRESHOLD),
]

values = ['neg', 'neu', 'pos']
sentiment_df['label'] = np.select(conditions, values)  # Insertion of values according to the polarity score using numpy
# nums = np.select(conditions, values)
# pprint(sentiment_df)

count = sentiment_df.label.value_counts()  # Doing the count of the polarity scores according to the values
pprint(count)

# ---------- To create a word cloud ---------#
# ---------- Tokenization ----------#

stop_words = stopwords.words('english')  # Gathering words to be removed


def custom_tokenize(text):  # Function to create tokens
    text = text.replace("'", "").replace("-", "").lower()  # removes single quotes and dashes

    # Splits words
    tk = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tk.tokenize(text)

    # Removing stop words
    wrds = [w for w in tokens if not w in stop_words]
    return wrds


def tokens_2_wrds(df, label):  # Function to create a list of relevant tokens
    titles = df[df['label'] == label].title
    tokens = titles.apply(custom_tokenize)
    wrds = list(chain.from_iterable(tokens))
    return wrds


pos_wrds = tokens_2_wrds(sentiment_df, 'pos')  # List of positive tokens
neg_wrds = tokens_2_wrds(sentiment_df, 'neg')  # List of negative tokens

pos_freq = nltk.FreqDist(pos_wrds)  # Counting the frequency of positive tokens
neg_freq = nltk.FreqDist(neg_wrds)  # Counting the frequency of negative tokens


# ---------- Word Cloud ----------#
# ---------- Assigning Color to Font in Word Cloud ----------#
def color_gen(col):  # Function to generate color
    color1 = col
    r, g, b = colors.to_rgb(color1)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return 'hsl(' + str(h*360) + ', 100%%, %d%%)'


# Assigning respective hsl values of the color
hsl_val = color_gen('xkcd:blood red')
hsl_val2 = color_gen('xkcd:navy blue')


# Functions to create a color gradient
def hsl_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return hsl_val % np.random.randint(0, 100)


def hsl_color_func2(word, font_size, position, orientation, random_state=None, **kwargs):
    return hsl_val2 % np.random.randint(0, 100)


# ---------- Generation of Word Cloud ----------#

wc_pos = WordCloud(                 # Word Cloud generation of Positive Sentiments
    background_color='Black',
    max_words=100,
    color_func=hsl_color_func,      # Assigning Color to Font
    stopwords=stop_words            # Removing common words
)

wc_neg = WordCloud(                 # Word Cloud generation of Negative Sentiments
    background_color='Black',
    max_words=100,
    color_func=hsl_color_func2,     # Assigning Color to Font
    stopwords=stop_words            # Removing common words
)

unique_string = " ".join(pos_wrds)
unique_string2 = " ".join(neg_wrds)


# ----------- Output ----------#

wc_pos.generate(unique_string)
wc_pos.to_file('pos_output.png')  # Positive Word Cloud being exported

wc_neg.generate(unique_string2)
wc_neg.to_file('neg_output.png')  # Negative Word Cloud being exported
sentiment_df.to_csv('sentiment_analysis_data.csv')  # Polarity Score DataFrame
