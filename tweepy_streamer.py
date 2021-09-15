from tweepy import API
from tweepy import OAuthHandler
from textblob import TextBlob
import twitter_credentials
import re
import os
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TwitterClient:

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


class TwitterAuthenticator:

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TweetAnalyzer:

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets):
        rdf = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        rdf['ID'] = np.array([tweet.id for tweet in tweets])  
        return rdf

    def tweets_to_data_frame2(self, tweets2):
        adf = pd.DataFrame(data=[tweet.text for tweet in tweets2], columns=['Tweets'])
        adf['ID'] = np.array([tweet.id for tweet in tweets2])   
        return adf




if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()

    tweets = api.search(q="@narendramodi", count=10000, lang="en")
    tweets2 = api.search(q="@RahulGandhi", count=10000, lang="en")

    rdf = tweet_analyzer.tweets_to_data_frame(tweets)
    adf = tweet_analyzer.tweets_to_data_frame2(tweets2)

    rdf['Sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in rdf['Tweets']])
    adf['Sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in adf['Tweets']])


    print("\nDonald Trump TWEETS\n")
    print(rdf)
    print("\nJoe Biden TWEETS\n")
    print(adf)

    exists = os.path.isfile(r'DonaldTrump_DB.csv')
    exists2 = os.path.isfile(r'JoeBiden_DB.csv')

    if exists:
        rdf.to_csv(r'DonaldTrump_DB.csv', mode='a', header=False)
    else:
        rdf.to_csv(r'DonaldTrump_DB.csv', mode='a')
    
    if exists2:
        adf.to_csv(r'JoeBiden_DB.csv', mode='a', header=False)
    else:
        adf.to_csv(r'JoeBiden_DB.csv', mode='a')

    rdf2 = pd.read_csv(r"DonaldTrump_DB.csv")
    adf2 = pd.read_csv(r"JoeBiden_DB.csv")

    rdf2 = rdf2.drop_duplicates(subset='Tweets')
    adf2 = adf2.drop_duplicates(subset='Tweets')

    rdf2.to_csv(r'DonaldTrump_DB.csv', index=False, mode='w')
    adf2.to_csv(r'JoeBiden_DB.csv', index=False, mode='w')

   
# Sentiment Distribution Graph
    a1 = rdf2[rdf2['Sentiment'] == -1].shape[0]
    a2 = adf2[adf2['Sentiment'] == -1].shape[0]
    b1 = rdf2[rdf2['Sentiment'] == 0].shape[0]
    b2 = adf2[adf2['Sentiment'] == 0].shape[0]
    c1 = rdf2[rdf2['Sentiment'] == 1].shape[0]
    c2 = adf2[adf2['Sentiment'] == 1].shape[0]
    data = {'Donald Trump': {'-1': a1, '0': b1, '1': c1}, 'Joe Biden': {'-1': a2, '0': b2, '1': c2}}
    df = pd.DataFrame(data)
    df.plot(kind='bar', figsize=(10, 4))
    plt.xlabel("Sentiment")
    plt.ylabel("Tweets Count")
    plt.title("Sentiment Distribution Graph")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\2.png', bbox_inches='tight')

# Popular hashtags
    hashs_r = rdf2["Tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
    hashs_r.columns = ["Hashtags", "Count"]
    hashs_a = adf2["Tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
    hashs_a.columns = ["Hashtags", "Count"]
    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    ax = sns.barplot(x="Count", y="Hashtags", data=hashs_r[:25], palette="seismic", linewidth=1, edgecolor="k" * 25)
    plt.grid(True)
    for i, j in enumerate(hashs_r["Count"][:25].values):
        ax.text(3, i, j, fontsize=10, color="white")
    plt.title("Popular hashtags used for Donald Trump")
    plt.subplot(212)
    ax1 = sns.barplot(x="Count", y="Hashtags", data=hashs_a[:25], palette="seismic", linewidth=1, edgecolor="k" * 25)
    plt.grid(True)
    for i, j in enumerate(hashs_a["Count"][:25].values):
        ax1.text(.3, i, j, fontsize=10, color="white")
    plt.title("Popular hashtags used for Joe Biden")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\6.png', bbox_inches='tight')

# Popular words in tweets
    pop_words_r = (rdf2["Tweets"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index().sort_values(by=[0], ascending=False))
    pop_words_r.columns = ["Word", "Count"]
    pop_words_r["Word"] = pop_words_r["Word"].str.upper()

    pop_words_a = (adf2["Tweets"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index().sort_values(by=[0], ascending=False))
    pop_words_a.columns = ["Word", "Count"]
    pop_words_a["Word"] = pop_words_a["Word"].str.upper()

    plt.figure(figsize=(12, 25))
    plt.subplot(211)
    ax = sns.barplot(x="Count", y="Word", data=pop_words_r[:30], linewidth=1, edgecolor="k" * 30, palette="Reds")
    plt.title("popular words in tweets - Donald Trump")
    plt.grid(True)
    for i, j in enumerate(pop_words_r["Count"][:30].astype(int)):
        ax.text(.8, i, j, fontsize=9)
    plt.subplot(212)
    ax1 = sns.barplot(x="Count", y="Word", data=pop_words_a[:30], linewidth=1, edgecolor="k" * 30, palette="Blues")
    plt.title("Popular words in tweets - Joe Biden")
    plt.grid(True)
    for i, j in enumerate(pop_words_a["Count"][:30].astype(int)):
        ax1.text(.8, i, j, fontsize=9)
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\7.png', bbox_inches='tight')

    fields=['ID','Tweets']
    field=['ID']
    temp1=pd.read_csv('DonaldTrump_DB.csv',index_col = 'ID', usecols=fields)
    temp2=pd.read_csv('JoeBiden_DB.csv',index_col = 'ID', usecols=fields)
    temp1.to_csv(r'DonaldTrump_Test.csv')
    temp2.to_csv(r'JoeBiden_Test.csv')
   


