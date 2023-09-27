import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk
import re  # library for regular expression operations
import string  # for string operations

from nltk.corpus import stopwords  # module for stop words that come with NLTK
from nltk.stem import PorterStemmer  # module for stemming
from nltk.tokenize import TweetTokenizer  # module for tokenizing strings


# nltk.download("stopwords")  # Set of stopwords
# nltk.download('punkt')  # Split the text into number of sentences
# nltk.download('wordnet')  # Lemmatize


# Create a function to clean the tweets
def nltk_preprocess(text):
    text = re.sub(r'^RT[\s]+', '', text)  # remove old style retweet text "RT"
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)  # remove hyperlinks
    text = re.sub(r'#', '', text)  # Removing '#' hashtag
    text = re.sub(r'[^a-zA-Z]', " ", text)
    text = re.sub(r'@[A-Za-z0–9]+', '', text)  # Removing @mentions
    # Tokenize
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text = tokenizer.tokenize(text)
    stopwords_english = stopwords.words('english')  # Import the english stop words list from NLTK
    # Lemmatize
    lemmatize = nltk.WordNetLemmatizer()
    text = [lemmatize.lemmatize(word) for word in text if
            word not in stopwords_english and word not in string.punctuation]
    # Instantiate stemming class
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    # Combine words
    text = " ".join(text)
    return text


# Create a percentage function
def percentage(part, whole):
    return 100 * float(part) / float(whole)


def twt_scrap_nlp_sentiment(year_list):
    data = pd.DataFrame()
    df_sent = pd.DataFrame({'ticker': [],
                            'year': [],
                            'Number_of_tweets': [],
                            'num_pos': [],
                            'num_neg': [],
                            'num_neu': [],
                            'percent_pos': [],
                            'percent_neg': [],
                            'percent_neu': [],
                            'mean_pos': [],
                            'mean_neg': [],
                            'mean_neu': [],
                            'mean_compound_sentiment': []})
    query_list = input("Enter the ticker symbol (or the list of tickers) that should be mentioned \n Query: ")
    # As long as the query is valid (not empty or equal to '#')...
    if query_list != '':
        # year_list = list(input("Enter the last day of the year (or the list of them) that
        # you want to Scrape Twitter for \n Year in 'YYYY-12-31' format: "))
        # As long as the query is valid (not empty or equal to '#')...
        # if year_list != '':
        # noOfTweet = input("Enter the number of tweets you want to Analyze: ")
        # if noOfTweet != '' :
        n_of_days = input("Enter the number of last days in a year you want to Scrape Twitter for: ")
        if n_of_days != '':
            for query in query_list:
                for year in year_list:
                    # Creating list to append tweet data
                    tweets_list = []
                    now = pd.to_datetime(year)
                    now = now.strftime('%Y-%m-%d')
                    yesterday = pd.to_datetime(year) - dt.timedelta(days=int(n_of_days))
                    yesterday = yesterday.strftime('%Y-%m-%d')
                    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' lang:en since:' +
                                                                             yesterday + ' until:' +
                                                                             now + ' -filter:links').get_items()):
                        # if i > int(noOfTweet):
                        # break
                        tweets_list.append(
                            [tweet.date, tweet.content, tweet.likeCount, tweet.replyCount, tweet.retweetCount])
                        # time.sleep(0.1)
                    # Creating a dataframe from the tweets list above
                    df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text', 'Number_of_likes',
                                                            'Number_of_replies', 'Number_of_retweets'])
                    df['ticker'] = query
                    df['year'] = year

                    # Clean text data
                    df["Text"] = df["Text"].apply(nltk_preprocess)

                    # Sentiment Analysis

                    # Assigning Initial Values
                    positive = 0
                    negative = 0
                    neutral = 0
                    # Creating empty lists
                    tweet_list1 = []
                    sent_dict = pd.DataFrame()
                    neutral_list = []
                    negative_list = []
                    positive_list = []

                    # Iterating over the tweets in the dataframe
                    for tweet in df['Text']:
                        tweet_list1.append(tweet)
                        analyzer = SentimentIntensityAnalyzer().polarity_scores(tweet)
                        neg = analyzer['neg']
                        neu = analyzer['neu']
                        pos = analyzer['pos']
                        comp = analyzer['compound']
                        sent_dict_1 = pd.DataFrame([analyzer])
                        sent_dict = pd.concat([sent_dict, sent_dict_1])

                        if neg > pos:
                            negative_list.append(tweet)  # appending the tweet that satisfies this condition
                            negative += 1  # increasing the count by 1
                        elif pos > neg:
                            positive_list.append(tweet)  # appending the tweet that satisfies this condition
                            positive += 1  # increasing the count by 1
                        elif pos == neg:
                            neutral_list.append(tweet)  # appending the tweet that satisfies this condition
                            neutral += 1  # increasing the count by 1

                    df = df.reset_index(drop=True)
                    sent_dict = sent_dict.reset_index(drop=True)
                    tweet_df = pd.concat([df, sent_dict], axis=1, ignore_index=True)

                    positive1 = percentage(positive, len(df))  # percentage is the function defined above
                    negative1 = percentage(negative, len(df))
                    neutral1 = percentage(neutral, len(df))

                    now = pd.to_datetime(now)
                    year = now.to_period("Y")

                    mean_sent_dict = pd.DataFrame(sent_dict.mean()).T

                    data = pd.concat([data, tweet_df])

                    df_sent1 = pd.DataFrame({'ticker': [query],
                                             'year': [year],
                                             'Number_of_tweets': [len(df)],
                                             'num_pos': [positive],
                                             'num_neg': [negative],
                                             'num_neu': [neutral],
                                             'percent_pos': [positive1],
                                             'percent_neg': [negative1],
                                             'percent_neu': [neutral1],
                                             'mean_pos': [round(float(mean_sent_dict['pos']), 3)],
                                             'mean_neg': [round(float(mean_sent_dict['neg']), 3)],
                                             'mean_neu': [round(float(mean_sent_dict['neu']), 3)],
                                             'mean_compound_sentiment': [round(float(mean_sent_dict[
                                                                                         'compound']), 3)]})

                    df_sent = pd.concat([df_sent, df_sent1])
                    print("Collecting and analyzing tweets with" + query + "mentions during" +
                          year + ". Keep calm and wait, please:)")

                    positive = percentage(positive, len(df))  # percentage is the function defined above
                    negative = percentage(negative, len(df))
                    neutral = percentage(neutral, len(df))

                    # Converting lists to pandas dataframe
                    tweet_list1 = pd.DataFrame(tweet_list1)
                    neutral_list = pd.DataFrame(neutral_list)
                    negative_list = pd.DataFrame(negative_list)
                    positive_list = pd.DataFrame(positive_list)

                    # using len(length) function for counting
                    print("Since " + n_of_days + " days, there have been", len(tweet_list1),
                          "tweets on " + query, end='\n*')
                    print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n*')
                    print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n*')
                    print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n*')

                    # Creating PieCart

                    labels = ['Positive [' + str(round(positive)) + '%]', 'Neutral [' + str(round(neutral)) + '%]',
                              'Negative [' + str(round(negative)) + '%]']
                    sizes = [positive, neutral, negative]
                    colors = ['yellowgreen', 'blue', 'red']
                    patches, texts = plt.pie(sizes, colors=colors, startangle=90)
                    plt.style.use('default')
                    plt.legend(labels)
                    plt.title("Sentiment Analysis Result for keyword= " + query + "")
                    plt.axis('equal')
                    plt.show()
            return df_sent, data


year_list = ['2022-12-31']
df_sent_riot, data_2023_1 = twt_scrap_nlp_sentiment(year_list)
'''
# 326 22 november
# Get user input

Query1 = ['AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI', 'AVGO',
          'AZN', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CEG', 'CHTR', 'CMCSA',
          'COST', 'CPRT', 'CRWD', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'DDOG',
          'DLTR', 'DOCU', 'DXCM', 'EA', 'EBAY', 'EXC', 'FAST', 'FISV',
          'FTNT', 'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC',
          'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU',
          'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MRVL',
          'MSFT', 'MTCH', 'MU', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'ODFL',
          'OKTA', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL',
          'QCOM', 'REGN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS', 'SPLK',
          'SWKS', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRSN', 'VRTX',
          'WBA', 'WDAY', 'XEL', 'ZM', 'ZS']

Year = ['2022-12-31', '2021-12-31', '2020-12-31', '2019-12-31', '2018-12-31', '2017-12-31', '2016-12-31'] '''
