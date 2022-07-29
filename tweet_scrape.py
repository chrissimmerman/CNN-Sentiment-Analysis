import tweepy
import pandas as pd

consumerKey = "XXX"
consumerSecret = "XXX"
accessToken = "XXX"
accessTokenSecret = "XXX"
bearerToken = "XXX"

client = tweepy.Client(
    consumer_key = consumerKey,
    consumer_secret = consumerSecret,
    access_token = accessToken,
    access_token_secret = accessTokenSecret,
    bearer_token = bearerToken,
    wait_on_rate_limit = True
)
cnnID = 759251

tweets = tweepy.Paginator(client.get_users_tweets, id = cnnID, max_results = 100).flatten(limit = 1000)

list = []
for page in tweets:
    list.append(page.data)

df = pd.DataFrame(list)

df.to_csv("tweets.csv", index=False)