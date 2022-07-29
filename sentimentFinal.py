import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import altair as alt
import re
import datapane as dp
import contractions
from string import punctuation
import numpy as np

def regexClean(data):
    cleanList = []
    for sentence in data['cleaned_text']:
        if re.search('\. \.', sentence) != None:
            cleanList.append(re.sub('\. \.$', '.', sentence))
        elif re.search("\..", sentence) != None:
            cleanList.append(re.sub("\..$", ".", sentence))
        elif re.search("\. \. ", sentence) != None:
            cleanList.append(re.sub("\. \. $", ".", sentence))
        elif re.search(" \.", sentence) != None:
            cleanList.append(re.sub(" \.$", ".", sentence))
    return cleanList

def listToString(data):
    emptyString = ""
    for sentence in data:
        if not isinstance(sentence, str):
            emptyString += str(sentence) + " "
        else:
            emptyString += sentence + " "
    return emptyString

def sentimentAnalysis(dataframe):
    #instantiating sentiment analysis object
    sia = SentimentIntensityAnalyzer()

    #creating temp dataframe for capturing sentiment analysis variables
    dfTemp = pd.DataFrame()
    dfTemp['sentence'] = 0
    dfTemp['string'] = ['0']
    dfTemp['sentiment_type'] = 'blah'
    dfTemp['sentiment_score'] = 0
    
    #creating copy of temp df to collect all data in loop
    dfCollection = dfTemp
    
    #loop writes sentence and sentiment data to dfCollection
    i = 0
    for index, row in dataframe.iterrows():
        scores = sia.polarity_scores(row[0])
        dfTemp['sentence'] = i
        i += 1
        for key, value in scores.items():
            dfTemp['string'] = row[0]
            dfTemp['sentiment_type'] = key
            dfTemp['sentiment_score'] = value
            dfCollection = pd.concat([dfCollection, dfTemp])
        
    #dfSentiment is dfCollection without the dummy row and it only has the sentiment type 'compound'
    dfSentiment = dfCollection[dfCollection.sentence != '0']
    dfSentiment = dfCollection[dfCollection.sentiment_type == 'compound']
    
    #removing any duplicate rows
    dfSentiment = dfSentiment.drop_duplicates()

    #resetting index of df
    dfSentiment = dfSentiment.reset_index(drop=True)
    
    return dfSentiment

def pieChart(dataframe):
    positive = 0
    neutral = 0
    negative = 0
    
    for number in dataframe['sentiment_score']:
        if number < 0:
            negative += 1
        elif number > 0:
            positive += 1
        else:
            neutral += 1
    twoDeeArray = [["Negative", negative], ["Neutral", neutral], ["Positive", positive]]
    tsdf = pd.DataFrame(twoDeeArray)
    tsdf.columns = ["Sentiment", "Quantity"]
    pie = alt.Chart(tsdf).mark_arc().encode(
        theta = "Quantity",
        color = alt.Color("Sentiment", scale = alt.Scale(scheme = "set1")),
    )
    return pie

def scatterPlot(dataframe):
    scatter = alt.Chart(dfSentiment).mark_circle().encode(
        x = alt.X("sentence:Q", axis = alt.Axis(title = "")),
        y = alt.Y("sentiment_score:Q", axis = alt.Axis(title = "")),
        color = alt.Color('sentiment_score', scale = alt.Scale(scheme = 'spectral')),
        tooltip = ['string', 'sentiment_score']
    ).interactive()
    
    return scatter

def sentBarChart(dataframe):
    bars = alt.Chart(dataframe).mark_bar().encode(
        x = alt.X("sentiment_score:Q", axis = alt.Axis(title="")),
        y = alt.Y("count()", axis = alt.Axis(title="")),
        color = alt.Color(
            'sentiment_score:Q', 
            scale=alt.Scale(scheme='spectral'), 
            legend = alt.Legend(orient = "left"),
        ),
        tooltip = ["sentiment_score", "count()"]
    ).properties(
        title = "Sentiment Distribution"
    )
    return bars

def wordPlot(dataframe):
    wordPlot = alt.Chart(dataframe).mark_bar().encode(
        x = "Frequency",
        y = alt.Y("Word",
        sort = alt.EncodingSortField(field = "Frequency", order = "descending"),
        ),
        color = alt.Color("Frequency:Q", scale = alt.Scale(scheme = "spectral"), legend = None),
        tooltip = "Frequency"
    ).properties(
        title = "Overall Word Frequency"
    )
    return wordPlot

def wordFiltering(list):
    stopwords = nltk.corpus.stopwords.words("english")
    filteredWords = []
    for word in list:
        if word not in stopwords and word not in punctuation and word.isalpha():
            filteredWords.append(word)
    return filteredWords

#reading csv
#cnnCSV = pd.read_csv('E:\\OneDrive\\CS Projects\\Python\\tweets.csv', encoding = "ISO-8859-1")
cnnCSV = pd.read_csv("/Users/chrissimmerman/Library/CloudStorage/OneDrive-Personal/CS Projects/Python/tweets.csv", encoding = "ISO-8859-1")

#cleaning up excess punctuation using regex
recleanedData = regexClean(cnnCSV)

#creating string to hold tweets
stringy = listToString(recleanedData)

#processing for frequency analysis: converting all characters to lowercase and separating contractions
stringyWords = stringy.lower()
stringyWords = contractions.fix(stringyWords)

#tokenizing text into words and sentences
sentences = nltk.tokenize.sent_tokenize(stringy)
words = nltk.tokenize.word_tokenize(stringyWords)

#removing stopwords and punctuation from list of words
words = wordFiltering(words)

#Creating frequency distribution of words
fd = nltk.FreqDist(words).most_common(20)
df1 = pd.DataFrame(fd, columns = ['Word', 'Frequency'])

#converting list of sentences into dictionary
dict = { i : sentences[i] for i in range(0, len(sentences))}

#converting dictionary to pandas dataframe
df2 = pd.DataFrame.from_dict(dict, orient = "index", columns = ["Sentence"])

#Performing sentiment analysis
dfSentiment = sentimentAnalysis(df2)

#Creating a histogram
hist = sentBarChart(dfSentiment)

#Creating scatter plot
scatter = scatterPlot(dfSentiment)

#Creeating word frequency plot
wordFrequency = wordPlot(df1)

#Creating DataPane Report
report = dp.Report(
    dp.Page(
        dp.Plot(hist, name = "Plot2", caption = "Histogram"),
    )
)

#report.save('chart.html')
report.upload(name = "Histogram")
