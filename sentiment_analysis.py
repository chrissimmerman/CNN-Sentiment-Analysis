import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import altair as alt
import re
import datapane as dp
import contractions
from string import punctuation

# Function to clean punctuation from a Pandas Series.
def regexClean(data):
    patterns = ['\. \.', "\..", "\. \. ", " \."]
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub('|'.join(patterns), '.', x))
    return data['cleaned_text'].tolist()

# Function to perform sentiment analysis on a Pandas DataFrame.
def sentimentAnalysis(dataframe):
    sia = SentimentIntensityAnalyzer()
    dfCollection = []

    for index, row in dataframe.iterrows():
        scores = sia.polarity_scores(row[0])
        for key, value in scores.items():
            dfCollection.append([index, row[0], key, value])

    dfSentiment = pd.DataFrame(dfCollection, columns=['sentence', 'string', 'sentiment_type', 'sentiment_score'])
    dfSentiment = dfSentiment[dfSentiment['sentiment_type'] == 'compound'].drop_duplicates().reset_index(drop=True)
    return dfSentiment

# Function to plot pie chart of sentiment scores
def pieChart(dataframe):
    sentiment_counts = dataframe['sentiment_score'].apply(lambda x: 'Positive' if x>0 else ('Neutral' if x==0 else 'Negative')).value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Quantity"]
    pie = alt.Chart(sentiment_counts).mark_arc().encode(
        theta = "Quantity",
        color = alt.Color("Sentiment", scale = alt.Scale(scheme = "set1"))
    )
    return pie

# Function to plot scatter chart of sentiment scores
def scatterPlot(dataframe):
    scatter = alt.Chart(dataframe).mark_circle().encode(
        x = alt.X("sentence:Q", axis = alt.Axis(title = "")),
        y = alt.Y("sentiment_score:Q", axis = alt.Axis(title = "")),
        color = alt.Color('sentiment_score', scale = alt.Scale(scheme = 'spectral')),
        tooltip = ['string', 'sentiment_score']
    ).interactive()
    return scatter

# Function to plot bar chart of sentiment scores
def sentBarChart(dataframe):
    dataframe['sentiment_score'] = pd.cut(dataframe['sentiment_score'], bins=np.arange(-1, 1, 0.1))
    bars = alt.Chart(dataframe).mark_bar().encode(
        x = alt.X("sentiment_score:Q", axis = alt.Axis(title="")),
        y = alt.Y("count()", axis = alt.Axis(title="")),
        color = alt.Color('sentiment_score:Q', scale=alt.Scale(scheme='spectral'), legend = alt.Legend(orient = "left")),
        tooltip = ["sentiment_score", "count()"]
    ).properties(title = "Sentiment Distribution")
    return bars

# Function to plot word frequency
def wordPlot(dataframe):
    wordPlot = alt.Chart(dataframe).mark_bar().encode(
        x = "Frequency",
        y = alt.Y("Word", sort = alt.EncodingSortField(field = "Frequency", order = "descending")),
        color = alt.Color("Frequency:Q", scale = alt.Scale(scheme = "spectral"), legend = None),
        tooltip = "Frequency"
    ).properties(title = "Overall Word Frequency")
    return wordPlot

# Function to filter stopwords and non-alphabetic words
def wordFiltering(list):
    stopwords = nltk.corpus.stopwords.words("english")
    filteredWords = [word for word in list if word not in stopwords and word not in punctuation and word.isalpha()]
    return filteredWords

# Main function
def main():
    #reading csv
    cnnCSV = pd.read_csv("/Users/chrissimmerman/Library/CloudStorage/OneDrive-Personal/CS Projects/Python/tweets.csv", encoding = "ISO-8859-1")

    #cleaning up excess punctuation using regex
    recleanedData = regexClean(cnnCSV)

    #creating string to hold tweets
    stringy = " ".join(recleanedData)

    #processing for frequency analysis: converting all characters to lowercase and separating contractions
    stringyWords = contractions.fix(stringy.lower())

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

    #Creating word frequency plot
    wordFrequency = wordPlot(df1)

    #Creating DataPane Report
    report = dp.Report(
        dp.Page(
            dp.Plot(hist, name = "Plot2", caption = "Histogram"),
        )
    )

    #report.save('chart.html')
    report.upload(name = "Histogram")

# Call the main function
if __name__ == "__main__":
    main()
