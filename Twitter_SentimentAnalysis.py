# Import Pandas library for Data manipulation and
# NLTK for performing NLP tasks
import pandas as pd
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')

# Import the file containing Tweet dataset using read command
tweets = pd.read_csv(r'C:\Users\VIKASH\Desktop\Tweets.csv')
print(tweets.shape)

# Drop/Remove the tweets with airline_sentiment_confidence less than .5(i.e 50%)
# Since classification of these tweets are not accurate
# axis = 0, specifies that we are dropping/removing the rows when confidence is <.5
tweets_df = tweets.drop(tweets[tweets['airline_sentiment_confidence'] < 0.5].index, axis=0)
print(tweets_df.shape)

# X -> stores input tweet text data
# y -> stores sentiments of the input tweet data
X = tweets_df['text']
y = tweets_df['airline_sentiment']

# Stopwords refer to words such as The,in,as,at etc which will be of no use for analysis
# Punctuation method is used to remove punctuation marks such as !,.'' etc which will be of no use for analysis
stop_words = stopwords.words('english')
punct = string.punctuation

# Stemmer actually finds the root of the words and replaces it.
# Ex Given,Giving are replaced with Root word "Give"
stemmer = PorterStemmer()

# Empty list to store cleaned data
cleaned_data = []

# Iterate through every tweet in the dataset
for i in range(len(X)):
    # replace every other character with white space if it's not an alphabet
    tweet = re.sub('[^a-zA-Z]', ' ', X.iloc[i])
    tweet = tweet.lower().split()
    # check each and every word and replace it with it's root word
    tweet = [stemmer.stem(word) for word in tweet if (word not in stop_words) and (word not in punct)]
    tweet = ' '.join(tweet)
    # Add the data to the cleaned_data list
    cleaned_data.append(tweet)

print(y)

# Specifies the ordering of the classification of sentiment
sentiment_ordering = ['negative', 'neutral', 'positive']

# For every word in y column, we are passing it through a function called as ‘sentiment_ordering.index()’
# The function replaces the word with it’s index in the sentiment_ordering list.
# For example, negative has index 0, neutral has index 1 and positive has index 2 in the list.
y = y.apply(lambda x: sentiment_ordering.index(x))

# CountVectorizer creates a matrix table, where each row represents a sentence and
# each word will have separate column for itself that represents it’s frequency.
cv = CountVectorizer(max_features=3000, stop_words=['virginamerica', 'unit'])
X_fin = cv.fit_transform(cleaned_data).toarray()
print(X_fin.shape)

# Multinomial NB model is a supervised learning algorithm.
model = MultinomialNB()

# Split the dataset into a training and testing section(testing size=30% of the actual data).
X_train, X_test, y_train, y_test = train_test_split(X_fin, y, test_size=0.25)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display the report
cf = classification_report(y_test, y_pred)
print(cf)
