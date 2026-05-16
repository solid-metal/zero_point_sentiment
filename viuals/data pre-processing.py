# IMPORTING LIBARIES

import pandas as pd
import numpy as np
import re
import nltk
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#import dataset

df = pd.read_excel("amazon.csv.xlsx")
print(df.head())

 # null values & missing reviews

print(df.isnull().sum())
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(inplace=True)

# Combine review title and content into a single column

df['review'] = (
    df['review_title'] + " " +
    df['review_content']
)

# convertin rating from 1-5 to 0-10

df['score'] = df['rating'] * 2

# Preprocessing the reviews(lowercase, remove urls,
# remove special characters, remove extra spaces, remove stop words, stemming)
df['review'] = df['review'].str.lower()
df['review'] = df['review'].apply(
    lambda x: re.sub(r'http\S+', '', str(x))
)
df['review'] = df['review'].apply(
    lambda x: re.sub(r'[^a-zA-Z\s]', '', x)
)
df['review'] = df['review'].apply(
    lambda x: " ".join(x.split())
)
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess(text):

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_review'] = df['review'].apply(preprocess)

# Function to assign sentiment
def get_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating == 3:
        return "Neutral"

    else:
        return "Negative"

# Create sentiment column

df['sentiment'] = df['rating'].apply(get_sentiment)

 # Show output


print(df[['clean_review', 'score', 'sentiment']].head(20))

#Save New CSV File

df.to_csv("amazon_sentiment.csv", index=False)

