import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('IMDB_Rating_2.csv').iloc[:1000]

# Preprocess the data
stopWords = set(stopwords.words('english'))  #english stopwords
stemmer = PorterStemmer()

def Preprocess(text):
    # Tokenize text
    words = nltk.word_tokenize(text)
    # Remove stop words and apply stemming
    processedWords = [stemmer.stem(word.lower()) for word in words if word.lower() not in stopWords and word.isalpha()]
    return ' '.join(processedWords)

# Apply preprocessing to the 'review' column
df['inputText'] = df['review'].apply(Preprocess)

# Define features (X) and labels (y)
X = df['inputText']
y = df['sentiment']

xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Bag of Words (BoW) vectorizer
vectorizer = CountVectorizer()
xTrain_bow = vectorizer.fit_transform(xTrain)
xTest_bow = vectorizer.transform(xTest)

model = LogisticRegression(max_iter=1000)
model.fit(xTrain_bow, yTrain)

yPred = model.predict(xTest_bow)
print("Logistic Regression Model Accuracy:", accuracy_score(yTest, yPred))