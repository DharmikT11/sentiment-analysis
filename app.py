import nltk
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request

# Download NLTK resources
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')

# Initialize NLTK stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)  # Join tokens into a single string

# Load Twitter samples dataset
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Combine the tweets and create labels
tweets = positive_tweets + negative_tweets
labels = ['pos'] * len(positive_tweets) + ['neg'] * len(negative_tweets)

# Preprocess tweets
processed_tweets = [preprocess_text(tweet) for tweet in tweets]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(processed_tweets)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Initialize Flask app
app = Flask(__name__)

# Index route
@app.route('/')
def index():
    return render_template('index.html', project_name='SentimentSense')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    processed_text = preprocess_text(text)
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    prediction = nb_classifier.predict(text_vectorized)
    
    # Map class labels to human-readable sentiments
    sentiment_map = {'neg': 'Negative', 'pos': 'Positive'}
    
    # Map the predicted class label to the corresponding sentiment
    predicted_sentiment = sentiment_map[prediction[0]]
    
    return render_template('result.html', prediction=predicted_sentiment)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
