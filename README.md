# SentimentSense - Twitter-Based Sentiment Analysis Web App

SentimentSense is a Flask web application that uses machine learning and NLP to classify the sentiment (positive or negative) of user-inputted text, trained on real Twitter data. The app provides an instant and interactive sentiment analysis experience in your browser.

## 📝 Description

- **Input:** Enter any text in the web interface.
- **Output:** The app predicts whether the sentiment is Positive or Negative, powered by a trained Naive Bayes classifier using a TF-IDF vectorizer.
- **Data:** Utilizes NLTK’s `twitter_samples` dataset for authentic tweet-based sentiment training.
- **Tech Stack:** Python, Flask, NLTK, scikit-learn, HTML/CSS (Bootstrap).

## 🚀 Features

- Clean web interface to enter and analyze text sentiment.
- Real-time prediction with instant feedback.
- Preprocessing with tokenization, stopword removal, and TF-IDF vectorization.
- Model training using Naive Bayes for effective classification.
- Ready to deploy and easy to customize with your own data or models.

## 🌐 Demo

1. Run the app and visit: `http://127.0.0.1:5000/`
2. Input your text in the provided box.
3. Instantly see if your text is "Positive" or "Negative"!

## 🛠️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
```

## 📁 Project Structure

```
sentiment-analysis/
├── app.py                  # Main Flask application and ML logic
├── static/
│   └── styles.css          # Custom styles
├── templates/
│   ├── index.html          # Main input page
│   └── result.html         # Output/result page
├── requirements.txt        # Python dependencies
└── README.md
```

## 🤖 Model Details

- **Dataset:** NLTK `twitter_samples` (positive & negative tweets)
- **Vectorizer:** `TfidfVectorizer` from scikit-learn
- **Classifier:** `MultinomialNB` (Naive Bayes)
- **Preprocessing:** Tokenization, lowercasing, stopword removal

## ✍️ Example

Try entering:
- "I love this product!" → Positive
- "This is the worst experience ever." → Negative

## 🎨 Screenshots

![Main Page Example](assets/example_main.png)
![Result Example](assets/example_result.png)

## 🤝 Contribution

Feel free to fork, enhance, or open issues/PRs for improvements!

## 📄 License

MIT License

---

**Short Description:**  
A Flask-based web app for real-time sentiment analysis of text using machine learning, trained on Twitter data.  
