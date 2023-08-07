from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the vectorizer and classifier
vectorizer = joblib.load('tfidf_vectorizer.pkl')
classifier = joblib.load('tfidf_classifier.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        # Transform the input text using the fitted vectorizer
        text_tfidf = vectorizer.transform([text])

        # Make prediction using the loaded classifier
        prediction = classifier.predict(text_tfidf)

        # Interpret the prediction label
        if prediction[0] == 0:
            result = "hate-speech"
        elif prediction[0] == 1:
            result = "offensive-speech"
        else:
            result = "neither"

        return render_template('result.html', result=result, text=text)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
