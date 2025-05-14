from flask import Flask, render_template, request
import torch
from sentiment import DistilBERTSentimentAnalyzer  # Aynı dizinde olmalı

app = Flask(__name__)

# Modeli yükle
model_wrapper = DistilBERTSentimentAnalyzer()
model_wrapper.model.load_state_dict(torch.load('distilbert_sentiment.pt', map_location=torch.device('cpu')))
model_wrapper.model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    user_text = ""
    if request.method == "POST":
        user_text = request.form["text"]
        if user_text.strip() != "":
            prediction = model_wrapper.predict([user_text])[0]
    return render_template("index.html", prediction=prediction, user_text=user_text)

if __name__ == "__main__":
    app.run(debug=True)