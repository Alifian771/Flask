from flask import Flask, request, jsonify, render_template
from transformers import pipeline

# Inisialisasi Flask
app = Flask(__name__)

# Load model BERT untuk analisis sentimen
sentiment_analyzer = pipeline("sentiment-analysis on Movie script", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route('/')
def index():
    # Render halaman HTML (frontend)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        # Ambil data JSON dari permintaan POST
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Analisis sentimen menggunakan BERT
        result = sentiment_analyzer(text)
        return jsonify(result[0])  # Kembalikan hasil dalam format JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
