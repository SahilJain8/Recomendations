import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

df = pd.read_csv('Dataset.csv')
feature_df = df[['category', 'product']]

cat_dict = {'stationary': 1, 'accesories': 2, 'clothing': 3, 'decorative': 4, 'handicrafts': 5, 'homecare': 6,
            'selfcare': 7,
            'kitchen': 8, 'food': 9, 'toys': 10, 'Technology': 11, 'Office Supplies': 12, 'Furniture': 13}

df["cat_ordinal"] = df.category.map(cat_dict)

tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words='english')

tvf_matrix = tfv.fit_transform(df['category'])

sig = sigmoid_kernel(tvf_matrix, tvf_matrix)

indices = pd.Series(df.index, index=df['product']).drop_duplicates()

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return "hello"


@app.route('/predict', methods=['POST'])
def give_rec():
    if request.method == 'POST':
        text = request.form["text"]
        idx = indices[text]
        sig_scores = list(enumerate(sig[idx]))
        print(sig_scores)
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:6]
        prod_indices = [i[0] for i in sig_scores]
        return jsonify(list(df['product'].iloc[prod_indices]))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
