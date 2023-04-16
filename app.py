from flask import Flask, render_template, request
import pandas as pd
import nltk
import pickle

app = Flask(__name__)

# Load dataset
amazon = pd.read_csv("amazon_product.csv")

amazon.drop('id', axis=1, inplace=True)

from nltk.stem.snowball import SnowballStemmer

stemer = SnowballStemmer('english')


def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemming = [stemer.stem(w) for w in tokens]

    return " ".join(stemming)


amazon['stemmed_tokens'] = amazon.apply(lambda row: tokenize_stem(row['Title'] + ' ' + row['Description']), axis=1)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfvector = TfidfVectorizer(tokenizer=tokenize_stem)


# functinos=========================
def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stemming = [stemer.stem(w) for w in tokens]

    return " ".join(stemming)


amazon['stemmed_tokens'] = amazon.apply(lambda row: tokenize_stem(row['Title'] + ' ' + row['Description']), axis=1)

def cos_sim(txt1, txt2):
    tfidmatrix = tfvector.fit_transform([txt1, txt2])
    similar_vectors = cosine_similarity(tfidmatrix)[0][1]
    return similar_vectors


def recommend_product(query):
    tokenized_query = tokenize_stem(query)
    amazon['similarity'] = amazon['stemmed_tokens'].apply(lambda x: cos_sim(tokenized_query, x))
    final_df = amazon.sort_values(by=['similarity'], ascending=False).head(10)[['Title', 'Description', 'Category']]
    return final_df



# create app
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    query = request.form['query']

    products = recommend_product(query)
    print(products.head())
    return render_template('index.html',products=products)



if __name__ == '__main__':
    app.run(debug=True)