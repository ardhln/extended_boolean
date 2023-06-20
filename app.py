from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from math import log10

app = Flask(__name__)

class ExtendedBooleanRetrieval:
    def __init__(self, documents):
        self.documents = documents
        self.tokenized_docs = []
        self.filtered_docs = []
        self.stemmed_docs = []
        self.inverted_index = {}
        self.term_freq = {}
        self.df = {}
        self.idfi = {}
        self.tfidf = {}
        self.doc_length = {}
        self.stop_words = set(stopwords.words("indonesian"))
        self.stemmer = PorterStemmer()
        self.N = 0
        self.classes = []

    def preprocess_documents(self):
        self.tokenized_docs = [word_tokenize(doc.lower()) for doc in self.documents]
        self.filtered_docs = [[word for word in doc if word not in self.stop_words] for doc in self.tokenized_docs]
        self.stemmed_docs = [[self.stemmer.stem(word) for word in doc] for doc in self.filtered_docs]

    def build_inverted_index(self):
        for i, doc in enumerate(self.stemmed_docs):
            for term in doc:
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(i+1)
                if term not in self.term_freq:
                    self.term_freq[term] = {}
                if i+1 not in self.term_freq[term]:
                    self.term_freq[term][i+1] = 0
                self.term_freq[term][i+1] += 1

    def calculate_df_idfi(self):
        self.N = len(self.documents)
        for term, postings in self.inverted_index.items():
            self.df[term] = len(postings)
            self.idfi[term] = log10(self.N/self.df[term])

    def calculate_tfidf(self):
        for term, doc_freq in self.term_freq.items():
            self.tfidf[term] = {}
            for doc_id, freq in doc_freq.items():
                self.tfidf[term][doc_id] = freq * self.idfi[term]

    def normalize_tfidf(self):
        for doc_id, doc in enumerate(self.stemmed_docs):
            self.doc_length[doc_id+1] = sum(self.tfidf[term][doc_id+1]**2 for term in doc)
            self.doc_length[doc_id+1] = self.doc_length[doc_id+1]**0.5
            for term in doc:
                self.tfidf[term][doc_id+1] /= self.doc_length[doc_id+1]

    def process_query(self, query, operator='OR'):
        query_tokens = word_tokenize(query.lower())
        query_terms = [self.stemmer.stem(word) for word in query_tokens if word not in self.stop_words]

        result_docs = set(range(1, len(self.documents)+1))

        if operator == 'AND':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.intersection(self.inverted_index[term])
        elif operator == 'OR':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.union(self.inverted_index[term])
        elif operator == 'NOT':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.difference(self.inverted_index[term])

        return result_docs

    def calculate_rsv(self, query, doc_id):
        query_tokens = word_tokenize(query.lower())
        query_terms = [self.stemmer.stem(word) for word in query_tokens if word not in self.stop_words]

        rsv = 0
        for term in query_terms:
            if term in self.tfidf and doc_id in self.tfidf[term]:
                rsv += self.tfidf[term][doc_id]

        return rsv

    def normalize_weight(self, weight, operator):
        if operator == 'OR':
            return weight / len(self.documents)
        elif operator == 'AND':
            return (1 - weight) / len(self.documents)
        elif operator == 'NOT':
            return 1 - weight

    def process_query_with_weight(self, query, operator='OR'):
        query_tokens = word_tokenize(query.lower())
        query_terms = [self.stemmer.stem(word) for word in query_tokens if word not in self.stop_words]

        result_docs = set(range(1, len(self.documents)+1))
        weight = 0

        if operator == 'AND':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.intersection(self.inverted_index[term])
                    weight += self.idfi[term]
            weight = self.normalize_weight(weight, 'AND')
        elif operator == 'OR':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.union(self.inverted_index[term])
                    weight += self.idfi[term]
            weight = self.normalize_weight(weight, 'OR')
        elif operator == 'NOT':
            for term in query_terms:
                if term in self.inverted_index:
                    result_docs = result_docs.difference(self.inverted_index[term])
                    weight += self.idfi[term]
            weight = self.normalize_weight(weight, 'NOT')

        return result_docs, weight

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    operator = request.form['operator']

    result, weight = retrieval.process_query_with_weight(query, operator)

    response = []
    for doc_id in result:
        rsv = retrieval.calculate_rsv(query, doc_id)
        rsv = retrieval.normalize_weight(rsv, operator)
        total_rsv = weight + rsv
        response.append({'document_id': doc_id, 'rsv': total_rsv, 'class': retrieval.classes[doc_id-1]})

    return render_template('search_results.html', response=response)


if __name__ == '__main__':
    # Dokumen yang akan diindeks
    documents = [
        ("Partai Golkar Demokrat tanding kampanye 2009", "politik"),
        ("Tanding pertama Persema Persebaya Malang", "olahraga"),
        ("Besar wasit tanding sepakbola adil", "olahraga"),
        ("Partai Demokrat menang pemilu 2009 figur SBY", "politik"),
        ("Tanding sepakbola Persebaya kampanye pemilu 2009 tunda", "?")
    ]

    retrieval = ExtendedBooleanRetrieval([doc[0] for doc in documents])
    retrieval.preprocess_documents()
    retrieval.build_inverted_index()
    retrieval.calculate_df_idfi()
    retrieval.calculate_tfidf()
    retrieval.normalize_tfidf()
    retrieval.classes = [doc[1] for doc in documents]

    app.run()
