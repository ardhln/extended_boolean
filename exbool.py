import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def create_feature_vector(tokens, word_set, query_expression):
    feature_vector = []
    for word in word_set:
        if query_expression == "AND":
            if word in tokens:
                feature_vector.append(1)
            else:
                feature_vector.append(0)
        elif query_expression == "OR":
            if word in tokens:
                feature_vector.append(1)
            else:
                feature_vector.append(1)  # Kata tidak ada dalam dokumen tetapi dianggap cocok karena operasi OR
        elif query_expression == "NOT":
            if word in tokens:
                feature_vector.append(0)  # Kata ada dalam dokumen tetapi dianggap tidak cocok karena operasi NOT
            else:
                feature_vector.append(1)
    return feature_vector

def calculate_match_percentage(vector_doc, vector_query):
    matches = sum(doc_word and query_word for doc_word, query_word in zip(vector_doc, vector_query))
    total_words = sum(vector_query)
    match_percentage = (matches / total_words) * 100
    return match_percentage

# Ambil input dokumen dan query
D = input("Masukkan dokumen: ")
query = input("Masukkan query: ")
logic_operator = input("Masukkan operator logika (AND/OR/NOT): ")

# Lakukan tokenisasi dan normalisasi teks
tokens_D = word_tokenize(D)
tokens_query = word_tokenize(query)

# Buat set kata unik dari dokumen dan query
word_set = set(tokens_D + tokens_query)

# Konstruksi vektor fitur
vector_D = create_feature_vector(tokens_D, word_set, logic_operator)
vector_query = create_feature_vector(tokens_query, word_set, "AND")  # Menggunakan operasi AND sebagai default

# Hitung persentase kecocokan
percentage = calculate_match_percentage(vector_D, vector_query)

# Analisis hasil
print("Persentase kecocokan dokumen dengan query:", percentage)
