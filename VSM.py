import re
import math
import spacy

spacy_nlp = spacy.load('en_core_web_sm')
# Getting stopwords from the file
stopwords = []
file = open("Stopword-List.txt", 'r')
stopwords = file.read().splitlines()
def tokenization():
    tokens = []
    for i in range(1, 448):
        doc_id = i
        f = open("Abstracts/" + str(doc_id) + ".txt", 'r')
        next(f)
        s = f.read()
        s = s.replace('\n', ' ')
        s = re.sub(r"can\'t", "can not", s)
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'re", " are", s)
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"\'t", " not", s)
        s = re.sub(r"\'ve", " have", s)
        s = re.sub(r"\'m", " am", s)
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub("[^A-Za-z0-9]+", " ", s)
        s = re.sub('  ', ' ', s)
        # converting each term into lowercase characters
        s = s.lower()
        # removing stopwords from the documents
        s = [words if words not in stopwords else '' for words in s.split(' ')]
        s = list(filter(None, s))
        s = ' '.join(s)
        # lemmatization
        doc_lemma = spacy_nlp(s)
        file_token = [token.lemma_ for token in doc_lemma]
        # creating posting list
        for x in file_token:
            tokens.append(x)

    # removing duplicates
    tokens = list(set(tokens))
    tokens = sorted(tokens)

    return tokens
'''
Retrieving all tokens in each document and storing in dictionary
'''
def doc_tokenization():
    tokens = {}
    for i in range(1, 448):
        doc_no = i
        f = open("Abstracts/" + str(doc_no) + ".txt", 'r')
        next(f)
        s = f.read()
        s = s.replace('\n', ' ')
        s = re.sub(r"can\'t", "can not", s)
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\'re", " are", s)
        s = re.sub(r"\'s", " is", s)
        s = re.sub(r"\'d", " would", s)
        s = re.sub(r"\'ll", " will", s)
        s = re.sub(r"\'t", " not", s)
        s = re.sub(r"\'ve", " have", s)
        s = re.sub(r"\'m", " am", s)
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub("[^A-Za-z0-9]+", " ", s)
        s = re.sub('  ', ' ', s)
        # converting each term into lowercase characters
        s = s.lower()
        # removing stopwords from the documents
        s = [word for word in s.split(' ') if word not in stopwords]
        s = list(filter(None, s))
        s = ' '.join(s)
        doci = spacy_nlp(s)
        file_token = [token.lemma_ for token in doci]
        file_token = sorted(file_token)
        key = i
        tokens[key] = file_token
    return tokens


'''
Calculating term frequency of the tokens in each documents
'''


def term_frequency(all_tokens, docu_tokens):
    tf = {}
    for i in range(1, 448):
        print(all_tokens)
        tf[i] = dict.fromkeys(all_tokens, 1)
        for j in docu_tokens[i]:
            tf[i][j] = tf[i][j] + 1
    return tf


'''
Calculating Inverse Document frequency
'''


def inverse_doument_frequency(tf, all_tokens):
    df = {}
    for i in all_tokens:
        df[i] = 0
        for j in range(1, 448):
            if (tf[j][i] > 0):
                df[i] += 1

    idf = {}
    for i in all_tokens:
        idf[i] = math.log(df[i] / 56)

    return idf


'''
Calculating TFIDF
'''


def tfidf(tf, idf, all_tokens):
    tfidf = {}

    for i in range(1, 448):
        tfidf[i] = {}
        for j in all_tokens:
            tfidf[i][j] = tf[i][j] * idf[j]

    return tfidf


'''
Preprocess query and Making Query vector
'''


def query_processing(query, token, idf):
    #q = pre_processing(query)
    s = query
    s = re.sub(r"can\'t", "can not", s)
    s = re.sub(r"n\'t", " not", s)
    s = re.sub(r"\'re", " are", s)
    s = re.sub(r"\'s", " is", s)
    s = re.sub(r"\'d", " would", s)
    s = re.sub(r"\'ll", " will", s)
    s = re.sub(r"\'t", " not", s)
    s = re.sub(r"\'ve", " have", s)
    s = re.sub(r"\'m", " am", s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub("[^A-Za-z0-9]+", " ", s)
    s = re.sub('  ', ' ', s)
    # converting each term into lowercase characters
    s = s.lower()
    # removing stopwords from the documents
    s = [word for word in s.split(' ') if word not in stopwords]
    s = list(filter(None, s))
    s = ' '.join(s)
    # lemmatization
    doc_lemma = spacy_nlp(s)
    q = [token.lemma_ for token in doc_lemma]
    qv = dict.fromkeys(token, 0)

    for i in q:
        if (i in token):
            qv[i] += 1
        else:
            print(i + " does not exists in dictionary!")

    for i in qv:
        qv[i] = qv[i] * idf[i]

    return qv
