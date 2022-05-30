import numpy as np
import VSM
import time
from flask import Flask, render_template,request
import operator
import json

app = Flask(__name__)

# '''
# Creating Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
# # '''
# all_tokens = VSM.tokenization()
# docu_tokens = VSM.doc_tokenization()
# tf = VSM.term_frequency(all_tokens,docu_tokens)
# idf = VSM.inverse_doument_frequency(tf,all_tokens)
# tfidf = VSM.tfidf(tf,idf,all_tokens)

# '''
# Saving Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
# # '''
# f = open('all_tokens.json', 'w')
# json.dump(all_tokens, f)
#
# f = open('docu_tokens.json', 'w')
# json.dump(docu_tokens, f)
#
# f = open('term_frequency.json', 'w')
# json.dump(tf, f)
#
# f = open('idf.json', 'w')
# json.dump(idf, f)
#
# f = open('tfidf.json', 'w')
# json.dump(tfidf, f)
#

'''
Loading Tokens,Document Tokens, Term Frequency, IDF, and TFIDF
'''

f_tokens = open('all_tokens.json')
for token in f_tokens:
    all_tokens = json.loads(token)

f_doc_token = open('docu_tokens.json')
docu_tokens = json.load(f_doc_token)

f_tf = open('term_frequency.json')
tf = json.load(f_tf)

f_idf = open('idf.json')
idf = json.load(f_idf)

f_tfidf = open('tfidf.json')
tfidf = json.load(f_tfidf)


def cosine_sim(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


print("Enter Query and 'quit' for exit")
query = input('Enter query:')
alpha = input('Enter alpha value:')


# Returning Relevant document retrieved
def documents_ret(a):
    docu = {}
    for i in range(1, 448):
        doc_no = i
        f = open("Abstracts/" + str(doc_no) + ".txt", 'r')
        next(f)
        s = f.read().replace('\n', ' ')

        key = 'Doc ' + str(doc_no)

        docu.setdefault(key, [])
        docu[key].append(s)

    documents = {}
    if (a):
        keys = list(a.keys())
        values = list(a.values())
        for i in range(len(keys)):
            speech = "Doc " + str(keys[i])
            documents.setdefault(speech, [])
            documents[speech].append(docu.get(speech))
            documents[speech].append(values[i])
    else:
        documents = {}

    return documents


'''
Default/Home Page is loaded
'''


@app.route('/')
def dictionary():
    return render_template('home.html')
'''
Query Processing Function take query and display result
'''
@app.route("/query", methods=['POST'])
def upload():
    # query processing start time
    start = time.time()
    query = request.form['query']
    alpha = request.form['alpha']
    print(alpha)
    if alpha == '':
        alpha = 0.001
    q = VSM.query_processing(query, all_tokens, idf)
    res = {}
    temp = 0
    vec1 = list(q.values())

    for x in range(1, 448):
        vec2 = list(tfidf[str(x)].values())
        sim = cosine_sim(vec1, vec2)
        if sim > float(alpha):
            temp = sim
            res[x] = temp

    res = dict(sorted(res.items(), key=operator.itemgetter(1), reverse=True))
    documents = documents_ret(res)

    print(res)
    return render_template('dictionary.html', dictionary=documents, num_docs=len(documents), quer=query)


if __name__ == '__main__':
    app.run()