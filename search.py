import tokenizer
import pickle
import math
from index import to_postingDict, to_string
import numpy as np
from numpy.linalg import norm
#import time

# load hash table and term index
hash_table = pickle.load(open('hashtable.p', 'rb'))
term_index = pickle.load(open('termindex.p', 'rb'))
# total number of terms
N = len(hash_table)
# top K results to display
K = 10

def ask():
    print('Enter: ', end='')
    query = input()
    return query

def parseQuery(query):
    return tokenizer.tokenize(query)

def retrieveRelevantPostings(terms):
    query_postings = list()
    with open('index.txt', 'r', encoding='utf-8') as f:
        # retrieve lines associated with each term
        for t in terms:
            f.seek(term_index[t])
            line = f.readline()
            query_postings.append(line)
    # convert lines into a postingDict
    return to_postingDict(query_postings)

def queryVector(terms, postingDict):
    tfidfDict = dict()
    # compute tf
    for t in terms:
        if t not in tfidfDict:
            tfidfDict[t] = 0
        tfidfDict[t] += 1
    # compute tfidf with log normalization
    for t in tfidfDict:
        tfidf = (1.0 + math.log10(tfidfDict[t])) * math.log10(N/len(postingDict[t]))
        tfidfDict[t] = tfidf
    # convert values to numpu array
    return np.array([v for v in tfidfDict.values()])

def docVectors(postingDict):
    docVectDict = dict()
    # accumulate score for each doc
    for term in postingDict:
        for p in postingDict[term]:
            # each term's frequency in each doc's vector is initialized to 0
            if p.docID not in docVectDict:
                docVectDict[p.docID] = dict(zip(postingDict.keys(), [0]*len(postingDict)))
            docVectDict[p.docID][term] = p.tfidf
    return docVectDict

def rank(queryVect, docVectDict):
    results = list()
    for doc in docVectDict:
        # convert each doc's vector to a numpy array
        docVect = np.array([v for v in docVectDict[doc].values()])
        # compute cosine similarity between two the query and the doc and store in the form (docID, score)
        results.append((doc, np.dot(queryVect, docVect)/(norm(queryVect)*norm(docVect))))
    # sort results by the scores from highest to lowest
    results.sort(key = lambda x: x[1], reverse=True)
    return results

if __name__ == '__main__':
    while (True):
        # prompt for user input
        query = ask()

        #start = time.time()
        # exit search if user enters exit search
        if (query == 'exit search'):
            break

        # tokenize query and remove terms that are not present in the term index
        terms = [t for t in parseQuery(query) if t in term_index]

        # get postings for each query term
        postingDict = retrieveRelevantPostings(terms)

        # calculate tfidf vector for query
        queryVect =  queryVector(terms, postingDict)

        # calculate tfidf vector for each doc
        docVectDict = docVectors(postingDict)

        # rank docs using cosine similarity with normalization
        results = rank(queryVect, docVectDict)
        
        # display results
        for doc, score in results[:K]:
            print(hash_table[doc])

        #end = time.time()
        #print(end - start)


