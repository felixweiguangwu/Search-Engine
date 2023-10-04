import json
from bs4 import BeautifulSoup
from tokenizer import tokenize
from posting import Posting
import os
import math
from decimal import Decimal, ROUND_UP
from contextlib import ExitStack
import pickle

def find_files(directory):
    files = list()
    # find all files in the given directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # if a path is a file, append to the list of files
        # else if a path is a directory, call itself on the path
        if os.path.isfile(f):
            files.append(f)
        elif os.path.isdir(f):
            for x in find_files(f):
                files.append(x)
    return files


def computePostings(docID, tokens, bonus):
    postingDict = dict()
    # compute term frequency of each token
    for token in tokens:
        if token not in postingDict:
            postingDict[token] = Posting(docID)
        # add bonus to token's tfidf
        if token in bonus:
            postingDict[token].tfidf += bonus[token]
        # update token's frequency
        postingDict[token].tfidf += 1
    # apply log weighting on each token's tf
    for p in postingDict:
        postingDict[p].tfidf = Decimal(str(1 + math.log10(postingDict[p].tfidf))).quantize(Decimal('.01'), rounding=ROUND_UP)
    return postingDict


def update_inverted_index(postingDict, inverted_index):
    # add all postings in postingDict to the inverted index
    for k, v in postingDict.items():
        if k not in inverted_index:
            inverted_index[k] = list()
        inverted_index[k].append(v)


def parse_doc(filename, hash_num, hash_table, inverted_index):
    with open(filename, 'r') as json_file:
        # parse json file
        f = json.load(json_file)
        url = f['url']
        content = f['content']

        # add url to hash table
        hash_table[hash_num] = url

        # parse html file
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()

        # get words in fields
        title = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('title')]))
        strong = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('strong')]))
        b = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('b')]))
        h1 = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('h1')]))
        h2 = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('h2')]))
        h3 = tokenize(' '.join([tag.text.strip() for tag in soup.find_all('h3')]))
        # compute bonus:
        # words in title, h1, h2, and h3 tags get 2 bonus points to its tf
        # words in strong and b tags get 1 bonus points to its tf
        bonus = dict()
        for t in title + h1 + h2 + h3:
            if t not in bonus:
                bonus[t] = 0
            bonus[t] += 2
        for t in strong + b:
            if t not in bonus:
                bonus[t] = 0
            bonus[t] += 1
        
        # convert to tokens
        tokens = tokenize(text)
        # compute postings for each token
        postingDict = computePostings(hash_num, tokens, bonus)
        # add the postings to the inverted index
        update_inverted_index(postingDict, inverted_index)


def to_string(inverted_index):
    # convert index to a list of strings where each term and its corresponding postings is a string
    postings = []
    for k in inverted_index:
        s = '{}:'.format(k)
        if len(inverted_index[k]) == 0: # term has no posting
            s += ' '
        else: # term has postings
            for p in inverted_index[k]:
                # each posting is separated with a space
                posting = ' {},{}'.format(p.docID, p.tfidf)
                s += posting
        s += '\n'
        postings.append(s)
    return postings


def to_postingDict(lines):
    # convert a list of strings to a postingDict
    postingDict = dict()
    for line in lines:
        # separate the term and its postings
        pair = line.split(': ')
        # get the term
        term = pair[0]
        # initialize each term's postings to an empty list
        if term not in postingDict:
            postingDict[term] = list()
        # term has no posting if pair[1] is a new line character
        if pair[1] != '\n':
            # split postings into a list
            postings = pair[1].split(' ')
            for p in postings:
                # split docID and tfidf
                docID, tfidf = p.split(',')
                # reconstruct posting and insert to the postingDict
                x = Posting(int(docID))
                x.tfidf = float(tfidf)
                postingDict[term].append(x)
    return postingDict


def build_index():
    inverted_index = dict()
    hash_table = dict()
    hash_num = 0
    # names for all files waiting to be parsed
    files = find_files('DEV')
    # total number of files
    N = len(files)
    dump_count = 0
    dumped = 0
    finished = 0
    # store index to disk when threshold is reached
    threshold = int(N/10)
    # stores the names of the partial index files
    partials = list()
    
    # parse each file 
    for f in files:
        hash_num += 1
        parse_doc(f, hash_num, hash_table, inverted_index)
        finished += 1
        print('Progress: {}/{}'.format(finished, N))
        # store partial index to disk
        if finished - dumped == threshold or finished == N:
            dump_count += 1
            dumped = finished
            filename = 'index{}.txt'.format(dump_count)
            partials.append(filename)
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(to_string(inverted_index))
            # make each term's postings to an empty list in the index
            for k in inverted_index:
                inverted_index[k] = list()
    
    # store hash table
    pickle.dump(hash_table, open('hashtable.p', 'wb'))

    # merge partial indexes
    with ExitStack() as stack:
        files = [stack.enter_context(open(f, "r", encoding='utf-8')) for f in partials]
        end = []
        while True:
            lines = list()
            for f in files:
                if f in end:
                    continue
                line = f.readline()
                if line == '':
                    end.append(f)
                else:
                    lines.append(line)
            # turn lines into a postingDict
            postingDict = to_postingDict(lines)
            # calcuate tfidf
            for t in postingDict:
                for p in postingDict[t]:
                    p.tfidf = Decimal(str(p.tfidf * math.log10(N/len(postingDict[t])))).quantize(Decimal('.01'), rounding=ROUND_UP)
            # turn postingDict into text
            line = to_string(postingDict)
            if len(line) == 0:
                line = ''
            else:
                line = line[0]
            # write to index
            with open('index.txt', 'a', encoding='utf-8') as f:
                f.write(line)
            # exit loop when all partial indexes are merged
            if len(end) == len(files):
                break

        # remove partial index filees
        for f in end:
            os.remove(f)
    
    # build index of index
    term_index = dict()
    with open('index.txt', 'r', encoding='latin-1') as f:
        byte_size = 0
        for i in range(N):
            # start read at 0
            f.seek(byte_size)
            line = f.readline()
            term = line.split(': ')[0]
            # size of the current line in bytes
            term_index[term] = byte_size
            # increment byte_size so on the next iteration we can read the next line
            # +1 for the new line character
            byte_size += len(line.encode('utf-8'))+1

    # store term index
    pickle.dump(term_index, open('termindex.p', 'wb'))


if __name__ == '__main__':
    build_index()