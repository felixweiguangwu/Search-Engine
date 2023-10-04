from posting import Posting
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

def tokenize(text):
    # decode text
    text = text.encode('utf8').decode('utf8')
    # create stemmer
    stemmer = PorterStemmer()
    # generate tokens
    raw_tokens = word_tokenize(text)
    # combine contractions except 's
    for i in range(len(raw_tokens)):
        if "'" in raw_tokens[i] and raw_tokens[i] not in ["'s", "'"]:
            raw_tokens[i-1] = raw_tokens[i-1] + raw_tokens[i]
            raw_tokens[i] = ''
    # using stemming on tokens that start with an alphanumeric character
    tokens = [stemmer.stem(t) for t in raw_tokens if re.match(r'\w', t)]
    return tokens