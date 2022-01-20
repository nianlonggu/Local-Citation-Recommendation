import re
from tqdm import tqdm
import nltk
from nltk.tokenize import  RegexpTokenizer
from nltk.stem import WordNetLemmatizer 
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

class SentenceTokenizer:
    def __init__(self ):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()

        self.general_stopwords = set(stopwords.words('english')) | set([ "<num>", "<cit>" ])

    @lru_cache(100000)
    def lemmatize( self, w ):
        return self.lemmatizer.lemmatize(w)
    
    def tokenize(self, sen, remove_stopwords = True ):
        if remove_stopwords:
            sen = " ".join( [ w for w in sen.lower().split() if w not in self.general_stopwords ] )
        else:
            sen = sen.lower()
        sen = " ".join([ self.lemmatize(w) for w in self.tokenizer.tokenize( sen )   ])
        return sen

class JaccardSim:
    def __init__(self):
        self.sent_tok = SentenceTokenizer()

    def compute_sim( self, textA, textB ):

        textA_words = set(self.sent_tok.tokenize( textA.lower(), remove_stopwords = True ).split()  )
        textB_words = set(self.sent_tok.tokenize( textB.lower(), remove_stopwords = True ).split()  )
        
        AB_words = textA_words.intersection( textB_words )
        return float(len( AB_words ) / (  len(textA_words) + len( textB_words )  -  len( AB_words ) + 1e-12  ))