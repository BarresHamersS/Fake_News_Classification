#basic libraries

import pandas as pd 
import numpy as np


#Libraries used to compete QA's
import sys
import gzip

#Spacy libraries
import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy_readability import Readability

#NLTK libraries for sentiment analysis VADER
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


nlp = spacy.load('en_core_web_md')

# define readability function
def readability_computation(doc):
    read = Readability()
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease

    return doc

nlp.add_pipe(readability_computation, last=True)




Doc.set_extension('compressed_size', default=None,force=True)

def compress_doc(doc):
    serialized_doc = doc.to_bytes()
    compressed_data = gzip.compress(serialized_doc)
    compressed_size = sys.getsizeof(compressed_data)
    doc._.compressed_size = compressed_size

    return doc

nlp.add_pipe(compress_doc, last=True)




def VADER_score(text):
    analyzer = SentimentIntensityAnalyzer()
    # i can probabily optimze computaiton time by processing the doc in the text process function
    doc = nlp(text)
    vader_scores = analyzer.polarity_scores(text)
    
    return vader_scores
    
def process_text(text):
    doc = nlp(text)

    doc = readability_computation(doc)
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease

    doc = compress_doc(doc)
    compressed_size = doc._.compressed_size
    

    vader_scores = VADER_score(text)
    vader_neg = vader_scores['neg']
    vader_neu = vader_scores['neu']
    vader_pos = vader_scores['pos']
    vader_compound = vader_scores['compound']


    
    return flesch_kincaid_reading_ease, compressed_size, vader_neg, vader_neu, vader_pos, vader_compound




