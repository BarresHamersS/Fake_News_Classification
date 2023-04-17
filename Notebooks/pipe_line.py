#   ███████  █████  ██   ██ ███████     ███    ██ ███████ ██     ██ ███████     ██      ██ ███    ██  ██████  ██    ██ ██ ███████ ████████ ██  ██████     ██      ██ ██████  ██████   █████  ██████  ██    ██ 
#   ██      ██   ██ ██  ██  ██          ████   ██ ██      ██     ██ ██          ██      ██ ████   ██ ██       ██    ██ ██ ██         ██    ██ ██          ██      ██ ██   ██ ██   ██ ██   ██ ██   ██  ██  ██  
#   █████   ███████ █████   █████       ██ ██  ██ █████   ██  █  ██ ███████     ██      ██ ██ ██  ██ ██   ███ ██    ██ ██ ███████    ██    ██ ██          ██      ██ ██████  ██████  ███████ ██████    ████   
#   ██      ██   ██ ██  ██  ██          ██  ██ ██ ██      ██ ███ ██      ██     ██      ██ ██  ██ ██ ██    ██ ██    ██ ██      ██    ██    ██ ██          ██      ██ ██   ██ ██   ██ ██   ██ ██   ██    ██    
#   ██      ██   ██ ██   ██ ███████     ██   ████ ███████  ███ ███  ███████     ███████ ██ ██   ████  ██████   ██████  ██ ███████    ██    ██  ██████     ███████ ██ ██████  ██   ██ ██   ██ ██   ██    ██  

import pandas as pd 
import numpy as np
import sys
import gzip
import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy_readability import Readability
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import norm
import statistics

nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_md')


# This function computes the FKE readability of a object with the build in spaCy functionality
def readability_computation(doc):
    read = Readability()
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease

    return doc

nlp.add_pipe(readability_computation, last=True)

# This function computes the compressed size of the string, this is the approximation of the kolmogorov complexity
Doc.set_extension('compressed_size', default=None,force=True)
def compress_doc(doc):
    serialized_doc = doc.to_bytes()
    compressed_data = gzip.compress(serialized_doc)
    compressed_size = sys.getsizeof(compressed_data)
    doc._.compressed_size = compressed_size

    return doc

nlp.add_pipe(compress_doc, last=True)


# This function counts the Named Entities in the text
def count_named_entities(text):
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents]
    return len(entities)

# This function computes the POS groups based on the work of (hans et al 2021)
pos_to_group = {
    'CC': 'C',
    'CD': 'C',
    'DT': 'D',
    'EX': 'E',
    'FW': 'F',
    'IN': 'I',
    'JJ': 'J',
    'JJR': 'J',
    'JJS': 'J',
    'MD': 'M',
    'NN': 'N',
    'NNS': 'N',
    'NNP': 'N',
    'NNPS': 'N',
    'PDT': 'P',
    'POS': 'P',
    'PRP': 'P',
    'PRP$': 'P',
    'RB': 'R',
    'RBR': 'R',
    'RBS': 'R',
    'RP': 'R',
    'TO': 'T',
    'UH': 'U',
    'VB': 'V',
    'VBD': 'V',
    'VBG': 'V',
    'VBN': 'V',
    'VBP': 'V',
    'VBZ': 'V',
    'WDT': 'W',
    'WP': 'W',
    'WP$': 'W',
    'WRB': 'W'
}


def count_pos(text):
    pos_counts = {}
    doc = nlp(text)
    for token in doc:
        pos = token.pos_
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1
    return pos_counts

# This functions computes the different VADER scores and makes use of the NLTK library
def VADER_score(text):
    analyzer = SentimentIntensityAnalyzer()
    doc = nlp(text)
    vader_scores = analyzer.polarity_scores(text)
    return vader_scores


# This function applies the Readability, compressed size and vader scores and applies them to a given pandas dataframe    
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

#                               #
# Stastical functions           #   
#                               #
#                               #
#                               #

# This function computes the labels per feature and can be used to compute the anova score
def values_by_label(df, feature):
    labels = [0, 1, 2]
    values_label = []
    for label in labels:
        values_label.append(df.loc[df['binary label'] == label, feature])
    return values_label

# Basic statistical information
def compute_statistics(list):
    mo = statistics.mode(list)
    mu = np.mean(list)
    sigma = np.std(list)
    me = statistics.median(list)
    
    return mo, mu, sigma, me