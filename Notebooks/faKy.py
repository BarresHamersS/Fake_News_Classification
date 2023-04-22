#   ███████  █████  ██   ██ ██    ██     ██      ██ ██████  ██████   █████  ██████  ██    ██ 
#   ██      ██   ██ ██  ██   ██  ██      ██      ██ ██   ██ ██   ██ ██   ██ ██   ██  ██  ██  
#   █████   ███████ █████     ████       ██      ██ ██████  ██████  ███████ ██████    ████   
#   ██      ██   ██ ██  ██     ██        ██      ██ ██   ██ ██   ██ ██   ██ ██   ██    ██    
#   ██      ██   ██ ██   ██    ██        ███████ ██ ██████  ██   ██ ██   ██ ██   ██    ██    
                                                                                         
                                                                                         
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




'''
 to add basic info
'''
def readability_computation(doc):
    read = Readability()
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease
    return doc

nlp.add_pipe(readability_computation, last=True)

# This function applies the Readability, compressed size and vader scores and applies them to a given pandas dataframe    
def process_text_readability(text):
    doc = nlp(text)
    doc = readability_computation(doc)
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease
    return flesch_kincaid_reading_ease

# This function computes the compressed size of the string, this is the approximation of the kolmogorov complexity
Doc.set_extension('compressed_size', default=None,force=True)
def compress_doc(doc):
    serialized_doc = doc.to_bytes()
    compressed_data = gzip.compress(serialized_doc)
    compressed_size = sys.getsizeof(compressed_data)
    doc._.compressed_size = compressed_size
    return doc

nlp.add_pipe(compress_doc, last=True)

# This function computes the compressed size of the string, this is the approximation of the kolmogorov complexity

def process_text_complexity(text):
    doc = nlp(text)
    doc = compress_doc(doc)
    compressed_size = doc._.compressed_size
    return compressed_size

# This functions computes the different VADER scores and makes use of the NLTK library
def VADER_score(text):
    analyzer = SentimentIntensityAnalyzer()
    doc = nlp(text)
    vader_scores = analyzer.polarity_scores(text)
    return vader_scores

def process_text_vader(text):
    vader_scores = VADER_score(text)
    vader_neg = vader_scores['neg']
    vader_neu = vader_scores['neu']
    vader_pos = vader_scores['pos']
    vader_compound = vader_scores['compound']
    return vader_neg, vader_neu, vader_pos, vader_compound  

# This function counts the Named Entities in the text
def count_named_entities(text):
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents]
    return len(entities)

def count_ner_labels(text):
    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents]
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

ner_labels = ['PERSON', 'ORG', 'GPE', 'DATE', 'NORP', 'CARDINAL']

def create_input_vector_NER(ner_count_dict):
    input_vector = np.zeros(len(ner_labels))
    for i, label in enumerate(ner_labels):
        if label in ner_count_dict:
            input_vector[i] = ner_count_dict[label]
    return input_vector



#### POS ############
# This function computes the POS groups based on the work of (hans et al 2021)
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

pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
def create_input_vector_pos(pos_count_dict):
    input_vector = np.zeros(len(pos_tags))
    for i, tag in enumerate(pos_tags):
        if tag in pos_count_dict:
            input_vector[i] = pos_count_dict[tag]
    return input_vector

############################




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

# post hoc table to reject; true false
def dunn_table(dunn_results):
    reject_h0_table = pd.DataFrame(columns=pd.MultiIndex.from_product([dunn_results.columns, ['value', 'reject']]))
    for i in dunn_results.index:
        for j in dunn_results.columns:
            p_value = dunn_results.loc[i, j]
            reject_h0 = p_value < 0.05 
            reject_h0_table.loc[i, (j, 'value')] = p_value
            reject_h0_table.loc[i, (j, 'reject')] = reject_h0
    reject_h0_table.columns.names = ['group', 'metric']
    return reject_h0_table