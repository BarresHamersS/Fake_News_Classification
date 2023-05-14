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
from spacy.tokens import Doc
from spacy_readability import Readability
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics



nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_md')


#   _______        _                                       _             
#  |__   __|      | |                                     (_)            
#     | | _____  _| |_   _ __  _ __ ___   ___ ___  ___ ___ _ _ __   __ _ 
#     | |/ _ \ \/ / __| | '_ \| '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |
#     | |  __/>  <| |_  | |_) | | | (_) | (_|  __/\__ \__ \ | | | | (_| |
#     |_|\___/_/\_\\__| | .__/|_|  \___/ \___\___||___/___/_|_| |_|\__, |
#                       | |                                         __/ |
#                       |_|                                        |___/ 



'''
 The function readability_computation computes the Flesch-Kincaid Reading Ease score for a spaCy document using the Readability class. 
 It returns the original document object. 
 It is added to the spaCy pipeline using nlp.add_pipe with last=True.
'''
def readability_computation(doc):
    read = Readability()
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease
    return doc
nlp.add_pipe(readability_computation, last=True)

'''
The function process_text_readability takes a text string as input, processes it with spaCy's NLP pipeline, 
computes the Flesch-Kincaid Reading Ease score using the readability_computation function, and returns the score.
''' 
def process_text_readability(text):
    doc = nlp(text)
    doc = readability_computation(doc)
    flesch_kincaid_reading_ease = doc._.flesch_kincaid_reading_ease
    return flesch_kincaid_reading_ease

'''
The first line creates a custom extension attribute called "compressed_size" for spaCy's Doc object. 
The function compress_doc compresses the serialized form of a spaCy Doc object using gzip, calculates the compressed size,
and sets the compressed size to the custom "compressed_size" attribute of the Doc object before returning the Doc object.
'''
Doc.set_extension('compressed_size', default=None,force=True)
def compress_doc(doc):
    serialized_doc = doc.to_bytes()
    compressed_data = gzip.compress(serialized_doc)
    compressed_size = sys.getsizeof(compressed_data)
    doc._.compressed_size = compressed_size
    return doc
nlp.add_pipe(compress_doc, last=True)

'''
The function process_text_complexity takes a text string as input, processes it with spaCy's custom NLP pipeline, 
computes the compressed size using the compressed_doc function, and returns the compressed size of the string in bits.
'''
def process_text_complexity(text):
    doc = nlp(text)
    doc = compress_doc(doc)
    compressed_size = doc._.compressed_size
    return compressed_size

'''
The function VADER_score() takes a text input and calculates the sentiment scores using the VADER sentiment analysis tool. 
It uses the SentimentIntensityAnalyzer() from the nltk package to compute the polarity scores. 
The function returns a dictionary containing the negative, neutral, positive, and compound sentiment scores
'''
def VADER_score(text):
    analyzer = SentimentIntensityAnalyzer()
    doc = nlp(text)
    vader_scores = analyzer.polarity_scores(text)
    return vader_scores

'''
The function takes a text input, applies the VADER sentiment analysis model to it, and returns 
the negative, neutral, and positive sentiment scores, as well as the overall compound score. 
These scores are returned as separate variables.
'''
def process_text_vader(text):
    vader_scores = VADER_score(text)
    vader_neg = vader_scores['neg']
    vader_neu = vader_scores['neu']
    vader_pos = vader_scores['pos']
    vader_compound = vader_scores['compound']
    return vader_neg, vader_neu, vader_pos, vader_compound  

'''
The function takes a text input, transforms it to a spacy doc object and then counts the named entities for the doc object.
The function subsequently returns the total count of NER objects in the analyzed text objects.
'''
def count_named_entities(text):
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents]
    return len(entities)

'''
The function takes in a text and uses spaCy to identify named entities, retrieve their labels, and count the occurrences of each label. 
The output is a dictionary where the keys are the labels and the values are the corresponding counts.
'''


def count_ner_labels(text):
    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents]
    counts = {}
    for label in labels:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

'''
The function takes in a dictionary of named entity recognition (NER) label counts and creates an input vector with the count for each NER label. 
The input vector is initialized as an array of zeros with a length equal to the number of NER labels. 
The function then iterates through each label, setting the count in the input vector if the label was found in the input dictionary. 
The resulting input vector is returned.
'''
ner_labels = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']
def create_input_vector_NER(ner_count_dict):
    input_vector = np.zeros(len(ner_labels))
    for i, label in enumerate(ner_labels):
        if label in ner_count_dict:
            input_vector[i] = ner_count_dict[label]
    return input_vector

'''
The function count_pos counts the number of parts of speech (POS) in a given text. 
It initializes an empty dictionary to store the POS counts and uses spaCy's POS tagger to identify the POS of each token in the input text. 
Finally, it returns a dictionary with the count of each POS.
'''
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

'''
This function takes a dictionary of POS tag counts and creates an input vector of zeros. 
Subsequently it counts values at the corresponding positions of the tag in a list of all possible POS tags. 
The function returns the input vector.
'''
pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
def create_input_vector_pos(pos_count_dict):
    input_vector = np.zeros(len(pos_tags))
    for i, tag in enumerate(pos_tags):
        if tag in pos_count_dict:
            input_vector[i] = pos_count_dict[tag]
    return input_vector


#   _____ _            _   _           _    __                  _   _                 
#  / ____| |          | | (_)         | |  / _|                | | (_)                
# | (___ | |_ __ _ ___| |_ _  ___ __ _| | | |_ _   _ _ __   ___| |_ _  ___  _ __  ___ 
#  \___ \| __/ _` / __| __| |/ __/ _` | | |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#  ____) | || (_| \__ \ |_| | (_| (_| | | | | | |_| | | | | (__| |_| | (_) | | | \__ \
# |_____/ \__\__,_|___/\__|_|\___\__,_|_| |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
                                                                                                                                                                        
'''
he function takes a dataframe and a feature as input, and returns a list of three lists where each list contains the values of the given feature for each of the binary labels present in the dataframe. 
It does so by filtering the dataframe for each label and extracting the values of the specified feature for each subset.
'''
def values_by_label(df, feature, labels, df_label):
    values_label = []
    for label in labels:
        values_label.append(df.loc[df[df_label] == label, feature])
    return values_label

'''
This function takes a list of numerical values as input and returns its mode, mean, standard deviation, and median as a tuple. 
The mode is calculated using the statistics module, while the mean, standard deviation, and median are calculated using numpy.
'''

'''
The dunn_table function takes a DataFrame of Dunn's test results and creates a new DataFrame where each row represents a group and each column represents a pairwise comparison with another group. 
The new DataFrame contains two sub-columns: value which holds the p-value for each comparison, 
and reject which holds a Boolean indicating whether the null hypothesis (that the two groups have the same distribution) should be rejected at a significance level of 0.05. 
The function then returns this new DataFrame.
'''
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