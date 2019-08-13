### UCSB Project 3: Predicting Litigated Claims
### Author: Aaron Barel, Mingxi Chen, Syen Yang Lu
### Descriptions: Modules used for Text Preprocessing

# Import basic modules
import pandas as pd

# Packages that deal with strings
import string
import re

# Text preprocessing packages from nltk and sklearn
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# Download necessary dictionary from nltk
nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)
nltk.download(['averaged_perceptron_tagger',
               'universal_tagset',
               'wordnet'], quiet = True)

def text_preprocessing(data):
    '''Define a function (a master function) to perform the text preprocessing
    Input: A data frame that contains loss description column and injury description column
    Output: A list of string that contains combined description'''
    
    # Make loss description and injury desription to two lists
    loss_de=data['LOSS_DESCRIPTION']
    injury_de=data['INJURY_DESCRIPTION']
    
    # Lowercase conversion
    loss_de=loss_de.str.lower()
    injury_de=injury_de.str.lower()
    
    # Remove special charaters and meaningless contractions
    replaced=["/","@","\\","-","'d","'s"]
    loss_de=remove_special_charac(loss_de,replaced)
    injury_de=remove_special_charac(injury_de,replaced)
    
    # Expand contractions
    loss_de=expand_contraction(loss_de)
    injury_de=expand_contraction(injury_de)
    
    # Extract abbreviations
    loss_de=loss_abbr(loss_de)
    injury_de=injury_abbr(injury_de)
    
    # Remove numbers
    loss_de=remove_number(loss_de)
    injury_de=remove_number(injury_de)
    
    # Remove old unit numbers
    loss_de=remove_old_unit(loss_de)
    injury_de=remove_old_unit(injury_de)
    
    # Remove punctuations
    loss_de=[remove_punctation(loss_de[i]) for i in range(0,len(loss_de))]
    injury_de=[remove_punctation(injury_de[i]) for i in range(0,len(injury_de))]
    
    # Remove English stopwords
    loss_de=[remove_stopwords(loss_de[i],ENGLISH_STOP_WORDS) for i in range(0,len(loss_de))]
    injury_de=[remove_stopwords(injury_de[i],ENGLISH_STOP_WORDS) for i in range(0,len(injury_de))]
    
    # Word lemmatization
    loss_de=[lemmatize_text(loss_de[i]) for i in range(0,len(loss_de))]
    injury_de=[lemmatize_text(injury_de[i]) for i in range(0,len(injury_de))]
    
    # Remove insurance stopwords
    insurance_stopwords=["claimant","claim","incident","report","insure","insured"]
    loss_de=remove_ins_stopwords(loss_de,insurance_stopwords)
    injury_de=remove_ins_stopwords(injury_de,insurance_stopwords)
    
    # Word stemming and join stemmed tokens as strings
    loss_de=[" ".join(text_stemming(loss_de[i])) for i in range(0,len(loss_de))]
    injury_de=[" ".join(text_stemming(injury_de[i])) for i in range(0,len(injury_de))]
    
    # Combine loss description and injury description
    combined_text=[" ".join([loss_de[i],injury_de[i]]) for i in range(0,len(loss_de))]
    
    # Save preprocessed text and other columns into dataframe
    text_df = pd.DataFrame(combined_text).rename(columns={0:"TEXT"})
    other_df = data.drop(columns=['LOSS_DESCRIPTION', 'INJURY_DESCRIPTION'])
    
    # Combine the dataframe and return the output
    all_df = [other_df, text_df]
    return pd.concat(all_df, axis=1)

def removewords(text,pattern):
    '''Define a function that removes words from a string
    Input: text - The original string; pattern - A list of words to be removed
    Output: The modified string'''
    
    # Remove pattern from text
    output= str(text)
    for i in range(0, len(pattern)):
        output=output.replace(pattern[i]," ")
    
    return output

def replacewords(text, pattern, replacement):
    '''A function that replaces words in a string 
    Input: text - Original text; pattern - A list of words to be replaced, replacement - A single word
    Output: the modified string'''
    
    output=str(text) # Conversion from numbers to strings
    
    # Words replacement
    for i in range(0,len(pattern)):
        output=output.replace(pattern[i],replacement)
    
    return output

def replace_list(text, pattern, replacements):
    '''A function that replace words in a string, dealing with two lists in pairwise
    Input: text - original text, pattern - a list of words to be replaced, replacements - a list of words
    Output: the modified string'''

    output=str(text) # Conversion from numbers to strings
    
    # List of words replacement
    for i in range(0,len(pattern)):
        output=output.replace(pattern[i],replacements[i])
        
    return output

def expand_contraction(string_list):
    '''A function that expands contractions 
    Input: a list of string
    Output: a modified list of string'''
    
    contractions=["n't","'ll","'ve","'re"] # Define a list of contractions we want to expand
    expanded=[" not"," will"," have"," are"] # Define a list of corresponding expansion
    
    return [replace_list(string_list[i],contractions,expanded) for i in range(0,len(string_list))]
    
def remove_special_charac (string_list, spec_charac):
    '''A function that remvoes special charater
    Input: A string list, a list of character to be removed
    Output: A modified string list'''
    
    return [removewords(string_list[i],spec_charac) for i in range(0,len(string_list))]

def loss_abbr(loss_descr):
    '''Define functions to replace abbreviations for each loss description
    Input: Loss description containing abbreviations
    Output: Loss description with abbreviations expanded'''
    
    # Replace single word abbreviations
    loss_descr=[replacewords(loss_descr[i],[" w "," w/ "],"with") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],["rec ","recd "," recvd "," recv"],"receive") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],[" rep "]," representative") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],["repts"],"report") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],[" atty "]," attorney ") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],[" ins "," insd "]," insured ") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],[" clmt "]," claimant ") for i in range(0,len(loss_descr))]
    loss_descr=[replacewords(loss_descr[i],[" ltr "],"letter") for i in range(0,len(loss_descr))]
    
    return loss_descr

#For injury description
def injury_abbr(injury_descr):
    '''Define functions to replace abbreviations for each injury description
    Input: Injury description containing abbreviations
    Output: Injury description with abbreviations expanded'''
    
    # Replace single words
    injury_descr=[replacewords(injury_descr[i],[" w "," w/ "],"with") for i in range(0,len(injury_descr))]
    injury_descr=[replacewords(injury_descr[i],[" atty "]," attorney ") for i in range(0,len(injury_descr))]
    injury_descr=[replacewords(injury_descr[i],[" ins "," insd "]," insured ") for i in range(0,len(injury_descr))]

    # Replace a group of words
    injury_abbre=[" s/t "," fx "," lt "," rt ","1st","2nd","3rd","4th","5th","6th"]
    injury_replacement=[" skin tear "," fracture "," left "," right ","first","second","third","fourth","fifth","sixth"]
    injury_descr=[replace_list(injury_descr[i],injury_abbre,injury_replacement) for i in range(0,len(injury_descr))]
    
    return injury_descr

def remove_number(string_list):
    '''A function that removes numbers
    Input: A string list
    Output: A modified string list'''
    
    string_list=["".join(i for i in string_list[s] if not i.isdigit()) for s in range(0,len(string_list))]
    string_list=[removewords(string_list[i],[" nd "," th "," rd "]) for i in range(0,len(string_list))]
    
    return string_list

def remove_old_unit(string_list):
    '''A function that removes old unit numbers (e.g. "usw....")
    Input: A string list
    Output: A modified string list'''
    
    return [re.compile("usw...").sub(" ",string_list[i]) for i in range(0,len(string_list))] 
    
def tokenize_text(text):
    '''A function that tokenizes a string
    Input: A string
    Output: A list of tokens'''
    
    tokens=nltk.word_tokenize(text)
    tokens=[token.strip() for token in tokens]
    
    return tokens

def remove_punctation(text):
    '''A function that remove punctuation
    Input: A string
    Output: A modified string'''
    
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
    
def remove_stopwords(text, stopword_list):
    '''Define a function that removes stopwords
    Input: text - the original text; stopword_list - a list of stopwords from sklearn
    Output: Modified string'''
    
    tokens = tokenize_text(text)
    tokens = [token for token in tokens if token not in stopword_list]
    output = " ".join(tokens)
    
    return output

def pos_tag_text(text):
    '''Annotate text tokens with POS tags
    Input: A string 
    Output: A list of tokens that are tagged'''
    
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    
    return tagged_lower_text

def lemmatize_text(text):
    '''A function that lemmatize the input string
    Input: A string
    Output: A lemmatized string'''
    
    wnl = WordNetLemmatizer()
    text = tokenize_text(text)
    pos_tagged_text = pos_tag_text(text)
    
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text

def remove_ins_stopwords(string_list,ins_stopwords):
    '''Define a function that removes insurance stopwords
    Input: A list of string, a list of insurance stopwords
    Output: A modified string list'''
    
    string_list=[removewords(string_list[i],ins_stopwords) for i in range(0,len(string_list))]
    
    return string_list

def text_stemming(text):
    '''Define a function that do stemming on a string
    Input: A text string
    Output: A list of stemmed tokens'''
    
    ps=PorterStemmer()
    tokens=tokenize_text(text)
    
    return [ps.stem(tokens[i]) for i in range(0,len(tokens))]