from cmath import nan
import pandas as pd
import numpy as np

# Data viz packages
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from nltk import ngrams
from wordcloud import WordCloud
#>-------------------------------------------
import re
import string
import ast
from tqdm.auto import tqdm
#spelling correction library
import itertools
from autocorrect import Speller
#spacy for language detection
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
# factory decorator for spacy language detetion below
@Language.factory("language_detector") # uncomment for the first run
def get_lang_detector(nlp, name):
  return LanguageDetector()

nlp=spacy.load("en_core_web_sm") #"en_core_web_sm"
nlp.add_pipe('language_detector', last=True)
#function to remove emoji
def remove_emojis(data):
  emoj = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002500-\U00002BEF"  # chinese char
      u"\U00002702-\U000027B0"
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u"\U00010000-\U0010ffff"
      u"\u2640-\u2642" 
      u"\u2600-\u2B55"
      u"\u200d"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\ufe0f"  # dingbats
      u"\u3030"
                    "]+", re.UNICODE)
  return re.sub(emoj, '', data)
#<-------------------------------------------
def readdata(filepath):
  df = pd.read_csv(filepath)
  return df
#<-------------------------------------------
def preprocessing(video_corpus, stem_or_lemma = 'lemma'):
  cleaned_text = []
  commentlen = 0 #
  for i, comments in enumerate(video_corpus):
    # convert string object to a list of separate comments
    list_comments = ast.literal_eval(comments)
    #print(comments)
    cleaned_comments = []
    for comment in list_comments:
      result = comment
      #removing punctuation
      result = result.translate(str.maketrans('','', string.punctuation)) 
      #removing URLs
      result = re.sub(r'http\S+', '', result)
      #removing emails
      result = re.sub(r'\S*@\S*\s?', '', result)
      #removing emojis
      result = remove_emojis(result)
      #detect and remove non english comments
      lang = nlp(result)._.language['language']
      if lang != 'en':
        continue
      #spelling correction
      spell = Speller(lang='en')
      result = spell(result)
      #tokenize
      result = word_tokenize(result)
      #removing stop words
      stopset = set( stopwords.words('english') )
      stopset |= {"Aaa","Bbb"} # custom stopwords if any
      result = [wrd for wrd in result if not wrd in stopset]
      #stemming or lematization
      if stem_or_lemma == 'stem':
        #stemming
        st = SnowballStemmer("english")
        result = [st.stem(i) for i in result]
      else:
        #lematizing
        lm = WordNetLemmatizer()
        result = [lm.lemmatize(i) for i in result]
      #cleaned_comments.append(result)
      cleaned_comments.append(" ".join([com for com in result]))
    #print(len(cleaned_comments),":",cleaned_comments)
    cleaned_text.append(cleaned_comments)
    commentlen += len(cleaned_text) #
    #if i == 10:
      #break
  print(commentlen) #
  return pd.DataFrame(cleaned_text)
#>-------------------------------------------
def preprocessing_training(data):
  progress_bar = tqdm(range(len(data)))
  for idx in range(len(data)):
    progress_bar.update(1)
    result = data["text"][idx]
    new_comment = ""
    if(pd.isna(result)):
      continue
    #removing punctuation
    result = result.translate(str.maketrans('','', string.punctuation)) 
    #removing URLs
    result = re.sub(r'http\S+', '', result)
    #removing emails
    result = re.sub(r'\S*@\S*\s?', '', result)
    #removing emojis
    result = remove_emojis(result)
    #spelling correction
    spell = Speller(lang='en')
    result = spell(result)
    #tokenize
    result = nltk.tokenize.word_tokenize(result)
    for idx2 in range(len(result)):
      word = result[idx2]
       #removing stop words
      word = word.lower()
      if(word not in nltk.corpus.stopwords.words('english')):
        if(idx2 != 0):
          new_comment += " "
        new_comment += word
    data["text"][idx] = new_comment

#>-------------------------------------------
def main():
  path = 'Comments.csv'
  data = readdata(path)
  #print(data['comments'][0])
#<-------------------------------------------  
  print(len(data))
  Comments_prep = preprocessing(data['comments'],"lemma")
  #print(Comments_prep)
  Comments_prep.to_csv('Comments_prep.csv', encoding='utf-8')
#<-------------------------------------------
  tweets = readdata('Tweets.csv')
  tweets = tweets.drop(columns=['textID', 'selected_text'])
  tweets = tweets.rename(columns={"sentiment": "labels"})
  tweets = preprocessing_training(tweets)
  tweets.to_csv('Tweets_prep.csv', encoding='utf-8')
#<-------------------------------------------

if __name__ == "__main__":
    main()