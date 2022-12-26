import pandas as pd
import numpy as np

# Data viz packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from nltk import ngrams
from wordcloud import WordCloud

def readdata(filepath):
  df = pd.read_csv(filepath)
  return df


def main():
  path = 'Comments.csv'
  data = readdata(path)
  print(data['comments'][0])


if __name__ == "__main__":
    main()
