from pprint import pprint
import re
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pymorphy2
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import pymorphy2
from nltk.corpus import stopwords
from typing import List
# stopwords from nltk
stop_words = stopwords.words('russian')
stop_words.extend(['это', 'этот', 'так', 'такой', 'такая'])
# pymorphy
morph = pymorphy2.MorphAnalyzer()

# function to get rid of stopwords
def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# Function to turn all the words to the normal form
def lemmatization(texts: List[List[str]]) -> List[List[str]]:
    output = []
    for sent in texts:
        sub_arr = []
        for word in sent:
            sub_arr.append(morph.parse(word)[0].normal_form)
        output.append(sub_arr)
    return output
        

# Reading file
text_data = ""
with open('text.txt', encoding="utf-8") as fp:
    text_data = fp.readlines()
data = [gensim.utils.simple_preprocess(str(sent), deacc=True) for sent in text_data]
# Removing stopwords
data = remove_stopwords(data)
# Normalizing
data_lemmatized = lemmatization(data)
# Kind of Bag Of Words
id2word = corpora.Dictionary(data_lemmatized)
# Building corpus
corpus = [id2word.doc2bow(text) for text in data_lemmatized]
# Building LDA model (there are 30 topics, according to the menu in VK)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=9, 
                                           random_state=42,
                                           update_every=1,
                                           chunksize=150,
                                           passes=30,
                                           alpha='auto',
                                           per_word_topics=True)

pprint(lda_model.print_topics())
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#pip install pyLDAvis==2.1.2
pyLDAvis.save_html(vis, 'LDA.html')

# pyLDAvis.show(data = vis, open_browser = True)