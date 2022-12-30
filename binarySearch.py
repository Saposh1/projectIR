import re
import nltk
from nltk.corpus import stopwords
from search_frontend import *
from inverted_index_colab import *
import numpy

def binarySearch(query,title_index,titlePath):
  dict_results={}
  query=numpy.unique(query)
  for term in query:
    if term in title_index.df:
      postings=title_index.read_posting_list(term,titlePath)
      for doc_id, tf in postings:
        dict_results[doc_id]=dict_results.get(doc_id,0)+1
  # sort result by the number of tokens appearance
  res = sorted(dict_results.items(), key=lambda x: x[1], reverse=True)
  return [(int(doc_id), docs_titles[str(doc_id)]) for doc_id, score in res]
