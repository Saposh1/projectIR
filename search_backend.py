''''THIS FILE INCLUDES ALL THE METHODS USED BY SEARCH FUNCTIONS'''
import re
import nltk
from nltk.corpus import stopwords
from search_frontend import *
from inverted_index_colab import *
import numpy
import math
from collections import defaultdict,Counter
import numpy
from numpy.linalg import norm
import inverted_index_colab
from search_frontend import *
import pandas as pd
import heapq

''''QUERY TOKANIZATION'''
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)

def queryRepresent(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


''''BINARY SEARCH METHOD'''
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


''''COSINE SIMILARITY METHOD'''
def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_query_size = len(query_to_search)
    Q = numpy.zeros((total_query_size))
    counter = Counter(query_to_search)
    ind=0
    for token in numpy.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing
            Q[ind] = tf * idf
            ind+=1
    return Q


def get_posting_iter(index):
    """
    This function returning the iterator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls

def get_candidate_documents_and_scores(query_to_search, index, path):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    path: path to posting list.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in numpy.unique(query_to_search):
        try:
          posting_list=index.read_posting_list(term,path)
        except:
          continue
          normlized_tfidf = [(doc_id, (freq / index.DL[str(doc_id)]) * math.log(len(index.DL) / index.df[term], 10)) for
                              doc_id, freq in posting_list]

          for doc_id, tfidf in normlized_tfidf:
              candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates

def cosine_similarity(Doc_vector, Q_vector):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    denominator = (norm(Q_vector) * norm(Doc_vector))
    if denominator==0.0:
      cosine=0.0
    else:
      cosine = numpy.dot(Q_vector, Doc_vector) / denominator
    return cosine

def top_n_cosine_sim(query,index,bodyPath='.',N=100):
    invertedIndex_term=get_candidate_documents_and_scores(query,index,bodyPath)
    #init all sim (q,d)=0
    dict_docs={}
    for doc, term in invertedIndex_term.keys():
      dict_docs[doc]=[0]*len(query)
    #init query vector tfidf
    query_vector=generate_query_tfidf_vector(query,index)
    #init term query index
    unique_terms={}
    i=0
    for term in query:
      unique_terms[term]=i
      i+=1
    #init tfidf for doc_id,term
    for doc_id, term in invertedIndex_term.keys():
      indexTerm=unique_terms[term]
      tfidf=invertedIndex_term[(doc_id,term)]
      dict_docs[doc_id][indexTerm]=tfidf
    #create heap for tfidf scores
    top_n_docs=[]
    for doc, doc_tfidf_vector in invertedIndex_term.items():
      cos_sim=cosine_similarity(doc_tfidf_vector,query_vector)
      if len(top_n_docs)==N:
        heapq.heapify(top_n_docs)
        mini=heapq.heappop(top_n_docs)
        if cos_sim>mini[0]:
          heapq.heappush(top_n_docs,(cos_sim,doc))
        else:
          heapq.heappush(top_n_docs,mini)
      else:
        heapq.heapify(top_n_docs)
        heapq.heappush(top_n_docs,(cos_sim,doc))
      top_n_docs=list(top_n_docs)
    #sorted scores results
    return sorted([(int(doc_id), docs_titles[str(doc_id)]) for score, doc_id in top_n_docs], key=lambda x: x[1], reverse=True)


''''BM25 METHOD'''
class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        # self.fname
        self.N = len(self.index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N
        # self.words, self.pls = zip(*self.index.posting_lists_iter())

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: iterator for working with posting.

    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / index.DL[str(doc_id)]) * math.log(len(index.DL) / index.df[term], 10))
                               for doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates

    def search(self, query, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        scores = {}
        for q in queries:
            self.idf = self.calc_idf(queries[q])
            docs = get_candidate_documents_and_scores(queries[q], self.index, words, pls)
            docDict = {}
            for i in docs:
                docDict[i[0]] = self._score(queries[q], i[0])
            scores[q] = get_top_n(docDict, N)
        return scores

    def _score(self, query, doc_id):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = DL[str(doc_id)]

        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(self.pls[self.words.index(term)])
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score