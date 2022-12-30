import nltk
import numpy as np
import inverted_index_colab
nltk.download('stopwords')

# def bm25_preprocess(data):
#     """
#     This function goes through the data and saves relevant information for the calculation of bm25.
#     Specifically, in this function, we will create 3 objects that gather information regarding document length, term frequency and
#     document frequency.
#     Parameters
#     -----------
#     data: list of lists. Each inner list is a list of tokens.
#     Example of data:
#     [
#         ['sky', 'blue', 'see', 'blue', 'sun'],
#         ['sun', 'bright', 'yellow'],
#         ['comes', 'blue', 'sun'],
#         ['lucy', 'sky', 'diamonds', 'see', 'sun', 'sky'],
#         ['sun', 'sun', 'blue', 'sun'],
#         ['lucy', 'likes', 'blue', 'bright', 'diamonds']
#     ]
#
#     Returns:
#     -----------
#     three objects as follows:
#                 a) doc_len: list of integer. Each element represents the length of a document.
#                 b) tf: list of dictionaries. Each dictionary corresponds to a document as follows:
#                                                                     key: term
#                                                                     value: normalized term frequency (by the length of document)
#
#
#                 c) df: dictionary representing the document frequency as follows:
#                                                                     key: term
#                                                                     value: document frequency
#     """
#     doc_len = []
#     tf = []
#     df = {}
#     it = 0
#     # YOUR CODE HERE
#     for list1 in data:
#         it += 1
#         dict1 = {}
#         len1 = len(list1)
#         doc_len.append(len1)
#         for word in list1:
#             if word not in df:
#                 df[word] = 1
#             elif df[word] < it:
#                 df[word] += 1
#             if word not in dict1:
#                 dict1[word] = 1 / len1
#             else:
#                 dict1[word] += 1 / len1
#         tf.append(dict1)
#     return doc_len, tf, df
# import math


import math
from itertools import chain
import time
import search_frontend


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
        #self.fname
        self.N = len(self.index.DL)
        self.AVGDL = sum(index.DL.values()) / self.N
        #self.words, self.pls = zip(*self.index.posting_lists_iter())

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

def get_candidate_documents_and_scores(query_to_search,index,words,pls):
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
            normlized_tfidf = [(doc_id,(freq/index.DL[str(doc_id)])*math.log(len(index.DL)/index.df[term],10)) for doc_id, freq in list_of_doc]
            
            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               

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