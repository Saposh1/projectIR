import nltk
nltk.download('stopwords')

def bm25_preprocess(data):
    """
    This function goes through the data and saves relevant information for the calculation of bm25.
    Specifically, in this function, we will create 3 objects that gather information regarding document length, term frequency and
    document frequency.
    Parameters
    -----------
    data: list of lists. Each inner list is a list of tokens.
    Example of data:
    [
        ['sky', 'blue', 'see', 'blue', 'sun'],
        ['sun', 'bright', 'yellow'],
        ['comes', 'blue', 'sun'],
        ['lucy', 'sky', 'diamonds', 'see', 'sun', 'sky'],
        ['sun', 'sun', 'blue', 'sun'],
        ['lucy', 'likes', 'blue', 'bright', 'diamonds']
    ]

    Returns:
    -----------
    three objects as follows:
                a) doc_len: list of integer. Each element represents the length of a document.
                b) tf: list of dictionaries. Each dictionary corresponds to a document as follows:
                                                                    key: term
                                                                    value: normalized term frequency (by the length of document)


                c) df: dictionary representing the document frequency as follows:
                                                                    key: term
                                                                    value: document frequency
    """
    doc_len = []
    tf = []
    df = {}
    it = 0
    # YOUR CODE HERE
    for list1 in data:
        it += 1
        dict1 = {}
        len1 = len(list1)
        doc_len.append(len1)
        for word in list1:
            if word not in df:
                df[word] = 1
            elif df[word] < it:
                df[word] += 1
            if word not in dict1:
                dict1[word] = 1 / len1
            else:
                dict1[word] += 1 / len1
        tf.append(dict1)
    return doc_len, tf, df
import math


class BM25:
    """
    Best Match 25.

    Parameters to tune
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.
        The frequnecy is normilzied by the max term frequency for each document.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.
    """

    def __init__(self, doc_len, df, tf=None, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.tf_ = tf
        self.doc_len_ = doc_len
        self.df_ = df
        self.N_ = len(doc_len)
        self.avgdl_ = sum(doc_len) / len(doc_len)

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

        # dict1={}
        # for k in query:
        #   if k in df.keys():
        #     dict1[k]=math.log((self.N_+1)/df[k]) #idf of BM25 slide 46 = log((N+1)/dfi)
        # return dict1

    def search(self, queries, N=100):
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