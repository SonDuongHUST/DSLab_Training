from collections import defaultdict
import numpy as np


# generate vocabulary and calculate idf
def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1.0 / df)
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)

    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split(' ')))
        for word in words:
            doc_count[word] += 1

    words_idfs = [(word, compute_idf(document_freq, corpus_size))
                    for word, document_freq in
                    zip(doc_count.keys(), doc_count.values())
                  if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda idf: -idf[1])
    print('Vocabulary size: {}'.format(len(words_idfs)))
    with open('20news-bydate/words_idfs.txt', 'w') as f:
        f.write('\n'.join([word + '<fff>' + str(idf) for word, idf in words_idfs]))

# compute tf_idf
def get_tf_idf(data_path):
    # get pre_computed idf values
    with open('20news-bydate/words_idfs.txt') as f:
        words_idfs = [(line.split('<fff>')[0], float(line.split('<fff>')[1]))
                      for line in f.read().splitlines()]
        word_IDs = dict([(word, index)
                         for index, (word, idf) in enumerate(words_idfs)])
        idfs = dict(words_idfs)

    # compute if_idf
    with open(data_path) as f:
        documents = [
            (int(line.split('<fff>')[0]),
             int (line.split('<fff>')[1]),
             line.split('<fff>')[2])
            for line in f.read().splitlines()]
    # data_tf_idf = []
    words_tfidfs = []
    sum_squares = 0.0
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word)
                             for word in word_set])
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq * 1.0 / max_term_freq * idfs[word]
            words_tfidfs.append((word_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value ** 2

    words_tfidfs_normalized = [str(index) + ':'
                               + str(tf_idf_value / np.sqrt(sum_squares))
                               for index, tf_idf_value in words_tfidfs]
    sparse_rep = ' '.join(words_tfidfs_normalized)
    # data_tf_idf.append((label, doc_id, sparse_rep))
    with open('20news-bydate/data_tfidf.txt', 'w') as f:
        f.write('\n'.join([str(label) + '<fff>' + str(doc_id) + '<fff>' + sparse_rep]))


generate_vocabulary('20news-bydate/20news-full-processed.txt')
get_tf_idf('20news-bydate/20news-full-processed.txt')
