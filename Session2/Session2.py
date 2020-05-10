from collections import defaultdict
import numpy as np
import random


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []

    def reset_members(self):
        self._members = []

    def add_members(self, member):
        self._members.append(member)


class Kmeans:
    def __init__(self, num_clusters=5):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = []  # list of centroids
        self._S = 0  # overall similarity

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(":")[0])
                tfidf = float(index_tfidf.split(":")[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open('words_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())

        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)

            self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def random_init(self):
        X = np.array(self._data)
        k = self._num_clusters
        cluster_init = X[np.random.choice(X.shape[0], k, replace=True)]
        for i in range(self._num_clusters):
            self._clusters[i]._centroid = cluster_init[i]

    def compute_similarity(self, member, centroid):
        # su dung khoang cach euclid de tinh su tuong dong
        return 1./(np.linalg.norm(member._r_d - centroid)+1)

    def run(self, criterion, threshold):
        self.random_init()

        # continually update clusters until convergence
        self._iterations = 0
        NMI_score = []
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)

            NMI_score.append(self.compute_NMI())
            self._iterations += 1
            if self.stopping_condition(criterion, threshold):
                return np.array(NMI_score)

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_members(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis=0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])
        # new_centroid = Member(new_centroid_r_d, label=cluster._centroid._label, doc_id=cluster._centroid._doc_id)

        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        critetia = ['centroid', 'similarity', 'max_iters']
        assert criterion in critetia
        if criterion == "max_iters":
            if self._iterations >= threshold:
                return True
            else:
                return False
        elif criterion == "centroid":
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new
                             if centroid not in self._E]
            self._E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

    def compute_purify(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1. / len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members) * 1.
            H_omega += -wk / N * np.log10(wk / N)
            member_labels = [member._label
                             for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label) * 1.
                cj = self._label_count[label]
                I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
        for label in range(20):
            cj = self._label_count[label] * 1.
            H_C += - cj / N * np.log10(cj / N)
        return I_value * 2. / (H_C + H_omega)


kmeans = Kmeans(20)
kmeans.load_data(data_path="data_tfidf.txt")
NMI_score = kmeans.run("max-iters", 1000)
print(NMI_score[-1])
print(kmeans.compute_purify())
