from data_utils import Data
import numpy as np


class Preprocess(object):
    def __init__(self, train=0):
        self.data = Data()
        print('Extracting data from file...')
        self.dataset = self.data.get_data(train)
        print('Done')
        self.n_classes = 2 * len(self.data.deps) + 1

    def get_word_features(self, words, wordlist):
        word_feature, pos_feature, dep_feature = [], [], []
        for word in words:
            word_feat = np.zeros(len(self.data.vocab))
            pos_feat = np.zeros(len(self.data.postags))
            dep_feat = np.zeros(len(self.data.deps))

            if not word == 'None':
                for worddict in wordlist:
                    if word == worddict['id']:
                        word_feat[worddict['word']] = 1
                        pos_feat[worddict['pos']] = 1
                        dep_feat[worddict['deprel']] = 1
                        break

            word_feature.append(word_feat)
            pos_feature.append(pos_feat)
            dep_feature.append(dep_feat)

        return np.concatenate((np.array(word_feature).flatten(), np.array(
            pos_feature).flatten(), np.array(dep_feature).flatten()))

    def get_features10(self, wordlist, configs):
        features = []
        for config in configs:
            sigma, beta, edges = config
            words = []

            for i in range(3):
                words.append('Null') if i >= len(sigma) else words.append(sigma[i])
                words.append('Null') if i >= len(beta) else words.append(beta[i])

            sigma_childs, beta_childs = [], []
            for edge in edges:
                if edge[0] == sigma[0]:
                    sigma_childs.append(edge[1])
                elif edge[0] == beta[0]:
                    beta_childs.append(edge[1])
            words.append(sigma_childs[0]) if sigma_childs else words.append('Null')
            words.append(sigma_childs[-1]) if sigma_childs else words.append('Null')
            words.append(beta_childs[0]) if beta_childs else words.append('Null')
            words.append(beta_childs[-1]) if beta_childs else words.append('Null')

            features.append(self.get_word_features(words, wordlist))
        return np.array(features)

    def get_features18(self, wordlist, configs):
        features = []
        for config in configs:
            sigma, beta, edges = config
            words = []

            for i in range(3):
                words.append('Null') if i >= len(sigma) else words.append(sigma[i])
                words.append('Null') if i >= len(beta) else words.append(beta[i])

            sigma0_childs, sigma1_childs = [], []
            for edge in edges:
                if edge[0] == sigma[0]:
                    sigma0_childs.append(edge[1])
                elif edge[0] == sigma[1]:
                    sigma1_childs.append(edge[1])

            sigma0_gchilds, sigma1_gchilds = [], []
            for edge in edges:
                if edge[0] in sigma0_childs:
                    sigma0_gchilds.append(edge[1])
                elif edge[0] in sigma1_childs:
                    sigma1_gchilds.append(edge[1])

            words.append(sigma0_childs[0]) if sigma0_childs else words.append('Null')
            words.append(sigma0_childs[1]) if len(sigma0_childs) > 1 else words.append('Null')
            words.append(sigma0_childs[-1]) if sigma0_childs else words.append('Null')
            words.append(sigma0_childs[-2]) if len(sigma0_childs) > 1 else words.append('Null')

            words.append(sigma1_childs[0]) if sigma1_childs else words.append('Null')
            words.append(sigma1_childs[1]) if len(sigma1_childs) > 1 else words.append('Null')
            words.append(sigma1_childs[-1]) if sigma1_childs else words.append('Null')
            words.append(sigma1_childs[-2]) if len(sigma1_childs) > 1 else words.append('Null')

            words.append(sigma0_gchilds[0]) if sigma0_gchilds else words.append('Null')
            words.append(sigma0_gchilds[-1]) if sigma0_gchilds else words.append('Null')
            words.append(sigma1_gchilds[0]) if sigma1_gchilds else words.append('Null')
            words.append(sigma1_gchilds[-1]) if sigma1_gchilds else words.append('Null')

            features.append(self.get_word_features(words, wordlist))
        return np.array(features)

    def get_classes(self, trans):
        tran_classes = []
        for tran in trans:
            tran_class = 0
            if tran[0] == 1:
                tran_class += tran[1] + 1
            elif tran[0] == 2:
                tran_class += len(self.data.deps) + tran[1] + 1
            tran_classes.append(tran_class)
        return np.array(tran_classes)

    def get_data(self, batch):
        wordlist, configs, trans = batch
        X_train = self.get_features18(wordlist, configs)
        Y_train = self.get_classes(trans)
        return X_train, Y_train

    def test_features(self):
        wordlist, configs, trans = self.dataset[0]
        features = self.get_features18(wordlist, configs)
        trans = self.get_classes(trans)
        print(features.shape, trans.shape)


def try_preprocess():
    pre = Preprocess()
    pre.test_features()


# try_preprocess()
