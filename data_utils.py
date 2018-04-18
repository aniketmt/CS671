import xml.etree.ElementTree as ET
import numpy as np


class Data(object):
    """Class to get data in required format"""

    def __init__(self, data_dir='UD_English-EWT/'):
        self.data_dir = data_dir
        train_file = self.data_dir + 'en_ewt-ud-train.conllu'
        test_file = self.data_dir + 'en_ewt-ud-test.conllu'
        dev_file = self.data_dir + 'en_ewt-ud-dev.conllu'
        self.data_file = [train_file, test_file, dev_file]

        self.vocab = []
        self.postags = []
        self.deps = []
        self.init_lists()

    def init_lists(self):
        root = ET.parse(self.data_dir + 'stats.xml').getroot()

        for child in root:
            if child.tag == 'tags':
                for gchild in child:
                    self.postags.append(gchild.attrib['name'])
            elif child.tag == 'deps':
                for gchild in child:
                    self.deps.append(gchild.attrib['name'])
        self.postags.append('_')
        self.deps.append('_')

    def get_file_data(self, filename):
        data_list = []

        with open(filename, 'r') as fdata:
            ddicts, dtree = [], []
            for line in fdata:
                if line == '\n':
                    data_list.append([ddicts, dtree])
                    ddicts = []
                    dtree = []
                else:
                    attr = line.split('\t')
                    if len(attr) > 1:
                        # Update lists
                        if attr[2] not in self.vocab:
                            self.vocab.append(attr[2])
                        # Make dictionary
                        ddicts.append({'id': attr[0], 'word': self.vocab.index(attr[2]),
                                       'pos': self.postags.index(attr[3]), 'deprel': self.deps.index(attr[7])})
                        # Generate tree
                        edge = {'head': attr[6], 'deprel': self.deps.index(attr[7]), 'dep': attr[0]}
                        if edge not in dtree:
                            dtree.append(edge)

        return data_list

    def get_transitions(self, word_list, edge_list):
        sigma = ['Root']
        beta = [word['id'] for word in word_list]
        edges = [(edge['head'], edge['dep']) for edge in edge_list]  # to check right-arc condition
        cedges = []
        configs, trans = [], []  # 0 for shift, 1 for left-arc, 2 for right arc

        while len(beta) > 1:
            configs.append([sigma, beta, cedges])

            if len(sigma) == 1:
                trans.append((0, None))
                sigma.append(beta.pop())
                continue

            sigma0 = sigma.pop()
            beta0 = beta.pop()
            arc_flag = False

            for edge in edge_list:
                if beta0 == edge['head'] and sigma0 == edge['dep']:
                    # Left-arc
                    trans.append((1, edge['deprel']))
                    edges.remove((beta0, sigma0))
                    cedges.append((beta0, sigma0))
                    beta.append(beta0)
                    arc_flag = True
                    break

                elif sigma0 == edge['head'] and beta0 == edge['dep']:
                    if beta0 in [edge2[0] for edge2 in edges]:
                        continue
                    else:
                        # Right-arc
                        trans.append((2, edge['deprel']))
                        edges.remove((sigma0, beta0))
                        cedges.append((sigma0, beta0))
                        beta.append(sigma0)
                        arc_flag = True
                        break

            if not arc_flag:
                # Shift
                trans.append((0, None))
                sigma.append(sigma0)
                sigma.append(beta0)

        configs.append([sigma, beta, cedges])
        trans.append((2, self.deps.index('root')))

        return np.array(configs), np.array(trans)

    def get_data(self, train=0):
        train_list = self.get_file_data(self.data_file[train])
        train_data = []
        for sentence in train_list:
            word_list, edge_list = sentence
            configs, trans = self.get_transitions(word_list, edge_list)
            tdata = []
            for i in range(configs.shape[0]):
                tdata.append([configs[i], trans[i]])
            train_data.append([word_list, tdata])
        return train_data


def test_Data():
    data = Data()
    data_list = data.get_data()
    print(data_list[49][0])
    print(data_list[49][1])


test_Data()
