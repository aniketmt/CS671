import numpy as np
from conllu import parse, parse_tree, print_tree
import xml.etree.ElementTree as ET


class Data(object):
    """docstring for ClassName"""

    def __init__(self, data_dir='UD_English-EWT/'):
        self.data_dir = data_dir
        self.train_file = self.data_dir + 'en_ewt-ud-train.conllu'
        self.test_file = self.data_dir + 'en_ewt-ud-test.conllu'
        self.dev_file = self.data_dir + 'en_ewt-ud-dev.conllu'
        self.vocab = []
        self.postags = []
        self.deps = []
        self.init_lists()

    def init_lists(self):
        tree = ET.parse(self.data_dir + 'stats.xml')
        root = tree.getroot()

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
        data_dicts, data_trees = [], []
        with open(filename, 'r') as ftrain:
            data_raw = ftrain.read()
            data_dicts_raw = parse(data_raw)

            # Create data dict with categorical variables
            for ddicts in data_dicts_raw:
                sentence_dicts = []
                for ddict in ddicts:
                    if ddict['lemma'] not in self.vocab:
                        self.vocab.append(ddict['lemma'])
                    sentence_dicts.append(
                        {'id': ddict['id'], 'lemma': self.vocab.index(ddict['lemma']),
                         'postag': self.postags.index(ddict['upostag']), 'dep': self.deps.index(ddict['deprel'])})
                data_dicts.append(sentence_dicts)

            # Create data-tree
            data_trees_raw = parse_tree(data_raw)
            for tree in data_trees_raw:
                data_trees.append(self.add_tree_edges(tree, []))

        return data_dicts, data_trees

    def add_tree_edges(self, tree_node, data_tree):
        # Recursively travel tree and add edges to data_tree
        if tree_node.children:
            for child in tree_node.children:
                data_tree.append(tree_node.data['id'], child.data['deprel'], child.data['id'])
                data_tree = self.add_tree_edges(child, data_tree)
        return data_tree


data = Data()
data.get_file_data(data.train_file)
