import numpy as np
import scipy.sparse as sp
import torch



def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(0))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output
    correct = preds.eq(labels).double()
    correct = correct.sum()
    # preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    # correct = correct.sum()
    return correct / len(labels)



# def my_features():
#     vectors = np.load('vectors.npy')
#     return torch.from_numpy(vectors)

import json
def adj_max(file, i, is_train_test):
    import javalang
    import networkx as nx
    from javalang.ast import Node
    import matplotlib.pyplot as plt
    if i == 1:
        if is_train_test == 'train':
            program = open('./data/camel1.6/train/isbug/'+file, 'r')
        else:
            program = open('./data/camel1.6/test/isbug/' + file, 'r')
    else:
        if is_train_test == 'train':
            program = open('./data/camel1.6/train/nobug/'+file, 'r')
        else:
            program = open('./data/camel1.6/test/nobug/' + file, 'r')

    programtext = program.read()

    tree = javalang.parse.parse(programtext)
    g = nx.DiGraph()

    def build_graph(node, G):
        if isinstance(node, Node):
            if node.__class__.__name__ == "ForStatement":
                G.add_edge(node.control.__class__.__name__, node.body.__class__.__name__)
            if node.__class__.__name__ == "IfStatement":
                G.add_edge(node.condition.__class__.__name__, node.then_statement.__class__.__name__)
                G.add_edge(node.condition.__class__.__name__, node.else_statement.__class__.__name__)
            if node.__class__.__name__ == "WhileStatement":
                G.add_edge(node.condition.__class__.__name__, node.body.__class__.__name__)
            if node.__class__.__name__ == "DoStatement":
                G.add_edge(node.condition.__class__.__name__, node.body.__class__.__name__)

            for i in range(len(node.children)):

                if isinstance(node.children[i], Node):

                    G.add_edge(node.__class__.__name__, node.children[i].__class__.__name__)
                    # values.append(node.__class__.__name__, node.children[i].__class__.__name__])
                    build_graph(node.children[i], G)
                elif node.children[i] == None:
                    continue
                elif isinstance(node.children[i], str):
                    continue
                elif isinstance(node.children[i], bool):
                    continue
                elif isinstance(node.children[i], set):
                    continue
                # G.add_node(node.__class__.__name__, value='Modifi')

                else:  # isinstance(node.children[i], list):
                    # print('aa',node.children[i],len(node.children[i]))
                    for j in range(len(node.children[i])):
                        # print('aa', node.children[i][j], )
                        if isinstance(node.children[i][j], Node):
                            # G.add_node(node.__class__.__name__, desc=str(node.__class__.__name__))
                            # G.add_node(node.children[i][j].__class__.__name__, desc=str(node.children[i][j].__class__.__name__))
                            G.add_edge(node.__class__.__name__, node.children[i][j].__class__.__name__)
                            # values.append([dicts[node.__class__.__name__], dicts[node.children[i][j].__class__.__name__]])
                            build_graph(node.children[i][j], G)

    build_graph(tree, g)
    sentences = ['CompilationUnit', 'Import', 'Documented', 'Declaration', 'TypeDeclaration', 'PackageDeclaration',
                 'ClassDeclaration', 'EnumDeclaration', 'InterfaceDeclaration', 'AnnotationDeclaration', 'Type',
                 'BasicType', 'ReferenceType', 'TypeArgument', 'TypeParameter', 'Annotation', 'ElementValuePair',
                 'ElementArrayValue', 'Member', 'MethodDeclaration', 'FieldDeclaration', 'ConstructorDeclaration',
                 'ConstantDeclaration', 'ArrayInitializer', 'VariableDeclaration', 'LocalVariableDeclaration',
                 'VariableDeclarator', 'FormalParameter', 'InferredFormalParameter', 'Statement', 'IfStatement',
                 'WhileStatement', 'DoStatement', 'ForStatement', 'AssertStatement', 'BreakStatement',
                 'ContinueStatement', 'ReturnStatement', 'ThrowStatement', 'SynchronizedStatement', 'TryStatement',
                 'SwitchStatement', 'BlockStatement', 'StatementExpression', 'TryResource', 'CatchClause',
                 'CatchClauseParameter', 'SwitchStatementCase', 'ForControl', 'EnhancedForControl', 'Expression',
                 'Assignment', 'TernaryExpression', 'BinaryOperation', 'Cast', 'MethodReference', 'LambdaExpression',
                 'Primary', 'Literal', 'This', 'MemberReference', 'Invocation', 'ExplicitConstructorInvocation',
                 'SuperConstructorInvocation', 'MethodInvocation', 'SuperMethodInvocation', 'SuperMemberReference',
                 'ArraySelector', 'ClassReference', 'VoidClassReference', 'Creator', 'ArrayCreator', 'ClassCreator',
                 'InnerClassCreator', 'EnumBody', 'EnumConstantDeclaration', 'AnnotationMethod']

    f = open('my_src_vocab.json', 'r')
    dict = json.load(f)
    f.close()
    build_graph(tree, g)

    for i in range(len(sentences)):
        g.add_node(sentences[i], attr=dict[sentences[i]])


    a = list(g.nodes)
    if "NoneType" in a:
        g.remove_node("NoneType")
        a.remove("NoneType")
    adj = nx.adjacency_matrix(g)
    nor_adj = normalize_adj(adj)
    adj_ = nor_adj.todense()

    feature = []
    for i in range(len(a)):
        feature.extend([dict[a[i]]])

    return feature, adj_


def list_features_adj(filenames, j,  is_train_test):
    features, adjs,  = [], []
    for i in range(len(filenames)):
        feature, adj = adj_max(filenames[i], j, is_train_test)
        features.append(feature)
        adjs.append(adj)
    #labels.append()

    return features, adjs


