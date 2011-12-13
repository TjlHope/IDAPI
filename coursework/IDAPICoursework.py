#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python

from __future__ import division

import sys
import os
from operator import itemgetter
import logging as log

#from numpy import *
import numpy as np

#from IDAPICourseworkLibrary import *
import IDAPICourseworkLibrary as IDAPI


#
# Coursework 1 begins here
#


def Prior(data, root, states):
    """Function to compute the prior distribution of the variable root from the
    data set."""
    prior = np.zeros((states[root]), float)
    # Coursework 1 task 1 should be inserted here...
    inc = 1 / data.shape[0]     # increment by proportion as sums to 1
    for row in data:
        prior[row[root]] += inc
    # end of Coursework 1 task 1.
    return prior


def CPT(data, child, parent, states):
    """Function to compute a CPT with parent node and child node varC
    from the data array it is assumed that the states are designated by
    consecutive integers starting with 0.
    """
    cpt = np.zeros((states[child], states[parent]), dtype='f8')
    # Coursework 1 task 2 should be inserted here...
    # cpt/ link matrix: P(C|P) = [[P(c0|p0), P(c0|p1), ..., P(c0|pn)],
    #                             [P(c1|p0), P(c1|p1), ..., P(c1|pn)],
    #                             ...,
    #                             [P(cn|p0), P(cn|p1), ..., P(cn|pn)]]
    for r in data:
        cpt[r[child], r[parent]] += 1   # Increment necessary points
    for c in cpt.T:
        s = sum(c)              # test for divide by zero
        if s:
            c /= s              # normalise the columns
    # end of coursework 1 task 2.
    return cpt


def JPT(data, row, col, states):
    """Function to calculate the joint probability table of two variables in
    the data set.
    """
    jpt = np.zeros((states[row], states[col]), dtype='f8')
    # Coursework 1 task 3 should be inserted here...
    inc = 1 / data.shape[0]     # all sums to 1, so increment by proportion
    for r in data:
        jpt[r[row], r[col]] += inc
    # end of coursework 1 task 3.
    return jpt


def JPT2CPT(jpt):
    """Function to convert a joint probability table to a conditional
    probability table. Note: modifies array in place.
    """
    # Coursework 1 task 4 should be inserted here...
    for c in jpt.T:
        s = sum(c)              # test for divide by zero
        if s:
            c /= s              # normalise the columns
    # end of coursework 1 task 4.
    return jpt


def Query(query, network):
    """Function to query a naive Bayesian network."""
    root_pdf = np.zeros(network[0].shape[0], dtype='f8')
    # Coursework 1 task 5 should be inserted here...
    for i, p in enumerate(network[0]):
        root_pdf[i] = p         # as P(P|C0&...&Cn) = alpha P(D)P(C0|D)...
        for j, cpt in enumerate(network[1:]):
            root_pdf[i] *= cpt[query[j], i]     # The 'P(Ci|D)'s
    # normalise for alpha
    root_pdf /= sum(root_pdf)
    # end of coursework 1 task 5.
    return root_pdf


def cw1():
    """main() part of Coursework 01."""
    fl = "IDAPIResults01.txt"
    if os.path.exists(fl):
        os.remove(fl)
    
    (variables, roots, states,
        points, datain) = IDAPI.ReadFile("Neurones.txt")
    data = np.array(datain)
    IDAPI.AppendString(fl, "Coursework One Results by:")
    IDAPI.AppendString(fl, "\ttjh08\tThomas Hope")
    IDAPI.AppendString(fl, "")

    # Task 1
    for root in [0]:
        IDAPI.AppendString(fl,
                           "The prior probability of node {0}".format(root))
        prior = Prior(data, root, states)
        IDAPI.AppendList(fl, prior)
    IDAPI.AppendString(fl, "")

    for child in [2]:
        # Task 2
        IDAPI.AppendString(fl,
                           "The conditional probability table P({0}|0)."
                               .format(child))
        cpt = CPT(data, child, 0, states)
        IDAPI.AppendArray(fl, cpt)

        # Task 3
        IDAPI.AppendString(fl,
                           "The joint probability table P({0}&0)."
                               .format(child))
        jpt = JPT(data, child, 0, states)
        IDAPI.AppendArray(fl, jpt)

        # Task 4
        IDAPI.AppendString(fl,
                           "The cpt P({0}|0) from the jpt P({0}&0)."
                               .format(child))
        cpt = JPT2CPT(jpt)
        IDAPI.AppendArray(fl, cpt)

        IDAPI.AppendString(fl, "")

    # Task 5
    network = ([Prior(data, 0, states)] +
               [CPT(data, i, 0, states) for i in range(1, variables)])
    for query in [[4, 0, 0, 0, 5], [6, 5, 2, 5, 5]]:
        IDAPI.AppendString(fl, "The pdf from the query {0}.".format(query))
        pdf = Query(query, network)
        IDAPI.AppendList(fl, pdf)

    IDAPI.AppendString(fl, "\nEND")


#
# End of Coursework 1
#
# Coursework 2 begins here
#


def MutualInformation(jpt):
    """Calculate the mutual information from the joint probability table of two
    variables.
    """
    mi = 0.0
    # Coursework 2 task 1 should be inserted here...
    # Dep(A, B) = Sum(P(ai&bj) * log2(P(ai&bj) / (P(ai) * P(bj))))
    column_totals = []  # Initiate list for independent column probablities.
    for i, row in enumerate(jpt):
        # Get independent probablity for rows (marginalise)
        row_total = sum(row)
        for j, cell in enumerate(row):
            # Get independent probablilty for columns (marginalise)
            if i == 0:
                column_totals.append(sum(jpt[:, j]))
            # Calculate Mutual Information
            try:
                # if either P(ai), P(bj) or P(ai&bj) is 0...
                mi += cell * np.log2(cell / (row_total * column_totals[j]))
            # ... this should increment by zero (i.e. don't increment)
            except (ZeroDivisionError, ValueError, FloatingPointError):
                continue
    # end of coursework 2 task 1.
    return mi


def DependencyMatrix(data, variables, states):
    """Constructs a dependency matrix for all the variables."""
    mi_matrix = np.zeros((variables, variables))
    # Coursework 2 task 2 should be inserted here...
    for i in range(variables):
        for j in range(i, variables):
            jpt = JPT(data, i, j, states)
            mi_matrix[i, j] = mi_matrix[j, i] = MutualInformation(jpt)
    # end of coursework 2 task 2.
    return mi_matrix


def DependencyList(dep_matrix):
    """Function to compute an ordered list of dependencies."""
    dep_list = []
    # Coursework 2 task 3 should be inserted here...
    for i in range(len(dep_matrix)):
        for j in range(i, len(dep_matrix[i])):
            dep_list.append((dep_matrix[i, j], i, j))
    dep_list.sort(reverse=True, key=itemgetter(0))
    # end of coursework 2 task 3.
    return np.array(dep_list)


# Coursework 2 task 4 begins here...
def connected(tree, src, dst, path=[]):
    for item in tree:
        #if item[0] in path and item[2] in path:
        if ((item[1] == src and item[2] == dst) or
            (item[2] == src and item[1] == dst)):
            return True
        elif item[1] == src and item[2] not in path:
            if connected(tree, item[2], dst, path + [item[1]]):
                return True
        elif item[2] == src and item[1] not in path:
            if connected(tree, item[1], dst, path + [item[2]]):
                return True


def SpanningTreeAlgorithm(dep_list, variables):
    """Function implementing the spanning tree algorithm."""
    spanning_tree = []
    for item in dep_list:
        if item[1] != item[2] and not(connected(spanning_tree, item[1], item[2])):
            spanning_tree.append(item)
    return np.array(spanning_tree)
# End of Coursework 2 task 4.


def dot(l, name="network.dot"):
    """Output dot file of list with items (weight, node0, node1)."""
    with open(name, 'w') as f:
        f.write("graph {0}{{".format(name.rsplit('.dot')[0]))
        for i in l:
            if i[0]:
                f.write("    {1} -- {2} [label={0:.5f}]".format(*i))
        f.write("}")


def cw2():
    """main() part of Coursework 02."""
    fl = "Results02.rst"
    if os.path.exists(fl):
        os.remove(fl)
    
    (variables, roots, states,
        points, datain) = IDAPI.ReadFile("HepatitisC.txt")
    data = np.array(datain)
    IDAPI.AppendString(fl, "Coursework Two Results by:\n")
    IDAPI.AppendString(fl, "* tjh08 - Thomas Hope")
    IDAPI.AppendString(fl, "* jzy08 - Jason Ye")
    IDAPI.AppendString(fl, "")

    dep_matrix = DependencyMatrix(data, variables, states)
    IDAPI.AppendString(fl, "The dependency matrix of the HepatitisC data.")
    IDAPI.AppendString(fl, "\n::\n")
    IDAPI.AppendArray(fl, dep_matrix)

    dep_list = DependencyList(dep_matrix)
    IDAPI.AppendString(fl, "The dependency list of the HepatitisC data.")
    IDAPI.AppendString(fl, "\n::\n")
    IDAPI.AppendArray(fl, dep_list)

    spanning_tree = SpanningTreeAlgorithm(dep_list, variables)
    IDAPI.AppendString(fl, "The Spanning Tree of the HepatitisC data.")
    IDAPI.AppendString(fl, "\n::\n")
    IDAPI.AppendArray(fl, spanning_tree)
    dot(spanning_tree, 'span.dot')
    os.system("neato span.dot -Tpng > span.png")

    IDAPI.AppendString(fl, ".. image:: span.png\n   :scale: 50%\n")
    
    os.system("rst2latex.py {0} IDAPIResults02.tex".format(fl))
    os.system("pdflatex IDAPIResults02.tex")


#
# End of coursework 2
#
# Coursework 3 begins here
#


def CPT_2(data, child, parent1, parent2, states):
    """Function to compute a CPT with multiple parents from the data set it is
    assumed that the states are designated by consecutive integers starting
    with 0.
    """
    cpt = np.zeros((states[child], states[parent1], states[parent2]),
                   dtype='f8')
    # Coursework 3 task 1 should be inserted here...
    for s in range(states[parent2]):
        d = [row for row in data if row[parent2] == s]
        cpt[:, :, s] = CPT(d, child, parent1, states)
    # end of Coursework 3 task 1.
    return cpt


def ExampleBayesianNetwork(data, states):
    """Definition of a Bayesian Network."""
    arcs = [[0], [1], [2, 0], [3, 2, 1], [4, 3], [5, 3]]
    cpt0 = Prior(data, 0, states)
    cpt1 = Prior(data, 1, states)
    cpt2 = CPT(data, 2, 0, states)
    cpt3 = CPT_2(data, 3, 2, 1, states)
    cpt4 = CPT(data, 4, 3, states)
    cpt5 = CPT(data, 5, 3, states)
    cpts = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcs, cpts


# Coursework 3 task 2 begins here...
def most_connected(tree, nodes):
    """Return the most connected node, so it can be removed."""
    # get a nice representation of immediate connections
    connections = tree[:, (1, 2)].astype(np.int)
    # dict mapping node names to positions
    node_names = dict(zip(nodes[:]['name'], range(nodes.shape[0])))
    # array of node connections
    node_connections = np.zeros(nodes.shape, dtype=np.int)
    # check each connection against the nodes
    for connection in connections:
        if connection[0] in node_names and connection[1] in node_names:
            # if between the nodes, use node_names to increment
            node_connections[node_names[connection[0]]] += 1
            node_connections[node_names[connection[1]]] += 1
    if node_connections.any():
        # If there are some connections return the position of the most
        # connected node.
        return node_connections.argmax()
    else:
        return None


def bayesian_roots(tree, roots, variables):
    """Finds the num_roots maximally weighted root nodes without a direct
    connection."""
    # initiate array of nodes
    count_type = np.dtype([('name', np.int), ('count', np.int),
                           ('weight', np.float)])
    nodes = np.zeros(max(variables) + 1, dtype=count_type)
    for i, n in enumerate(nodes):
        # have sparse array to allow indexing
        if i in variables:
            n['name'] = i
        else:
            nodes[i] = (-1, -10, -10)
    #nodes[:]['name'] = np.array(list(variables), dtype=np.int)
    log.debug("Nodes:\n{0}".format(nodes))
    # populate node data
    for branch in tree:
        nodes[int(branch[1])]['count'] += 1
        nodes[int(branch[1])]['weight'] += branch[0]
        nodes[int(branch[2])]['count'] += 1
        nodes[int(branch[2])]['weight'] += branch[0]
    # Loop to find root nodes
    while True:
        # Order to try and find the best match
        nodes.sort(order=('count', 'weight'))
        # Can only order ascending, so get last num_roots
        root_nodes = nodes[-roots:]
        if root_nodes.shape[0] > 1:
            # See if there's a connecting node in the root selection
            joining_node = most_connected(tree, root_nodes)
            if joining_node is not None:
                # If it's there, give it minimum priority...
                nodes[- roots + joining_node]['count'] = -1
                nodes[- roots + joining_node]['weight'] = -1
                continue        # ... and go back and sort again.
        break           # Have the root nodes
    return root_nodes[:]['name']


def split_tree(tree, nodes0, nodes1):
    """Splits the tree in two, each only containing the given nodes.
    nodes{0,1} should be disjoint and exhaustive."""
    # Get the tree in a nice format
    connections = tree[:, (1, 2)].astype(np.int)
    # Initiate the two trees
    tree0 = []
    tree1 = []
    # Check each connection, putting it in correct tree
    for i, connection in enumerate(connections):
        if connection[0] in nodes0 and connection[1] not in nodes1:
            tree0.append(tree[i])
        elif connection[0] in nodes1 and connection[1] not in nodes0:
            tree1.append(tree[i])
        elif connection[0] not in nodes0 and connection[0] not in nodes1:
            raise ValueError("Node arrays are not exhaustive: {0} not found."
                                .format(connection[0]))
        elif connection[1] not in nodes0 and connection[1] not in nodes1:
            raise ValueError("Node arrays are not exhaustive: {0} not found."
                                .format(connection[1]))
        else:
            raise ValueError("Node arrays are not disjoint: {0} spans arrays."
                                .format(connection))
    # Return two new trees as arrays
    return np.array(tree0), np.array(tree1)


def bayesian_arcs(tree, roots, nodes):
    """Finds the directional arc connections from the root nodes."""
    # Initiate arcs with root nodes of the tree
    root_nodes = bayesian_roots(tree, roots, nodes)
    log.debug("Roots:\t{0}".format(root_nodes))
    arcs = [[root] for root in root_nodes]
    # Get the tree in a nice format
    connections = tree[:, (1, 2)].astype(np.int) if tree.shape[0] else []
    # Controll sets (only one needed, but the two lend clarity)
    visited = set(root_nodes)
    remaining = set(nodes) - visited
    while remaining:                # We haven't visited them all yet.
        for node in remaining:
            node_cons = []
            for connection in connections:      # Find valid connections
                if node == connection[0] and connection[1] in visited:
                    node_cons.append(connection[1])
                elif node == connection[1] and connection[0] in visited:
                    node_cons.append(connection[0])
            if node_cons:       # We have [a] valid connection[s]
                arcs.append([node] + node_cons)         # Add group to arcs...
                visited.add(node)       # ... and add node to visited.
        if remaining.isdisjoint(visited):
            # No nodes visited this iteration
            log.warn("No nodes visited this iteration")
            log.info("... root nodes do not cover whole tree, so split tree.")
            tree0, tree1 = split_tree(tree, visited, remaining)
            log.debug("tree 0:\t{0}\n{1}".format(visited, tree0))
            log.debug("tree 1:\t{0}\n{1}".format(remaining, tree1))
            # TODO: work for more root values other than 2
            arcs = (bayesian_arcs(tree0, roots - 1, visited) +
                    bayesian_arcs(tree1, 1, remaining))
            break
        else:
            remaining -= visited        # Update node list removing visited.
    return arcs


def bayesian_CPTs(data, arcs, states):
    """Generate the CPTs for a Bayesian Network from a list of arcs."""
    cpts = []
    # Mapping of arc sizes to geration functions.
    cpt_func = {1: Prior, 2: CPT, 3: CPT_2}
    # Generate CPTs from arcs using mapping.
    cpts = [cpt_func[len(arc)](*[data] + arc + [states]) for arc in arcs]
    return cpts


def BayesianNetwork(data, tree, variables, states, roots):
    """Generate a Bayesian Network from data."""
    log.info("Bayesian Network for tree:\n{0}".format(tree))
    arcs = bayesian_arcs(tree, roots, range(variables))
    log.info("has arcs:\n{0}".format(arcs))
    cpts = bayesian_CPTs(data, arcs, states)
    return arcs, cpts
# end of coursework 3 task 2.


def MDLSize(arcs, cpts, points, states):
    """Function to calculate the MDL size of a Bayesian Network."""
    mdl_size = 0.0
    # Coursework 3 task 3 begins here...
    link_size = {1: lambda p: p.shape[0] - 1,
                 2: lambda c: (c.shape[0] - 1) * c.shape[1],
                 3: lambda c2: (c2.shape[0] - 1) * c2.shape[1] * c2.shape[2]}
    modB = sum([link_size[cpt.ndim](cpt) for cpt in cpts])
    mdl_size = modB * np.log2(points)/2
    # end of coursework 3 task 3.
    return mdl_size


def JointProbability(data_point, arcs, cpts):
    """Function to calculate the joint probability of a single data point in a
    Network."""
    jp = np.ones(1, dtype='f8')
    # Coursework 3 task 4 begins here...
    point = np.array(data_point)
    for i, arc in enumerate(arcs):
        address = point[arc]
        value = cpts[i].item(*address)
        jp *= value
    # end of coursework 3 task 4.
    return jp


def MDLAccuracy(data, arcs, cpts):
    """Function to calculate the MDLAccuracy from a data set."""
    mdl_accuracy = 0.0
    # Coursework 3 task 5 begins here...
    for d in data:
        jp = JointProbability(d, arcs, cpts)
        jp_l2 = np.log2(jp) if jp else 0        # Prevent log(0)
        mdl_accuracy += jp_l2
    # end of coursework 3 task 5.
    return mdl_accuracy


# Coursework 3 task 6 begins here...
def best_network(data, tree, variables, points, states, roots):
    scores = []
    for i in range(len(tree)):
        t = tree[(range(i) + range(i + 1, len(tree)))]
        arcs, cpts = BayesianNetwork(data, t, variables, states, roots)
        scores.append(MDLSize(arcs, cpts, points, states) -
                      MDLAccuracy(data, arcs, cpts))
    return min(scores)
# end of coursework 3 task 6.


def cw3():
    """main() part of Coursework 03."""
    fl = "IDAPIResults03.txt"
    if os.path.exists(fl):
        os.remove(fl)
    
    (variables, roots, states,
        points, datain) = IDAPI.ReadFile("HepatitisC.txt")
    data = np.array(datain)
    # p1
    IDAPI.AppendString(fl, "Coursework Three Results by:")
    IDAPI.AppendString(fl, "\ttjh08 - Thomas Hope")
    IDAPI.AppendString(fl, "\tjzy08 - Jason Ye")
    IDAPI.AppendString(fl, "")
    # Generate the network
    #arcs, cpts = ExampleBayesianNetwork(data, states)
    dep_matrix = DependencyMatrix(data, variables, states)
    dep_list = DependencyList(dep_matrix)
    tree = SpanningTreeAlgorithm(dep_list, variables)
    arcs, cpts = BayesianNetwork(data, tree, variables, states, roots)
    # p2
    mdl_size = MDLSize(arcs, cpts, points, states)
    IDAPI.AppendString(fl, "The MDLSize of the network:")
    IDAPI.AppendList(fl, np.array([mdl_size]))
    # p3
    mdl_acc = MDLAccuracy(data, arcs, cpts)
    IDAPI.AppendString(fl, "The MDLAccuracy of the network:")
    IDAPI.AppendList(fl, np.array([mdl_acc]))
    # p4
    mdl_score = mdl_size - mdl_acc
    IDAPI.AppendString(fl, "The MDLScore of the network:")
    IDAPI.AppendList(fl, np.array([mdl_score]))
    # p5
    best_score = best_network(data, tree, variables, points, states, roots)
    IDAPI.AppendString(fl, "The best score of the network with one arc removed:")
    IDAPI.AppendList(fl, np.array([best_score]))
    # pn
    IDAPI.AppendString(fl, "END")
    ####
    os.system('cat {0}'.format(fl))


#
# End of coursework 3
#
# Coursework 4 begins here
#


def Mean(data):
    # not needed because of 'from __future__ import division'
    #real_data = data.astype(float)
    #num_vars = data.shape[1]
    #mean = []
    # Coursework 4 task 1 begins here...
    #if data.shape[0]:
        #mean = data.sum(0) / data.shape[0]
    #else:
        #mean = np.zeros(data.shape[0])
    mean = data.mean(0)
    # end of coursework 4 task 1.
    return mean


def Covariance(data):
    # not needed because of 'from __future__ import division'
    #real_data = data.astype(float)
    #num_vars = data.shape[1]
    #covar = np.zeros((num_vars, num_vars), float)
    # Coursework 4 task 2 begins here...
    #mean = Mean(data)
    #for i, i_var in enumerate(data.T):
        #for j, j_var in enumerate(data.T):
            #covar[i, j] = (((i_var - mean[i]) *
                            #(j_var - mean[j])).sum(0) /
                           #(num_vars - 1))
    covar = np.cov(data.T)
    # end of coursework 4 task 2.
    return covar


def CreateEigenfaceFiles(basis):
    # Coursework 4 task 3 begins here...
    for i, component in enumerate(basis):
        IDAPI.SaveEigenface(component, "PrincipalComponent{0}.jpg".format(i))
    # end of coursework 4 task 3.


def ProjectFace(basis, mean, face_image):
    #magnitudes = []
    # Coursework 4 task 4 begins here...
    magnitudes = np.matrix(face_image - mean) * np.matrix(basis.T)
    # end of coursework 4 task 4.
    return np.array(magnitudes)[0]


def CreatePartialReconstructions(basis, mean, magnitudes):
    # Coursework 4 task 5 begins here...
    b_matrix = np.matrix(basis)
    len_m = len(magnitudes)
    fl_name = "PartialReconstruction{0}.jpg"
    IDAPI.SaveEigenface(mean, fl_name.format('Mean'))
    for i in range(len_m):
        m_matrix = np.matrix(magnitudes * ([1] * i + [0] * (len_m - i)))
        IDAPI.SaveEigenface(np.array((m_matrix * b_matrix + mean))[0],
                            fl_name.format(i))
    # end of coursework 4 task 5.


def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 6 begins here...
    # The first part is almost identical to the above Covariance function, but
    # because the data has so many variables you need to use the Kohonen Lowe
    # method described in lecture 15 The output should be a list of the
    # principal components normalised and sorted in descending order of their
    # eignevalues magnitudes.

    # end of coursework 4 task 6.
    return np.array(orthoPhi)


def cw4():
    """main() part of Coursework 04."""
    fl = "IDAPIResults04.rst"
    if os.path.exists(fl):
        os.remove(fl)
    
    (variables, roots, states,
        points, datain) = IDAPI.ReadFile("HepatitisC.txt")
    data = np.array(datain)
    # p1
    IDAPI.AppendString(fl, "Coursework Four Results by:\n")
    IDAPI.AppendString(fl, "* tjh08 - Thomas Hope")
    IDAPI.AppendString(fl, "* jzy08 - Jason Ye")
    IDAPI.AppendString(fl, "")
    # p2
    hepC_mean = Mean(data)
    IDAPI.AppendString(fl, "The Mean vector of the HepatitisC data set:")
    IDAPI.AppendString(fl, "\n::\n")
    IDAPI.AppendList(fl, hepC_mean)
    # p3
    hepC_covar = Covariance(data)
    IDAPI.AppendString(fl, "The Covariance matrix of the HepatitisC data set:")
    IDAPI.AppendString(fl, "\n::\n")
    IDAPI.AppendArray(fl, hepC_covar)
    # p4
    basis = np.array(IDAPI.ReadEigenfaceBasis())
    mean_face = np.array(IDAPI.ReadOneImage("MeanImage.jpg"))
    face = np.array(IDAPI.ReadOneImage('c.pgm'))
    magnitudes = ProjectFace(basis, mean_face, face)
    CreatePartialReconstructions(basis, mean_face, magnitudes)

    # pn
    IDAPI.AppendString(fl, "END")
    ####
    #os.system('cat {0}'.format(fl))
    #os.system("rst2latex.py {0}.rst {0}.tex".format(fl.rpartition('.')[0]))
    #os.system("pdflatex {0}.tex".format(fl.rpartition('.')[0]))
    #os.system("xdg-open {0}.pdf".format(fl.rpartition('.')[0]))


#
# main() part of program
#


if __name__ == '__main__':
    # Raise all errors from numpy functions.
    old_settings = np.seterr(all='raise')
    np.set_printoptions(precision=3)
    log.basicConfig(level=log.ERROR)
    cw4()
    sys.exit(0)
