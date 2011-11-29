#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python

from __future__ import division

import os
from operator import itemgetter

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
    cpt = np.zeros((states[child], states[parent]), dtype="f4")
    # Coursework 1 task 2 should be inserted here...
    # cpt/ link matrix: P(C|P) = [[P(c0|p0), P(c0|p1), ..., P(c0|pn)],
    #                             [P(c1|p0), P(c1|p1), ..., P(c1|pn)],
    #                             ...,
    #                             [P(cn|p0), P(cn|p1), ..., P(cn|pn)]]
    for r in data:
        cpt[r[child], r[parent]] += 1   # Increment necessary points
    print cpt
    for c in cpt.T:
        s = sum(c)              # test for divide by zero
        if s:
            c /= s              # normalise the columns
    # end of coursework 1 task 2.
    print cpt
    return cpt


def JPT(data, row, col, states):
    """Function to calculate the joint probability table of two variables in
    the data set.
    """
    jpt = np.zeros((states[row], states[col]), dtype="f4")
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
    root_pdf = np.zeros(network[0].shape[0], dtype="f4")
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


# Coursework 2 task 4
def connected(tree, src, dst, path=[]):
    print path
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
# End of Coursework 2 task 4


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
                   dtype="f4")
    # Coursework 3 task 1 should be inserted here...
    cpt_1 = CPT(data, child, parent1, states)
    cpt_2 = CPT(data, child, parent2, states)

    # end of Coursework 3 task 1.
    return cpt


def ExampleBayesianNetwork(theData, noStates):
    """Definition of a Bayesian Network."""
    arcList = [[0], [1], [2, 0], [3, 2, 1], [4, 3], [5, 3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList


# Coursework 3 task 2 begins here

# end of coursework 3 task 2


def MDLSize(arcList, cptList, noDataPoints, noStates):
    """Function to calculate the MDL size of a Bayesian Network."""
    mdlSize = 0.0
    # Coursework 3 task 3 begins here...

    # end of coursework 3 task 3.
    return mdlSize


def JointProbability(dataPoint, arcList, cptList):
    """Function to calculate the joint probability of a single data point in a
    Network."""
    jP = 1.0
    # Coursework 3 task 4 begins here...

    # end of coursework 3 task 4.
    return jP


def MDLAccuracy(theData, arcList, cptList):
    """Function to calculate the MDLAccuracy from a data set."""
    mdlAccuracy = 0
    # Coursework 3 task 5 begins here...

    # end of coursework 3 task 5.
    return mdlAccuracy


def cw3():
    """main() part of Coursework 03."""
    fl = "Results03.txt"
    if os.path.exists(fl):
        os.remove(fl)
    
    (variables, roots, states,
        points, datain) = IDAPI.ReadFile("HepatitisC.txt")
    data = np.array(datain)
    IDAPI.AppendString(fl, "Coursework Two Results by:\n")
    IDAPI.AppendString(fl, "* tjh08 - Thomas Hope")
    IDAPI.AppendString(fl, "* jzy08 - Jason Ye")
    IDAPI.AppendString(fl, "")
    
    cpt = CPT_2(data, 2, 1, 0, states)

    IDAPI.AppendString(fl, "\nEND")


#
# End of coursework 3
#
# Coursework 4 begins here
#


def Mean(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    mean = []
    # Coursework 4 task 1 begins here...

    # end of coursework 4 task 1.
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here...

    # end of coursework 4 task 2.
    return covar


def CreateEigenfaceFiles(theBasis):
    pass        # delete this when you do the coursework
    # Coursework 4 task 3 begins here...

    # end of coursework 4 task 3.


def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here...

    # end of coursework 4 task 4.
    return np.array(magnitudes)


def CreatePartialReconstructions(aBasis, aMean, componentMags):
    pass        # delete this when you do the coursework
    # Coursework 4 task 5 begins here...

    # end of coursework 4 task 5.


def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here...
    # The first part is almost identical to the above Covariance function, but
    # because the data has so many variables you need to use the Kohonen Lowe
    # method described in lecture 15 The output should be a list of the
    # principal components normalised and sorted in descending order of their
    # eignevalues magnitudes.

    # end of coursework 4 task 6.
    return np.array(orthoPhi)


#
# main() part of program
#


if __name__ == '__main__':
    # Raise all errors from numpy functions.
    old_settings = np.seterr(all='raise')
    cw3()
