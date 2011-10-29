#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python

from __future__ import division

import os

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
    cpt = np.zeros((states[child], states[parent]), float)
    # Coursework 1 task 2 should be inserted here...
    # cpt/ link matrix: P(C|P) = [[P(c0|p0), P(c0|p1), ..., P(c0|pn)],
    #                             [P(c1|p0), P(c1|p1), ..., P(c1|pn)],
    #                             ...,
    #                             [P(cn|p0), P(cn|p1), ..., P(cn|pn)]]
    for r in data:
        cpt[r[child], r[parent]] += 1   # Increment necessary points
    for c in cpt.T:
        c /= sum(c)                     # normalise
    # end of coursework 1 task 2.
    return cpt


def JPT(data, row, col, states):
    """Function to calculate the joint probability table of two variables in
    the data set.
    """
    jpt = np.zeros((states[row], states[col]), float)
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
        c /= sum(c)     # normalise the columns
    # end of coursework 1 task 4.
    return jpt


def Query(query, network):
    """Function to query a naive Bayesian network."""
    root_pdf = np.zeros(network[0].shape[0], float)
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


def MutualInformation(jP):
    """Calculate the mutual information from the joint probability table of two
    variables.
    """
    mi = 0.0
    # Coursework 2 task 1 should be inserted here...

    # end of coursework 2 task 1.
    return mi


def DependencyMatrix(theData, noVariables):
    """Constructs a dependency matrix for all the variables."""
    MIMatrix = np.zeros((noVariables, noVariables))
    # Coursework 2 task 2 should be inserted here...

    # end of coursework 2 task 2.
    return MIMatrix


def DependencyList(depMatrix):
    """Function to compute an ordered list of dependencies."""
    depList = []
    # Coursework 2 task 3 should be inserted here...

    # end of coursework 2 task 3.
    return np.array(depList2)


# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    """Functions implementing the spanning tree algorithm."""
    spanningTree = []

    return np.array(spanningTree)


#
# End of coursework 2
#
# Coursework 3 begins here
#


def CPT_2(theData, child, parent1, parent2, noStates):
    """Function to compute a CPT with multiple parents from he data set it is
    assumed that the states are designated by consecutive integers starting
    with 0.
    """
    cPT = np.zeros((noStates[child], noStates[parent1], noStates[parent2]),
                   float)
    # Coursework 3 task 1 should be inserted here...

    # end of Coursework 3 task 1.
    return cPT


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
    cw1()
