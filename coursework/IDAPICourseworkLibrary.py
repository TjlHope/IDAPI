#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob

import Image
import numpy


def ReadFile(filename):
    """Function to read data from a file in the format defined by Duncan."""
    f = open(filename)
    # The first line contains the number of variables. the function int
    # converts a string to an integer.
    noVariables = int(f.readline())
    # The second line contains the number of root nodes.
    noRoots = int(f.readline())
    # The third line contains the number of states of each variable
    # This command extracts a list of integers. The split method breaks the
    # line into a list of substrings.
    # The map function applies a function (int) to a list (the substrings) to
    # produce a list of integers.
    noStates = map(int, f.readline().split())
    # The fourth line contains a single integer the number of data points.
    noDataPoints = int(f.readline())
    # All the subsequent lines of the file are data points. Each line is
    # extracted as a list of integers which is appended to the list datain.
    datain = []
    for x in range(noDataPoints):
        datain.append(map(int, f.readline().split()))
    f.close()
    return [noVariables, noRoots, noStates, noDataPoints, datain]


def AppendArray(filename, anArray):
    """Function to write an array to a results file the array is assumed to be
    either of proababilities of dependencies.
    """
    f = open(filename, 'a')
    for row in range(anArray.shape[0]):
        for col in range(anArray.shape[1]):
            f.write('%6.3f ' % (anArray[row, col]))
        f.write('\n')
    f.write('\n\n')
    f.close()


def AppendList(filename, aList):
    """Function to write a list to a results file."""
    f = open(filename, 'a')
    for row in range(aList.shape[0]):
        f.write('%6.3f ' % (aList[row]))
    f.write('\n\n')
    f.close()


def AppendString(filename, aString):
    """Function to write a string to a results file."""
    f = open(filename, 'a')
    f.write('%s\n' % (aString))
    f.close()


#
# Image handline functions
#
# These functions turn images into data sets and vice versa
#
def SaveEigenface(component, filename):
    """Function to turn a principal component into an image and save it. The
    assumed resolution is 92 by 112 pixels. The component is a one dimensional
    representation of an image with each row concatinated.
    """
    theMax = max(component)
    theMin = min(component)
    scale = 255.0 / (theMax - theMin)
    eigenfaceImage = map(int, (component - theMin) * scale)
    im = Image.new('L', (92, 112))
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            im.putpixel((x, y), eigenfaceImage[x + 92 * y])
    im.save(filename)


def ReadOneImage(filename):
    """Function to convert images into a data format equivalent to the above
    format where each row of an array is one image with rows concatinated into
    a single vector.
    The images for this project are assumed to be all of resolution 92 by 112
    pixels and are taken from the current directory in .pgm format.
    """
    datain = []
    im = Image.open(filename)
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            datain.append(im.getpixel((x, y)))
    return datain


def ReadImages():
    """Function to convert all images in current working directory using the
    previous function.
    """
    datain = []
    [datain.append(ReadOneImage(infile)) for infile in glob.glob("*.pgm")]
    return datain


# Needed for testing tasks 4.4 to 4.6 in the event that task 4.3 is unattempted
# or unsucessful.

def WriteEigenfaceBasis(pcBasis):
    """Function to save an eigenface basis to a file."""
    f = open("EigenfaceBasis.txt", "w")
    for row in range(pcBasis.shape[0]):
        for col in range(pcBasis.shape[1]):
            f.write('%12.10f ' % (pcBasis[row, col]))
        f.write('\n')
    f.write('\n\n')
    f.close()


def ReadEigenfaceBasis():
    """Function to read an eigenface basis from a file."""
    f = open("PrincipalComponents.txt")
    datain = []
    for line in range(10):
        datain.append(map(float, f.readline().split()))
    f.close()
    return numpy.array(datain)


if __name__ == '__main__':
    import sys
    sys.stderr.write("Library of functions for IDAPI coursework\n")
    sys.exit(1)
