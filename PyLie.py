__author__ = 'florian'

"""
This is the python implementation of the Group Theory method of Susyno.
"""

import pudb
import sys

sys.path.insert(0, '/Applications/HEPtools/sympy-0.7.6')
import numpy as np
from sympy import *
import copy as cp


class CartanMatrix(object):
    """
    Represents a Cartan Matrix
    """

    def __init__(self, name, Id):

        self._translation = {"SU": "A", "SP": "C", "SO": ("B", "D")}
        self._classicalLieAlgebras = ["A", "B", "C", "D"]
        self._name = name
        self._id = Id
        self.cartan = []  # store the Cartan matrix
        self._constructCartanMatrix()

    def _constructCartanMatrix(self):

        if not (self._name in self._classicalLieAlgebras):
            if self._name in self._translation:
                if self._name == "SU":
                    self._id -= 1
                    self._name = self._translation[self._name]
                    return self._constructCartanMatrix()
                elif self._name == "SP":
                    if self._id % 2 == 0:
                        self._id /= 2
                        self._name = self._translation[self._name]
                        return self._constructCartanMatrix()
                    else:
                        print("error 'SP' Id number must be even")
                        return
                elif self._name == "SO" and self._id % 2 == 0:
                    if self._id < 5:
                        print("Error n >=3 or > 4 for SO(n)")
                        return
                    self._id /= 2
                    self._name = self._translation[self._name]
                    return self._constructCartanMatrix()
                elif self._name == "SO" and self._id % 2 == 1:
                    self._id = (self._id - 1) / 2
                    self._name = self._translation[self._name]
                    return self._constructCartanMatrix()
                else:
                    print"Error unknown Lie Algebra, try 'A', 'B','C' or 'D'"
                    return

        if self._name in ["A", "B", "C"] and self._id == 1:
            self.cartan = SparseMatrix([2])
        if self._name == "A" and self._id > 1:
            self.cartan = SparseMatrix(self._id, self._id, lambda i, j: self._fillUpFunctionA(i, j))
        elif self._name == "B" and self._id > 1:
            self.cartan = SparseMatrix(self._id, self._id, lambda i, j: self._fillUpFunctionB(i, j))
            self.cartan[self._id - 2, self._id - 1] = -2
        elif self._name == "C" and self._id > 1:
            self.cartan = SparseMatrix(self._id, self._id, lambda i, j: self._fillUpFunctionC(i, j))
            self.cartan[self._id - 1, self._id - 2] = -2
        elif self._name == "D" and self._id > 1:
            self.cartan = SparseMatrix(self._id, self._id, lambda i, j: self._fillUpFunctionD(i, j))
            self.cartan[self._id - 1, self._id - 2] = 0
            self.cartan[self._id - 2, self._id - 1] = 0
            self.cartan[self._id - 1, self._id - 3] = -1
            self.cartan[self._id - 3, self._id - 1] = -1

    def _fillUpFunctionA(self, i, j):
        if i == j:
            return 2
        elif i == j + 1 or j == i + 1:
            return -1
        else:
            return 0

    def _fillUpFunctionB(self, i, j):
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionC(self, i, j):
        return self._fillUpFunctionA(i, j)

    def _fillUpFunctionD(self, i, j):
        return self._fillUpFunctionA(i, j)


class LieAlgebra(object):
    """
    This is the central class implmenting all the method that one can perform on the lie algebra
    """

    def __init__(self, cartanMatrix):
        self.cartan = cartanMatrix
        self.cm = self.cartan.cartan
        self.n = self.cm.shape[0]  # size of the n times n matrix
        self.cminv = self.cm.inv()  # inverse of the Cartan matrix
        self.ncminv = matrix2numpy(self.cminv)  # numpy version of the inverse of the Cartan matrix
        self.ncm = matrix2numpy(self.cm)  # numpy version of the Cartan Matrix
        self.matD = self._matrixD()  # D matrix
        self.smatD = self._specialMatrixD()  # Special D matrix
        self.cmID = np.dot(self.ncminv, self.matD)  # matrix product of the inverse Cartan matrix and D matrix
        self.proots = self._positiveRoots()  # calc the positive roots

    def _matrixD(self):
        """
        Returns a diagonal matrix with the values <root i, root i>
        """
        positions = sum(
            [[(irow, icol) for icol, col in enumerate(row) if (col in [-1, -2, -3]) and (irow < icol)] for irow, row in
             enumerate(self.ncm)], [])
        result = np.ones((1, self.n))[0]
        for coord1, coord2 in positions:
            result[coord2] = self.cm[coord2, coord1] / self.cm[coord1, coord2] * result[coord1]
        return np.diagflat(result)

    def _simpleProduct(self, v1, v2, cmID):
        # Scalar product from two vector and a matrix
        if type(v2) == list:
            v2 = np.array(v2)
        if type(v1) == list:
            v1 = np.array(v1)
        return 1 / 2. * np.dot(np.dot(v1, cmID), v2.transpose())[0, 0]

    def _longestWeylWord(self):
        # returns the longest Weyl word: from the Lie manual see Susyno
        weight = [-1] * self.n
        result = []
        while map(abs, weight) != weight:
            for iel, el in enumerate(weight):
                if el < 0:
                    break
            weight = self._reflectWeight(weight, iel + 1)
            result.insert(0, iel + 1)
        return result

    def _reflectWeight(self, weight, i):
        """
        Reflects a given weight. WARNING The index i is from 1 to n
        """
        result = cp.deepcopy(weight)
        result[i - 1] = -weight[i - 1]
        for ii in range(1, 5):
            if self.smatD[i - 1, ii - 1] != 0:
                result[self.smatD[i - 1, ii - 1] - 1] += weight[i - 1]
        return result

    def _specialMatrixD(self):
        result = SparseMatrix(self.n, 4, 0)
        for i in range(1, self.n + 1):
            k = 1
            for j in range(1, self.n + 1):
                if self.cm[i - 1, j - 1] == -1:
                    result[i - 1, k - 1] = j
                    k += 1
                if self.cm[i - 1, j - 1] == -2:
                    result[i - 1, k - 1] = j
                    result[i - 1, k - 1 + 1] = j
                    k += 2
                if self.cm[i - 1, j - 1] == -3:
                    result[i - 1, k - 1] = j
                    result[i - 1, k - 1 + 1] = j
                    result[i - 1, k - 1 + 2] = j
                    k += 3
        return result

    def _weylOrbit(self, weight):
        """
        Creates the weyl orbit i.e. the system of simple root
        """
        counter = 0
        result, wL = [], []
        wL.append([weight])
        result.append(weight)
        while len(wL[counter]) != 0:
            counter += 1
            wL.append([])
            for j in range(1, len(wL[counter - 1]) + 1):
                for i in range(1, self.n + 1):
                    if wL[counter - 1][j - 1][i - 1] > 0:
                        aux = self._reflectWeight(wL[counter - 1][j - 1], i)[i + 1 - 1:self.n + 1]
                        if aux == map(abs, aux):
                            wL[counter].append(self._reflectWeight(wL[counter - 1][j - 1], i))
            result = result + wL[counter]  # Join the list
        return result

    def _findM(self, ex, el, ind):
        aux1 = cp.copy(el[ind - 1])
        aux2 = cp.copy(el)
        aux2[ind - 1] = 0
        auxMax = 0
        for ii in range(1, aux1 + 2):
            if ex.count(aux2) == 1:
                auxMax = aux1 - ii + 1
                return auxMax
            aux2[ind - 1] = cp.copy(aux2[ind - 1] + 1)
        return auxMax

    def _positiveRoots(self):
        """
        Returns the positive roots of a given group
        """
        aux1 = [[KroneckerDelta(i, j) for j in range(1, self.n + 1)] for i in range(1, self.n + 1)]
        count = 0
        weights = cp.copy(self.cm)
        while count < weights.rows:
            count += 1
            aux2 = cp.copy(aux1[count - 1])
            for inti in range(1, self.n + 1):
                aux3 = cp.copy(aux2)
                aux3[inti - 1] += 1
                if self._findM(aux1, aux2, inti) - weights[count - 1, inti - 1] > 0 and aux1.count(aux3) == 0:
                    weights = weights.col_join(weights.row(count - 1) + self.cm.row(inti - 1))
                    aux1.append(aux3)
        return matrix2numpy(weights)

    def _dominantConjugate(self, weight):
        weight = weight[0]
        if self.cm == np.array([[2]]):  # SU2 code
            if weight[0] < 0:
                return [-weight, 1]
            else:
                return [weight, 0]
        else:
            index = 0
            dWeight = weight
            i = 1
            while i <= self.n:
                if (dWeight[i - 1] < 0):
                    index += 1
                    dWeight = self._reflectWeight(dWeight, i)
                    i = min([self.smatD[i - 1, 0], i + 1])
                else:
                    i += 1
            return [dWeight, index]

    def _dominantWeights(self, weight):
        """
        Generate the dominant weights without dimentionality information
        """
        # convert the weight
        weight = np.array([weight])
        listw = [weight]
        counter = 1
        while counter <= len(listw):
            aux = [listw[counter - 1] - self.proots[i] for i in range(len(self.proots))]
            aux = [el for el in aux if np.all(el == abs(el))]
            listw = listw + aux
            # remove duplicates this is actually a pain since numpy are not hashable
            listw = {array.tostring(): array for array in listw}.values()
            counter += 1
        # need to sort listw
        def sortList(a, b):
            tp1 = list(np.dot(-(a - b), self.ncminv)[0])
            return cmp(tp1, [0] * a.shape[1])
        # The Sorting looks to be identical to what was done in SUSYNO willl require further checking at some point
        listw.sort(sortList)
        # listw = [np.array([[1,1]]),np.array([[0,0]])]
        # Sum the positive roots
        deltaTimes2 = self.proots.sum(axis=0)
        functionaux = {tuple(listw[0].ravel()): 1}
        result = [[listw[0], 1]]
        for j in range(2, len(listw) + 1):
            for i in range(1, len(self.proots) + 1):
                k = 1
                aux1 = functionaux[tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0])]
                key = tuple(listw[j - 1].ravel())
                while aux1 != 0:
                    aux2 = k * (self.proots[i - 1] + listw[j - 1])
                    if key in functionaux:
                        functionaux[key] += 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self.cmID)
                    else:
                        functionaux[key] = 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self.cmID)
                    k += 1
                    # update aux1 value
                    kkey = tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0])
                    if kkey in functionaux:
                        aux1 = functionaux[kkey]
                    else:
                        aux1 = 0
            functionaux[key] /= self._simpleProduct(listw[0] + listw[j - 1] + deltaTimes2,
                                                    listw[0] - listw[j - 1], self.cmID)
            result.append([listw[j - 1], functionaux[tuple(listw[j - 1].ravel())]])
        return result
