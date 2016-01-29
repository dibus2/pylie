__author__ = 'florian'

"""
PyLie is a python module for Lie group calculation for particle physics. In particular, it can manipulate any of the Lie
algebra, calculate the system of roots, weights, casimir, dynkin, Matrix representation and Invariants of the product of
several irrep.
It is a python implementation of the Susyno group method.
"""

import pudb
import sys

sys.path.insert(0, '/Applications/HEPtools/sympy-0.7.6')
import numpy as np
from sympy import *
from sympy.combinatorics import Permutation

init_printing(use_latex=True)
import copy as cp
import operator
import itertools


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
                    self._name = self._translation[self._name][1]
                    return self._constructCartanMatrix()
                elif self._name == "SO" and self._id % 2 == 1:
                    self._id = (self._id - 1) / 2
                    self._name = self._translation[self._name][0]
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
        self._n = self.cm.shape[0]  # size of the n times n matrix
        self.cminv = self.cm.inv()  # inverse of the Cartan matrix
        self.ncminv = matrix2numpy(self.cminv)  # numpy version of the inverse of the Cartan matrix
        self.ncm = matrix2numpy(self.cm)  # numpy version of the Cartan Matrix
        self._matD = self._matrixD()  # D matrix
        self._smatD = self._specialMatrixD()  # Special D matrix
        self._cmID = np.dot(self.ncminv, self._matD)  # matrix product of the inverse Cartan matrix and D matrix
        self._cmIDN = self._cmID / np.max(self._matD)  # same as cmID but normalized to the max of matD
        self.proots = self._positiveRoots()  # calc the positive roots
        # Sum the positive roots
        self._deltaTimes2 = self.proots.sum(axis=0)
        self.adjoint = self._getAdjoint()
        self.longestWeylWord = self._longestWeylWord()
        # store the matrices for speeding up multiple calls
        self._repMinimalMatrices = {}
        self._repMatrices = {}
        self._dominantWeightsStore = {}
        self._invariantsStore = {}
        self.a, self.b, self.c, self.d, self.e = map(IndexedBase, ['a', 'b', 'c', 'd', 'e'])
        self.f, self.g, self.h, self.i = map(IndexedBase, ['f', 'g', 'h', 'i'])
        self._symblist = [self.a, self.b, self.c, self.d, self.e]
        self._symbdummy = [self.f, self.g, self.h, self.i]
        self.p, self.q = map(Wild, ['p', 'q'])
        self.pp = Wild('pp', exclude=[IndexedBase])
        # create an Sn object for all the manipulation on the  Sn  group
        self.Sn = Sn()
        # create a MathGroup object for the auxiliary functions
        self.math = MathGroup()

    def _matrixD(self):
        """
        Returns a diagonal matrix with the values <root i, root i>
        """
        positions = sum(
            [[(irow, icol) for icol, col in enumerate(row) if (col in [-1, -2, -3]) and (irow < icol)] for irow, row in
             enumerate(self.ncm)], [])
        result = np.ones((1, self._n), dtype=object)[0]

        for coord1, coord2 in positions:
            result[coord2] = Rational(self.cm[coord2, coord1], self.cm[coord1, coord2]) * result[coord1]
        return np.diagflat(result)

    def _simpleProduct(self, v1, v2, cmID):
        # Scalar product from two vector and a matrix
        if type(v2) == list:
            v2 = np.array(v2)
        if type(v1) == list:
            v1 = np.array(v1)
        return Rational(1, 2) * (np.dot(np.dot(v1, cmID), v2.transpose())[0, 0])

    def _longestWeylWord(self):
        # returns the longest Weyl word: from the Lie manual see Susyno
        weight = [-1] * self._n
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
            if self._smatD[i - 1, ii - 1] != 0:
                result[self._smatD[i - 1, ii - 1] - 1] += weight[i - 1]
        return result

    def _specialMatrixD(self):
        result = SparseMatrix(self._n, 4, 0)
        for i in range(1, self._n + 1):
            k = 1
            for j in range(1, self._n + 1):
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
                for i in range(1, self._n + 1):
                    if wL[counter - 1][j - 1][i - 1] > 0:
                        aux = self._reflectWeight(wL[counter - 1][j - 1], i)[i + 1 - 1:self._n + 1]
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
        aux1 = [[KroneckerDelta(i, j) for j in range(1, self._n + 1)] for i in range(1, self._n + 1)]
        count = 0
        weights = cp.copy(self.cm)
        while count < weights.rows:
            count += 1
            aux2 = cp.copy(aux1[count - 1])
            for inti in range(1, self._n + 1):
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
            while i <= self._n:
                if (dWeight[i - 1] < 0):
                    index += 1
                    dWeight = self._reflectWeight(dWeight, i)
                    i = min([self._smatD[i - 1, 0], i + 1])
                else:
                    i += 1
            return [dWeight, index]

    def _dominantWeights(self, weight):
        """
        Generate the dominant weights without dimentionality information
        """
        keyStore = tuple(weight)
        if keyStore in self._dominantWeightsStore:
            return self._dominantWeightsStore[keyStore]
        # convert the weight
        weight = np.array([weight], dtype=int)
        listw = [weight]
        counter = 1
        while counter <= len(listw):
            aux = [listw[counter - 1] - self.proots[i] for i in range(len(self.proots))]
            aux = [el for el in aux if np.all(el == abs(el))]
            listw = listw + aux
            # remove duplicates this is actually a pain since numpy are not hashable
            tp = []
            listw = [self._nptokey(el) for el in listw]
            for el in listw:
                if not (el) in tp:
                    tp.append(el)
            listw = [np.array([el], dtype=int) for el in tp]
            counter += 1
        # need to sort listw
        def sortList(a, b):
            tp1 = list(np.dot(-(a - b), self.ncminv)[0])
            return cmp(tp1, [0] * a.shape[1])

        # The Sorting looks to be identical to what was done in SUSYNO willl require further checking at some point
        listw.sort(sortList)
        # listw = [np.array([[1,1]]),np.array([[0,0]])]
        functionaux = {self._nptokey(listw[0]): 1}
        result = [[listw[0], 1]]
        for j in range(2, len(listw) + 1):
            for i in range(1, len(self.proots) + 1):
                k = 1
                aux1 = self._indic(functionaux,
                                   tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0]))
                key = self._nptokey(listw[j - 1])
                while aux1 != 0:
                    aux2 = k * self.proots[i - 1] + listw[j - 1]
                    if key in functionaux:
                        functionaux[key] += 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self._cmID)
                    else:
                        functionaux[key] = 2 * aux1 * self._simpleProduct(aux2, [self.proots[i - 1]], self._cmID)
                    k += 1
                    # update aux1 value
                    kkey = tuple(self._dominantConjugate(k * self.proots[i - 1] + listw[j - 1])[0])
                    if kkey in functionaux:
                        aux1 = functionaux[kkey]
                    else:
                        aux1 = 0
            functionaux[key] /= self._simpleProduct(listw[0] + listw[j - 1] + self._deltaTimes2,
                                                    listw[0] - listw[j - 1], self._cmID)
            result.append([listw[j - 1], self._indic(functionaux, self._nptokey(listw[j - 1]))])
        self._dominantWeightsStore[keyStore] = result
        return result

    def casimir(self, irrep):
        """
        Returns the casimir of a given irrep
        """
        irrep = np.array([irrep])
        return self._simpleProduct(irrep, irrep + self._deltaTimes2, self._cmIDN)

    def dimR(self, irrep):
        """
        Returns the dimention of representation irrep
        """
        if not (type(irrep) == np.ndarray):
            irrep = np.array([irrep])
        delta = Rational(1, 2) * self._deltaTimes2
        if self.cartan._name == 'A' and self.cartan._id == 1:
            delta = np.array([delta])
        result = np.prod([
                             self._simpleProduct([self.proots[i - 1]], irrep + delta, self._cmID) / self._simpleProduct(
                                 [self.proots[i - 1]], [delta], self._cmID)
                             for i in range(1, len(self.proots) + 1)], axis=0)
        return int(result)

    def _representationIndex(self, irrep):
        delta = np.ones((1, self._n), dtype=int)
        # Factor of 2 ensures is due to the fact that SimpleProduct is defined such that Max[<\[Alpha],\[Alpha]>]=1 (considering all positive roots), but we would want it to be =2
        return Rational(self.dimR(irrep), self.dimR(self.adjoint)) * 2 * self._simpleProduct(irrep, irrep + 2 * delta,
                                                                                             self._cmID)

    def _getAdjoint(self):
        # returns the adjoint of the gauge group
        return np.array([self.proots[-1]])  # recast the expression

    def _repsUpToDimNAuxMethod(self, weight, digit, max, reap):
        """
        This is a recursive auxiliary method of repsUpToDimN
        """
        waux = cp.deepcopy(weight)
        waux[digit] = 0
        if digit == len(weight) - 1:
            while self.dimR(np.array([waux])) <= max:
                reap.append(cp.deepcopy([int(el) for el in waux.ravel()]))
                waux[digit] += 1
        else:
            while self.dimR(np.array([waux])) <= max:
                self._repsUpToDimNAuxMethod(waux, digit + 1, max, reap)
                waux[digit] += 1

    def repsUpToDimN(self, maxdim):
        """
        returns the list of irreps of dim less or equal to maxdim
        """
        reap = []
        self._repsUpToDimNAuxMethod(np.zeros((1, self._n))[0], 0, maxdim, reap)
        # sort the list according to dimension
        def sortByDimension(a, b):
            dma, dmb = self.dimR(a), self.dimR(b)
            repa, repb = self._representationIndex(np.array([a])), self._representationIndex(np.array([b]))
            conja, conjb = self._conjugacyClass(a), self._conjugacyClass(b)
            return cmp(tuple(flatten([dma, repa, conja])), tuple(flatten([dmb, repb, conjb])))

        reap.sort(sortByDimension)
        return reap

    def _getGroupWithRankNsqr(self, n):
        """
        returns the list of algebra with rank equal to n^2
        """
        res = []
        if n > 0: res.append(("A", n))
        if n > 2: res.append(("D", n))
        if n > 1: res.append(("B", n))
        if n > 2: res.append(("C", n))
        if n == 2: res.append(("G", 2))
        if n == 4: res.append(("F", 4))
        if n == 6: res.append(("E", 6))
        if n == 7: res.append(("E", 7))
        if n == 8: res.append(("E", 8))
        return res

    def _cmToFamilyAndSeries(self):
        aux = self._getGroupWithRankNsqr(self._n)
        # create the corresponding Cartan matrix
        cartans = [CartanMatrix(*el).cartan for el in aux]
        # find the position of the current group in this list
        ind = [iel for iel, el in enumerate(cartans) if el == self.cm][0]
        return aux[ind]

    def _conjugacyClass(self, irrep):
        if not (type(irrep) == np.ndarray):
            irrep = np.array(irrep)
        series, n = self._cmToFamilyAndSeries()
        if series == "A":
            return [np.sum([i * irrep[i - 1] for i in range(1, n + 1)]) % (n + 1)]
        if series == "B":
            return [irrep[n - 1] % 2]
        if series == "C":
            return [np.sum([irrep[i - 1] for i in range(1, n + 1, 2)]) % 2]
        if series == "D" and n % 2 == 1:
            return [(irrep[-2] + irrep[-1]) % 2,
                    (2 * np.sum([irrep[i - 1] for i in range(1, n - 1, 2)])
                     + (n - 2) * irrep[-2] + n * irrep[-1]) % 4]
        if series == "D" and n % 2 == 0:
            return [(irrep[-2] + irrep[-1]) % 2,
                    (2 * np.sum([irrep[i - 1] for i in range(1, n - 2, 2)])
                     + (n - 2) * irrep[-2] + n * irrep[-1]) % 4]
        if series == "E" and n == 6:
            return [(irrep[0] - irrep[1] + irrep[3] - irrep[4]) % 3]
        if series == "E" and n == 7:
            return [(irrep[3] + irrep[5] + irrep[6]) % 2]
        if series == "E" and n == 8:
            return [0]
        if series == "F":
            return [0]
        if series == "G":
            return [0]

    def conjugateIrrep(self, irrep):
        """
        returns the conjugated irrep
        """
        lbd = lambda weight, ind: self._reflectWeight(weight, ind)
        res = -reduce(lbd, [np.array([irrep])[0]] + self.longestWeylWord)
        return res

    def _weights(self, weights):
        """
        Reorder the weights of conjugate representations
        so that RepMatrices[group,ConjugateIrrep[group,w]]=-Conjugate[RepMatrices[group,w]]
        and Invariants[group,{w,ConjugateIrrep[group,w]},{False,False}]=a[1]b[1]+...+a[n]b[n]
        """
        if (cmp(list(weights), list(self.conjugateIrrep(weights))) in [-1, 0]) and not (np.all(
                    (self.conjugateIrrep(weights)) == weights)):
            return [np.array([-1, 1], dtype=int) * el for el in self._weights(self.conjugateIrrep(weights))]
        else:
            dw = self._dominantWeights(weights)
            result = sum(
                [[[np.array(el, dtype=int), dw[ii][1]] for el in self._weylOrbit(self._tolist(dw[ii][0][0]))] for ii in
                 range(len(dw))], [])

            def sortList(a, b):
                tp1 = list(np.dot(-(a[0] - b[0]), self.ncminv).ravel())
                return cmp(tp1, [0] * a[0].shape[0])

            result.sort(sortList)
            return result

    def repMinimalMatrices(self, maxW):
        """
        1) The output of this function is a list of sets of 3 matrices:
            {{E1, F1, H1},{E2, F2, H2},...,{En, Fn, Hn}}, where n is the group's rank.
            Hi are diagonal matrices, while Ei and Fi are raising and lowering operators.
            These matrices obey the Chevalley-Serre relations: [Ei, Ej]=delta_ij Hi, [Hi,Ej]= AjiEj, [Hi,Fj]= -AjiFj and [Hi,Hj=0
            here A is the Cartan matrix of the group/algebra.
        2) With the exception of SU(2) [n=1], these 3n matrices Ei, Fi, Hi do not generate the Lie algebra,
            which is bigger, as some raising and lowering operators are missing.
            However, these remaining operators can be obtained through simple commutations: [Ei,Ej], [Ei,[Ej,Ek]],...,[Fi,Fj], [Fi,[Fj,Fk]].
        3) This method clearly must assume a particular basis for each representation so the results are basis dependent.
        4) Also, unlike RepMatrices, the matrices given by this function are not Hermitian and therefore they do not conform with the usual requirements of model building in particle physics.
            However, for some applications, they might be all that is needed.
        """
        # check whether it s not been calculated already
        if type(maxW) == np.ndarray:
            tag = self._nptokey(maxW)
        else:
            tag = tuple(maxW)
            maxW = np.array([maxW])
        if tag in self._repMinimalMatrices:
            return self._repMinimalMatrices[tag]

        # auxiliary function for the repMatrices method base on the Chevalley-Serre relations

        cmaxW = self.conjugateIrrep(self._tolist(maxW))
        if cmp(self._tolist(maxW), self._tolist(cmaxW)) in [-1, 0] and not (np.all(cmaxW == maxW)):
            return [[-1 * el[1], -1 * el[0], -1 * el[2]] for el in
                    self.repMinimalMatrices(cmaxW)]
        else:
            listw = self._weights(self._tolist(maxW))
            up, dim, down = {}, {}, {}
            for i in range(len(listw)):
                dim[self._nptokey(listw[i][0])] = listw[i][1]
            up[self._nptokey(listw[0][0])] = np.zeros((1, self._n), dtype=int)
            for element in range(1, len(listw)):
                matrixT = [[]]
                for j in range(self._n):
                    col = [[]]
                    for i in range(self._n):
                        key1 = self._nptokey(listw[element][0] + self.ncm[i])
                        key2 = self._nptokey(listw[element][0] + self.ncm[i] + self.ncm[j])
                        key3 = self._nptokey(listw[element][0] + self.ncm[j])
                        dim1 = self._indic(dim, key1)
                        dim2 = self._indic(dim, key2)
                        dim3 = self._indic(dim, key3)
                        ax = 1 if col == [[]] else 0
                        if dim1 != 0 and dim3 != 0:
                            if dim2 != 0:
                                aux1 = up[self._nptokey(listw[element][0] + self.ncm[i])][j]
                                aux2 = down[self._nptokey(listw[element][0] + self.ncm[i] + self.ncm[j])][i]
                                if i != j:
                                    if type(col) != np.ndarray:
                                        col = np.dot(aux1, aux2)
                                    else:
                                        col = np.append(col, np.dot(aux1, aux2), axis=ax)
                                else:
                                    if type(col) != np.ndarray:
                                        col = np.dot(aux1, aux2) + (
                                                                       listw[element][0][i] + self.ncm[i, i]) * np.eye(
                                            dim1, dtype=object)
                                    else:
                                        col = np.append(col, np.dot(aux1, aux2) + (
                                            listw[element][0][i] + self.ncm[i, i]) * np.eye(
                                            dim1, dtype=object), axis=ax)
                            else:
                                if i != j:
                                    if type(col) != np.ndarray:
                                        col = np.zeros((dim1, dim3), dtype=object)
                                    else:
                                        col = np.append(col, np.zeros((dim1, dim3)), axis=ax)
                                else:
                                    tmp = (listw[element][0][i] + self.ncm[i, i]) * np.eye(dim1, dtype=object)
                                    if type(tmp) != np.ndarray:
                                        tmp = np.array([[tmp]])
                                    if type(col) != np.ndarray:
                                        col = tmp
                                    else:
                                        col = np.append(col, tmp, axis=ax)
                    if type(col) != list:
                        if type(matrixT) != np.ndarray:
                            matrixT = col.transpose()
                        else:
                            matrixT = np.append(matrixT, col.transpose(), axis=0)
                if matrixT == [[]]:
                    matrix = np.zeros((1, 1), dtype=object)
                else:
                    matrix = matrixT.transpose()
                aux1 = sum([self._indic(dim, self._nptokey(listw[element][0] + self.ncm[i])) for i in range(self._n)])
                aux2 = self._indic(dim, self._nptokey(listw[element][0]))
                cho = self.math._decompositionTypeCholesky(matrix)
                if cho.shape == (0,):
                    aux3 = np.array([[0]])
                else:
                    aux3 = np.pad(cho, pad_width=((0, max(aux1 - cho.shape[0], 0)), (0, max(aux2 - cho.shape[1], 0))),
                                  mode='constant')
                aux4 = aux3.transpose()
                if np.all((np.dot(aux3, aux4)) != matrix):
                    print("Error in repminimal matrices:", aux3, " ", aux4, " ", matrix)
                # Obtain the blocks in  (w+\[Alpha]i)i and wj. Use it to feed the recursive algorith so that we can calculate the next w's
                aux1 = np.array(
                    [[0, 0]])  # format (+-): (valid cm raise index i - 1, start position of weight w+cm[[i-1]]-1)
                for i in range(self._n):
                    key = self._nptokey(listw[element][0] + self.ncm[i])
                    if key in dim:
                        aux1 = np.append(aux1, np.array([[i + 1, aux1[-1, 1] + dim[key]]], dtype=object), axis=0)
                for i in range(len(aux1) - 1):
                    index = aux1[i + 1, 0]
                    posbegin = aux1[i, 1] + 1
                    posend = aux1[i + 1, 1]
                    key = self._nptokey(listw[element][0] + self.ncm[index - 1])
                    aux2 = down[key] if key in down else [[]] * self._n
                    aux2[index - 1] = aux3[posbegin - 1:posend]
                    down[key] = aux2
                    key2 = self._nptokey(listw[element][0])
                    aux2 = up[key2] if key2 in up else [[]] * self._n
                    aux2[index - 1] = (aux4.transpose()[posbegin - 1:posend]).transpose()
                    up[key2] = aux2
            # Put the collected pieces together and build the 3n matrices: hi,ei,fi
            begin, end = {self._nptokey(listw[0][0]): 1}, {self._nptokey(listw[0][0]): listw[0][1]}
            for element in range(1, len(listw)):
                key = self._nptokey(listw[element][0])
                key1 = self._nptokey(listw[element - 1][0])
                begin[key] = begin[key1] + listw[element - 1][1]
                end[key] = end[key1] + listw[element][1]
            aux2 = sum([listw[i][1] for i in range(len(listw))])
            aux3 = np.zeros((aux2, aux2), dtype=object)
            matrixE, matrixF, matrixH = [], [], []
            for i in range(self._n):
                aux6, aux7, aux8 = np.zeros((aux2, aux2), dtype=object), np.zeros((aux2, aux2), dtype=object), np.zeros(
                    (aux2, aux2), dtype=object)  # e[i], f[i], h[i]
                for element in range(len(listw)):
                    key = self._nptokey(listw[element][0] + self.ncm[i])
                    key2 = self._nptokey(listw[element][0])
                    key3 = self._nptokey(listw[element][0] - self.ncm[i])
                    if key in dim:
                        b1, e1 = begin[key], end[key]
                        b2, e2 = begin[key2], end[key2]
                        aux6[b1 - 1:e1, b2 - 1:e2] = (up[key2][i]).transpose()
                    if key3 in dim:
                        b1, e1 = begin[key3], end[key3]
                        b2, e2 = begin[key2], end[key2]
                        aux7[b1 - 1:e1, b2 - 1:e2] = (down[key2][i]).transpose()
                    b1, e1 = begin[key2], end[key2]
                    aux8[b1 - 1:e1, b1 - 1:e1] = listw[element][0][i] * np.eye(listw[element][1], dtype=object)
                matrixE.append(SparseMatrix(aux6))  # sparse matrix transfo
                matrixF.append(SparseMatrix(aux7))  # sparse matrix transfo
                matrixH.append(SparseMatrix(aux8))  # sparse matrix transfo
            aux1 = [[matrixE[i], matrixF[i], matrixH[i]] for i in range(self._n)]
            self._repMinimalMatrices[tag] = aux1
            return aux1

    def invariants(self, reps, conj=[]):
        """
        Calculates the linear combinations of the components of rep1 x rep2 x ... which are invariant under the action of the group.
        These are also known as the Clebsch-Gordon coefficients.
        The invariants/Clebsch-Gordon coefficients returned by this function follow the following general normalization convention.
        Writing each invariant as Sum_i,j,...c^ij... rep1[i] x rep2[j] x ..., then the normalization convention is  Sum_i,j,...|c_ij...|^2=Sqrt[dim(rep1)dim(rep2)...]. Here, i,j, ... are the components of each representation.
        conj represents wether or not the irrep should be conjugated.
        """
        storedkey = tuple([(tuple(el), el1) for el, el1 in zip(reps, conj)])
        key = tuple([tuple(el) for el in reps])
        if storedkey in self._invariantsStore:
            return self._invariantsStore[storedkey]
        if conj != []:
            assert len(conj) == len(reps), "Length `conj` should match length `reps`!"
            assert np.all([type(el) == bool for el in conj]), "`conj` should only contains boolean!"
        else:
            conj = [False] * len(reps)
        # Let's re-order the irreps
        skey = sorted(key)
        alreadyTaken = []
        order = []
        for el in skey:
            pos = [iel for iel, elem in enumerate(key) if el == elem]
            for ell in pos:
                if not (ell in alreadyTaken):
                    alreadyTaken.append(ell)
                    order.append(ell)
                    break
        alreadyTaken = []
        inverseOrder = []
        for el in key:
            pos = [iel for iel, elem in enumerate(skey) if el == elem]
            for ell in pos:
                if not (ell in alreadyTaken):
                    alreadyTaken.append(ell)
                    inverseOrder.append(ell)
                    break
        # let's reorder the conj accordingly
        conj = [conj[el] for el in order]
        # replacement rules
        subs = [(self._symbdummy[i], self._symblist[j]) for i, j in zip(range(len(self._symblist)), order)]
        if len(reps) == 2:
            cjs = not (conj[0] == conj[1])
            invs, maxinds = self._invariants2Irrep(skey, cjs)
        elif len(reps) == 3:
            if (conj[0] and conj[1] and conj[2]) or (not (conj[0]) and not (conj[1]) and not (conj[2])):
                invs, maxinds = self._invariants3Irrep(skey, False)
            if (conj[0] and conj[1] and not (conj[2])) or (not (conj[0]) and not (conj[1]) and conj[2]):
                invs, maxinds = self._invariants3Irrep(skey, True)
            if (conj[0] and not (conj[1]) and conj[2]) or (not (conj[0]) and conj[1] and not (conj[2])):
                invs, maxinds = self._invariants3Irrep([skey[0], skey[2], skey[1]], True)
                # do the substitutions c->b b->c
                invs = [self._safePermutations(el, ((self.c, self.b), (self.b, self.c))) for el in invs]
                tp = cp.deepcopy(maxinds[0])
                maxinds[0] = cp.deepcopy(maxinds[-1])
                maxinds[-1] = tp
            if (not (conj[0]) and conj[1] and conj[2]) or (conj[0] and not (conj[1]) and not (conj[2])):
                invs, maxinds = self._invariants3Irrep([skey[2], skey[1], skey[0]], True)
                invs = [self._safePermutations(el, ((self.c, self.a), (self.a, self.c))) for el in invs]
                tp = cp.deepcopy(maxinds[0])
                maxinds[0] = cp.deepcopy(maxinds[-1])
                maxinds[-1] = tp
        elif len(reps) == 4:
            invs, maxinds = self._invariants4Irrep([], skey, conj)
        else:
            exit("Error, only 2, 3 or 4 irrep should be passed.")
        # Let's now obtain the tensor expression of the result TODO I am actully not sure this is needed
        # tensor = []
        # for i, el in enumerate(maxinds):
        #    tensor.append([(self._symblist[i][j + 1], j) for j in range(el)])
        # tensorInd = itertools.product(*tensor)
        # tensorExp = np.zeros([1] + maxinds, dtype=object)
        # for iel, el in enumerate(invs):
        #    for elem in tensorInd:
        #        tpmatch = el.match(self.pp * reduce(operator.mul, [xelem[0] for xelem in elem]) + self.q)
        #        if tpmatch != None:
        #            if len(elem) == 2:
        #                tensorExp[iel, elem[0][1], elem[1][1]] = tpmatch[self.pp]
        #            elif len(elem) == 3:
        #                tensorExp[iel, elem[0][1], elem[1][1], elem[2][1]] = tpmatch[self.pp]
        #            elif len(elem) == 4:
        #                tensorExp[iel, elem[0][1], elem[1][1], elem[2][1], elem[3][1]] = tpmatch[self.pp]
        #            else:
        #                exit("Error, cannot determin the tensor form for more than 4 fields contracted together")
        ## TODO normalize the invariants
        # tensorExp = self._normalizeInvariantsTensor([reps[i] if not(cjs[i]) else self.conjugateIrrep(reps[i]) for i in range(len(reps))], tensorExp)
        invs = self._normalizeInvariants(reps, invs)
        # TODO symmetrize the invariants
        invs = self._symmetrizeInvariants(reps, invs, conj)
        # restore the ordering of the representations which was changed above
        subsdummy = [(self._symblist[i], self._symbdummy[i]) for i in range(len(subs))]
        invs = [el.subs(tuple(subsdummy)) for el in invs]
        invs = [el.subs(tuple(subs)) for el in invs]
        self._invariantsStore[key] = invs
        return invs

    def _invariants2Irrep(self, reps, cjs):
        """
        return the invariants of the the irreps
        """
        w1, w2 = self._weights(reps[0]), self._weights(reps[1])
        reps = [np.array([el]) for el in reps if type(el) != np.array]
        # Warning, because the results are stored they need to be copied otherwise modifying one modifies the other one
        r1, r2 = cp.deepcopy(self.repMinimalMatrices(reps[0])), cp.deepcopy(self.repMinimalMatrices(reps[1]))
        if cjs:
            for i in range(len(w2)):
                w2[i][0] = - w2[i][0]
            for i in range(self._n):
                for j in range(3):
                    r2[i][j] = - r2[i][j].transpose()

        array1, array2 = {}, {}
        for i in range(len(w1)):
            array1[self._nptokey(w1[i][0])] = w1[i][1]
        for i in range(len(w2)):
            array2[self._nptokey(w2[i][0])] = w2[i][1]
        aux1 = []
        for i in range(len(w1)):
            if self._indic(array2, self._nptokey(-w1[i][0])) != 0:
                aux1.append([w1[i][0], -w1[i][0]])
        dim1 = [0]
        for i in range(1, len(aux1) + 1):  # WARNING dim1 is aligned with mathematica
            dim1.append(dim1[i - 1] + self._indic(array1, self._nptokey(aux1[i - 1][0])) * self._indic(array2,
                                                                                                       self._nptokey(
                                                                                                           aux1[i - 1][
                                                                                                               1])))
        b1, e1 = {}, {}
        for i in range(len(aux1)):
            key = tuple([self._nptokey(el) for el in aux1[i]])
            b1[key] = dim1[i] + 1
            e1[key] = dim1[i + 1]
        bigMatrix = []
        for i in range(self._n):
            aux2 = []
            keysaux2 = []
            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    val = [aux1[j][0] + self.ncm[i], aux1[j][1]]
                    key = tuple([self._nptokey(el) for el in val])
                    if not (key in keysaux2):
                        aux2.append(val)
                        keysaux2.append(key)
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1] + self.ncm[i]]
                    key = tuple([self._nptokey(el) for el in val])
                    if not (key in keysaux2):
                        aux2.append(val)
                        keysaux2.append(key)

            if len(w1) == 1 and len(w2) == 1:  # Special care is needed if both reps are singlets
                aux2 = aux1
            dim2 = [0]
            for k in range(1, len(aux2) + 1):
                dim2.append(dim2[k - 1] + self._indic(array1, self._nptokey(aux2[k - 1][0])) * self._indic(array2,
                                                                                                           self._nptokey(
                                                                                                               aux2[
                                                                                                                   k - 1][
                                                                                                                   1])))
            b2, e2 = {}, {}
            for k in range(len(aux2)):
                key = tuple([self._nptokey(el) for el in aux2[k]])
                b2[key] = dim2[k] + 1
                e2[key] = dim2[k + 1]
            if dim2[len(aux2)] != 0 and dim1[len(aux1)] != 0:
                matrixE = SparseMatrix(zeros(dim2[len(aux2)], dim1[len(aux1)]))
            else:
                matrixE = []
            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0] + self.ncm[i], aux1[j][1]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])
                    matrixE[self._indic(b2, kaux4) - 1:self._indic(e2, kaux4),
                    self._indic(b1, kaux3) - 1:self._indic(e1, kaux3)] = np.kron(
                        self._blockW(aux1[j][0] + self.ncm[i], aux1[j][0], w1, r1[i][0]),
                        np.eye(self._indic(array2, self._nptokey(aux1[j][1])), dtype=object))
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1] + self.ncm[i]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])
                    matrixE[self._indic(b2, kaux4) - 1:self._indic(e2, kaux4),
                    self._indic(b1, kaux3) - 1:self._indic(e1, kaux3)] = np.kron(
                        np.eye(self._indic(array1, self._nptokey(aux1[j][0])), dtype=object),
                        self._blockW(aux1[j][1] + self.ncm[i], aux1[j][1], w2, r2[i][0]))
            if bigMatrix == [] and matrixE != []:
                bigMatrix = SparseMatrix(matrixE)

            elif bigMatrix != [] and matrixE != []:
                bigMatrix = bigMatrix.col_join(matrixE)
        dim1 = [0]
        dim2 = [0]
        for i in range(1, len(w1) + 1):
            dim1.append(dim1[i - 1] + w1[i - 1][1])
        for i in range(1, len(w2) + 1):
            dim2.append(dim2[i - 1] + w2[i - 1][1])
        for i in range(len(w1)):
            b1[self._nptokey(w1[i][0])] = dim1[i]
        for i in range(len(w2)):
            b2[self._nptokey(w2[i][0])] = dim2[i]
        result = []
        aind, bind = [], []
        if len(bigMatrix) != 0:
            dt = 100 if len(bigMatrix) < 10000 else 500
            aux4 = self._findNullSpace(bigMatrix, dt)
            # let's construct the invariant combination from the null space solution
            # declare the symbols for the output of the invariants
            expr = []
            for i0 in range(len(aux4)):
                expr.append(0)
                cont = 0
                for i in range(len(aux1)):
                    for j1 in range(self._indic(array1, self._nptokey(aux1[i][0]))):
                        for j2 in range(self._indic(array2, self._nptokey(aux1[i][1]))):
                            cont += 1
                            aind.append(1 + b1[self._nptokey(aux1[i][0])] + j1)
                            bind.append(1 + b2[self._nptokey(aux1[i][1])] + j2)
                            expr[i0] += aux4[0][cont - 1] * self.a[1 + b1[self._nptokey(aux1[i][0])] + j1] * self.b[
                                1 + b2[self._nptokey(aux1[i][1])] + j2]
            result = [expr[ii] for ii in range(len(aux4))]
        # Special treatment - This code ensures that well known cases come out in the expected form
        if self.cartan._name == "A" and self.cartan._id == 1 and reps[0] == reps[1] and reps[0] == [1] and not (cjs):
            # Todo check that this is needed here as well
            result = [-el for el in result]
        if aind != [] and bind != []:
            return result, [max(aind), max(bind)]
        else:
            return result, [0, 0]

    def _invariants3Irrep(self, reps, cjs):
        """
        Returns the invariant for three irreps
        """
        w1, w2, w3 = self._weights(reps[0]), self._weights(reps[1]), self._weights(reps[2])
        reps = [np.array([el]) for el in reps if type(el) != np.array]
        # Warning, because the results are stored they need to be copied otherwise modifying one modifies the other one
        r1, r2, r3 = cp.deepcopy(self.repMinimalMatrices(reps[0])), cp.deepcopy(
            self.repMinimalMatrices(reps[1])), cp.deepcopy(self.repMinimalMatrices(reps[2]))
        if cjs:
            for i in range(len(w3)):
                w3[i][0] = - w3[i][0]
            for i in range(self._n):
                for j in range(3):
                    r3[i][j] = - r3[i][j].transpose()

        array1, array2, array3 = {}, {}, {}
        for i in range(len(w1)):
            array1[self._nptokey(w1[i][0])] = w1[i][1]
        for i in range(len(w2)):
            array2[self._nptokey(w2[i][0])] = w2[i][1]
        for i in range(len(w3)):
            array3[self._nptokey(w3[i][0])] = w3[i][1]
        aux1 = []
        for i in range(len(w1)):
            for j in range(len(w2)):
                if self._indic(array3, self._nptokey(-w1[i][0] - w2[j][0])) != 0:
                    aux1.append([w1[i][0], w2[j][0], -w1[i][0] - w2[j][0]])
        dim1 = [0]
        for i in range(1, len(aux1) + 1):  # WARNING dim1 is aligned with mathematica
            dim1.append(dim1[i - 1] + self._indic(array1, self._nptokey(aux1[i - 1][0])) * (
                self._indic(array2, self._nptokey(aux1[i - 1][1])) * self._indic(array3, self._nptokey(aux1[i - 1][2]))
            ))
        b1, e1 = {}, {}
        b3 = {}
        for i in range(len(aux1)):
            key = tuple([self._nptokey(el) for el in aux1[i]])
            b1[key] = dim1[i] + 1
            e1[key] = dim1[i + 1]
        bigMatrix = []
        for i in range(self._n):
            aux2 = []
            keysaux2 = []
            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    val = [aux1[j][0] + self.ncm[i], aux1[j][1], aux1[j][2]]
                    key = tuple([self._nptokey(el) for el in val])
                    if not (key in keysaux2):
                        aux2.append(val)
                        keysaux2.append(key)
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1] + self.ncm[i], aux1[j][2]]
                    key = tuple([self._nptokey(el) for el in val])
                    if not (key in keysaux2):
                        aux2.append(val)
                        keysaux2.append(key)
                if self._indic(array3, self._nptokey(aux1[j][2] + self.ncm[i])) != 0:
                    val = [aux1[j][0], aux1[j][1], aux1[j][2] + self.ncm[i]]
                    key = tuple([self._nptokey(el) for el in val])
                    if not (key in keysaux2):
                        aux2.append(val)
                        keysaux2.append(key)
            if len(w1) == 1 and len(w2) == 1 and len(w3) == 1:  # Special care is needed if both reps are singlets
                aux2 = aux1
            dim2 = [0]
            for k in range(1, len(aux2) + 1):
                dim2.append(dim2[k - 1] + self._indic(array1, self._nptokey(aux2[k - 1][0])) * self._indic(array2,
                                                                                                           self._nptokey(
                                                                                                               aux2[
                                                                                                                   k - 1][
                                                                                                                   1])) * self._indic(
                    array3, self._nptokey(aux2[k - 1][2])))
            b2, e2 = {}, {}
            for k in range(len(aux2)):
                key = tuple([self._nptokey(el) for el in aux2[k]])
                b2[key] = dim2[k] + 1
                e2[key] = dim2[k + 1]
            if dim2[len(aux2)] != 0 and dim1[len(aux1)] != 0:
                matrixE = SparseMatrix(zeros(dim2[len(aux2)], dim1[len(aux1)]))
            else:
                matrixE = []
            for j in range(len(aux1)):
                if self._indic(array1, self._nptokey(aux1[j][0] + self.ncm[i])) != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0] + self.ncm[i], aux1[j][1], aux1[j][2]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])
                    matrixE[self._indic(b2, kaux4) - 1:self._indic(e2, kaux4),
                    self._indic(b1, kaux3) - 1:self._indic(e1, kaux3)] = np.kron(np.kron(
                        self._blockW(aux1[j][0] + self.ncm[i], aux1[j][0], w1, r1[i][0]),
                        np.eye(self._indic(array2, self._nptokey(aux1[j][1])), dtype=object)),
                        np.eye(self._indic(array3, self._nptokey(aux1[j][2])), dtype=object))
                if self._indic(array2, self._nptokey(aux1[j][1] + self.ncm[i])) != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1] + self.ncm[i], aux1[j][2]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])
                    matrixE[self._indic(b2, kaux4) - 1:self._indic(e2, kaux4),
                    self._indic(b1, kaux3) - 1:self._indic(e1, kaux3)] = np.kron(np.kron(
                        np.eye(self._indic(array1, self._nptokey(aux1[j][0])), dtype=object),
                        self._blockW(aux1[j][1] + self.ncm[i], aux1[j][1], w2, r2[i][0])),
                        np.eye(self._indic(array3, self._nptokey(aux1[j][2])), dtype=object))
                if self._indic(array3, self._nptokey(aux1[j][2] + self.ncm[i])) != 0:
                    aux3 = aux1[j]
                    aux4 = [aux1[j][0], aux1[j][1], aux1[j][2] + self.ncm[i]]
                    kaux4 = tuple([self._nptokey(el) for el in aux4])
                    kaux3 = tuple([self._nptokey(el) for el in aux3])
                    matrixE[self._indic(b2, kaux4) - 1:self._indic(e2, kaux4),
                    self._indic(b1, kaux3) - 1:self._indic(e1, kaux3)] = np.kron(np.kron(
                        np.eye(self._indic(array1, self._nptokey(aux1[j][0])), dtype=object),
                        np.eye(self._indic(array2, self._nptokey(aux1[j][1])), dtype=object)),
                        self._blockW(aux1[j][2] + self.ncm[i], aux1[j][2], w3, r3[i][0]))

            if bigMatrix == [] and matrixE != []:
                bigMatrix = SparseMatrix(matrixE)
            elif bigMatrix != [] and matrixE != []:
                bigMatrix = bigMatrix.col_join(matrixE)
        dim1 = [0]
        dim2 = [0]
        dim3 = [0]
        for i in range(1, len(w1) + 1):
            dim1.append(dim1[i - 1] + w1[i - 1][1])
        for i in range(1, len(w2) + 1):
            dim2.append(dim2[i - 1] + w2[i - 1][1])
        for i in range(1, len(w3) + 1):
            dim3.append(dim3[i - 1] + w3[i - 1][1])
        for i in range(len(w1)):
            b1[self._nptokey(w1[i][0])] = dim1[i]
        for i in range(len(w2)):
            b2[self._nptokey(w2[i][0])] = dim2[i]
        for i in range(len(w3)):
            b3[self._nptokey(w3[i][0])] = dim3[i]

        result = []
        if len(bigMatrix) != 0:
            dt = 100 if len(bigMatrix) < 10000 else 500
            aux4 = self._findNullSpace(bigMatrix, dt)
            # let's construct the invariant combination from the null space solution
            # declare the symbols for the output of the invariants
            expr = []
            aind, bind, cind = [], [], []
            for i0 in range(len(aux4)):
                expr.append(0)
                cont = 0
                for i in range(len(aux1)):
                    for j1 in range(self._indic(array1, self._nptokey(aux1[i][0]))):
                        aux2 = 0
                        for j2 in range(self._indic(array2, self._nptokey(aux1[i][1]))):
                            for j3 in range(self._indic(array3, self._nptokey(aux1[i][2]))):
                                cont += 1
                                aind.append(1 + b1[self._nptokey(aux1[i][0])] + j1)
                                bind.append(1 + b2[self._nptokey(aux1[i][1])] + j2)
                                cind.append(1 + b3[self._nptokey(aux1[i][2])] + j3)
                                aux2 += aux4[i0][cont - 1] * self.a[1 + b1[self._nptokey(aux1[i][0])] + j1] * self.b[
                                    1 + b2[self._nptokey(aux1[i][1])] + j2] * self.c[
                                            1 + b3[self._nptokey(aux1[i][2])] + j3]
                        expr[i0] += aux2
            result = [expr[ii] for ii in range(len(aux4))]
        return result, [max(aind), max(bind), max(cind)]

    def _invariants4Irrep(self, otherStuff, reps, cjs):
        """
        Returns the invariants for four irreps
        """
        result = []
        if len(reps) == 3:
            aux1 = self.invariants(reps, cjs)
            subs = tuple([(self.a, self._symblist[len(otherStuff)]),
                          (self.b, self._symblist[1 + len(otherStuff)]),
                          (self.c, self._symblist[2 + len(otherStuff)]),
                          (self.d, self._symblist[3 + len(otherStuff)]),
                          ])
            # do the permutations
            aux1 = [self._safePermutations(el, subs) for el in aux1]
            aux2 = otherStuff[0]
            for i in range(2, len(otherStuff) + 1):
                # TODO this has to be tested
                pudb.set_trace()
                aux2 = sum(otherStuff[i - 1].subs(aux2), [])
            for el in aux1:
                el = self._safePermutations(el, tuple(aux2)).expand()
                result.append(el)
            return result

        trueReps = [tuple(self.conjugateIrrep(el)) if cjs[iel] else el for iel, el in enumerate(reps)]
        # find the irreps in the product of the first two representations
        aux1 = [el[0] for el in self.reduceRepProduct([trueReps[0], trueReps[1]])]
        # conjugate them
        aux1 = [self.conjugateIrrep(el) for el in aux1]
        # do the same for the rest of the irreps
        aux2 = [el[0] for el in self.reduceRepProduct([el for el in trueReps[2:]])]
        # get the intersection and sort it by dimension
        aux1 = sorted([list(elem) for
                       elem in list(set([tuple(el) for el in aux1]).intersection(set([tuple(el) for el in aux2])))],
                      key=lambda x: self.dimR(x))
        for i in range(len(aux1)):
            aux2 = self._irrepInProduct([reps[0], reps[1], aux1[i]], cjs=[cjs[0], cjs[1], False])
            subs = tuple([(self.a, self._symblist[len(otherStuff)]),
                          (self.b, self._symblist[1 + len(otherStuff)]),
                          (self.c, self._symblist[2 + len(otherStuff)]),
                          (self.d, self._symblist[3 + len(otherStuff)]),
                          ])
            aux2 = [[self._safePermutations(ell, subs) for ell in el] for el in aux2]
            aux3 = [self._symblist[len(otherStuff) + 1][el] for el in range(1, self.dimR(aux1[i]) + 1)]
            aux2 = [(el1, el2) for el1, el2 in zip(aux3, sum(aux2, []))]
            # warning otherstuff should not be appended in this scope
            otherStuffcp = cp.deepcopy(otherStuff)
            otherStuffcp.append(aux2)
            tp1 = reps[2:]
            tp1.insert(0, aux1[i])
            # do some conversion type
            tp1 = [list(el) for el in tp1]
            tp2 = cjs[2:]
            tp2.insert(0, True)
            result.append(self._invariants4Irrep(otherStuffcp, tp1, tp2))
        return flatten(result), [self.dimR(el) for el in reps]

    def _irrepInProduct(self, reps, cjs=[]):
        """
        calculate the combination of rep1xrep2 that transform as rep3
        """
        if cjs == []:
            cjs = [False] * len(reps)
        aux = self.invariants(reps, cjs)
        vector = reduce(lambda x, y: x.union(y), [el.find(self.c[self.p]) for el in aux])
        vector = sorted(list(vector), key=lambda x: x.args[1])
        aux = [[el.coeff(ell) for ell in vector] for el in aux]
        return aux

    def _safePermutations(self, exp, permutations):
        """
        Safely performs the permutations given in permutations. In sympy one needs to copy
        all the variable into temporary ones before permuting.
        """
        # different equations in function of the type of transformation e.g b[1] -> f(i[j]), or b->c
        if type(permutations[0][0]) == IndexedBase:
            for iel, (old, new) in enumerate(permutations):
                exp = exp.replace(old[self.p], self._symbdummy[0][iel, self.p])
            for iel, (old, new) in enumerate(permutations):
                exp = exp.replace(self._symbdummy[0][iel, self.p], new[self.p])
        else:
            for iel, (old, new) in enumerate(permutations):
                exp = exp.replace(old, self._symbdummy[0][iel])
            for iel, (old, new) in enumerate(permutations):
                exp = exp.replace(self._symbdummy[0][iel], new)
        return exp

    def _symmetrizeInvariants(self, reps, invs, cjs):
        #  TODO
        return invs

    def _normalizeInvariants(self, representations, invs):
        """
         returns the invariants normalized to Sum |c_ij|^2 = Sqrt(dim(irrep1)dim(irrep2)...dim(irrepn))
        """
        # Sqrt(Prod(dim(irrep[i])),{i,1,n})
        repDims = sqrt(reduce(operator.mul, [self.dimR(el) for el in representations], 1))
        for iel, el in enumerate(invs):
            norm = sum([ell.replace(self.a[self.q], 1).replace(self.b[self.q], 1).replace(self.c[self.q], 1).replace(
                self.d[self.q], 1) ** 2 for ell in el.args])
            invs[iel] = (el / sqrt(norm) * sqrt(repDims)).expand()

        return invs

    def _normalizeInvariantsTensor(self, representations, invariantsTensors):
        """
        normalize the invariants according to TODO
        """

        return invariantsTensors

    def repMatrices(self, maxW):
        """
        This method returns the complete set of matrices that make up a representation, with the correct casimir and trace normalizations
        1) The matrices {M_i} given by this method are in conformity with the usual requirements in particle physics: \!\(
            M_a^\Dagger = M_a ; Tr(M_a M_b = S(rep) \Delta_ab; Sum_a M_a M_a = C(rep) 1.
        """
        # check if its been calculated already
        if type(maxW) == np.array:
            tag = self._nptokey(maxW)
        else:
            tag = tuple(maxW)
        if tag in self._repMatrices:
            return self._repMatrices[tag]
        # Let's gather the minimal rep matrices
        if type(maxW) == list:
            maxW = np.array([maxW])
        rep = self.repMinimalMatrices(maxW)
        dimG = 2 * len(self.proots) + len(self.ncm)
        dimR = self.dimR(maxW)
        sR = Rational(self.casimir(self._tolist(maxW)) * dimR, dimG)
        if dimR == 1:
            #  Trivial representation, the matrices are null
            listTotal = [rep[0][0] for i in range(dimG)]
            return listTotal
        listE, listF, listH = [el[0] for el in rep], [el[1] for el in rep], [el[2] for el in rep]
        # If it's not the trivial rep, generate the matrices of the remaining algebra elements.
        #  The positive roots of the algebra serve as a guide in this process of doing comutators
        for i in range(self._n, len(self.proots)):
            j = 0
            aux = []
            while aux == []:
                aux = [iel for iel, el in enumerate(self.proots[:i]) if np.all(el == self.proots[i] - self.proots[j])]
                if aux == []:
                    j += 1
            listE.append(listE[aux[0]].multiply(listE[j]).add(-listE[j].multiply(listE[aux[0]])))
            listF.append(listF[aux[0]].multiply(listF[j]).add(- listF[j].multiply(listF[aux[0]])))
        for i in range(len(listE)):
            # Change from the operadores T+, T- to Tx,Ty
            aux = listE[i]
            listE[i] = listE[i].add(listF[i])
            listF[i] = aux.add(-listF[i])
            # Control the normalization of the Tx,Ty matrices with the trace condition
            listE[i] = SparseMatrix(listE[i] * sqrt(sR) / sqrt((listE[i].multiply(listE[i])).trace()))
            listF[i] = SparseMatrix(listF[i] * sqrt(sR) / sqrt((listF[i].multiply(listF[i])).trace()))
        matrixCholesky = np.dot(self.ncminv, self._matD)  # See the casimir expression in a book on lie algebras
        aux = (SparseMatrix(matrixCholesky).cholesky()).transpose()  # get the actual cholesky decomposition from sympy
        listH = [reduce(operator.add, [listH[j] * aux[i, j] for j in range(self._n)]) for i in range(self._n)]
        # Up to multiplicative factors, Tz are now correct. We fix again the normalization with the trace condition
        listH = [listH[i] * (sqrt(sR) / sqrt((listH[i].multiply(listH[i])).trace())) for i in range(self._n)]
        listTotal = [listE, listF, listH]
        self._repMatrices[tag] = listTotal
        return listTotal

    def _nptokey(self, array):
        return tuple(array.ravel())

    def _tolist(self, array):
        return list(array.ravel())

    def _indic(self, dict, key):
        if key in dict:
            return dict[key]
        else:
            return 0

    def _blockW(self, w1, w2, listW, repMat):
        """
        aux function to construct the invariants
        """
        dim = [0]
        for i in range(1, len(listW) + 1):
            dim.append(dim[i - 1] + listW[i - 1][1])
        b, e = {}, {}
        for i in range(len(listW)):
            key = self._nptokey(listW[i][0])
            b[key] = dim[i] + 1
            e[key] = dim[i + 1]
        aux1 = repMat[b[self._nptokey(w1)] - 1:e[self._nptokey(w1)], b[self._nptokey(w2)] - 1:e[self._nptokey(w2)]]
        return aux1

    def _findNullSpace(self, matrixIn, dt):
        """
        This is the aux function to determin the invariants.
        """
        # TODO for some reason when we split in several bits res we find two vectors instead of one that need to be summed up...
        sh = matrixIn.shape
        matrixInlist = matrixIn.row_list()
        aux1, gather = [], {}
        for iel, el in enumerate(matrixInlist):
            if el[0] in gather:
                gather[el[0]].append(el)
            else:
                gather[el[0]] = [el]
        aux1 = sorted(gather.values())
        if len(aux1) == 0:
            return eye(len(matrixIn[0]))
        preferredOrder = flatten(
            [[iel for iel, el in enumerate(aux1) if len(el) == i] for i in range(1, max(map(len, aux1)) + 1)])
        matrix = {}
        for iel, el in enumerate(preferredOrder):
            for ell in aux1[el]:
                matrix[(iel, ell[1])] = ell[2]
        matrix = SparseMatrix(iel + 1, sh[1], matrix)  # the number of columns is kept fix
        n, n2 = matrix.shape
        v = IndexedBase('v')
        varnames = [v[i] for i in range(n2)]
        var = Matrix([varnames])
        varSol = Matrix([varnames])
        for i in range(1, n + 1, dt):
            #  To determine the replacement rules we need to create a system of linear equations
            sys = matrix[i - 1:min(i + dt, n), :]
            sys = sys.col_insert(sh[1] + 1, zeros(sys.shape[0], 1))
            res = solve_linear_system(sys, *varSol.tolist()[0])
            #  substitute the solution
            varSol = varSol.subs(res)
        # now we need to extract the vector again
        tally = []
        for el in varSol.tolist()[0]:
            tp = [ell.indices[0] for ell in list(el.find(v[self.p]))]
            tally.append(tp)
        tally = list(set(flatten(tally)))
        res = []
        for el in tally:
            tp = cp.deepcopy(varSol)
            for ell in tally:
                if ell == el:
                    tp = tp.subs(v[ell], 1)
                else:
                    tp = tp.subs(v[ell], 0)
            res.append(tp)
        return res

    def dynkinIndex(self, rep):
        """
        returns the dynkin index of the corresponding representation
        """
        return self.casimir(rep) * Rational(self.dimR(rep), self.dimR(self.adjoint))

    def reduceRepProduct(self, repslist):
        """
        Reduces a direct product of representation to its irreducible parts
        """
        if len(repslist) == 1:
            return [repslist, 1]
        # order the list by dimension
        orderedlist = sorted(repslist, key=lambda x: self.dimR(x))
        n = len(orderedlist)
        result = self._reduceRepProductBase2(orderedlist[n - 2], orderedlist[n - 1])
        for i in range(2, n):
            result = self._reduceRepProductBase1(orderedlist[n - i - 1], result)
        return result

    def _reduceRepProductBase1(self, rep1, listReps):
        res = sum([[(ell[0], el[1] * ell[1]) for ell in self._reduceRepProductBase2(rep1, el[0])] for el in listReps],
                  [])
        final = []
        togather = cp.deepcopy(res)
        while togather != []:
            gathering = togather.pop(0)
            temp = [gathering[1]]
            for iel, el in enumerate(togather):
                if el[0] == gathering[0]:
                    temp.append(el[1])
            togather = [el for el in togather if el[0] != gathering[0]]
            final.append((gathering[0], sum(temp)))
        return final

    def _reduceRepProductBase2(self, w1, w2):
        l1 = self._dominantWeights(w1)
        delta = np.ones(self._n, dtype=int)
        dim = {}
        allIrrep, added = [], []
        for i in range(len(l1)):
            wOrbit = np.array(self._weylOrbit(self._tolist(l1[i][0])))
            for j in range(len(wOrbit)):
                aux = self._dominantConjugate([wOrbit[j] + np.array(w2) + delta])
                if np.all(aux[0] - 1 == abs(aux[0] - 1)):
                    key = self._nptokey(aux[0] - delta)
                    if key in dim:
                        dim[key] += (-1) ** aux[1] * l1[i][1]
                    else:
                        dim[key] = (-1) ** aux[1] * l1[i][1]
                    val = self._tolist(aux[0] - delta)
                    if not (val in allIrrep):
                        allIrrep.append(val)
        result = [(el, self._indic(dim, tuple(el))) for el in allIrrep]
        result = [el for el in result if el[1] != 0]
        return result

    def permutationSymmetryOfInvariants(self, listofreps):
        """
        Computes how many invariant combinations there are in the product of the representations of the gauge group
         provided, together with the information on how these invariants change under a permutation of the representations
         - The output is rather complex (see the examples below). It is made of two lists: {indices, SnRepresentations}.
          The first one (indices) indicates the position of equal representations in the  input list. So indices={G1, G2, \[CenterEllipsis]}
          where each GI lists the positions of a group of equal representations. For example, if the input list is {Subscript[R, 1], Subscript[R, 2],Subscript[R, 1], Subscript[R, 2]} for some representation Subscript[R, 1], Subscript[R, 2] of the gauge group, indices will be {{1,3},{2,4}} (the representations in positions 1 and 3 are the same, as well as the ones in the positions 2 and 4). The second list (SnRepresentations) is itself a list {SnRep1, SnRep2, \[CenterEllipsis]} with the break down of the gauge invariants according to how they change under permutations of equal representations. Specifically, each SnRepI is of the form {{SnRepIG1, SnRepIG2, \[CenterEllipsis]}, multiplicity} where each SnRepIGJ is the irreducible representation of an Subscript[S, n] induced when the same fields in the grouping GJ are permuted. multiplicity indicates how many times such a gauge invariant is contained in the product of the representations of the gauge group provided.
        :param listofreps:
        :return:
        """

        indices, invariants = self._permutationSymmetryOfInvariantsProductParts(listofreps)
        invariants = [el for el in invariants if np.all(np.array(el[0][0])*0 == np.array(el[0][0]))]
        invariants = [[el[0][1],el[1]] for el in invariants]
        return [indices, invariants]

    def _permutationSymmetryOfInvariantsProductParts(self, listofreps):
        """
        This calculates the Plethysms in a tensor product of different fields/representations *)
        """
        listofreps = [[el] for el in listofreps]
        aux1 = self.math.tally(listofreps)
        plesthysmFields = [[i + 1 for i, el in enumerate(listofreps) if el == ell[0]] for ell in aux1]
        aux2 = [self._permutationSymmetryOfInvariantsProductPartsAux(aux1[i][0], aux1[i][1]) for i in range(len(aux1))]
        aux2 = self.math._tuplesWithMultiplicity(aux2)
        aux3 = [[self.reduceRepProduct([el[0] for el in ell[0]][0])]for ell in aux2]
        aux3 = sum([[[[aux3[i][j][0], [el[1] for el in aux2[i][0]]], aux3[i][j][1] * aux2[i][1]]
                for j in range(len(aux3[i]))]
                for i in range(len(aux3))], [])
        aux3 = self.math._tallyWithMultiplicity(aux3)
        return [plesthysmFields, aux3]

    def _permutationSymmetryOfInvariantsProductPartsAux(self, rep, n):
        intPartitionsN = list(self.math._partitionInteger(n))
        # this differs from the original algo because we only consider a single group factor
        aux = self.math._tuples(intPartitionsN, 1)
        snPart = [self.Sn.decomposeSnProduct(el) for el in aux]
        aux = [self._plethysms(rep[j], i[j]) for i in aux for j in range(len(i))]
        aux = [self.math._tuplesWithMultiplicity([el]) for el in aux]
        aux = [[[[[aux[i][j][0], intPartitionsN[k]], aux[i][j][1] * snPart[i][k]]
                 for k in range(len(intPartitionsN))]
                for j in range(len(aux[i]))]
               for i in range(len(aux))]
        aux = [el for el in sum(sum(aux, []), []) if not (el[-1] == 0)]
        result = self.math._tallyWithMultiplicity(aux)
        return result

    def _plethysms(self, weight, partition):
        n = sum(partition)
        kList = list(self.math._partitionInteger(n))
        summing = []
        for i in range(len(kList)):
            factor = 1 / factorial(n) * self.Sn.snClassOrder(kList[i]) * self.Sn.snClassCharacter(partition, kList[i])
            aux = [self._adams(el, weight) for el in kList[i]]
            aux = self._reduceRepPolyProduct(aux)
            aux = [(el[0], factor * el[1]) for el in aux]
            summing.append(aux)
        summing = self._gatherWeightsSingle(summing)
        return summing

    def _reduceRepPolyProduct(self, polylist):
        """
        (* This method calculates the decompositions of a product of sums of irreps: (R11+R12+R13+...) x (R21+R22+R23+...) x ... *)
        (* polyList = list of lists of representations to be multiplied. The method outputs the decomposition of such a product *)
        """
        n = len(polylist)
        aux = polylist[0]
        if n <= 1:
            return aux
        for i in range(n - 1):
            aux = list(self.math._tuplesList([aux, polylist[i + 1]]))
            aux2 = [self.reduceRepProduct([ell[0] for ell in el[0:2]]) for el in aux]
            aux = self._gatherWeights(aux2, [el[0][1] * el[1][1] for el in aux])
        return aux

    def _gatherWeightsSingle(self, llist):
        aux = sum(llist, [])
        aux = self.math._gatherAux(aux)
        aux = [[el[0][0], sum([ell[1] for ell in el])] for el in aux]
        aux = [el for el in aux if el[1] != 0]
        return aux

    def _gatherWeights(self, listW, listMult):
        aux = [[[el[0], listMult[i] * el[1]] for el in listW[i]] for i in range(len(listW))]
        aux = sum(aux, [])
        aux = self.math._gatherAux(aux)
        aux = [[el[0][0], sum([ell[1] for ell in el[0:]])] for el in aux]
        aux = [el for el in aux if el[1] != 0]
        return aux

    def _adams(self, n, rep):
        aux = self._dominantWeights(rep)
        aux = [((el[0] * n).tolist()[0], el[1]) for el in aux]
        result = [[self._vdecomp(aux[i][0]), aux[i][1]] for i in range(len(aux))]
        result = [[result[i][0][j][0], result[i][0][j][1] * result[i][1]] for j in range(len(result[i][0])) for i in
                  range(len(result))]
        return result

    def _vdecomp(self, dominantWeight):
        # return self._altDom1Arg([el.tolist()[0] for el in self._weylOrbit(dominantWeight)])
        return self._altDom1Arg(self._weylOrbit(dominantWeight))

    def _altDom1Arg(self, weights):
        return self._altDom(weights, self.longestWeylWord)

    def _altDom(self, weights, weylWord):
        prov = [[el, 1] for el in weights]
        for i in range(len(weylWord)):
            for j in range(len(prov)):
                if prov[j][1] != 0:
                    if prov[j][0][weylWord[i] - 1] >= 0:
                        pass
                    elif prov[j][0][weylWord[i] - 1] == -1:
                        prov[j][1] = 0
                    elif prov[j][0][weylWord[i] - 1] <= -2:
                        prov[j][1] = - prov[j][1]
                        prov[j][0] = [prov[j][0][0] - (prov[j][0][weylWord[i] - 1] + 1) * self.cm[weylWord[i] - 1]]
        prov = [el for el in prov if not (el[1] == 0)]
        return prov


class Sn:
    def __init__(self):
        # declare a MathGroup object to access the standard method
        self.math = MathGroup()

    def snIrrepGenerators(self, Lambda, orthogonalize=True):
        """
        returns the matrices that generate Sn
        - representation must be a partition of some integer n, as irreducible representations of  Subscript[S, n] are specified in this way;
        - Note with the (12) and (12...n) elements of the Subscript[S, n] group alone, it is possible to generate all remaining group elements, for any n;
        - This function returns two real orthogonal/unitary matrices which are the representation matrices of the elements (12) and (12...n) elements of the Subscript[S, n] group. If orthogonality is not required, the option OrthogonalizeGenerators->False can be used \[LongDash] the resulting matrices have less complicated values, and the code is executed faster.
        """
        n = sum(Lambda)
        sts = self.generateStandardTableaux(Lambda)
        basicPermutations = [Permutation(1, 2), Permutation(*range(1, n + 1))]
        # because the length of the lists on which we apply the permutations is not constant we need to resize them for each element
        tabloids, stsX = [], []
        for perm in basicPermutations:
            stscp = cp.deepcopy(sts)
            sts_nosort = cp.deepcopy(sts)
            for iel, el in enumerate(stscp):
                for iell, ell in enumerate(el):
                    for ixel, xel in enumerate(ell):
                        if xel in perm.args[0]:
                            stscp[iel][iell][ixel] = perm(xel)
                            sts_nosort[iel][iell][ixel] = perm(xel)
                    stscp[iel][iell] = sorted(stscp[iel][iell])
            # TODO IMPLEMENT THE DELETE DUPLICATES
            tabloids.append(stscp)
            stsX.append(sts_nosort)
        X, Y = np.zeros((2, len(sts), len(tabloids[0])), dtype=int), np.zeros((2, len(sts), len(tabloids[0])),
                                                                              dtype=int)
        for alpha in range(2):
            for i in range(len(sts)):
                for j in range(len(tabloids[alpha])):
                    startingTableauxY = sts[i]
                    startingTableauxX = stsX[alpha][i]
                    targetTabloid = tabloids[alpha][j]
                    tmp = [[self.math._position_in_array(targetTabloid, ell)[0][0] for ell in el] for el in
                           self._transposeTableaux(startingTableauxY)]
                    Y[alpha][i][j] = reduce(operator.mul,
                                            [0 if sorted(el) != range(len(el)) else Permutation(el).signature() for el
                                             in tmp])
                    tmp = [[self.math._position_in_array(targetTabloid, ell)[0][0] for ell in el] for el in
                           self._transposeTableaux(startingTableauxX)]
                    X[alpha][i][j] = reduce(operator.mul,
                                            [0 if sorted(el) != range(len(el)) else Permutation(el).signature() for el
                                             in tmp])
        result = [(SparseMatrix(X[i]) * SparseMatrix(Y[i]).inv()).transpose() for i in range(2)]
        # Finally let's orthogonalize the generators P_i
        # Oi = B.Pi.Inverse[B], Oi are ortho and B the change of basis
        # since Pi are real Pi^T.(B^T.B).Pi = B^T.B
        # If both Pi are taken into consideration, this fixes completely B^T.B as the Pi are generators of the group in
        # an irreducible representation
        # With KroneckerProduct and NullSpace, B^T.B can be found, and B can be obtained with the CholeskyTypeDecomposition
        if orthogonalize:
            Id = eye((result[0].shape[0]) ** 2)
            aux = [np.transpose(np.kron(np.conjugate(el), el)) for el in result]
            ns = Matrix(np.concatenate((aux[0] - Id, aux[1] - Id), axis=0)).nullspace()
            if ns != []:
                ns = ns[0]
            else:
                exit("Impossible to find null space in SnGenerator.")
            BcB = self.math._inverseFlatten(ns, [result[0].shape[0], result[0].shape[0]])
            B = SparseMatrix(self.math._decompositionTypeCholesky(np.array(BcB))).transpose()
            result = [B * el * B.inv() for el in result]
        return result

    def decomposeSnProduct(self, partitionsList):
        """
        This method decomposes the product of a list of Sn rep into its irreducible parts
        """
        n = sum(partitionsList[0])
        result = [1 / factorial(n) * sum([
                                             self.snClassOrder(i) * reduce(operator.mul, [
                                                 self.snClassCharacter(inputPartition, list(i)) for inputPartition in
                                                 partitionsList])
                                             * self.snClassCharacter(list(j), list(i)) for i in
                                             list(self.math._partitionInteger(n))]) for j in
                  list(self.math._partitionInteger(n))]
        return result

    def snClassOrder(self, partition):
        """
        size of a given conjugacy class of Sn. The formula is easy but see for example
         Enumerative Combinatorics", Richard P.Stanley, http://math.mit.edu/~rstan/ec/ec1.pdf, 1.3.2 Proposition"
        """
        n = sum(partition)
        aux = self.math.tally(partition)
        return factorial(n) / (
            reduce(operator.mul, [aux[i][0] ** aux[i][1] * factorial(aux[i][1]) for i in range(len(aux))]))

    def snClassCharacter(self, partitionL, partitionM):
        """
        (* See arXiv:math/0309225v1[math.CO] for the way to compute SnClassCharacter from the Murnaghan-Nakayama rule  *)
(* \[Lambda] is the representation; \[Mu] is the conjugacy class. This method computes the character of conjugacy class \mu in the irreducible representation \[Lambda]  *
        """
        if len(partitionL) == 0:
            return 1
        n = sum(partitionL)
        if n != sum(partitionM):
            exit("Error in SnClassCharacter method: both partitions must be of the same order.")
            return
        newL = self.rimHooks(partitionL, partitionM[0])
        newM = partitionM[1:]
        result = sum([(-1) ** newL[i][1] * self.snClassCharacter(newL[i][0], newM) for i in range(len(newL))])
        return result

    def rimHooks(self, partition, l):
        """
        (* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
        (* This method finds all the rim hooks \[Xi] with length l and returns a list with all the possibilities {partition\\[Xi], leg length of rim hook \[Xi]} which is writen as {partition\\[Xi],ll(\[Xi])}*)
        """
        sequence = self._partitionSequence(partition)
        result = []
        for i in range(len(sequence) - l):
            if sequence[i] == 1 and sequence[i + l] == 0:
                seqMinusHook = cp.deepcopy(sequence)
                seqMinusHook[i] = 0
                seqMinusHook[i + l] = 1
                length = sequence[i:i + l + 1].count(0) - 1
                result.append((self._rebuildPartitionFromSequence(seqMinusHook), length))
        return result

    def checkStandardTableaux(self, tab):
        """
        Returns True if tab is a standard tableau i.e. it grows on each line and each columns
        """
        transpose = self._transposeTableaux(tab)
        return all([self._issorted(el) for el in tab] + [self._issorted(el) for el in transpose])

    def _transposeTableaux(self, tab):
        """
        Transpose a tableaux
        """
        tabcp = cp.deepcopy(tab)
        for iel, el in enumerate(tabcp):
            tabcp[iel] = el + [None] * (len(tabcp[0]) - len(el))
        tabcp = np.array(tabcp).T.tolist()
        for iel, el in enumerate(tabcp):
            tabcp[iel] = [ell for ell in el if ell is not None]
        return tabcp

    def generateStandardTableaux(self, Lambda):
        """
        Generates all the standard tableaux given by the partition LAmbda
        """
        result = self._generateStandardTableauxAux([[None] * el for el in Lambda])
        return result

    def _generateStandardTableauxAux(self, tab):
        """
        Aux function for the recursion algo
        """
        if not (self.checkStandardTableaux(tab)):
            return []
        # stop criterion for the recursion
        # flatten tab
        flttab = sum(tab, [])
        # stopping creterion for the recursion
        if not (None in flttab):
            return [tab]
        n = len(flttab)
        # flatten removes Nones
        temp = [el for el in flttab if el is not None]
        missingNumbers = [el for el in range(1, n + 1) if not (el in temp)]
        stop = False
        for idi, i in enumerate(tab):
            if stop:
                idi -= 1
                break
            for idj, j in enumerate(i):
                if j == None:
                    stop = True
                    break
        if stop:
            positionNone = [idi, idj]
        else:
            positionNone = []
        result = []
        for el in missingNumbers:
            newT = cp.deepcopy(tab)
            newT[positionNone[0]][positionNone[1]] = el
            tp = self._generateStandardTableauxAux(newT)
            result += tp
        return result

    def hookContentFormula(self, partition, nMax):
        """
        1) Applies the Hook Content Formula to a semi-standard Young tableau with cells filled with the numbers 0, ...,n (repetitions are allowed) - see reference below
        2) Recall that a partition {Lambda_1, Lambda_2, ...} is associated with a Young tableau where row i contains Lambda_i cells - for example the partition {4,3,1,1} of 9 yields the tableau
        3) In a semi-standard Young tableau, the x_i which fill it must increase from top to bottom and must not decrease from left to right.
        4) The number of semi-standard Young tableau given by the partition \[Lambda], where the cell can have any positive integer value smaller or equal to n is given by hookContentFormula(Lambda, n).
        5)The application in model building of this is the following: consider a parameter M_f1f2, ... where the f_i =1,...,n are flavor indices. If Mu is known to have some symmetry (given by a partition Lambda) under a permutation of these indices, then the number of degrees of freedom in Mu is given by  hookContentFormula(Lambda_n) (see examples below).
        """
        n1 = partition[0]
        n2 = len(partition)
        inverseP = [len([ell for ell in partition if ell >= el]) for el in range(1, n1 + 1)]
        if type(nMax) != Symbol:
            aux = [[Rational((nMax + i - j), partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1)
                    if partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1 > 0 else 1 for j in range(1, n2 + 1)]
                   for
                   i in range(1, n1 + 1)]
        else:
            aux = [[(nMax + i - j) / (partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1)
                    if partition[j - 1] + inverseP[i - 1] - (j - 1) - (i - 1) - 1 > 0 else 1 for j in range(1, n2 + 1)]
                   for
                   i in range(1, n1 + 1)]
        result = reduce(operator.mul, flatten(aux))
        return result

    def _partitionSequence(self, partition):
        sequence = [1] * partition[-1]
        sequence.append(0)
        for i in range(1, len(partition)):
            sequence = sequence + [1] * (partition[-i - 1] - partition[-i])
            sequence.append(0)
        return sequence

    def _rebuildPartitionFromSequence(self, sequence):
        """
        (* See arXiv:math/0309225v1[math.CO] - this is an auxiliar method to calculate SnClassCharacter *)
        (* RebuiltPartitionFromSequence[PartitionSequence[partition]]=partition *)
        """
        counter1s = 0
        result = []
        for i in range(len(sequence)):
            if sequence[i] == 0:
                result.insert(0, counter1s)
            else:
                counter1s += 1
        return [el for el in result if el != 0]


class MathGroup:
    def __init__(self):
        pass

    def decompositionTypeCholesky(self, matrix):
        """
        falls back to the regular Cholesky for sym matrices
        """
        n = len(matrix)
        shape = matrix.shape
        matrix = np.array([int(el) if int(el) == el else el for el in matrix.ravel()], dtype=object).reshape(shape)
        matD = np.zeros((n, n), dtype=object)
        matL = np.eye(n, dtype=object)
        for i in range(n):
            for j in range(i):
                if matD[j, j] != 0:
                    if type(matD[j, j]) in [Add, Mul]:
                        coeff = 1 / matD[j, j]
                    else:
                        coeff = Rational(1, matD[j, j])
                    matL[i, j] = coeff * (
                        matrix[i, j] - sum([matL[i, k] * np.conjugate(matL[j, k]) * matD[k, k]
                                            for k in range(j)])
                    )
                else:
                    matL[i, j] = 0
            matD[i, i] = matrix[i, i] - sum([matL[i, k] * np.conjugate(matL[i, k]) * matD[k, k] for k in range(i)])
        # get the sqrt of the diagonal matrix:
        if np.all(matD.transpose() != matD):
            exit("Error, the matD is not diagonal cannot take the sqrt.")
        else:
            matDsqr = diag(*[sqrt(el) for el in matD.diagonal()])
            result = (matL * matDsqr).transpose()
            #  Make the resulting matrix as small as possible by eliminating null columns
            result = np.array(
                [np.array(result.row(i))[0] for i in range(result.rows) if result.row(i) != zeros(1, n)]).transpose()
        return result

    def _partition(self, llist, llen):
        # partition llist into sublist of length len
        res = []
        llistcp = cp.deepcopy(llist)
        while len(llistcp) >= llen:
            print(llistcp)
            res.append(llistcp[:llen])
            llistcp = llistcp[llen:]
        print(res, llist, llen)
        return res

    def _inverseFlatten(self, flattenedList, dims):
        lbd = lambda x, y: self._partition(x, y)
        return reduce(lbd, [flattenedList] + dims[::-1][:-1])

    def _position_in_array(self, target, elem):
        # returns the positions (x,y) of elem in target assume only one occurence
        # e.g. {{1,2,3},{3,4}} -> position_in_array(1) -> (0,0)
        pos = []
        for iel, el in enumerate(target):
            if elem in el:
                pos.append([iel, el.index(elem)])
                break
        return pos

    def _rotateleft(self, llist, n):
        return llist[n:] + llist[:n]

    def _issorted(self, llist):
        # returns wether a list is sorted
        return all([llist[i] <= llist[i + 1] or llist[i + 1] is None for i in xrange(len(llist) - 1)])

    def _tuples(self, llist, n):
        """
        returns all the possible tuples of length n from elementes of llist
        """
        return sorted(list(set(
            sum([list(itertools.permutations(el)) for el in itertools.combinations_with_replacement(llist, n)], []))))

    def _tuplesList(self, llist):
        return itertools.product(*llist)

    def _tuplesWithMultiplicity(self, listoflists):
        aux1 = list(self._tuplesList(listoflists))
        aux2 =  [reduce(operator.mul,[ell[1] for ell in el]) for el in aux1]
        aux1 = [[ell[0] for ell in el] for el in aux1]
        res = zip(aux1, aux2)
        return res

    def _yieldParts(self, num, lt):
        if not num:
            yield ()
        for i in range(min(num, lt), 0, -1):
            for parts in self._yieldParts(num - i, i):
                yield (i,) + parts

    def _partitionInteger(self, num):
        # returns all the partition of num
        for part in self._yieldParts(num, num):
            yield part

    def tally(self, llist):
        tally, mul = [], []
        for el in llist:
            if not (el in tally):
                mul.append(llist.count(el))
                tally.append(el)
        return zip(tally, mul)

    def _tallyWithMultiplicity(self, listoflists):
        aux1 = self._gatherAux(listoflists)
        aux2 = [sum([ell[1] for ell in el]) for el in aux1]
        aux1 = [el[0][0] for el in aux1]
        result = zip(aux1, aux2)
        return result

    def _gatherAux(self, llist):
        gather = []
        gathered = []
        for el in llist:
            if el[0] in gathered:
                iel = gathered.index(el[0])
                gather[iel].append(el)
            else:
                gather.append([el])
                gathered.append(el[0])
        return gather
