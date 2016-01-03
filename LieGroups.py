"""
This is some functions taken from Susyno for a far far away future.

"""
import sys
sys.path.insert(0,'/Applications/HEPtools/sympy-0.7.6')
import numpy as np
from sympy import *
import copy as cp
import pudb


def ALiefillupfunction(i,j):

    if i==j : 
        return 2
    elif i==j+1 or j==i+1:
        return -1
    else :
        return 0
    
def BLiefillupfunction(i,j):
   return ALiefillupfunction(i,j)

def CLiefillupfunction(i,j):
    return ALiefillupfunction(i,j)

def DLiefillupfunction(i,j) : 
    return ALiefillupfunction(i,j)

def CartanMatrix(name,Id):

    Translation = {"SU": "A", "SP": "C", "SO": ("B","D")}
    ClassicalLieAlgebras = ["A","B","C","D"]
    
    if name =="U" and Id ==1 : 
        cartan = SparseMatrix(0,0,[0])

    if not(name in ClassicalLieAlgebras) :
        if name in Translation:
            if name == "SU" :
              return CartanMatrix(Translation[name],Id-1)
            elif name == "SP":
                if Id%2==0 :
                    return CartanMatrix(Translation[name],Id/2) 
                else :
                    print("error 'SP' Id number must be even")
                    return
            elif name == "SO" and Id%2==0 :
                if Id < 5:
                    print("Error n >=3 or > 4 for SO(n)")
                    return
                return CartanMatrix(Translation[name][1],Id/2)
            elif name == "SO" and Id%2==1 :
                return CartanMatrix(Translation[name][0],(Id-1)/2)
            else :
                print"Error unknown Lie Algebra, try 'A', 'B','C' or 'D'"
                return

    if name in ["A","B","C"] and Id ==1 : 
        cartan = SparseMatrix([2])
    if name == "A" and Id > 1  :
        cartan = SparseMatrix(Id,Id,lambda i,j : ALiefillupfunction(i,j)) 
    elif name == "B" and Id > 1 :
        cartan = SparseMatrix(Id,Id,lambda i,j : BLiefillupfunction(i,j)) 
        cartan[Id-2,Id-1] = -2
    elif name == "C" and Id > 1 :
        cartan = SparseMatrix(Id,Id,lambda i,j : CLiefillupfunction(i,j)) 
        cartan[Id-1,Id-2] = -2
    elif name == "D" and Id > 1 :
        cartan = SparseMatrix(Id,Id,lambda i,j : DLiefillupfunction(i,j)) 
        cartan[Id-1,Id-2] = 0
        cartan[Id-2,Id-1] = 0
        cartan[Id-1,Id-3] = -1
        cartan[Id-3,Id-1] = -1
    return cartan


"""
All the functions take a cartan matrix as input as it is the fundamental representation of the Lie Algebras
"""

def PositieRoots(group):

    """
    Returns the positive roots of a given group
    """
    #number of roots (The cartan are squarred matrices
    n = group.rows
    aux1 = [[KroneckerDelta(i,j) for j in range(1,n+1)] for i in range(1,n+1)]
    count = 0
    weights = cp.copy(group)

    while count < weights.rows :
        count+=1
        aux2 = cp.copy(aux1[count-1])
        for inti in range(1,n+1):
            aux3 = cp.copy(aux2)
            aux3[inti-1] += 1
            if (FindM(aux1,aux2,inti) - weights[count-1,inti-1] > 0 and aux1.count(aux3)==0):
                weights = weights.col_join(weights.row(count-1)+group.row(inti-1))
                aux1.append(aux3)
    return matrix2numpy(weights,dtype=int)



def FindM(ex,el,indice):
    aux1 = cp.copy(el[indice-1])
    aux2 = cp.copy(el)
    aux2[indice-1]=0
    auxMax=0
    for ii in range(1,aux1+2):
        if ex.count(aux2) == 1 :
            auxMax=aux1-ii+1
            return auxMax
        aux2[indice-1]=cp.copy(aux2[indice-1]+1)
    return auxMax 


def ReflectWeight(group,weight,i):
    """
    Reflects a given weight. WARNING The index i is from 1 to n
    """
    result = cp.deepcopy(weight)
    result[i-1] = -weight[i-1]
    mD = SpecialMatrixD(group)
    for ii in range(1,5):
        if mD[i-1,ii-1] != 0 :
            result[mD[i-1,ii-1]-1]+= weight[i-1]
    return result


def SpecialMatrixD(group):

    n = group.rows
    result = SparseMatrix(n,4,0)
    for i in range(1,n+1):
        k=1
        for j in range(1,n+1):
            if group[i-1,j-1] == -1:
                result[i-1,k-1] = j 
                k+=1
            if group[i-1,j-1] == -2:
                result[i-1,k-1] = j
                result[i-1,k-1+1] = j
                k +=2
            if group[i-1,j-1] == -3:
                result[i-1,k-1] = j
                result[i-1,k-1+1]=j
                result[i-1,k-1+2]=j
                k+=3
    return result


def WeylOrbit(group,weight):
    """
    Creates the weyl orbit i.e. the system of simple root
    """
    n = group.rows
    counter = 0
    result,wL = [],[]
    wL.append([weight])
    result.append(weight)
    while len(wL[counter])!=0 :
        counter += 1
        wL.append([])
        for j in range(1,len(wL[counter-1])+1):
            for i in range(1,n+1):
                    if wL[counter-1][j-1][i-1] > 0 :
                        aux = ReflectWeight(group,wL[counter-1][j-1],i)[i+1-1:n+1]
                        if aux == map(abs,aux):
                            wL[counter].append(ReflectWeight(group,wL[counter-1][j-1],i))
        result = result + wL[counter]#Join the list
    return result



def DominantWeights(group, weight):
    """
    Generate the dominant weights without dimentionality information
    """
    #convert the weight 
    weight = np.array([weight])
    # cal the inverse
    cminv = np.array(group.inv())
    # get teh positive roots
    proots = PositieRoots(group)
    listw = []
    listw.append(weight)
    counter =1
    while counter <=len(listw):
        aux = [listw[counter-1]-proots[i] for i in range(len(proots))]
        aux = [el for el in aux if np.all(el==abs(el))]
        listw = listw + aux
        # remoev duplicates this is actually a pain since numpy are not hashable
        listw = {array.tostring(): array for array in listw}.values()
        counter += 1
    # need to sort listw
    def SortList(a,b) :
        tp1 = list(np.dot(-(a-b),cminv)[0])
        return cmp(tp1,[0]*a.shape[1])
    #The Sorting looks to be identical to what was done in SUSYNO willl require further checking at some point
    listw.sort(SortList)
    #listw = [np.array([[1,1]]),np.array([[0,0]])]
    n = group.shape[0]
    matD = MatrixD(group)
    cmID = np.dot(cminv,matD)
    # Sum the positive roots
    deltaTimes2 = proots.sum(axis=0)
    functionaux = {}
    functionaux[tuple(listw[0].ravel())] = 1
    result = [[listw[0],1]]
    for j in range(2,len(listw)+1):
        for i in range(1,len(proots)+1):
            k=1
            aux1 = functionaux[tuple(DominantConjugate(group,k*proots[i-1]+listw[j-1])[0])]
            key = tuple(listw[j-1].ravel())
            while aux1 != 0 :
               aux2 = k * (proots[i-1] + listw[j-1])
               if key in functionaux:
                   functionaux[key] += 2 * aux1 * SimpleProduct(aux2, [proots[i-1]], cmID)
               else :
                   functionaux[key] = 2 * aux1 * SimpleProduct(aux2, [proots[i-1]], cmID)
               k+=1
               #update aux1 value
               kkey = tuple(DominantConjugate(group,k*proots[i-1]+listw[j-1])[0])
               if kkey in functionaux:
                   aux1 = functionaux[kkey]
               else :
                   aux1 = 0
        functionaux[key] = functionaux[key] / SimpleProduct(listw[0]+listw[j-1]+deltaTimes2,listw[0]-listw[j-1],cmID)
        result.append([listw[j-1],functionaux[tuple(listw[j-1].ravel())]])
    return result
                
    
def LongestWeylWord(group):
    """ from the Lie Manual see Susyno"""
    n = group.shape[0] 
    weight = [-1]*n
    result = []
    while map(abs,weight) != weight:
        for iel,el in enumerate(weight):
            if el<0:
                break
        weight = ReflectWeight(group,weight,iel+1)
        result.insert(0,iel+1)
    return result




def DominantConjugate(group, weight):
    weight = weight[0]
    if group == np.array([[2]]):#SU2 code
        if weight[0] < 0 :
            return [-weight,1]
        else :
            return [weight,0]
    else :#else
        index = 0
        dWeight = weight
        i=1
        mD = SpecialMatrixD(group)
        while i<=group.shape[0]:
            if (dWeight[i-1] < 0):
                index+=1
                dWeight = ReflectWeight(group,dWeight,i)
                i=min([mD[i-1,0],i+1])
            else :
                i+=1
        return [dWeight,index]



def SimpleProduct (v1, v2, cmID):
    return (1/2. * Matrix(v1)*cmID*Matrix(v2).transpose())[0,0]
    

def MatrixD(group):
    """
    Returns a diagonal matrix with the values <root i, root i> 
    """
    group = matrix2numpy(group)
    positions = sum([[(irow,icol) for icol,col in enumerate(row) if (col in [-1,-2,-3]) and (irow < icol) ]for irow,row in enumerate(group)],[])
    result = np.ones((1,len(group)))[0]
    for coord1,coord2 in positions:
        result[coord2] = group[coord2,coord1]/group[coord1,coord2]*result[coord1]
    return np.diagflat(result)

def Casimir(group,irrep):
    n = group.shape[0]
    proots = PositieRoots(group)
