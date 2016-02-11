from PyLie import *
import pudb
import time

a = CartanMatrix("SU", 2)
b = CartanMatrix("SU", 3)
c = CartanMatrix("SU", 9)
ALie = LieAlgebra(a)
BLie = LieAlgebra(b)
SU9 = LieAlgebra(c)
d = CartanMatrix("SU", 5)
SU5 = LieAlgebra(d)
SU4 = LieAlgebra(CartanMatrix("SU", 4))
SP12 = LieAlgebra(CartanMatrix("SP", 12))

# print("***** Check WeylOrbit function *****")
# print("aold [0]")
# print(lie.WeylOrbit(aold, [0]))
#
# print("a [0]")
# print(ALie._weylOrbit([0]))
##
## print("bold [1,0]")
## print(lie.WeylOrbit(bold,[1,0]))
##
## print("b [1,0]")
## print(BLie._weylOrbit([1,0]))
##
## print("bold [1,1]")
## print(lie.WeylOrbit(bold,[1,1]))
##
## print("b [1,1]")
## print(BLie._weylOrbit([1,1]))
##
## print("***** Check DominantWeights function *****")
## print("bold [1,0]")
###lie.DominantWeights(bold,[1,0])
## print("b [1,0]")
## BLie._dominantWeights([1,0])
## print("bold [1,1]")
## lie.DominantWeights(bold,[1,1])
## print("b [1,1]")
## BLie._dominantWeights([1,1])
##
##
## print("***** Check Casimir *****")
## print("SU2:")
## SU2casimir = [ALie.casimir([n]) for n in range(1,101)]
## print(SU2casimir)
## print("SU3:")
## SU3irreps = [[0, 0], [1, 0], [0, 1], [0, 2], [2, 0], [1, 1], [3, 0], [0, 3], [2,1], [1, 2], [4, 0], [0, 4], [0, 5], [5, 0], [1, 3], [3, 1], [2, 2], \
## [6, 0], [0, 6], [4, 1], [1, 4], [7, 0], [0, 7], [3, 2], [2, 3], [0,
## 8], [8, 0], [5, 1], [1, 5], [9, 0], [0, 9], [2, 4], [4, 2], [1, 6],
## [6, 1], [3, 3], [10, 0], [0, 10], [0, 11], [11, 0], [7, 1], [1, 7],
## [5, 2], [2, 5], [4, 3], [3, 4], [12, 0], [0, 12], [8, 1], [1, 8]]
## SU3casimir = [BLie.casimir(el) for el in SU3irreps]
## print(SU3casimir)
## print("SU9:")
## SU9irreps = [[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [2, 0, \
## 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]]
##
## SU9casimir = [SU9.casimir(el) for el in SU9irreps]
## print(SU9casimir)
##
## print("***** Check DimR *****")
## print("SU2:")
## SU2dimr = [ALie.dimR([n]) for n in range(1,101)]
## print(SU2dimr)
## print("SU3: dim irreps:")
## SU3dimr = [BLie.dimR(el) for el in SU3irreps]
## print(SU3dimr)
###print("SU9:")
###SU9dimr = [SU9.dimR(el) for el in SU9irreps]
###print(SU9dimr)
##
## print("representation index functio: ")
## print("SU3")
## irrepstocheckSU3 = [[1,n] for n in range(12)]
## repindex = [BLie._representationIndex(np.array([el])) for el in irrepstocheckSU3]
## for rep,index in zip(irrepstocheckSU3,repindex):
##    print(index," ", rep)
##
## print("RepsUpToDimN:")
## print("SU2")
## print(ALie.repsUpToDimN(5))
## print("SU3")
## print(BLie.repsUpToDimN(10))
## print("SU5")
## print(SU5.repsUpToDimN(10))
## print("dims:",[SU5.dimR(el) for el in SU5.repsUpToDimN(10)])
## print("Group with Rank N^2:",8)
## print(SU5._getGroupWithRankNsqr(8))
## print(SU5._cmToFamilyAndSeries())
## print("Conjugacy class")
## print("SU5")
## SU5irreps = SU5.repsUpToDimN(10)
## print(SU5irreps)
## print([SU5._conjugacyClass(el)for el in SU5irreps])
##
## print("SO10")
## e = CartanMatrix("SO",10)
## SO10 = LieAlgebra(e)
## SO10irreps = SO10.repsUpToDimN(100)
## print(SO10irreps)
## print([SO10._conjugacyClass(el) for el in SO10irreps[2:]])
##
## print("SO11")
## SO11 = LieAlgebra(CartanMatrix("SO", 11))
## SO11irreps = SO11.repsUpToDimN(100)
## print(SO11irreps)
## print([SO11._conjugacyClass(el) for el in SO11irreps])
#
## print("SP10")
## SP10 = LieAlgebra(CartanMatrix("SP", 10))
## SP10irreps = SP10.repsUpToDimN(10)
## print(SP10irreps)
## print([SP10._conjugacyClass(el) for el in SP10irreps])
##
##
## print("SP12")
## SP12irreps = SP12.repsUpToDimN(100)
## print(SP12irreps)
## print([SP12._conjugacyClass(el) for el in SP12irreps])
##
## print("Conjugated irrep")
## print("SU5")
## print(SU5.conjugateIrrep([1,0,0,0]))
## print(SU5.conjugateIrrep([1,1,0,1]))
#
#
# print("Weights")
# print("SU2")
## print(ALie._weights(np.array([[1]])))
# print("SU3")
# print(BLie._weights(np.array([0, 1])))
# print(BLie._weights(np.array([1, 1])))
# print(BLie._weights(np.array([1, 0])))
# print("rep minimal matrices: ")
##res = ALie.repMinimalMatrices(np.array([[2]]))
##for el in res:
##    print(el[0], " ", el[1], " ", el[2])
##print("SU5")
##res = SU5.repMinimalMatrices(np.array([[1, 0, 0, 0]]))
##for el in res:
##    print(el[0], " ", el[1], " ", el[2])
##    print("\n")
##print("SP12")
##res = SP12.repMinimalMatrices(np.array([[1, 0, 0, 0, 0, 0]]))
##for el in res:
##    print(el[0], " ", el[1], " ", el[2])
##    print("\n")
##
##print("SU3")
##res = BLie.repMinimalMatrices(np.array([[1,1]]))
##for el in res:
##    print(el[0],el[1],el[2])
##    print("\n")
##
##print("Rep Matrices")
##print("SU2")
##res = ALie.repMatrices([1])
##print(res)
##print("\n")
##res = ALie.repMatrices([2])
##print(res)
##print("\n")
##res = ALie.repMatrices([3])
##print(res)
##print("\n")
##print("SU3")
##res = BLie.repMatrices([1,1])
##print(res)
##print("SU5")
##res = SU5.repMatrices([1,0,0,0])
##print(res)
##print("\nInvariants")
# print("SU2")
# res = ALie.invariants([[1],[1]],conj=[True, False])
# res2 = ALie.invariants([[1],[1]],conj=[False, False])
# print(res,res2)
# res = ALie.invariants([[4],[4]],conj=[True, False])
# print(res)
# print("SU3")
# res = BLie.invariants([[1,0],[1,0]],conj=[True,True])
# print(res)
# print("SU4")
# begin = time.time()
# res = SU4.invariants([[1,1,0],[1,1,0]],conj=[False, True])
# end = time.time()-begin
# print(end)
# print(res)
# print("Hookcontent FOrmula:")
# Sn = Permutation()
# print(Sn.hookContentFormula([4,3,2],Symbol('n')))

print("Invariants of three Irreps:")
print("SU2:")
# res = ALie.invariants([[2], [1], [1]], conj=[False, False, False])
# print(res)
# print("SU5")
# start = time.time()
##pudb.set_trace()
##res = SU5._weights([0,0,1,1])
# res = SU5.invariants([[1,1,0,0],[0,0,1,1]],conj=[False,False])
##print(res)
##res = SU5.invariants([[1,1,1,0],[0,1,1,1]],conj=[False,False])
# end = time.time()-start
# print(res)
# print(end)

# print("Invariants of four Irreps:")
# print("SU2:")
##res = ALie.invariants([[2],[1],[1],[2]],conj=[True, True, True, True])
# print("SU3:")
##res = BLie.invariants([[1,1],[1,1],[1,0],[1,0]],conj=[True,False,True,False])
# res = BLie.invariants([[1,1],[1,1],[1,1]],conj=[False,False,False])
# pudb.set_trace()
print("Check the Sn")
Sn = Sn()
res = ALie.invariants([[2], [2], [2], [2]], conj=[False, False, False, True])
print(res)
pudb.set_trace()
print(res)
