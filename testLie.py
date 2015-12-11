from PyLie import *
import LieGroups as lie
import pudb

a = CartanMatrix("SU",2)
b = CartanMatrix("SU",3)
aold = lie.CartanMatrix("SU",2)
bold = lie.CartanMatrix("SU",3)
ALie = LieAlgebra(a)
BLie = LieAlgebra(b)

print("***** Check WeylOrbit function *****")
print("aold [0]")
print(lie.WeylOrbit(aold,[0]))

print("a [0]")
print(ALie._weylOrbit([0]))

print("bold [1,0]")
print(lie.WeylOrbit(bold,[1,0]))

print("b [1,0]")
print(BLie._weylOrbit([1,0]))

print("bold [1,1]")
print(lie.WeylOrbit(bold,[1,1]))

print("b [1,1]")
print(BLie._weylOrbit([1,1]))

print("***** Check DominantWeights function *****")
print("bold [1,0]")
#lie.DominantWeights(bold,[1,0])
print("b [1,0]")
BLie._dominantWeights([1,0])
print("bold [1,1]")
lie.DominantWeights(bold,[1,1])
print("b [1,1]")
BLie._dominantWeights([1,1])


