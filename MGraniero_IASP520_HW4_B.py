A = (115, 115, 115, -25, 110, 10) 
B = (125, 15, 115, -25, 70, 20)
C = (115, 55, 115, 25, -110, 30)
D = (15, 115, 115, -25, 70, 40)
E = (115, 55, 125, -25, 110, 50)

sA = set(A)
sB = set(B)
sC = set(C)
sD = set(D)
sE = set(E)

print("{} is converted to the set {}".format(A,sA))
print("{} is converted to the set {}".format(B,sB))
print("{} is converted to the set {}".format(C,sC))
print("{} is converted to the set {}".format(D,sD))
print("{} is converted to the set {}".format(E,sE))

AllSet = [sA,sB,sC,sD,sE]

for s in AllSet:
    union = AllSet[0] | s
    print("{} union {} = {}".format(AllSet[0],s,union))
    
for s in AllSet:
    intSec = AllSet[0] & s
    print("{} intersection {} = {}".format(AllSet[0],s,intSec))
        
for s in AllSet:
    jacSim = len(intSec) / len(union)
    union = AllSet[0] | s
    intSec = AllSet[0] & s
    print("The Jaccard similarity between {} and {} is {}".format(AllSet[0], s, jacSim))

