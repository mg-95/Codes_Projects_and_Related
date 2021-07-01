import math

def eucliDist(lst1, lst2):
    return math.dist(lst1, lst2)

def manhDist(lst1, lst2):
    diff = 0
    for i, data in enumerate(lst1):
        diff += abs(lst1[i] - lst2[i])
    return diff

def nDimDist(*lst):
    print("\t\t","*"*10,"Distance","*"*10)
    print("\t\t",end="")
    for item in lst[0]:
        print(item,"\t",end="")
    print("")
    for itm1 in lst[0]:
        print(itm1,"\t",end="")
        for itm2 in lst[0]:
            print("{:.2f}\t".format(eucliDist(itm1,itm2)),end="")
        print("")

def dist(f,a,b):
    print("The distance by {} is {}".format(f, f(a,b)))

def dist1(f,a):
    print("The distance by {} is {}".format(f, f(a)))

nPoints = [(115, 115, 115, -25, 110, 10),(125, 15, 115, -25, 70, 20),(115, 55, 115, 25, -110, 30),
           (15, 115, 115, -25, 70, 40),(115, 55, 125, -25, 110, 50),(115, 115, 115, -25, -110, 60),
           (115, 115, 55, 25, 70, 70),(115, 115, 15, -25, 110, 80),(55, 125, 115, 25, 70, 90),
           (15, 115, 115, -25, -110, 100)]

for n in nPoints:
    dist(eucliDist,nPoints[0],n)
for n in nPoints:
    dist(manhDist,nPoints[0],n)

dist1(nDimDist,nPoints)
