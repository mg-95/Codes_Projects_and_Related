import os
def reading(filLst):
    docLst = []
    for doc in filLst:
        f = open(doc)
        docName = doc.split('.')[0]
        doc = f.read().split()
        docLst.append({docName: doc})
    return docLst

ex49 = ["doc1.txt","doc2.txt","doc3.txt","doc4.txt","doc5.txt","doc6.txt",]

docLst = reading(ex49)

print("-"*20,"Set to compute Jaccard Similarity")
doc1 = docLst[0]['doc1']
doc2 = docLst[1]['doc2']
doc3 = docLst[2]['doc3']
doc4 = docLst[3]['doc4']
doc5 = docLst[4]['doc5']
doc6 = docLst[5]['doc6']

print(len(doc1),"\t",doc1)
print(len(doc2),"\t",doc2)
print(len(doc3),"\t",doc3)
print(len(doc4),"\t",doc4)
print(len(doc5),"\t",doc5)
print(len(doc6),"\t",doc6)
print()

print("-"*25, "Convert to Set")
sDoc1 = set(doc1)
sDoc2 = set(doc2)
sDoc3 = set(doc3)
sDoc4 = set(doc4)
sDoc5 = set(doc5)
sDoc6 = set(doc6)
print(len(sDoc1),"\t",sDoc1)
print(len(sDoc2),"\t",sDoc2)
print(len(sDoc3),"\t",sDoc3)
print(len(sDoc4),"\t",sDoc4)
print(len(sDoc5),"\t",sDoc5)
print(len(sDoc6),"\t",sDoc6)
print()

print("-"*25,"set UNION & set Intersection")
allDocs = [sDoc1,sDoc2,sDoc3,sDoc4,sDoc5,sDoc6]

int1 = 0

for a in allDocs:
    int1 += 1
    DocName = 'Doc'+str(int1)
    sUnion = allDocs[0] | a
    sIntersection = allDocs[0] & a
    print("Doc1 vs",DocName)
    print(sUnion, sIntersection)
    print("Jaccard Similarity: {:0.3f}".format(len(sIntersection)/len(sUnion)))
    print()





