list_of_words=[]
dictionary={}
dictionary_reduce={}
'''Creates a list of words by iterating over each doucment'''
for k in range(1,7):
        '''Opening and reading each doucment'''
        with open(f"doc{k}.txt") as f: 
                list_of_words.append({f'doc{k}':[]})
                for i in f.readlines():
                        for j in i.split():
                                if j not in list_of_words[k-1][f'doc{k}']:
                                        list_of_words[k-1][f'doc{k}'].append(j)


'''Creates a map by iterating over each document'''
for k in range(1,7):
        '''Opening and reading each doucment'''
        with open(f"doc{k}.txt") as f: 
                dictionary[f'doc{k}']={}
                for i in f.readlines():
                        for j in i.split():
                                if j not in dictionary[f'doc{k}'].keys():
                                        dictionary[f'doc{k}'][j]=1
                                else:
                                        dictionary[f'doc{k}'][j]+=1

'''creates map reduce by iteratiing over the existing map'''
for k1,v1 in dictionary.items():
        for k2,v2 in v1.items():
                if k2 not in dictionary_reduce.keys():
                        dictionary_reduce[k2]={k1:v2}
                else:
                        dictionary_reduce[k2][k1]=v2

print("\n\n-------------------------------------DISPLAYING LIST OF WORDS--------------------------------------\n\n")
print(list_of_words)

print("\n\n-------------------------------------DISPLAYING MAP--------------------------------------\n\n")
print(dictionary)

print("\n\n--------------------------------DISPLAYING MAP REDUCE-------------------------------------------\n\n")
print(dictionary_reduce)

print("\n\n----------------------------DISPLAYING MAP REDUCE IN SORTED ORDER-----------------------------------------------\n\n")
for i in sorted(dictionary_reduce):
        print(f"{i} => {dictionary_reduce[i]}")
list1 = list(dictionary_reduce.keys())
list2 = list(dictionary_reduce.values())

list3 = sorted(list(zip(list1, list2)))
print(list3)



