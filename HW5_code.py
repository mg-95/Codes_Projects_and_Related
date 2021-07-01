from collections import defaultdict
import operator

Support = 6

item_counts = defaultdict(int)
pair_counts = defaultdict(int)
triple_counts = defaultdict(int)

with open('C://Users//Matteo//Desktop//IASP 520//HW5_dataset.txt') as f:
    lines = f.readlines()
f.close()

def normalize_group(*args):
    return str(sorted(args))

def generate_pairs(*args):
    pairs = []
    for idx_1 in range(len(args) - 1):
        for idx_2 in range(idx_1 + 1, len(args)):
            pairs.append(normalize_group(args[idx_1], args[idx_2]))
    return pairs

for line in lines:
    for item in line.split():
        item_counts[item] += 1

frequent_items = set()
for key in item_counts:
    if item_counts[key] > Support:
        frequent_items.add(key)

for line in lines:
    items = line.split()
    for idx_1 in range(len(items) - 1):
        if items[idx_1] not in frequent_items:
            continue
        for idx_2 in range(idx_1 + 1, len(items)):
            if items[idx_2] not in frequent_items:
                continue
            pair = normalize_group(items[idx_1], items[idx_2]) 
            pair_counts[pair] += 1
            
frequent_pairs = set()
for key in pair_counts:
    if pair_counts[key] > Support:
        frequent_pairs.add(key)

for line in lines:
    items = line.split()
    for idx_1 in range(len(items) - 2):
        if items[idx_1] not in frequent_items:
            continue
        for idx_2 in range(idx_1 + 1, len(items) - 1):
            if items[idx_2] not in frequent_items:
                continue
            first_pair = normalize_group(items[idx_1], items[idx_2])
            if first_pair not in frequent_pairs:
                continue
            for idx_3 in range(idx_2 + 1, len(items)):
                if items[idx_3] not in frequent_items:
                    continue
                pairs = generate_pairs(items[idx_1], items[idx_2], items[idx_3])
                if any(pair not in frequent_pairs for pair in pairs):
                    continue
                triple = normalize_group(items[idx_1], items[idx_2], items[idx_3])
                triple_counts[triple] += 1

frequent_triples = set()
for key in triple_counts:
    if triple_counts[key] > THRESHOLD:
        frequent_triples.add(key)

item_counts = { k: v for k, v in item_counts.items() if v > Support }
sorted_items = sorted(item_counts.items(), key = operator.itemgetter(1))

for entry in sorted_items:
    print('{0}: {1}'.format(entry[0], entry[1]))

pair_counts = { k: v for k, v in pair_counts.items() if v > Support }
sorted_pairs = sorted(pair_counts.items(), key = operator.itemgetter(1))

for entry in sorted_pairs:
    print('{0}: {1}'.format(entry[0], entry[1]))

triple_counts = { k: v for k, v in triple_counts.items() if v > Support }
sorted_triples = sorted(triple_counts.items(), key = operator.itemgetter(1))

for entry in sorted_triples:
    print('{0}: {1}'.format(entry[0], entry[1]))


