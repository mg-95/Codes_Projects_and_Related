import xlrd

loc = ("C:\\Users\\Matteo\\Desktop\\IASP 520\\MGraniero_HW1_IASP520.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

X = []

for row in range(sheet.nrows):
    sheet.cell_value(row, 2)
    X.append(row)
    
def mean(X):
    s = 0
    for num in X:
        s = s + num
    return s / len(X)

def mode(X):
    max_count = (0,0)
    for num in X:
        occurences = X.count(num)
        if occurences > max_count[0]:
            max_count = (occurences, num)
    return max_count[1]

def median(X):
    X.sort()
    if len(X) % 2 != 0:
        middle_index = int((len(X) - 1) / 2)
        return X[middle_index]
    elif len(list_of_nums) % 2 == 0:
        mi_1 = int(len(X) / 2)
        mi_2 = int(len(X) / 2) - 1
        return int(mean([X[mi_1], X[mi_2]]))
    
print(mean(X), median(X), mode(X))
