#######   Question 1   #######

Integer = int(input("Enter an integer: "))

if Integer % 2 == 0:
    print("Your integer is even")
else:
    print("Your integer is odd")

#######   Question 2   #######

Seconds = int(input("Enter number of seconds: "))

Hours = (1/3600)*Seconds

Minutes = (Hours - (int(Hours)))*60

Seconds = (Minutes - (int(Minutes)))*60

print(int(Hours))
print(int(Minutes))
print(int(Seconds))

#######   Question 3   #######

Letter = input("Enter a letter: ")

if Letter in list(['a','A','e','E','i','I','o','O','u','U']):
    print("Your letter is a vowel")
else:
    print("Your letter is a consonant")

#######   Question 4   #######

Number = int(input("Enter a positive number or zero: "))

while Number < 0:
    Number = int(input("Enter a positive number or zero: "))
    if Number >= 0: break

Sum = 0

for a in range(Number + 1):
    factorial = 1

    for b in range(1, a+1):
        factorial = factorial * b

    Sum = factorial + Sum

    print(str(a) +'!', end = ' ')
    if a != Number:
            print('+', end = ' ')

print('= ' + str(Sum))

#######   Question 5   #######

number = int(input('Please enter an even number: '))

EvenNumbers = []

while number % 2 == 0:

    EvenNumbers.append(number)

    number = int(input('Please enter an even number or odd to stop: '))
    
    if number % 2 != 0: break

avg = (sum(EvenNumbers))/(len(EvenNumbers))

print(avg)

#######   Question 6   #######

myWordBank = []

for a in range(5):

    word = input('type a word: ')

    if 'A' in word or 'a' in word:

        myWordBank.append(word)

print(myWordBank)

#######   Question 7   #######

phrase = list(input("Please enter a sentence: "))

phrase.sort()

x = phrase[-1]

print("The last letter in the alphabetically sorted string is", x)


