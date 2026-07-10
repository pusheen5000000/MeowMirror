lines = int(input())
tee = 0
ess = 0

for i in range(lines):
    sentence = input()
    for letter in sentence:
        if letter == 't' or letter == 'T': #feedback: letter.lower from raymond.
            tee +=1
        elif letter == 's' or letter == 'S':
            ess +=1

if tee > ess: 
    print("English")
else:
    print("French")
        