import matplotlib.pyplot as plt

f = open("sorted_player.txt", "r")
content = f.read()

content = content.split(',')

generation_number = []
worst_case = []
best_case = []
mean_case = []

for value in content:
    string = value.split(':')
    if len(string) < 4:
        continue
    generation_number.append(int(string[0]))
    worst_case.append(int(string[1]))
    best_case.append(int(string[2]))
    mean_case.append(int(string[3]))

plt.plot(generation_number, worst_case, label="worst")
plt.plot(generation_number, best_case, label="best")
plt.plot(generation_number, mean_case, label="mean")
plt.legend()
plt.show()
