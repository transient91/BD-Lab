import random
import itertools

attr_num = 4
lattice_num = 2**attr_num

str = 'ABCDEFGHIJKLMNOPQRSTUVWKYZ'[:attr_num]
lst = []
for i in range(1, attr_num + 1):
    lst += [''.join(x) for x in itertools.combinations(str, i)]
# print(lst)
with open('data.txt', 'w', encoding='UTF-8') as f:
    size = 1
    f.write('{},{},{}\n'.format(1, 'NONE', size))
    for i in range(lattice_num - 1):
        size = random.randint(size + 1, size + 10)
        f.write('{},{},{}\n'.format(i + 2, lst[i], size))