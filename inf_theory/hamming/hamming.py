from typing import List
from random import choices,choice, randrange
from pprint import pprint


def get_dist_hamming(a: List[int], b: List[int]):
    """Возвращает расстояние Хэмминга для двух массивов"""
    return sum([1 for i in range(len(a)) if a[i] != b[i]])

def checking2(n: int):
    """Проверка на степень двойки"""
    if n & (n - 1) :
        return False
    else:
        return True

def get_bin_list(n: int, size=None):
    """Возвращает список"""
    arr = []
    while True:
        arr.insert(0, n % 2)
        n //= 2
        if n == 0:
            break
    if size is not None:
        return [0]*(size-len(arr))+arr
    return arr

def code_words(n):
    """Генерация всех кодовых слов длиной n"""
    for i in range(2 ** n):
        yield get_bin_list(i,n)

def transpose(arr: List[List]) -> List[List]:
    """Транспонирует матрицу"""
    return list(map(list,zip(*arr)))

def mult(mat1: List[List], mat2: List[List]) -> List[List]:
    """Умножает матрицу arr1 (слева) на матрицу arr2 (справа)"""
    result = []
    for i in range(0,len(mat1)):
        temp=[]
        for j in range(0,len(mat2[0])):
            s = 0
            for k in range(0,len(mat1[0])):
                k = 0 if mat1[i][k]*mat2[k][j] == 0 else 1
                s = (s+k) % 2
            temp.append(s)
        result.append(temp)
    return result

def ones(n: int):
    """Возвращает единичную матрицу размером n"""
    res = []
    for i in range(n):
        row = [0] * n
        row[i] = 1
        res.append(row)
    return res

def poscript(mat1: List[List], mat2: List[List]):
    """Дозаписывает справа от матрицы 1 матрицу 2"""
    res = []
    for i in range(len(mat1)):
        res.append(mat1[i][:]+mat2[i][:])
    return res

def get_h1(n: int, k: int):
    """Возвращает матрицу H1"""
    h1 = []
    numbers = []
    t = 1
    for i in range(k):
        while checking2(t) or t in numbers:
            t += 1
        h1.append(get_bin_list(t,r))
        numbers.append(t)
    return transpose(h1)

def get_h(n: int, k: int):
    """Возвращает проверочную матрицу"""
    return poscript(get_h1(n,k),ones(n-k))

def get_g(n: int, k: int):
    """Возвращает порождающую матрицу"""
    return poscript(ones(k),transpose(get_h1(n,k)))


        
        
n = 15
k = 11
r = n-k
H = get_h(n,k)
G = get_g(n,k)
print('Матрица H:')
pprint(H)
print('\nМатрица G:')
pprint(G)


print('\nПроверим проверочную и порождающую матрицы перемножением:')
pprint(mult(G,transpose(H)))

print('Сохраним все закодированные слова с исходными в словарь')
encoded_dict = {}
for i in code_words(11):
    encoded = tuple(mult([i], G)[0])
    encoded_dict[encoded] = i
    
    
random_encoded_word = choice(list(encoded_dict.keys()))
print('\n\nВыберем случайное кодовое слово:\n',random_encoded_word,sep='')

print('Проверим кодовое слово умножением на матрицу H:')
print(mult([random_encoded_word],transpose(H)))

print('Внесем ошибку в кодовое слово с помощью случайного XOR:')
print('Исходное:', random_encoded_word)
error_in = randrange(n)
errored_encoded_word = tuple([random_encoded_word[i] if i != error_in else random_encoded_word[i] ^ 1 for i in range(n)])
print('С ошибкой:', errored_encoded_word)


min_dist_hamming = n
min_word = []
print('Найдем слово методом исправления: найдем минимальное расстояние Хэмминга между словом с ошибкой и всем словарем')
for word in encoded_dict:
    dist_hamming = get_dist_hamming(word,errored_encoded_word)
    if min_dist_hamming > dist_hamming:
        min_word = word
        min_dist_hamming = dist_hamming
print('Найденное слово:', min_word)
print('Это слово совпадает с изначально закодированным?', min_word == random_encoded_word)
print('Это слово соответствует исходному сообщению:', encoded_dict[min_word])
print('Проверим слово методом обнаружения: попробуем найти слово с ошибкой в словаре напрямую')
check = errored_encoded_word in encoded_dict
print(f'Слово {"не" if not check else ""} обнаружено!\n\n')

print('Внесем 2 ошибки и проверим слово методом обнаружения')
errors_in = choices(range(0,n),k=2)
errored_encoded_word = tuple([random_encoded_word[i] if i not in errors_in else random_encoded_word[i] ^ 1 for i in range(n)])
print('С ошибкой:', errored_encoded_word)


print('Проверим слово методом обнаружения: попробуем найти слово с ошибкой в словаре напрямую')
check = errored_encoded_word in encoded_dict
print(f'Слово {"не" if not check else ""} обнаружено!')

print('Проверять слово методом исправления неправильно,\n'
      'т. к. согласно теореме с минимальным расстоянием d = 3,\n'
      'мы можем исправить максимум 1 ошибку.\n'
      'Следовательно, может быть более одного слова с минимальным расстоянием Хэмминга')
