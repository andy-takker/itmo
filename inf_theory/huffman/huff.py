from collections import Counter
import queue
import math
from typing import Dict
import os
import sys
from pathlib import Path

BYTES_BY_SYMBOL = 8

class ProbabilityCounter(Counter):
    """Счетчик для подсчета частот и вероятностей символов"""
    def probabilties(self):
        total = self.total()
        self.probs = {}
        for key in self:
            self.probs[key] = self[key] / total

    def most_possible(self):
        return list(sorted(self.probs.items(),key=lambda x: x[1],reverse=True))


class Node:
    """Узел дерева Хаффмана"""
    def __init__(self, freq, key = -1, left = None, right = None, code = ''):
        self.freq = freq
        self.key = key
        self.left = left
        self.right = right
        self.code = code

    def __lt__(self, otr: 'Node'):
        """Необходимо определить операцию 'меньше' для использования в PriorityQueue"""
        return self.freq < otr.freq


def calculate_entropy(freq_table: ProbabilityCounter):
    """Расчет энтропии"""
    return sum([-p * math.log2(p) for p in freq_table.probs.values()])

def make_huffman_code_table(freq_table: Counter) -> Dict:
    """Построение таблицы кодов Хаффмана"""
    nodes = []
    q = queue.PriorityQueue()
    code_table = {}

    for k, v in sorted(freq_table.items(),key=lambda x: x[0]):
        nodes.append(Node(freq=v,key=k))
        q.put(nodes[-1])

    while q.qsize() > 1:
        left_node = q.get()
        right_node = q.get()
        left_node.code = '1'
        right_node.code = '0'
        new_node = Node(left_node.freq+right_node.freq, left=left_node,right=right_node)
        nodes.append(new_node)
        q.put(nodes[-1])

    def tree_traversal(p,codestr=[]):
        """Обход дерева"""
        codestr.append(p.code)
        if p.left:
            tree_traversal(p.left,codestr.copy())
            tree_traversal(p.right,codestr.copy())
        else:
            code_table[p.key] = ''.join(codestr)
    tree_traversal(nodes[-1])
    return code_table

def encode_freq(freq_table: Dict[str,int] ) -> str:
    """Кодирование таблицы частот"""
    output_str = b''
    d = {}
    for i in freq_table:
        d[ord(i)] = freq_table[i].to_bytes(BYTES_BY_SYMBOL,'big')
    for i in range(0,256):
        if i in d:
            output_str += d[i]
        else:
            output_str += int(0).to_bytes(BYTES_BY_SYMBOL,'big')
    return output_str

def decode_freq(encoded_str: str) -> Dict[str,int]:
    """Декодиование таблицы частот"""
    freq_table = {}
    for i in range(0, 256):
        f = int.from_bytes(encoded_str[i*BYTES_BY_SYMBOL:(i+1)*BYTES_BY_SYMBOL],'big')
        if f != 0:
            freq_table[chr(i)] = f
    return freq_table

def calculate_avg_code_word_length(code_table: dict, freq_counter: ProbabilityCounter):
    """Расчет средней длины кодового слова"""
    return sum([len(code_table[k])* v for k,v in freq_counter.probs.items()])


def checking_code_redudancy(r: float, p_max: float) -> bool:
    """Оценка избыточности кода"""
    h = lambda x: -x * math.log2(x) - (1-x) * math.log2(1-x)
    if p_max < 0.5:
        return r <= p_max +0.087
    else:
        return r <= 2 - h(p_max)-p_max

def encode_huffman(code_table: Dict, data: str, buffer: int = 8) -> str:
    """Кодирование исходного текста Хаффманом"""
    output_str = ''
    for char in data:
        output_str += code_table[char]
    extra_discharges = buffer - len(output_str) % 8
    filler = max(code_table.values(),key=lambda x: len(x))
    output_str += filler[:extra_discharges]
    return output_str

def decode_huffman(code_table: Dict, binary_string: str) -> str:
    """Декодирование файла в исходный текст"""
    output_str = ''
    i = 0
    while True:
        for k, v in code_table.items():
            if binary_string.find(v,i, i+len(v)) != -1:
                output_str += k
                i += len(v)
                break
        else:
            break
    return output_str

def str_to_bytes(bit_string: str) -> str:
    """Преобразование полученной обычной строки в байтовую"""
    n = int(bit_string,2)
    return n.to_bytes(n.bit_length()//8,'big')

def bytes_to_str(byte_str: str) -> str:
    """Преобразование байтовой строки в бинарную"""
    n = int.from_bytes(byte_str,'big')
    return f'{n:b}'

def encode_file(input_filename: str, encoded_filename: str):
    """Закодировать файл"""

    with open(input_filename,'r') as file:
        text = file.read()

    p_counter = ProbabilityCounter(text)
    p_counter.probabilties()

    entropy = calculate_entropy(p_counter)
    print(f'Энтропия: {entropy:.2f} бит')

    code_table = make_huffman_code_table(p_counter)

    avg_word_length = calculate_avg_code_word_length(code_table,p_counter)
    print(f'Средняя длина кодового слова: {avg_word_length:.2f} бит(а)')

    file_size_after_encoding = avg_word_length * len(text) / 8 / 1024 / 1024
    print(f'Примерный размер файла после кодирования (без таблицы): {file_size_after_encoding:.2f} МБайт')
    print(f'Размер таблицы: {8 * 256 / 1024} КБайт')
    f = checking_code_redudancy(avg_word_length - entropy, p_counter.most_possible()[0][1])
    if f:
        print(f'Избыточность кода (r = {avg_word_length - entropy:.2f} бит на символ) удовлетворительна')
    else:
        print('Код не оптимален!')


    encoded_string = encode_huffman(code_table, text)
    encoded_frequencies = encode_freq(p_counter)
    encoded_byte_string = str_to_bytes(encoded_string)

    with open(encoded_filename,'wb') as bytes_file:
        bytes_file.write(encoded_frequencies+encoded_byte_string)

    real_file_size = os.path.getsize(encoded_filename) / 1024 / 1024
    input_file_size = os.path.getsize(input_filename) / 1024 / 1024
    print(f'Кодирование прошло успешно!')
    print(f'Исходный размер файла: {input_file_size:.2f} МБайт')
    print(f'Фактический размер файла после кодирования: {real_file_size:.2f} МБайт')
    print(f'Степень сжатия: {(input_file_size-real_file_size)/input_file_size*100:.2f}%')

def decode_file(encoded_filename: str, decoded_filename: str):
    """Раскодировать файл"""
    with open(encoded_filename, 'rb') as bytes_file:
        bytes_from_file = bytes_file.read()

    encoded_text = bytes_to_str(bytes_from_file[256*8:])
    encoded_frequencies = bytes_from_file[:256*BYTES_BY_SYMBOL]

    p_counter = ProbabilityCounter(decode_freq(encoded_frequencies))
    p_counter.probabilties()
    code_table = make_huffman_code_table(p_counter)
    decoded_text = decode_huffman(binary_string=encoded_text,code_table=code_table)

    with open(decoded_filename, 'w+') as decoded_file:
        decoded_file.write(decoded_text)
    print('Файл успешно раскодирован!')

def main():
    if len(sys.argv) == 1:
        print('Программа работает в двух режимах - кодирования и раскодирования.\n\n'
              'Для кодирования необходимо передать параметры \'e input_file.txt output_file.txt\' Например,\n'
              'python huff.py e file1.txt file2.txt\n\n'
              'Для декодирования необхоидмо ппередать параметры \'d encoded_file.txt decoded_file.txt\' Например, \n'
              'python huff.py d file1.txt file2.txt\n\n')
    else:
        if sys.argv[1] == 'e' and len(sys.argv) == 4:
            if Path(sys.argv[2]).is_file() and Path(sys.argv[3]).parent.is_dir():
                encode_file(input_filename=sys.argv[2],encoded_filename=sys.argv[3])
            else:
                raise Exception('File not found!\nПроверьте переданные параметры')
        elif sys.argv[1] == 'd' and len(sys.argv) == 4:
            if Path(sys.argv[2]).is_file() and Path(sys.argv[3]).parent.is_dir():
                decode_file(encoded_filename=sys.argv[2], decoded_filename=sys.argv[3])
            else:
                raise Exception('File not found!\nПроверьте переданные параметры')
        else:
            raise Exception('Некорректный ввод. Для получения справки запустите программу без аргументов')

if __name__ == '__main__':
    main()
