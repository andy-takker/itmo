import math
import os
import queue
import sys
from collections import Counter
from hashlib import md5
from pathlib import Path
from typing import Dict

BYTES_BY_SYMBOL = 8


class ProbabilityCounter(Counter):
    """Счетчик для подсчета частот и вероятностей символов"""

    def probabilties(self):
        total = self.total()
        self.probs = {}
        for key in self:
            self.probs[key] = self[key] / total

    def most_possible(self):
        return list(
            sorted(self.probs.items(), key=lambda x: x[1], reverse=True))


class Node:
    """Узел дерева Хаффмана"""

    def __init__(self, freq, key=-1, left=None, right=None, code=''):
        self.freq = freq
        self.key = key
        self.left = left
        self.right = right
        self.code = code

    def __lt__(self, otr: 'Node'):
        """Операция 'меньше' для использования в приоритетной очереди"""
        return self.freq < otr.freq


def calculate_entropy(freq_table: ProbabilityCounter):
    """Расчет энтропии"""
    return sum([-p * math.log2(p) for p in freq_table.probs.values()])


def make_huffman_code_table(freq_table: Counter) -> Dict:
    """Построение таблицы кодов Хаффмана"""
    nodes = []
    q = queue.PriorityQueue()
    code_table = {}

    for k, v in sorted(freq_table.items(), key=lambda x: x[0]):
        nodes.append(Node(freq=v, key=k))
        q.put(nodes[-1])

    while q.qsize() > 1:
        left_node = q.get()
        right_node = q.get()
        left_node.code = '1'
        right_node.code = '0'
        new_node = Node(left_node.freq + right_node.freq, left=left_node,
                        right=right_node)
        nodes.append(new_node)
        q.put(nodes[-1])

    def tree_traversal(node, code_str=[]):
        """Обход дерева"""
        code_str.append(node.code)
        if node.left:
            tree_traversal(node.left, code_str.copy())
            tree_traversal(node.right, code_str.copy())
        else:
            code_table[node.key] = ''.join(code_str)

    tree_traversal(nodes[-1])
    return code_table


def encode_freq(freq_table: Dict[str, int]) -> bytes:
    """Кодирование таблицы частот"""
    output_str = b''
    d = {}
    for i in freq_table:
        d[i] = freq_table[i].to_bytes(BYTES_BY_SYMBOL, 'big')
    for i in range(0, 256):
        if i in d:
            output_str += d[i]
        else:
            output_str += int(0).to_bytes(BYTES_BY_SYMBOL, 'big')
    return output_str


def decode_freq(encoded_str: bytes) -> Dict[str, int]:
    """Декодирование таблицы частот"""
    freq_table = {}
    for i in range(0, 256):
        f = int.from_bytes(
            encoded_str[i * BYTES_BY_SYMBOL:(i + 1) * BYTES_BY_SYMBOL], 'big')
        if f != 0:
            freq_table[chr(i)] = f
    return freq_table


def calculate_avg_code_word_length(code_table: dict,
                                   freq_counter: ProbabilityCounter):
    """Расчет средней длины кодового слова"""
    return sum([len(code_table[k]) * v for k, v in freq_counter.probs.items()])


def checking_code_redundancy(r: float, p_max: float) -> bool:
    """Оценка избыточности кода"""
    h = lambda x: -x * math.log2(x) - (1 - x) * math.log2(1 - x)
    if p_max < 0.5:
        return r <= p_max + 0.087
    else:
        return r <= 2 - h(p_max) - p_max


def encode_huffman(code_table: Dict, data: bytes) -> str:
    """Кодирование исходного текста Хаффманом"""
    output_str = ''
    for char in data:
        output_str += code_table[char]
    extra_discharges = BYTES_BY_SYMBOL - len(output_str) % BYTES_BY_SYMBOL
    filler = max(code_table.values(), key=lambda x: len(x))
    output_str += filler[:extra_discharges]
    return output_str


def decode_huffman(code_table: Dict, binary_string: str) -> str:
    """Декодирование файла в исходный текст"""
    output_str = ''
    i = 0
    while True:
        for k, v in code_table.items():
            if binary_string.find(v, i, i + len(v)) != -1:
                output_str += k
                i += len(v)
                break
        else:
            break
    return output_str


def str_to_bytes(bit_string: str) -> bytes:
    """Преобразование полученной обычной строки в байтовую"""
    bytes_str = b''
    for i in range(0, len(bit_string), 32):
        bs = bit_string[i:i + 32]
        n = int(bs, 2)
        bytes_str += n.to_bytes(len(bs) // 8, 'big')
        if i == 0:
            print(bs)
            print(n.to_bytes(len(bs) // 8, 'big'))
            print(n)
    return bytes_str


def bytes_to_str(byte_str: bytes) -> str:
    """Преобразование байтовой строки в бинарную"""
    n = int.from_bytes(byte_str, 'big')
    return f'{n:b}'


def encode_file(input_filename: str, encoded_filename: str):
    """Закодировать файл"""

    with open(input_filename, 'rb') as file:
        text = file.read()

    p_counter = ProbabilityCounter(text)
    p_counter.probabilties()

    entropy = calculate_entropy(p_counter)
    print(f'Энтропия: {entropy:.2f} бит')

    code_table = make_huffman_code_table(p_counter)

    print(len(max(code_table.values(), key=lambda x: len(x))))

    avg_word_length = calculate_avg_code_word_length(code_table, p_counter)
    print(f'Средняя длина кодового слова: {avg_word_length:.2f} бит(а)')

    file_size_after_encoding = avg_word_length * len(text) / 8
    print(f'Примерный размер файла после кодирования (без таблицы):'
          f' {file_size_after_encoding:.2f} Байт')
    print(f'Размер таблицы: {8 * 256} Байт')
    f = checking_code_redundancy(avg_word_length - entropy,
                                 p_counter.most_possible()[0][1])
    if f:
        print(f'Избыточность кода (r = {avg_word_length - entropy:.2f} '
              f'бит на символ) удовлетворительна')
    else:
        print('Код не оптимален!')

    encoded_string = encode_huffman(code_table, text)
    encoded_frequencies = encode_freq(p_counter)
    encoded_byte_string = str_to_bytes(encoded_string)

    with open(encoded_filename, 'wb') as bytes_file:
        bytes_file.write(encoded_frequencies + encoded_byte_string)

    real_file_size = os.path.getsize(encoded_filename)
    input_file_size = os.path.getsize(input_filename)
    print(f'Кодирование прошло успешно!')
    print(f'Исходный размер файла: {input_file_size:.2f} Байт')
    print(f'Фактический размер файла после кодирования: {real_file_size:.2f} '
          f'Байт')
    compression_ratio = (input_file_size - real_file_size) / input_file_size
    print(f'Степень сжатия: {compression_ratio * 100:.2f}%')


def decode_file(encoded_filename: str, decoded_filename: str):
    """Раскодировать файл"""
    with open(encoded_filename, 'rb') as bytes_file:
        bytes_from_file = bytes_file.read()

    encoded_text = bytes_to_str(bytes_from_file[256 * 8:])
    encoded_frequencies = bytes_from_file[:256 * BYTES_BY_SYMBOL]

    p_counter = ProbabilityCounter(decode_freq(encoded_frequencies))
    p_counter.probabilties()
    code_table = make_huffman_code_table(p_counter)
    decoded_text = decode_huffman(binary_string=encoded_text,
                                  code_table=code_table)

    with open(decoded_filename, 'w+') as decoded_file:
        decoded_file.write(decoded_text)
    print('Файл успешно раскодирован!')


def checking_files(original_filename: str, decoded_filename: str) -> None:
    with open(original_filename, 'rb') as original_file:
        original_bytes = original_file.read()
    with open(decoded_filename, 'rb') as decoded_file:
        decoded_bytes = decoded_file.read()
    original_hash = md5(original_bytes)
    decoded_hash = md5(decoded_bytes)
    print(f'Хэш первоначального файла:\t{original_hash.hexdigest()}')
    print(f'Хэш восстановленного файла:\t{decoded_hash.hexdigest()}')
    if original_hash.hexdigest() == decoded_hash.hexdigest():
        print('Файлы идентичны!')
    else:
        print('Файлы не совпадают!')


def main():
    if len(sys.argv) == 1:
        print(
            'Программа работает в двух режимах - кодирования и раскодирования.'
            '\n\nДля кодирования необходимо передать параметры \'e original_fil'
            'e.txt output_file.txt\'\nНапример, python huff.py e file1.txt file'
            '2.txt\n\nДля декодирования необходимо передать параметры \'d encod'
            'ed_file.txt decoded_file.txt\'\nНапример, python huff.py d file1.'
            'txt file2.txt\n\nТакже есть режим проверки хэша файлов для сравнен'
            'ия содержимого для этого необходимо\nпередать параметры \'c origin'
            'al_file.txt decoded_file.txt\'')
    else:
        if sys.argv[1] == 'e' and len(sys.argv) == 4:
            if Path(sys.argv[2]).is_file() and Path(
                    sys.argv[3]).parent.is_dir():
                encode_file(input_filename=sys.argv[2],
                            encoded_filename=sys.argv[3])
            else:
                print(sys.argv)
                raise Exception('File not found!\nПроверьте параметры')
        elif sys.argv[1] == 'd' and len(sys.argv) == 4:
            if Path(sys.argv[2]).is_file() and Path(
                    sys.argv[3]).parent.is_dir():
                decode_file(
                    encoded_filename=sys.argv[2],
                    decoded_filename=sys.argv[3],
                )
            else:
                raise Exception('File not found!\nПроверьте параметры')
        elif sys.argv[1] == 'c' and len(sys.argv) == 4:
            if Path(sys.argv[2]).is_file() and Path(sys.argv[3]).is_file():
                checking_files(
                    original_filename=sys.argv[2],
                    decoded_filename=sys.argv[3],
                )
            else:
                raise Exception('File not found!\nПроверьте параметры')
        else:
            raise Exception('Неверный ввод. Для получения справки запустите '
                            'программу без аргументов')


if __name__ == '__main__':
    main()
