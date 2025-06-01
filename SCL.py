import numpy as np

class SCL:
    def __init__(self, N: int, F: list[int], L: int):
        self.N = N # Полная длина кода
        self.F = F # Позиции замороженных бит
        self.F_inv = [x for x in range(N) if x not in F] # Позиции полезных бит
        self.L = L # Объём списка путей

    def encode(self, code: str) -> str:
        if len(code) != self.N // 2:
            raise Exception('Code rate is not 1/2')
        encoded = ['0'] * self.N
        for i, pos in enumerate(self.F_inv):
            encoded[pos] = code[i]

        return ''.join(encoded)

    @staticmethod
    def left(x, y):
        return np.sign(x*y) * min(abs(x), abs(y))

    @staticmethod
    def right(x, y, b):
        if b not in [0, 1]:
            raise ValueError("b not in [0, 1]")
        if b == 0:
            return x + y

        return y - x

    def decode(self, code: str) -> str:
        if len(code) != self.N:
            raise Exception('Code length is incorrect')

        