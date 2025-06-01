import numpy as np


class SCL:
    def __init__(self, N: int, F: list[int], max_paths: int):
        self.N = N # Полная длина кода
        self.F = F # Позиции замороженных бит
        self.max_paths = max_paths # Объём списка путей
        self.n = int(np.log2(N)) + 1 # Количество уровней дерева

    # Функция спуска влево
    @staticmethod
    def L(L1, L2):
        l = []
        n = len(L1)
        for i in range(n):
            l.append(np.sign(L1[i]) * np.sign(L2[i]) * min(np.abs(L1[i]), np.abs(L2[i])))

        return l


    # Функция спуска вправо
    @staticmethod
    def R(L1, L2, b):
        G = []
        for i in range(len(L2)):
            G.append(L2[i] + (1 - 2 * b[1][i]) * L1[i])

        return G


    # Функция XOR для объединения левого и правого детей узла
    @staticmethod
    def xor(u1, u2, u1_list, u2_list):
        res = []
        for i in range(len(u1[1])):
            res.append((u1[1][i] + u2[1][i]) % 2)
        for i in range(len(u2[1])):
            res.append(u2[1][i])
        u_resultantList = u1_list + u2_list
        ans = (u1[0] + u2[0], res, u_resultantList)
        return ans


    # Основная функция декодирования
    def decode(self, y, d = 0, node = 0, l=None):
        if l is None:
            l = [(0, []), []]

        if d == self.n - 1:
            # Базовый случай, в котором мы дошли до листьев
            decision = []
            decoded_list = []
            if node not in self.F:
                if (y[0] < 0):
                    # Вариант, если мы декодировали правильно
                    decision.append((0, [1])) # В метрику пути добавляем 0 и наш выбранный бит (1)
                    decoded_list.append([1]) # В декодированный результат добавляем 1

                    # Альтернативно, если примем здесь бит равным 0, то записываем ошибку в метрику пути
                    decision.append((np.abs(y[0]), [0])) # Добавляем модуль значения в метрику и выбранный бит (0)
                    decoded_list.append([0]) # В декодированный результат добавляем 0
                else:
                    # То же, что и выше, рассматриваем два варианта
                    decision.append((0, [0]))
                    decoded_list.append([0])
                    decision.append((np.abs(y[0]), [1]))
                    decoded_list.append([1])
            else:
                # Знаем, что бит у нас заморожен
                if (y[0] >= 0):
                    # Ошибки нет, всё круто, в метрику пути пишем 0
                    decision.append((0, [0]))
                    decoded_list.append([0])
                else:
                    # Точно знаем, что у нас тут ошибка, добавляем модуль к метрике пути
                    decision.append((np.abs(y[0]), [0]))
                    decoded_list.append([0])

            return (decision, decoded_list)
        else:
            # Случай, когда мы ещё спускаемся до листьев
            L1 = y[0: len(y) // 2]
            L2 = y[len(y) // 2:]

            left = self.L(L1, L2) # Подготавливаем список через L-функцию для передачи ниже по дереву

            Ldecision, Ldecoded_list = self.decode(left, d + 1, 2 * node, l) # Получаем декодированный вариант из левого нижнего поддерева

            # Теперь подготавливаем список для передачи в правое поддерево
            # При этом создаём вариант для каждого из возможных решений в левом
            right = []
            for i in range(len(Ldecision)):
                right.append(self.R(L1, L2, Ldecision[i]))

            # Декодируем правое поддерево для каждого из возможных случаев
            selection_list = []
            for i in range(len(right)):
                (Rdecision, Rdecoded_list) = self.decode(right[i], d + 1, 2 * node + 1, l)
                for j in range(len(Rdecision)):
                    selection_list.append(self.xor(Ldecision[i], Rdecision[j], Ldecoded_list[i], Rdecoded_list[j]))

            selection_list = sorted(selection_list, key = lambda x: x[0])[:self.max_paths]

            return_tuples = []
            return_decoded_list = []
            for i in range(len(selection_list)):
                return_tuples.append((selection_list[i][0], selection_list[i][1]))
                return_decoded_list.append(selection_list[i][2])

        return (return_tuples, return_decoded_list)
