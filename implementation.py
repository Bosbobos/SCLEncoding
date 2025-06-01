import numpy as np

# Left belief function
def L(L1, L2):
    l = []
    n = len(L1)
    for i in range(n):
        l.append(np.sign(L1[i]) * np.sign(L2[i]) * min(np.abs(L1[i]), np.abs(L2[i])))
    return l


# Right belief function
def R(L1, L2, b):
    G = []
    # print(b)
    # print(L1)
    for i in range(len(L2)):
        G.append(L2[i] + (1 - 2 * b[1][i]) * L1[i])
    # print(G)
    return G


# Xor function after left and right decoding completion
def xor(u1, u2, u1_list, u2_list):
    res = []
    for i in range(len(u1[1])):
        res.append((u1[1][i] + u2[1][i]) % 2)
    for i in range(len(u2[1])):
        res.append(u2[1][i])
    u_resultantList = u1_list + u2_list
    ans = (u1[0] + u2[0], res, u_resultantList)
    return ans


# Decoding function
def decode(y, d, node, l):
    if (d == n - 1):

        decision = []
        decoded_list = []
        if node not in F:
            if (y[0] < 0):
                decision.append((0, [1]))
                decoded_list.append([1])
                decision.append((np.abs(y[0]), [0]))
                decoded_list.append([0])
            else:
                decision.append((0, [0]))
                decoded_list.append([0])
                decision.append((np.abs(y[0]), [1]))
                decoded_list.append([1])
        else:

            if (y[0] >= 0):
                decision.append((0, [0]))
                decoded_list.append([0])
            else:
                decision.append((np.abs(y[0]), [0]))
                decoded_list.append([0])
        # print('decision ',decision)

        return (decision, decoded_list)
    else:

        L1 = y[0: len(y) // 2]
        L2 = y[len(y) // 2:]

        left = L(L1, L2)

        (Ldecision, Ldecoded_list) = decode(left, d + 1, 2 * (node), l)

        # print("this is arr1",arr1)

        # print(Ldecision)
        # print(Ldecoded_list)
        right = []
        for i in range(len(Ldecision)):
            # print(arr1)
            right.append(R(L1, L2, Ldecision[i]))  # sending tuple to g function ,right is list of list of values
        # print(right)

        selection_list = []
        for i in range(len(right)):
            (Rdecision, Rdecoded_list) = decode(right[i], d + 1, 2 * node + 1, l)
            for j in range(len(Rdecision)):
                selection_list.append(xor(Ldecision[i], Rdecision[j], Ldecoded_list[i], Rdecoded_list[j]))

        selection_list.sort()
        # print(selection_list)
        if (len(selection_list) > maxSize):
            for i in range(len(selection_list) - maxSize):
                selection_list.pop()

        return_tuples = []
        return_decoded_list = []
        for i in range(len(selection_list)):
            return_tuples.append((selection_list[i][0], selection_list[i][1]))
            return_decoded_list.append(selection_list[i][2])

        # print(return_tuples)

        # print(return_decoded_list)
    return (return_tuples, return_decoded_list)

N = 8 #Total bits
K = N//2 # Useful bits
n = 4 # Tree depth
maxSize = 4 # Number of routes to save
F = [0,1,2,4] # Frozen bits positions

encoded = [-1.3, -0.7, -0.15, 2.5, -0.18, 0.75, 1, -0.1]
l =[(0,[]),[]]
print(decode(encoded, 0, 0, l))