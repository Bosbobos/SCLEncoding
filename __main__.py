from SCL import SCL

if __name__ == '__main__':
    N = 16
    F = [0, 1, 2, 3, 4, 5, 8, 9]
    scl = SCL(N, F)
    code = '1'*8
    encoded = scl.encode(code)
    print(encoded)