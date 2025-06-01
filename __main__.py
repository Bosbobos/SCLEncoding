from SCL import SCL

if __name__ == '__main__':
    N = 8 #Total bits
    maxSize = 4 # Number of routes to save
    F = [0,1,2,4] # Frozen bits positions
    scl = SCL(N, F, maxSize)

    encoded = [-1.3, -0.7, -0.15, 2.5, -0.18, 0.75, 1, -0.1]
    result = scl.decode(encoded)
    print(result)
