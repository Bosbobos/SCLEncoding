from SCL import SCL

if __name__ == '__main__':
    N = 8 #Total bits
    maxSize = 4 # Number of routes to save
    F = [0,1,2,4] # Frozen bits positions
    scl = SCL(N, F, maxSize)

    encoded = [-1.3, -0.7, -0.15, 2.5, -0.18, 0.75, 1, -0.1]
    result = scl.decode(encoded)
    print(result)
    print(scl.get_decoded_results(*result))

    code = [1, 1, 1, 1]
    prepared = scl.prepare_input(code)
    encoded2 = scl.encode(prepared)
    decoded = scl.decode(encoded2)
    print(decoded)
    scl.print_decoded_results_table(*decoded)

