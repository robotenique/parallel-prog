import numpy as np
import sys

def convert(mtx):
    n = len(mtx)
    m = len(mtx[0])
    s = ""
    s+=f"{n} {m}\n"
    for i in range(n):
        for j in range(m):
            if mtx[i][j] != 0:
                s+=f"{i + 1} {j + 1} {mtx[i][j]}\n"
    return s

def main():
    mtx = []
    for _ in range(500):
        mtx.append(list(range(1,501)))
    a_rows = 2048
    a_cols = 2048
    b_cols = 2048
    if len(sys.argv) == 3:
        a_rows = int(sys.argv[1])
        a_cols = int(sys.argv[2])
        b_cols = int(sys.argv[1])
    a = np.floor(np.random.random((a_rows,a_cols))*10)
    b = np.floor(np.random.random((a_cols,b_cols))*10)
    mat_a = open("matriz_a.mat", "w")
    mat_b = open("matriz_b.mat", "w")
    mat_a.write(convert(a))
    mat_b.write(convert(b))
    mat_a.close()
    mat_b.close()

if __name__ == '__main__':
    main()
