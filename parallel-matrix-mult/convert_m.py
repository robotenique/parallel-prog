import numpy as np
def convert(mtx):
    n = len(mtx)
    m = len(mtx[0])
    print(f"{n} {m}")
    for i in range(n):
        for j in range(m):
            if mtx[i][j] != 0:
                print(f"{i + 1} {j + 1} {mtx[i][j]}")

def main():
    mtx = [[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]
          ]
    a = np.random.random((500,100))*10
    b = np.random.random((100,500))*10
    convert(a)


if __name__ == '__main__':
    main()
