def convert(mtx):
    n = len(mtx)
    m = len(mtx[0])
    print(f"{n} {m}")
    for i in range(n):
        for j in range(m):
            if mtx[i][j] != 0:
                print(f"{i + 1} {j + 1} {mtx[i][j]}")
    
def main():
    mtx = [[1, 1],
           [1, 0]
          ]
    convert(mtx)

if __name__ == '__main__':
    main()
