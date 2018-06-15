import numpy as np
import sys

def mat2str(mat):
    s = ""
    for i in range(3):
        for j in range(3):
            c = ' ' if j < 2 else '\n'
            s += f"{mat[i][j]}{c}"
    return s

def main():
    if len(sys.argv) < 3:
        print("Usage: python gen-test.py <num_matrices> <dest_file>")
        return
    num_mat = int(sys.argv[1])
    dest = sys.argv[2]
    r = 10
    f = open(dest, "w")
    f.write(f"{num_mat}\n***\n")
    for i in range(num_mat):
        f.write(mat2str(np.random.randint(r, size=(3,3)) - r//2))
        f.write("***\n")
    f.close()

if __name__ == '__main__':
    main()
