for i in `seq 16 16 64`; do
    make clean
    make CLFLAGS="-DMIN_SIZE=$i"
    for j in `seq 256 256 2048`; do
        for k in `seq 256 256 2048`; do
            echo $i $j $k
            python convert_m.py $j $k
            ./main p matriz_a.mat matriz_b.mat matriz_c.mat >> "test2_"$i".csv"
        done
    done
done
