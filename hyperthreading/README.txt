@author: João Gabriel Basi Nº USP: 9793801
@author: Juliano Garcia de Oliveira Nº USP: 9277086

Como rodar o programa:
Compilar usando 'make htest'

Executar:
./htest

Funcionamento:
Nosso programa utiliza 5 threads para escrever uma string em 5 arquivos (um para
cada thread) 180000 vezes. Assim, as threads concorrem pela escrita em arquivos,
já que o HD tem que ser compartilhado entre elas. Como no modo hyper-threading
existem mais processadores disputando a escrita mas só um pode fazê-la, há
vários conflitos, o que faz com que o programa fique lento. Como sem
hyper-threading há menos processadores, há menos conflitos e o programa roda
mais rápido.

Especificações do computador:
Intel Core i7-5500U @ 4x 3GHz
CPU(s):                     4
Thread(s) per núcleo:       2
Núcleo(s) por soquete:      2
Soquete(s):                 1

Tempos de execução com hyper-threading:
84.73s
83.32s
83.85s

Tempos de execução sem hyper-threading:
78.46s
78.78s
79.70s
