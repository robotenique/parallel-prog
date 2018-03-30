@author: João Gabriel Basi Nº USP: 9793801
@author: Juliano Garcia de Oliveira Nº USP: 9277086

Como rodar o programa:
Compilar usando 'make puzzle'

Executar:
./puzzle <numero de pedras>

Funcionamento:
A nossa solução cria N - 1 threads, que representam os sapos. Sapos machos tem
ID positivo, as fêmeas tem ID negativo, enquanto a pedra livre é representada
pelo número 0.
Utilizamos uma barreira de sincronização para tentar fazer com que as threads se
iniciem ao mesmo tempo. Cada thread do sapo tenta andar para frente, se conseguir
(se o espaço estiver livre), adquire o lock do mutex da posição que vai pular,
trava a posição que está no momento, troca os valores e libera o mutex.
Se não conseguir pular, soma 1 ao contador e retorna para o loop da thread, que só
para quando foi detectado deadlock ou quando o contador chegou no limite estipulado.

A thread que gerencia e detecta deadlock é a própria main. Quando é detectado deadlock,
uma variável booleana recebe o valor 'true', e as threads saem do seu loop.
OBS: Como as threads dos sapos são 'detached', não é preciso dar join delas na main.

A quantidade de vezes que o programa chega na solução é muito baixa. Com os testes
que fizemos, a partir de 7 pedras é praticamente impossível que o problema seja
solucionado na nossa implementação.
