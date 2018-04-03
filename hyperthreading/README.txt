@author: João Gabriel Basi Nº USP: 9793801
@author: Juliano Garcia de Oliveira Nº USP: 9277086

Como rodar o programa:
Compilar usando 'make htest'

Executar:
./htest

Funcionamento:



Com HT:
102.36s
101.15s
101.85s
101.92s

Sem HT:
99.15s
99.19s
99.68s
99.94s




A nossa solução cria N - 1 (N = número de pedras) threads, que representam os sapos. Sapos machos tem
ID positivo, as fêmeas tem ID negativo, enquanto a pedra livre é representada
pelo número 0.
Utilizamos uma barreira de sincronização para tentar fazer com que as threads se
iniciem ao mesmo tempo. Cada thread do sapo tenta andar para frente, se conseguir
(se o espaço estiver livre), adquire o lock do mutex da posição que vai pular,
trava a posição que está no momento, troca os valores e libera o mutex.
Se não conseguir pular, soma 1 ao contador e retorna para o loop da thread, que só
para quando foi detectado deadlock ou quando o contador chegou no limite estipulado.
Colocamos nosso contador em 10000*N, onde N é o número de pedras. Como nas nossas
simulações a escolha das threads não seguia um padrão uniforme, deixamos o contador
com um número alto para deixar o programa rodar por mais tempo.
A thread que gerencia e detecta deadlock é a própria main. Quando é detectado deadlock,
uma variável booleana recebe o valor 'true', e as threads saem do seu loop.
OBS: Como as threads dos sapos são 'detached', não é preciso dar join delas na main.

A quantidade de vezes que o programa chega na solução é muito baixa. Com os testes
que fizemos, a partir de 7 pedras é praticamente impossível que o problema seja
solucionado na nossa implementação.
