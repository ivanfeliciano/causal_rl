# Catcher

El agente debe atrapar fruta cayendo.

## Acciones

Izquierda y derecha

## Recompensas 

+1 por cada fruta que recoge o -1 si no la atrapa. El juego termina cuando 
se acaban las vidas.

## Estados

La representación no visual del juego incluye:

* posición *x* del jugador.
* velocidad del jugador.
* posición *x* de la fruta.
* posición *y* de la fruta.

## Modelo causal

X = {action A, currentPos X1, NewPos X2, fruitPos X3, currentDistance X4, newDistance X5, closer Z1}
A = {-1, 0, 1}
X1 = u1
X2 = A + X1 + u2
X3 = u3
X4 = |X1 - X3| + u4
X5 = |X2 - X3| + u5
Z1 = 1 si X5 < X4 0 en otro caso



# FlappyBird

## Acciones

Arriba

## Recompensas

Por cada tubería por la que pase gana +1. Si toca el piso, la tubería o el techo termina el episodio y la recompensa es -1.

## Estado

La representación no visual del juego incluyes:

* posición *y* del jugador.
* velocidad del jugador.
* distancia de la siguiente tubería al jugador.
* posición top *y* de la siguiente tubería.
* posición bottom  *y* de la siguiente tubería.
* distancia de la siguiente siguiente tubería al jugador.
* posición top *y* de la siguiente siguiente tubería.
* posición bottom *y* de la siguiente siguiente tubería.

## Modelo causal

X = {action A, currentYPlayerPos X1, nextPipeTopY X2, nextPipeBottomY X3, newPlayerYPos X4, currentSafePoint X5, closerToSafePoint Z1}
A  = u1
X1 = u2
X2 = u3
X3 = u4
X4 = X1 + A
X5 = (X2 + X3) / 2
Z1 = 1 si |X1 - X5| > |X4 - X5|



# Pixelcopter

Parecido al FlappyBird.

## Acciones

Arriba.

## Recompensas

Por cada bloque vertical que pasa +1. Por cada estado terminal (contacto con 
cualquier cosa verde) -1.

## Estados

La representación no visual de los estados incluye:

* posición *y* del jugador.
* velocidad del jugador.
* distancia del jugador al piso.
* distancia del jugador al techo.
* distancia *x* del siguiente bloque al jugador.
* posición top *y* del siguiente bloque al jugador.
* posición bottom *y* del siguiente bloque.

## Modelo causal

X = {action A, currentDistanceToCeil X1, currentDistanceToFloor X2, currentSafePointCF X3, nextBlockTopY X4, nextBlockBottomY X5, currentSafePointBlock X6, playerYPos X7}
A = u1
X1 = u2
X2 = u3
X3 = (X1 + X2) / 2
X4 = u4
X5 = u5
X6 = (X4 + X5)  / 2
X7 = u6
Z1 = 1 si |X3 - X7| > |X3 - X7 + A| 0 en otro caso
Z2 = 1 si |X6 - X7| > |X6 - X7 + A| 0 en otro caso

# Pong

Pong simula una tabla de tennis en 2D.

## Acciones

Arriba y abajo.

## Recompensas

El agente recibe una recompensa positiva +1, cada vez que pone la pelota
detrás de la paleta del oponente, -1, en el caso contrario.

## Estados

La representación no visual del juego tiene:

* posición *y* del jugador.
* velocidad del jugador.
* posición *y* del cpu.
* posición de la pelota *x*.
* posición de la pelota *y*.
* velocidad de la pelota en *x*.
* velocidad de la pelota en *y*.

## Modelo causal

X = {action A, ballX1Pos X1, ballY1Pos X2, ballX2Pos X3, ballY2Pos X4, currentPlayerYPos X5, newPlayerYPos X6, predictedBallY X7, hitBall Z1}
A = u1
X1 = u2
X2 = u3
X3 = u4
X4 = u5
X5 = u6
X6 = X5 + A
X7 = (X4 - X2) / (X3 - X1) (- X1) + X2
Z1 = 1 si |X7 - X5| > |X7 - X6|


# Snake

El clásico juego de la viborita.

## Acciones

Arriba, abajo, izquieda y derecha.

## Recompensas

Recompensa positiva +1 cuando la cabeza tiene contacto con un cuadro rojo. 
-1 cuando tiene un estado terminal (choca con la pared o con su propio cuerpo).

## Estados 

* posición *x* de la cabeza.
* posición *y* de la cabeza.
* posición *x* de la comida.
* posición *y* de la comida.
* distacia de la cabeza a cada segmento de la serpiente.

## Modelo causal

X = {up A1, down A2, left A3, right A4, limitsY X1, limitsX X2, snakeHeadX X3, snakeHeadY X4, 
newSnakeHeadX X5, newSnakeHeadY X6, notCollisionX X7, notCollisionY X8, bodyCollision X9, bodyPos X10, 
fruitX X11, fruitY X12, closeToFruit Z1}

X5 = A3 + A4 + X3
X6 = A1 + A2 + X4
X7 = 1 si (0 < X5 < X2) 0 en otro caso
X8 = 1 si (0 < X6 < X3) 0 en otro caso
X9 = 1 si dist((X5, X6), (X10_X, X10_Y)| < EPS 0 en otro caso
Z1 = 1 si |X11 - X5| < |X11 - X3| AND |X12 - X6| < |X12 - X4|
