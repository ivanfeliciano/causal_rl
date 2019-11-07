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

# Snake

El clásico juego de la viborita.

## Acciones

Arriba, abajo, izquieda y derecha.

## Recompensas

Recompensa positiva +1 cuando la cabez tiene contacto con un cuadro rojo. 
-1 cuando tiene un estado terminal (choca con la pared o con su propio cuerpo).

## Estados 

* posición *x* de la cabeza.
* posición *y* de la cabeza.
* posición *x* de la comida.
* posición *y* de la comida.
* distacia de la cabeza a cada segmento de la serpiente.


