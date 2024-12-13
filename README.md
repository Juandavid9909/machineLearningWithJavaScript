# Proceso para solucionar un problema

- Identificar la información que es relevante para el problema.
- Armar un set de datos relacionados al problema que estamos intentando solucionar.
- Decidir el tipo de salida que estamos intentando predecir.
- Basado en el tipo de salida, elegir un algoritmo que determinará una correlación entre nuestras "features" y "labels (etiquetas)".
- Usar el modelo generado por el algoritmo para hacer una predicción.


## Clasificación

El valor de nuestras etiquetas pertenece a un conjunto discreto. Se pueden obtener datos como: "PASS", "FAIL", "SPAM", "NOT SPAM", etc.


## Regresión

El valor de nuestras etiquetas pertenece a un conjunto continuo. Se pueden obtener datos como: 2.5, 4, 6, 9.3, etc

Para manejar toda esta información existen diferentes forma de validar la data en JavaScript. Una es teniendo un arreglo de objetos con todos los datos para poder iterar y validar la información. Otra forma es teniendo arreglos dentro de otros arreglos, donde cada uno de los arreglos internos va a representar cada uno de los valores, es importante tener la estructura de los elementos idéntica para que se puedan asociar correctamente.


# Algoritmos

## K-Nearest Neighbor (knn)

"Birds of a feather flock together".

Este algoritmo se utiliza en el proyecto Plink, digamos que tenemos la siguiente situación: Se quiere predecir el bucket en el que caerá una pelota si es lanzada desde la posición 300px. Este algoritmo se encargará de hacer la siguiente validación:

1. Lanzar la pelota varias veces y guardar el valor del bucket en el que cae.
2. Para cada uno de estos cálculos, se resta 300px del punto de origen y se obtiene el valor absoluto, junto con el bucket final.
3. Se ordenan los elementos de menor a mayor.
4. Valida los primero K elementos y define cuál fue el bucket más común.
5. En cualquiera que sea el bucket en el que más frecuentemente se ingresó, será al que posiblemente caiga la pelota.


# Plinko

Es una app donde se lanza una pelota desde cierta posición y puede caer en un bucket específico.


## Objetivo

Dados algunos datos sobre desde dónde se lanza una pelota, podemos predecir en qué bucket terminará?


## Datos relevantes para el problema

Uno de los datos más relevantes sería la posición desde la que se lanza la pelota, ya que esto nos permitirá hacer un cálculo basado en el origen del lanzamiento. Aquí también es muy importante la información del bucket donde termina el bucket, esto debido a que así podremos asociar o encontrar alguna relación entre estos datos para realizar nuestra predicción.

Por otra parte hay otros 2 datos importantes en este problema, y son el rango de la redondez de la pelota, y el rango de tamaño de la misma. Esto puede afectar drásticamente el resultado final, por lo que debemos tener en cuenta estos datos para realizar los cálculos.

| Features | Labels |
|--|--|
| Posición de lanzamiento | Cubo en el que cae la pelota |
| Redondez de la pelota |  |
| Tamaño de la pelota |  |

Tener en cuenta que cambiar alguno de los features probablemente cambiará los labels.

En este caso se utilizará la opción de tener arreglos dentro de un arreglo global.

```javascript
[
	[300, 0.4, 16, 4],
	[350, 0.4, 25, 5],
	[416, 0.4, 16, 4],
	[722, 0.4, 16, 7]
]
```

Cabe resaltar que el primer elemento es la posición de lanzamiento, el segundo la redondez de la pelota, el tercero el tamaño de la misma, y el 4 y último el bucket en el que finaliza la pelota.


## Tipo de salida

Sería una clasificación, debido a que tenemos un conjunto de datos fijos donde podrá caer la pelota, no habrán más ni menos, ni estará en puntos medios entre los buckets, etc.


## Implementación propia de Knn

```javascript
const outputs = [
	[10, .5, 16, 1],
	[200, .5, 16, 4],
	[350, .5, 16, 4],
	[600, .5, 16, 5]
];

const predictionPoint = 300;
const k = 3;

function distance(point) {
	return Math.abs(point - predictionPoint);
}

_.chain(outputs)
	.map((row) => [distance(row[0]), row[3]])
	.sortBy((row) => row[0])
	.slice(0, 3)
	.countBy((row) => row[1])
	.toPairs()
	.sortBy((row) => row[1])
	.last()
	.first()
	.parseInt()
	.value();
```


## Interpretar malos resultados

Debido a varios factores no se obtuvieron los resultados esperados, cuando sucede esto podemos seguir los siguientes pasos:

1. Ajustar los parámetros del análisis.
2. Agregar más features para explicar el análisis.
3. Cambiar el punto de predicción.
4. Acepte que tal vez no exista una buena correlación.

### Encontrar nuestro K ideal
- Guardar una buena cantidad de puntos de datos.
- Dividir la data en set de entrenamiento y set de pruebas.
- Para cada registro de prueba, correr nuestro KNN usando la data de entrenamiento.
- ¿El resultado de KNN es igual al del registro de prueba?