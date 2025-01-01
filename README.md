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


## Normalización

Delimitar valores de nuestra data original entre 0 y 1.

$$Dataset Normalizado = \frac{FeatureValue - minOfFeatureValues}{maxOfFeatureValues - minOfFeatureValues}$$


## Estandarización

Delimitar valores de nuestra data original entre -1 y 1.


# Lodash

| Ventajas | Desventajas |
|--|--|
| Proporciona métodos para todo lo que necesitamos | Extremadamente lento (relativo) |
| Excelente para diseños de APIs | No enfocado en números |
| Habilidades transferibles a otros proyectos de JavaScript | Algunas cosas son extrañas (obtener una columna de valores) |


# Tensorflow JS

| Ventajas | Desventajas |
|--|--|
| API similar a Lodash | Sigue en desarrollo (verificar) |
| Extremadamente rápido para cálculos numéricos |  |
| Tiene una API para álgebra lineal de bajo nivel y una API de alto nivel para Machine Learning |  |
| API similar a numpy (librería numérica muy popular en Python) |  |


## Tensor

Es esencialmente (en este caso) un objeto de JavaScript que almacena una colección de números, y estos números serán algún tipo de estructura de arreglo.

```javascript
// Ejemplo 1: 2 dimensiones
[
	[300, 0.4, 16, 4],
	[350, 0.4, 25, 5],
	[416, 0.4, 16, 4],
	[722, 0.4, 16, 7]
]

// Ejemplo 2: 1 dimensión
[200, 400, 600]

// Ejemplo 3: 3 dimensiones
[
	[
		[5, 10, 17]
	]
]
```


## Shape

Básicamente, estamos hablando de una métrica alrededor de un **tensor** o una propiedad de un **tensor** que describe cuántos registros hay o cuántos elementos hay en cada dimensión individual. Imaginemos llamar el método `.length` en cada una de las dimensiones (una sola vez por dimensión) de afuera hacia adentro.

```javascript
// Ejemplo
[5, 10, 17].length // Shape: [3]

[
	[5, 10, 17],
	[18, 4, 2].length
].length // Shape: [2, 3]

[
	[
		[5, 10, 17].length
	].length
].length // Shape: [1, 1, 3]
```

Los ejercicios en este caso serán de máximo 2 dimensiones, por lo tanto, una forma fácil de recordar cómo calcular el Shape será `[#rows, #columns]`.


## Broadcasting

El Broadcasting se aplica cuando se toman los shapes de nuestros tensors y de derecha a izquierda los shapes son iguales o 1. Cuando el shape es 1 el elemento se replica para las operaciones de todos los elementos del otro arreglo, por ejemplo $[1, 2, 3] + [4] = [5, 6, 7]$.

```javascript
// Ejemplo
const data = tf.tensor([[1, 2, 3], [4, 5, 6]]);
const otherData = tf.tensor([[1], [1]]);

data.add(otherData); // [[2, 3, 4], [5, 6, 7]]
```


## Slices

Nos permiten sacar la cantidad de datos (por ejemplo una columna) que necesitemos.

|  | 0 | 1 | 2 |
|--|--|--|--|--
| 0 | 20 | 30 | 40 |
| 1 | 50 | 60 | 70 |
| 2 | 50 | 60 | 70 |
| 3 | 50 | 60 | 70 |
| 4 | 50 | 60 | 70 |
| 5 | 50 | 60 | 70 |

Y los parámetros son:

| Start index |Size |
|--|--|
| [0, 1] | [6, 1] |


## Concatenation

Cuando queramos juntar 2 tensor lo podemos hacer de la siguiente forma:

```javascript
const tensorA = tf.tensor([
	[1, 2, 3],
	[4, 5, 6]
]);

const tensorB = tf.tensor([
	[7, 8, 9],
	[10, 11, 12]
]);

tensorA.concat(tensorB);

// Resultado
[
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],
	[10, 11, 12]
];

// Concatenar y que se queden en la misma fila
tensorA.concat(tensorB, 1); // el segundo parámetro por defecto es 0, y podemos colocar 0 o 1 para indicar en qué sentido queremos que se haga la concatenación

// Resultado
[
	[1, 2, 3, 7, 8, 9],
	[4, 5, 6, 10, 11, 12]
];
```


## Sum

Permite realizar la suma total de una fila de un tensor, y como segundo parámetro (opcional) podemos indicar si queremos que nos deje sólo el número total (por defecto false), o si queremos mantener nuestro shape.

```javascript
const jumpData = tf.tensor([
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70]
]);

const playerData = tf.tensor([
	[1, 160],
	[2, 160],
	[3, 160],
	[4, 160]
]);

jumpData.sum(1, true).concat(playerData, 1);
```


## ExpandDims

Nos permite agregar dimensiones (de a una) a nuestros tensors.

```javascript
const jumpData = tf.tensor([
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70]
]);

// Recibe un parámetro opcional para indicar el axis donde queremos que expanda la dimensión
jumpData.sum(1).expandDims(1).concat(playerData, 1);
```


## Unstack

Nos permite independizar los tensor para que sólo sean tensors los arreglos más internos dejando todo como un arreglo de JavaScript.

````javascript
const jumpData = tf.tensor([
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70],
	[70, 70, 70]
]);

jumpData.unStack()[0]; // [70, 70, 70]
```


## Práctica

```javascript
const data = tf.tensor([1, 2, 3]);
const otherData = tf.tensor([4, 5, 6]);

// Sumar elementos por columna y fila (no se altera el elemento en paréntesis)
data.add(otherData);

// Restar elementos por columna y fila
data.sub(otherData);

// Multiplicar
data.mul(otherData);

// Dividir
data.div(otherData);

// Obtener shape
data.shape;

// Imprimir datos
data.print();

// Imprimir elemento específico (index)
data.get(0);
data.get(0, 1); // para 2 dimensiones

// Slice
const data = tf.tensor([
	[10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60]
]);
// el -1 permitirá que tensorflow detecte la longitud de los elementos
data.slice([0, 1], [-1, 1]); // [[20], [50], [20], [50], [20], [50], [20], [50]]
```

Debemos tener en cuenta que cuando hacemos operaciones matemáticas entre 2 arreglos de distinto shape, los elementos que no hagan match con ningún elemento del otro arreglo darán como resultado `undefined`. Sin embargo, aquí entra un término conocido como Broadcasting que va muy de la mano con shapes.

Otra cosa a tener en cuenta es que una vez que se haya creado un tensor, ya no se pueden modificar sus valores, la única forma sería crear un nuevo tensor.


# Algoritmos

## K-Nearest Neighbor (knn)

"Birds of a feather flock together".

Este algoritmo se utiliza en el proyecto Plink, digamos que tenemos la siguiente situación: Se quiere predecir el bucket en el que caerá una pelota si es lanzada desde la posición 300px. Este algoritmo se encargará de hacer la siguiente validación:

1. Lanzar la pelota varias veces y guardar el valor del bucket en el que cae.
2. Para cada uno de estos cálculos, se resta 300px del punto de origen y se obtiene el valor absoluto, junto con el bucket final.
3. Se ordenan los elementos de menor a mayor.
4. Valida los primero K elementos y define cuál fue el bucket más común.
5. En cualquiera que sea el bucket en el que más frecuentemente se ingresó, será al que posiblemente caiga la pelota.


# Plinko (clasificación)

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


## KNN multidimensional

Debido a que no se han tenido en cuenta muchos parámetros, podemos hacer uso de el teorema de Pitágoras para hacer los cálculos. Para esto, debemos recordar la fórmula $C=\frac{A^2 + B^2}{2}$. Ahora, si necesitamos que sea en 3D nuestro teorema de Pitágoras, podemos tener la fórmula $D=\frac{A^2 + B^2 + C^2}{2}$.


## Normalización

| Posición de lanzamiento | Posición normalizada |
|--|--|
| 200 | .1 |
| 150 | 0 |
| 650 | 1 |
| 430 | .56 |

Valor mínimo 150, valor máximo 650.

| Redondez | Redondez normalizada |
|--|--|
| .55 | 1 |
| .53 | .6 |
| .53 | .6 |
| .5 | 0 |

Valor mínimo .5, valor máximo .55.


## Plan de migración a Tensorflow JS

- Aprender unas bases fundamentales sobre Tensorflow JS.
- Hacer algunos ejercicios con Tensorflow.
- Reconstruir el algoritmo KNN usando Tensorflow.
- Construir otros algoritmos con Tensorflow.


# Determinar el precio de una casa (regresión)

## Algoritmo KNN

- Encontrar la distancia entre las features y el punto de predicción.
- Ordenar del punto menor al mayor.
- Tomar los primeros $K$ registros.
- Porcentaje del valor del label para esos primeros $K$ registros.


## Datos relevantes para el problema

| Features | Labels |
|--|--|
| Longitud y latitud | Precio de la casa (en miles) |

### Ejemplos
| Longitud | Latitud | Precio de la casa (en miles) |
|--|--|--|
| 84 | 83 | 200 |
| 84.1 | 85 | 250 |
| 84.2 | 84 | 234 |
| 84.3 | 83.5 | 246 |
| 85 | 83.6 | 243 |

### Punto de predicción
| Longitud  | Latitud |
|--|--|
| 84.2 | 85.2 |

En este caso manejaremos 2 tensors, uno para los features, y otro para los labels, es decir que el primer tensor tendrá la longitud y latitud de las casas, y el segundo tensor su valor en miles.

```javascript
const tensor1 = tf.tensor([
	[84, 83],
	[84.1, 85],
	[84.2, 84],
	[84.3, 83.5],
	[85, 83.6]
]);

const tensor2 = tf.tensor([
	[200],
	[250],
	[234],
	[246],
	[243]
]);
```

Para calcular la distancia podemos usar la siguiente fórmula:

$$Distance=\sqrt{(Lat_2 - Lat_1)^2 + (Long_2 - Long_1)^2}$$

Debemos tener en cuenta que si hacemos el cálculo de la distancia y ordenamos los elementos sin haberlos conectado con nuestros labels, básicamente estos labels no harán match con los registros correspondientes, así que hasta no solucionar este problema no podemos hacer el ordenamiento.

```javascript
const features = tf.tensor([
	[-121, 47],
	[-121.2, 46.5],
	[-122, 46.4],
	[-120.9, 46.7]
]);

const labels = tf.tensor([
	[200],
	[250],
	[215],
	[240]
]);

const predictionPoint = tf.tensor([-121, 47]);
const k = 2;

// Encontrar la distancia, concatenar con nuestros labels y ordenar
features
	.sub(predictionPoint) // restar longitudes y latitudes con punto de predicción
	.pow(2) // elevar al cuadrado
	.sum(1) // sumar elementos en eje vertical
	.pow(0.5) // raíz cuadrada
	.expandDims(1) // poner el mismo shape que en nuestros labels
	.concat(labels, 1) // concatenar con labels
	.unstack() // disponer de métodos de JavaScript para arreglos
	.sort((a, b) => a.get(0) - b.get(0)) // ordenar
	.slice(0, k) // obtener primeros K elementos
	.reduce((acc, pair) => acc + pair.get(1), 0) / k; // sumar y calcular promedio
```