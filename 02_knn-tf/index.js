require("@tensorflow/tfjs-node");

const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
    return features
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
}

let { features, labels, testFeatures, testLabels } = loadCSV("kc_house_data.csv", {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"]
});

features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

console.log("Guess", result, testLabels[0][0]);