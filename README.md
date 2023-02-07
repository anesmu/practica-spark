# Resolución de entidades con Spark

## Enunciado de la práctica

Se pide implementar un algoritmo de resolución de entidades utilizando PySpark siguiendo los
pasos descritos en el aparatado “¿Cómo llevar a cabo la resolución de entidades?”. El
objetivo es averiguar sistemáticamente cuáles de los registros se refieren en realidad la misma
entidad.

Los pasos a dar son los siguientes:
### Normalización de fuentes
* Unificar los esquemas de datasets descargados creando un solo conjunto de datos.
* Limpiar los datos estandarizando el tipo de datos (por ejemplo, asegurándose de que los
nulos sean realmente nulos y no la cadena nula), unificando el formato (por ejemplo, todo en
minúsculas) y eliminando caracteres especiales (por ejemplo, corchetes y paréntesis).

### Creación de funciones para la caracterización y generación de claves de bloque 
Elegir una buena clave de bloque puede mejorar drásticamente la eficiencia del proceso de
resolución de entidades. No vale la pena evaluar todos los pares de candidatos, por ejemplo,
existen productos que solo observando su nombre claramente no son lo mismo y no deben
comprarse. El objetivo de este apartado es obtener una clave de bloque específica para crear
pares de candidatos. Si por ejemplo se utilizara el nombre de cadena exacto del producto solo
los productos que tengan exactamente la misma cadena de nombre normalizada se incluirán en
los pares de candidatos, esto es claramente demasiado restrictivo y se perderían muchas
coincidencias potenciales.

Existen muchas formas de caracterizar los datos y obtener así unas buenas claves. Para los
datos de texto hay una enorme variedad de algoritmos:
* Tokenización: Romper frases en palabras
* Estandarización de los tokens: Stemming o lematización.
* Algoritmos de frecuencia: TF-IDF.
* Algoritmos basados en funciones hash: Locality Sensitive Hashing
* Embeddings: Word embeddings, sentence embeddings

En este apartado es importante ser creativo con las funciones de transformación, de tal forma
que se obtenga un conjunto de pares candidatos que contengan productos que sean similares
por sus características.

### Generar pares de candidatos
La generación de pares de candidatos es una parte bastante sencilla de la resolución de
entidades, esencialmente es una unión por las claves de bloque. Se pide utilizar el paquete
GraphFrames para la generación de un grafo que contenga todos los pares candidatos obtenidos
a partir de las claves de bloque.

### Puntuar los pares de candidatos
Puntuar los pares candidatos, devueltos por el grafo, es crucial para eliminar aquellos no
coincidentes y crear las entidades finales. Este paso es bastante abierto y uno puede ser muy
creativo sobre las funciones y características específicas de puntuación a implementar. Algunas
métricas clásicas que se pueden utilizar en este paso son:
* La distancia de coseno: Típica cuando has convertido tus cadenas en vectores.
* La distancia de Levenshtein.
* La proximidad de los valores numéricos.
* 
Para puntuar los pares de candidatos también es posible utilizar un algoritmo de SparkML y
entrenarlo con las características deseadas, utilizando para el entrenamiento los datos del fichero
Amzon_GoogleProducts_perfectMapping.csv.

### Generar entidades

Dada la métrica resultante del apartado anterior, utilizar un criterio objetivo para decidir el umbral
qué hace que un par de candidatos se conviertan en la misma entidad y comparar los resultados
con los datos del fichero Amzon_GoogleProducts_perfectMapping.csv.