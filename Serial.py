#!/usr/bin/env python
# -*- coding: utf-8 -*-
from time import time
import re
from collections import defaultdict
import os
import numpy as np
import random

aloneDocuments = set({})

#Este metodo lee de disco y trae a memoria todos los documentos como arreglos
def archivos(filelist):
    '''
    Este metodo lee de disco y trae a memoria todos los documentos como arreglos
    :param filelist: arreglo con los nombre de los archivos
    :return: Arreglo que contiene un arreglo de las palabras de cada archivo
    '''

    # Arreglo de palabas a ignorar
    stopwords = set(
        ["a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi", "ahora",
         "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí", "alrededor", "ambos",
         "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas", "aproximadamente", "aquel", "aquella",
         "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla", "aquéllas", "aquéllos", "aquí", "arriba",
         "arribaabajo", "aseguró", "asi", "así", "atras", "aun", "aunque", "ayer", "añadió", "aún", "b", "bajo",
         "bastante", "bien", "breve", "buen", "buena", "buenas", "bueno", "buenos", "c", "cada", "casi", "cerca",
         "cierta", "ciertas", "cierto", "ciertos", "cinco", "claro", "comentó", "como", "con", "conmigo", "conocer",
         "conseguimos", "conseguir", "considera", "consideró", "consigo", "consigue", "consiguen", "consigues",
         "contigo", "contra", "cosas", "creo", "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto",
         "cuantos", "cuenta", "cuál", "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da",
         "dado", "dan", "dar", "de", "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante",
         "demasiado", "demás", "dentro", "deprisa", "desde", "despacio", "despues", "después", "detras", "detrás",
         "dia", "dias", "dice", "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio",
         "donde", "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
         "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra", "enfrente",
         "enseguida", "entonces", "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso",
         "esos", "esta", "estaba", "estaban", "estado", "estados", "estais", "estamos", "estan", "estar", "estará",
         "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex", "excepto", "existe", "existen",
         "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron", "fui", "fuimos", "g", "general", "gran",
         "grandes", "gueno", "h", "ha", "haber", "habia", "habla", "hablan", "habrá", "había", "habían", "hace",
         "haceis", "hacemos", "hacen", "hacer", "hacerlo", "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay",
         "haya", "he", "hecho", "hemos", "hicieron", "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó",
         "informo", "informó", "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "intento",
         "ir", "j", "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar",
         "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante", "medio",
         "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis", "misma",
         "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho", "muchos", "muy", "más", "mí",
         "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas", "ninguno", "ningunos",
         "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "nueva", "nuevas",
         "nuevo", "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro", "otros", "p", "pais", "para", "parece",
         "parte", "partir", "pasada", "pasado", "paìs", "peor", "pero", "pesar", "poca", "pocas", "poco", "pocos",
         "podeis", "podemos", "poder", "podria", "podriais", "podriamos", "podrian", "podrias", "podrá", "podrán",
         "podría", "podrían", "poner", "por", "porque", "posible", "primer", "primera", "primero", "primeros",
         "principalmente", "pronto", "propia", "propias", "propio", "propios", "proximo", "próximo", "próximos", "pudo",
         "pueda", "puede", "pueden", "puedo", "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes",
         "quiere", "quiza", "quizas", "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado",
         "realizar", "realizó", "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes",
         "salvo", "se", "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser", "sera", "será", "serán",
         "sería", "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
         "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su", "supuesto", "sus",
         "suya", "suyas", "suyo", "sé", "sí", "sólo", "t", "tal", "tambien", "también", "tampoco", "tan", "tanto",
         "tarde", "te", "temprano", "tendrá", "tendrán", "teneis", "tenemos", "tener", "tenga", "tengo", "tenido",
         "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda", "todas", "todavia", "todavía", "todo", "todos",
         "total", "trabaja", "trabajais", "trabajamos", "trabajan", "trabajar", "trabajas", "trabajo", "tras", "trata",
         "través", "tres", "tu", "tus", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una",
         "unas", "uno", "unos", "usa", "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va",
         "vais", "valor", "vamos", "van", "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera",
         "verdadero", "vez", "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y",
         "ya", "yo", "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
         "último", "últimos", "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually",
         "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
         "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody",
         "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate",
         "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated", "at", "available",
         "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before",
         "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between",
         "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can", "can't", "cannot", "cant", "cause",
         "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning",
         "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could",
         "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't", "different",
         "do", "does", "doesn't", "doing", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg",
         "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever",
         "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far",
         "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
         "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes",
         "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't",
         "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here", "here's", "hereafter",
         "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how",
         "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch",
         "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is",
         "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know",
         "known", "knows", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
         "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may",
         "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must",
         "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither",
         "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
         "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old",
         "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours",
         "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps",
         "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r",
         "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
         "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see",
         "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious",
         "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six", "so", "some",
         "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon",
         "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take",
         "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their",
         "theirs", "them", "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore",
         "therein", "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third",
         "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
         "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two",
         "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used",
         "useful", "uses", "using", "usually", "uucp", "v", "value", "various", "very", "via", "viz", "vs", "w", "want",
         "wants", "was", "wasn't", "way", "we", "we'd", "we'll", "we're", "we've", "welcome", "well", "went", "were",
         "weren't", "what", "what's", "whatever", "when", "whence", "whenever", "where", "where's", "whereafter",
         "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who",
         "who's", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without",
         "won't", "wonder", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've",
         "your", "yours", "yourself", "yourselves", "z", "zero"])
    allFiles = defaultdict(lambda: [])
    for i in filelist:
        with open(datasetLocation + i, 'r') as f:
            text = f.read().lower()
            f.close()
            text = re.sub('[^a-z\ \']+', " ", text)
            file = list(text.split())
        allFiles[i] = [word for word in file if word not in stopwords]
    return allFiles


#Este metodo calcula la distancia entre 2 documentos usando el algoritmo de Jaccard
def jaccardAlgorithm(fileA,fileB):

    collectionA = defaultdict(lambda: 0)
    collectionB = defaultdict(lambda: 0)

    totalFile = list(set(fileA + fileB))

    for i in range(len(fileA)):
        collectionA[fileA[i]]+=1

    for j in range(len(fileB)):
        collectionB[fileB[j]]+=1

    interception = list(set(set(fileA) & set(fileB)))

    countInterception = 0
    for k in range(len(interception)):
        countInterception = countInterception + collectionA[interception[k]] + collectionB[interception[k]]
    return countInterception / (len(totalFile) * 1.0)


#Este metodo conforma los clusters
def clustering(centroids, allFiles, listDocuments):
    # En este diccionario la clave es el nombre del archivo y su valor es un arreglo donde se encuentra el indice de similitud con cada centroide
    comparisons = {}

    comparisons = defaultdict(lambda: [], comparisons)

    # Este loop remueve los centroides de la lista para no ser comparados
    for centroid in centroids:
        listDocuments.remove(centroid)

    # Este loop itera sobre los documentos y los compara contra cada centroide.
    for doc in listDocuments:
        wordsDoc = allFiles[doc]
        for centroid in centroids:
            wordsCentroids = allFiles[centroid]
            index = jaccardAlgorithm(wordsDoc,wordsCentroids)
            comparisons[doc].append(index)
    # La clave de este diccionario va de 0 a k, y su valor es un arreglo de nombres de los archivos que pertenecen a este cluster
    clustersDictionary = {}

    clustersDictionary = defaultdict(lambda: [], clustersDictionary)
    # Este loop inicializa los valores del diccionario con un arreglo vacio.
    for initCluster in range(0, len(centroids)):
        clustersDictionary[initCluster] = []

    for doc in listDocuments:
        if max(comparisons[doc])>0:
            if doc in aloneDocuments:
                aloneDocuments.remove(doc)
            maximum = max(comparisons[doc])
            i = comparisons[doc].index(maximum)
            clustersDictionary[i].append(doc)
        else:
            aloneDocuments.add(doc)

    for centroid in centroids:
        clustersDictionary[centroids.index(centroid)].append(centroid)
        listDocuments.append(centroid)

    return clustersDictionary


#Este metodo redefine los centroides de cada cluster
def redefineCentroids(clustersDictionary,centroids, allFiles):
    docAverages = {}
    docAverages = defaultdict(lambda: [], docAverages)
    cumSum = []

    for centroid in centroids:
        centroidIndex = centroids.index(centroid)

        distanceMatrix = [[1 for x in range(0, len(clustersDictionary[centroidIndex]))] for y in range(0, len(clustersDictionary[centroidIndex]))]

        for y in range(0, len(clustersDictionary[centroidIndex])):

            for x in range(y + 1, len(clustersDictionary[centroidIndex])):
                nameDoc1 = clustersDictionary[centroidIndex][x]
                nameDoc2 = clustersDictionary[centroidIndex][y]

                wordsDoc1 = allFiles[nameDoc1]
                wordsDoc2 = allFiles[nameDoc2]

                distanceMatrix[x][y] = jaccardAlgorithm(wordsDoc1, wordsDoc2)
                distanceMatrix[y][x] = distanceMatrix[x][y]

        for j in range(0, len(distanceMatrix[0])):

            cumSum = np.cumsum(distanceMatrix[j])

            docAverages[centroidIndex].append((cumSum[len(cumSum) - 1]) / len(cumSum))

    for centroid in centroids:
        index = centroids.index(centroid)
        centroids[index] = clustersDictionary[index][docAverages[index].index(max(docAverages[index]))]

    return centroids

t0 = time()
k = 4
datasetLocation = "/opt/GutenbergP/txt/"
listDocuments = os.listdir(datasetLocation)
assert len(listDocuments) >= k
#centroids = random.sample(listDocuments, k)
#centroids = ["Abraham Lincoln___The Emancipation Proclamation.txt","Alfred Russel Wallace___The Malay Archipelago, Volume 2.txt","Ambrose Bierce___Cobwebs From an Empty Skull.txt", "Andrew Lang___Rhymes a la Mode.txt"]
centroids = listDocuments[0:k]
allfiles = archivos(listDocuments)
clusters = clustering(centroids, allfiles,listDocuments)
centroids = redefineCentroids(clusters, centroids,allfiles)
clusters = clustering(centroids, allfiles,listDocuments)
t1=time()

print "Clusters:", clusters
print "Centroids:",centroids
print "Documentos with no relation:",aloneDocuments
print"Time:", t1-t0
