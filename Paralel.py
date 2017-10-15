#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import re
from time import time
from mpi4py import MPI
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()

def distancia(archivoA, archivoB):
    '''
    Este metodo calcula y devuelve la distancia entre 2 archivos.
    :param archivoA: 1er archivo para calcular distancia. (String)
    :param archivoB: 2ndo archivo para calcular distancia.(String)
    :return: dot_product / (norm_a * norm_b): es la distancia entre los 2 archivos. (entero entre 0 y 1).
    '''

    stopwords = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi",
                 "ahora", "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí",
                 "alrededor", "ambos", "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas",
                 "aproximadamente", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla",
                 "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun",
                 "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas",
                 "bueno", "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
                 "claro", "comentó", "como", "con", "conmigo", "conocer", "conseguimos", "conseguir", "considera",
                 "consideró", "consigo", "consigue", "consiguen", "consigues", "contigo", "contra", "cosas", "creo",
                 "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuenta", "cuál",
                 "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar",
                 "de", "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás",
                 "dentro", "deprisa", "desde", "despacio", "despues", "después", "detras", "detrás", "dia", "dias",
                 "dice", "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
                 "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
                 "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra",
                 "enfrente", "enseguida", "entonces", "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa",
                 "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estados", "estais", "estamos",
                 "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
                 "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron",
                 "fui", "fuimos", "g", "general", "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla",
                 "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo",
                 "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos",
                 "hicieron", "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó",
                 "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "j",
                 "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar",
                 "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante",
                 "medio", "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis",
                 "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho", "muchos", "muy",
                 "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas",
                 "ninguno", "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
                 "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro",
                 "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada", "pasado", "paìs", "peor", "pero",
                 "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
                 "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque",
                 "posible", "primer", "primera", "primero", "primeros", "principalmente", "pronto", "propia", "propias",
                 "propio", "propios", "proximo", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
                 "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
                 "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó",
                 "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se",
                 "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser", "sera", "será", "serán", "sería",
                 "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
                 "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su",
                 "supuesto", "sus", "suya", "suyas", "suyo", "sé", "sí", "sólo", "t", "tal", "tambien", "también",
                 "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis", "tenemos",
                 "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda",
                 "todas", "todavia", "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos",
                 "trabajan", "trabajar", "trabajas", "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuvo",
                 "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos", "usa",
                 "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor",
                 "vamos", "van", "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
                 "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y", "ya", "yo",
                 "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
                 "último", "últimos", "a", "a's", "able", "about", "above", "according", "accordingly", "across",
                 "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                 "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and",
                 "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart",
                 "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking",
                 "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become",
                 "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
                 "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
                 "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes",
                 "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering",
                 "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                 "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                 "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
                 "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
                 "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
                 "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
                 "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
                 "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly",
                 "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
                 "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself",
                 "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                 "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
                 "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                 "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately",
                 "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
                 "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean",
                 "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself",
                 "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
                 "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
                 "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
                 "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise",
                 "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular",
                 "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
                 "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                 "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say",
                 "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                 "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
                 "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
                 "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
                 "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                 "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them",
                 "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein",
                 "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third",
                 "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                 "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
                 "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
                 "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various",
                 "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll",
                 "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when",
                 "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                 "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom",
                 "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would",
                 "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves", "z", "zero"}

    fileA = mapaDeArreglosDocs[archivoA]
    fileB = mapaDeArreglosDocs[archivoB]

    collectionA = {}

    collectionA = defaultdict(lambda: 0)

    collectionB = {}

    collectionB = defaultdict(lambda: 0)

    totalFile = list(set(fileA + fileB))

    for i in range(len(fileA)):
        collectionA[fileA[i]] += 1

    for j in range(len(fileB)):
        collectionB[fileB[j]] += 1

    interception = list(set(set(fileA) & set(fileB)))

    countInterception = 0
    for k in range(len(interception)):
        countInterception = countInterception + collectionA[interception[k]] + collectionB[interception[k]]
    return countInterception / (len(totalFile) * 1.0)

def documentosvscentroides(centroides, documentos, sinrelacion):
    '''
    Este metodo calcula las distancias que cada documento tiene con los centroides y a partir de esto genera el mapa de
    clusters.
    :param centroides: arreglo de centroides.
    :param documentos: arreglo de documentos a comparar contra los centroides (Strings).
    :param sinrelacion: set de documentos que no tienen relacion con ningun centroide.
    :return: mapaclusters: es el mapa que contiene como clave el numero(entero) de cluster y como valor un arreglo
    de documentos (strings) que pertenecen al cluster.
    :return: sinrelacion: set de documentos que no tienen relacion con ningun centroide.
    '''
    comparaciones = {}

    comparaciones = defaultdict(lambda: [])


    for doc in documentos:
        for centroid in centroides:
            comparaciones[doc].append(distancia(doc,centroid))


    mapaclusters={}

    mapaclusters = defaultdict(lambda: [])

    for initcluster in range(0,len(centroides)):
        mapaclusters[initcluster]=[]


    for doc in documentos:
        if max(comparaciones[doc]) != 0:
            if doc in sinrelacion:
                sinrelacion.remove(doc)
            mapaclusters[comparaciones[doc].index(max(comparaciones[doc]))].append(doc)
        else:
            sinrelacion.add(doc)

    if rank == 0:
        for cent in centroides:
            mapaclusters[centroides.index(cent)].append(cent)

    return mapaclusters,sinrelacion

def recentrarcentroide (arreglocluster):
    '''
    Este metodo se encarga de basado en un arreglo que contiene un cluster sacar un nuevo centroide de ese cluster.
    :param arreglocluster:es el arreglo que contiene los documentos pertenecientes a un cluster.
    :return: centroide: nuevo centroide.
    '''
    promedios=[]

    distancias = [[1 for x in range(0,len(arreglocluster))] for y in range(0,len(arreglocluster))]
    for y in range(0,len(arreglocluster)):
        for x in range(y+1,len(arreglocluster)):
            aux1=arreglocluster[x]
            aux2=arreglocluster[y]
            distancias[x][y] = distancia(aux1,aux2)
            distancias[y][x] = distancias[x][y]
    for j in range(0, len(distancias[0])):
        temporal = np.cumsum(distancias[j])
        promedios.append((temporal[len(temporal) - 1]) / len(temporal))


    centroide = arreglocluster[promedios.index(max(promedios))]

    return centroide

def sacarMapaDeArreglos(documents):
    '''
    Este metodo se encarga de abrir los documentos y calcular su arreglo de palabras (frecuencias) poniendolos en un mapa
    de tipo key:document(string) value:Array de frecuencias.
    :param documents: arreglo de documentos (Strings) a partir de los cuales se va a calcular los arreglos de frecuencias y
    el mapa.
    :return: mapa: el mapa que contiene los arreglos de los documentos asi: key:document(string) value:Array de frecuencias.
    :return: documents: arreglo de documentos (Strings) que ya estan en el mapa.
    '''
    stopwords = {"a", "actualmente", "acuerdo", "adelante", "ademas", "además", "adrede", "afirmó", "agregó", "ahi",
                 "ahora", "ahí", "al", "algo", "alguna", "algunas", "alguno", "algunos", "algún", "alli", "allí",
                 "alrededor", "ambos", "ampleamos", "antano", "antaño", "ante", "anterior", "antes", "apenas",
                 "aproximadamente", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui", "aquél", "aquélla",
                 "aquéllas", "aquéllos", "aquí", "arriba", "arribaabajo", "aseguró", "asi", "así", "atras", "aun",
                 "aunque", "ayer", "añadió", "aún", "b", "bajo", "bastante", "bien", "breve", "buen", "buena", "buenas",
                 "bueno", "buenos", "c", "cada", "casi", "cerca", "cierta", "ciertas", "cierto", "ciertos", "cinco",
                 "claro", "comentó", "como", "con", "conmigo", "conocer", "conseguimos", "conseguir", "considera",
                 "consideró", "consigo", "consigue", "consiguen", "consigues", "contigo", "contra", "cosas", "creo",
                 "cual", "cuales", "cualquier", "cuando", "cuanta", "cuantas", "cuanto", "cuantos", "cuenta", "cuál",
                 "cuáles", "cuándo", "cuánta", "cuántas", "cuánto", "cuántos", "cómo", "d", "da", "dado", "dan", "dar",
                 "de", "debajo", "debe", "deben", "debido", "decir", "dejó", "del", "delante", "demasiado", "demás",
                 "dentro", "deprisa", "desde", "despacio", "despues", "después", "detras", "detrás", "dia", "dias",
                 "dice", "dicen", "dicho", "dieron", "diferente", "diferentes", "dijeron", "dijo", "dio", "donde",
                 "dos", "durante", "día", "días", "dónde", "e", "ejemplo", "el", "ella", "ellas", "ello", "ellos",
                 "embargo", "empleais", "emplean", "emplear", "empleas", "empleo", "en", "encima", "encuentra",
                 "enfrente", "enseguida", "entonces", "entre", "era", "eramos", "eran", "eras", "eres", "es", "esa",
                 "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estados", "estais", "estamos",
                 "estan", "estar", "estará", "estas", "este", "esto", "estos", "estoy", "estuvo", "está", "están", "ex",
                 "excepto", "existe", "existen", "explicó", "expresó", "f", "fin", "final", "fue", "fuera", "fueron",
                 "fui", "fuimos", "g", "general", "gran", "grandes", "gueno", "h", "ha", "haber", "habia", "habla",
                 "hablan", "habrá", "había", "habían", "hace", "haceis", "hacemos", "hacen", "hacer", "hacerlo",
                 "haces", "hacia", "haciendo", "hago", "han", "hasta", "hay", "haya", "he", "hecho", "hemos",
                 "hicieron", "hizo", "horas", "hoy", "hubo", "i", "igual", "incluso", "indicó", "informo", "informó",
                 "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "intento", "ir", "j",
                 "junto", "k", "l", "la", "lado", "largo", "las", "le", "lejos", "les", "llegó", "lleva", "llevar",
                 "lo", "los", "luego", "lugar", "m", "mal", "manera", "manifestó", "mas", "mayor", "me", "mediante",
                 "medio", "mejor", "mencionó", "menos", "menudo", "mi", "mia", "mias", "mientras", "mio", "mios", "mis",
                 "misma", "mismas", "mismo", "mismos", "modo", "momento", "mucha", "muchas", "mucho", "muchos", "muy",
                 "más", "mí", "mía", "mías", "mío", "míos", "n", "nada", "nadie", "ni", "ninguna", "ningunas",
                 "ninguno", "ningunos", "ningún", "no", "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro",
                 "nuestros", "nueva", "nuevas", "nuevo", "nuevos", "nunca", "o", "ocho", "os", "otra", "otras", "otro",
                 "otros", "p", "pais", "para", "parece", "parte", "partir", "pasada", "pasado", "paìs", "peor", "pero",
                 "pesar", "poca", "pocas", "poco", "pocos", "podeis", "podemos", "poder", "podria", "podriais",
                 "podriamos", "podrian", "podrias", "podrá", "podrán", "podría", "podrían", "poner", "por", "porque",
                 "posible", "primer", "primera", "primero", "primeros", "principalmente", "pronto", "propia", "propias",
                 "propio", "propios", "proximo", "próximo", "próximos", "pudo", "pueda", "puede", "pueden", "puedo",
                 "pues", "q", "qeu", "que", "quedó", "queremos", "quien", "quienes", "quiere", "quiza", "quizas",
                 "quizá", "quizás", "quién", "quiénes", "qué", "r", "raras", "realizado", "realizar", "realizó",
                 "repente", "respecto", "s", "sabe", "sabeis", "sabemos", "saben", "saber", "sabes", "salvo", "se",
                 "sea", "sean", "segun", "segunda", "segundo", "según", "seis", "ser", "sera", "será", "serán", "sería",
                 "señaló", "si", "sido", "siempre", "siendo", "siete", "sigue", "siguiente", "sin", "sino", "sobre",
                 "sois", "sola", "solamente", "solas", "solo", "solos", "somos", "son", "soy", "soyos", "su",
                 "supuesto", "sus", "suya", "suyas", "suyo", "sé", "sí", "sólo", "t", "tal", "tambien", "también",
                 "tampoco", "tan", "tanto", "tarde", "te", "temprano", "tendrá", "tendrán", "teneis", "tenemos",
                 "tener", "tenga", "tengo", "tenido", "tenía", "tercera", "ti", "tiempo", "tiene", "tienen", "toda",
                 "todas", "todavia", "todavía", "todo", "todos", "total", "trabaja", "trabajais", "trabajamos",
                 "trabajan", "trabajar", "trabajas", "trabajo", "tras", "trata", "través", "tres", "tu", "tus", "tuvo",
                 "tuya", "tuyas", "tuyo", "tuyos", "tú", "u", "ultimo", "un", "una", "unas", "uno", "unos", "usa",
                 "usais", "usamos", "usan", "usar", "usas", "uso", "usted", "ustedes", "v", "va", "vais", "valor",
                 "vamos", "van", "varias", "varios", "vaya", "veces", "ver", "verdad", "verdadera", "verdadero", "vez",
                 "vosotras", "vosotros", "voy", "vuestra", "vuestras", "vuestro", "vuestros", "w", "x", "y", "ya", "yo",
                 "z", "él", "ésa", "ésas", "ése", "ésos", "ésta", "éstas", "éste", "éstos", "última", "últimas",
                 "último", "últimos", "a", "a's", "able", "about", "above", "according", "accordingly", "across",
                 "actually", "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost",
                 "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and",
                 "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart",
                 "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking",
                 "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become",
                 "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside",
                 "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
                 "came", "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes",
                 "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering",
                 "contain", "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                 "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                 "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
                 "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
                 "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
                 "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
                 "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
                 "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly",
                 "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
                 "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself",
                 "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                 "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
                 "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                 "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "known", "knows", "l", "last", "lately",
                 "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
                 "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean",
                 "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself",
                 "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
                 "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
                 "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
                 "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise",
                 "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular",
                 "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
                 "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                 "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say",
                 "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                 "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
                 "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
                 "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
                 "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                 "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them",
                 "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein",
                 "theres", "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third",
                 "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                 "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
                 "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
                 "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various",
                 "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll",
                 "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever", "when",
                 "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "whereupon",
                 "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever", "whole", "whom",
                 "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't", "wonder", "would",
                 "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
                 "yourself", "yourselves", "z", "zero"}

    for cent in centroides:
        if cent not in documentos:
            documentos.append(cent)

    mapa = {}

    for doc in documents:
        with open(ruta + doc, 'r') as file:
            text = file.read().lower()
            file.close()
            text = re.sub('[^a-z\ \']+', " ", text)
            Arch = list(text.split())

        mapa[doc] = [word for word in Arch if word not in stopwords]


    for cent in centroides:
        if cent in documentos:
            documentos.remove(cent)

    return mapa,documentos

def separarDocumentosYObtenerCentroides(ruta,sacarcentroide,centroides):
    '''
    Este metodo se encarga de separar los documentos desde el rank 0 para darle a cada proceso un sub array de documentos
    a partir de los cuales calculara el cluster.
    :param ruta: ruta de donde se obtienen los documentos.
    :param sacarcentroide: entero que si es cero saca un random de centroides de los documentos y en caso contrario usa
    los centroides que se pasan en el parametro.
    :param centroides: arreglo de centroides que puede ser vacio en el caso de que se vayan a calular al azar.
    :return: splittedDocs: arreglo de arreglo de documentos en el que cada posicion es el sub array de docs.
    :return: centroides: son los centroides que se pasaron por parametro o los calculados al azar.
    '''
    splittedDocs = []
    if rank == 0:
        alldocuments = os.listdir(ruta)
        assert len(alldocuments) >= k
        if(sacarcentroide>0):
            centroides = alldocuments[0:k] #['Abraham Lincoln___Lincoln Letters.txt','Bret Harte___A Drift from Redwood Camp.txt','Charles Darwin___Coral Reefs.txt','Daniel Defoe___Memoirs of a Cavalier.txt'] random.sample(alldocuments, k)
        print "centroides", centroides

        for cent in centroides:
            alldocuments.remove(cent)

        if len(alldocuments)%size == 0:
            slice = len(alldocuments)//size
            temporal = 0
            for x in range(0,size):
                splittedDocs.append(alldocuments[temporal:slice+temporal])
                temporal += slice

        else:
            sobra = len(alldocuments)%size
            slice = (len(alldocuments)-sobra) / size
            temporal = 0
            for x in range(0, size):
                if (sobra>0):
                    slice += 1
                    sobra -= 1
                    splittedDocs.append(alldocuments[temporal:slice + temporal])
                    temporal +=1
                    slice -= 1
                else:
                    splittedDocs.append(alldocuments[temporal:slice + temporal])
                temporal += slice

        for cent in centroides:
            alldocuments.append(cent)

    else:
        centroides = None


    return splittedDocs,centroides

def imprimirClustersYSinRelacionesGlobales(centroides,mapaclusters,sinrelacion):
    '''
    Este metodo se encarga de imprimir el mapa que contiene los clusters y los documentos que no tienen relacion con ningun
    centroide o ningun cluster.
    :param centroides: arreglo de centroides.
    :param mapaclusters: es el mapa que contiene como clave el numero(entero) de cluster y como valor un arreglo
    de documentos (strings) que pertenecen al cluster.
    :param sinrelacion: set de los documentos que no tienen relacion con ningun otro.
    :return: sinrelacion: set de los documentos que no tienen relacion con ningun otro.
    :return: arreglodearreglodeclusters: es el mapa convertido en un arreglo de arreglos. esto se hace ya que las funciones
    gather y scatter no permiten el envio de mapas.
    '''
    arregloclusters = []

    for cent in centroides:
        arregloclusters.append(mapaclusters[centroides.index(cent)])

    arreglodearreglodeclusters = comm.gather(arregloclusters, root=0)

    solossinrelacion = comm.gather(sinrelacion, root=0)

    if rank == 0:

        supermapaclusters = {}
        supermapaclusters = defaultdict(lambda: [])

        for cent in centroides:
            for i in range(size):
                supermapaclusters[centroides.index(cent)] += arreglodearreglodeclusters[i][centroides.index(cent)]


        for i in range(size):
            sinrelacion |= solossinrelacion[i]

        print "soy el master y estos no tienen relación al pelo", sinrelacion
        print "soy el master y esto junte en el mapa",supermapaclusters

    return sinrelacion,arreglodearreglodeclusters

def separarDocumentosDelCluster(arreglo):

    splittedDocs = []
    if rank == 0:

        if len(arreglo) % size == 0:
            slice = len(arreglo) // size
            temporal = 0
            for x in range(0, size):
                splittedDocs.append(arreglo[temporal:slice + temporal])
                temporal += slice

        else:
            sobra = len(arreglo) % size
            slice = (len(arreglo) - sobra) / size
            temporal = 0
            for x in range(0, size):
                if (sobra > 0):
                    slice += 1
                    sobra -= 1
                    splittedDocs.append(arreglo[temporal:slice + temporal])
                    temporal += 1
                    slice -= 1
                else:
                    splittedDocs.append(arreglo[temporal:slice + temporal])
                temporal += slice

    return splittedDocs

def mayorPromedio(miArregloDeDocsPorCluster,docsDelCluster):
    promedioMayor = -1
    documentoMayor = ""
    for i in miArregloDeDocsPorCluster:
        suma = 0
        for j in docsDelCluster:
            suma += distancia(i,j)
        promedio = suma/len(docsDelCluster)
        if (promedio > promedioMayor):
            promedioMayor=promedio
            documentoMayor=i

    return documentoMayor,promedioMayor


splittedDocs = []
sinrelacion = set([])
centroides = []
k = 4
ruta = "/opt/GutenbergShort/txt/"

if rank ==0:
    t1 = time()

splittedDocs,centroides = separarDocumentosYObtenerCentroides(ruta,1,centroides)

documentos = comm.scatter(splittedDocs, root=0)
centroides = comm.bcast(centroides, root=0)

mapaDeArreglosDocs, documentos = sacarMapaDeArreglos(documentos)

mapaclusters,sinrelacion = documentosvscentroides(centroides,documentos,sinrelacion)

sinrelacion,arreglodearreglodeclusters = imprimirClustersYSinRelacionesGlobales(centroides,mapaclusters,sinrelacion)
sinrelacion = comm.bcast(sinrelacion, root=0)
arreglodearreglodeclusters = comm.bcast(arreglodearreglodeclusters, root=0)


# RECENTRAR


supermapaclusters = {}
supermapaclusters = defaultdict(lambda: [])

for cent in centroides:
    for i in range(size):
        supermapaclusters[centroides.index(cent)] += arreglodearreglodeclusters[i][centroides.index(cent)]

centroides = []

print supermapaclusters

for i in range(k):
    arregloDeDocumentosSeparadosPorCluster = separarDocumentosDelCluster(supermapaclusters[i])
    miArregloDeDocsPorCluster = comm.scatter(arregloDeDocumentosSeparadosPorCluster, root=0)
    mapaDeArreglosDocs, documentos = sacarMapaDeArreglos(supermapaclusters[i])
    documentoConMayorPromedioPorSubCluster, numeroDelMayorPromedio = mayorPromedio(miArregloDeDocsPorCluster,supermapaclusters[i])
    arregloDeMayoresPromedios = comm.gather(numeroDelMayorPromedio, root=0)
    arregloDeDocsPromedios = comm.gather(documentoConMayorPromedioPorSubCluster, root=0)
    if rank == 0:
        nuevoCentroide = arregloDeDocsPromedios[arregloDeMayoresPromedios.index(max(arregloDeMayoresPromedios))]
        centroides.append(nuevoCentroide)

centroides = comm.bcast(centroides, root = 0)


# REPITO

sinrelacion = set([])

splittedDocs,centroides = separarDocumentosYObtenerCentroides(ruta,0,centroides)
documentos = comm.scatter(splittedDocs, root=0)
centroides = comm.bcast(centroides, root=0)


mapaDeArreglosDocs, documentos = sacarMapaDeArreglos(documentos)

mapaclusters,sinrelacion = documentosvscentroides(centroides,documentos,sinrelacion)

sinrelacion,arreglodearreglodeclusters = imprimirClustersYSinRelacionesGlobales(centroides,mapaclusters,sinrelacion)

if rank ==0:
    t2 = time()
    print "time ", t2 - t1
    print "centroides: ", centroides
