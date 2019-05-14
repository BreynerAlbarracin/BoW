import cv2
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

# Permite obtener los descriptores de la imagen
def features(image, extractor):
    img = gray(image)
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)

    img2 = image
    cv2.drawKeypoints(img2 ,kp, img2, color=(0,255,0), flags=0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    print('Cantidad de descriptores por imagen')
    print(len(des))
    if(showIMG):
        plt.imshow(img2)
        plt.show()

    return kp, des

# Crea el histograma para un grupo de descriptores
def build_histogram(descriptor_list, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    kmeans_result =  kmeans.predict(descriptor_list)
    for i in kmeans_result:
        histogram[i] += 1.0
    histogram = histogram / len(kmeans.cluster_centers_)
    return histogram

# Combierte ua imagen a escala de grises
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#BoW
showIMG = False
showCodo = False
k = 20

if len(sys.argv) > 3:
    if(sys.argv[1] == 'true'):
        showIMG = True

    if(sys.argv[2] == 'true'):
        showCodo = True

    if(sys.argv[3] != 'none'):
        k = int(sys.argv[3])

root = os.getcwd() + '/img/'
dirs = os.listdir(root)

images_paths = []
for path in dirs:
    images_paths.append(root + path + '/')

neighbors = []
for path in images_paths:
    images = []
    if(path.find('test') < 0):
        for file in os.listdir(path):
            img = cv2.imread(path + file)
            if(img is not None):
                images.append(img)

        print('Cantidad de imagenes')
        print(len(images))

        # Obtenemos los descriptores para cada imagen en el arreglo determinado
        orb = cv2.ORB_create()
        descriptor_list = []
        for image in images:
            keypoint, descriptors = features(image, orb)
            for descriptor in descriptors:
                descriptor_list.append(descriptor)

        print('Cantidad de descriptores acumulados')
        print(len(descriptor_list))

        # Calcular K por medoto Codo
        if(showCodo):
            print('Computando grafica para Metodo Codo')
            Nc = range(1, k)
            kmeans = [KMeans(n_clusters=i) for i in Nc]
            score = [kmeans[i].fit(descriptor_list).score(descriptor_list) for i in range(len(kmeans))]
            plt.plot(Nc,score)
            plt.xlabel('Number of Clusters')
            plt.ylabel('Score')
            plt.title('Elbow Curve')
            plt.show()

        # Creamos los clusters
        print('Creando kmeans con ' + str(k) + ' clusters')
        kmeans = KMeans(n_clusters = k).fit(descriptor_list)
        centroids = kmeans.cluster_centers_
        histogram = build_histogram(descriptor_list, kmeans)

        plt.subplot(121)
        plt.plot(centroids, 'ro')

        plt.subplot(122)
        plt.bar(list(range(len(centroids))), histogram)

        plt.show()

        neighbors.append(histogram)

print('Prueba con Test1')
data = cv2.imread('img/test/test2.jpg')
keypoint, descriptors = features(data, orb)
histogram2 = build_histogram(descriptors, kmeans)

neighbor = NearestNeighbors(n_neighbors = len(neighbors))
neighbor.fit(neighbors)
dist, result = neighbor.kneighbors([histogram2])

print(dist)
print(result)
