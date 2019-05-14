import cv2
import numpy as np
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
    plt.imshow(img2)
    plt.show()

    return kp, des

# Crea el histograma para un grupo de descriptores
def build_histogram(descriptor_list, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    kmeans_result =  kmeans.predict(descriptor_list)
    for i in kmeans_result:
        histogram[i] += 1.0
    return histogram

# Combierte ua imagen a escala de grises
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#BoW
k = 800;

images_paths = [
    'gsxs1.jpg',
    'gsxs2.jpg',
    'gsxs3.jpg',
    'gsxs4.jpg',
    'gsxs5.jpg',
    'gsxs6.jpg',
    'gsxs7.jpg'
]

images = []
for path in images_paths:
    img = cv2.imread('img/gsxs150/' + path)
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
Nc = range(1, k)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(descriptor_list).score(descriptor_list) for i in range(len(kmeans))]
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

# Creamos los clusters
kmeans = KMeans(n_clusters = k).fit(descriptor_list)

histogram = build_histogram(descriptor_list, kmeans)

subplot(211)
subplot(212)
plt.bar(list(range(kmeans.cluster_centers_)), histogram)

plt.show()

# data = cv2.imread('img/gsxs-calle')
# data = gray(data)
# keypoint, descriptor = features(data, extractor)
# histogram = build_histogram(descriptor, kmeans)
# neighbor = NearestNeighbors(n_neighbors = 20)
# neighbor.fit(preprocess_image)
# dist, result = neighbor.kneighbors([histogram])
