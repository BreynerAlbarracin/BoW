import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

images = [cv2.imread('img/gsxs-blanco.png'), cv2.imread('img/gsxs-blanco2.jpg')]

extractor = cv2.xfeatures2d.SIFT_create()

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

kmeans = KMeans(n_clusters = 800)

preprocessed_image = []
for image in images:
    image = gray(image)
    keypoint, descriptor = features(image, extractor)
    if (descriptor is not None):
        histogram = build_histogram(descriptor, kmeans)
        preprocessed_image.append(histogram)

data = cv2.imread('img/gsxs-calle')
data = gray(data)
keypoint, descriptor = features(data, extractor)
histogram = build_histogram(descriptor, kmeans)
neighbor = NearestNeighbors(n_neighbors = 20)
neighbor.fit(preprocess_image)
dist, result = neighbor.kneighbors([histogram])
