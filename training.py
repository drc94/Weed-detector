import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from sklearn import svm

import cv2
import glob
import sys

PATCH_SIZE = 100

# load the image, convert it to grayscale, blur it slightly, and threshold it
images = [cv2.imread(file) for file in glob.glob("Images/*.jpeg")]

i = int(sys.argv[1])
scale_percent = 40
width = int(images[i].shape[1]*scale_percent/100)
height = int(images[i].shape[0]*scale_percent/100)
dim = (width, height)
resize = cv2.resize(images[i], dim, interpolation = cv2.INTER_AREA)
image = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

# select some patches from weed areas of the image
weed_locations = [(200, 150), (300, 150), (300, 240), (400, 236)]
weed_patches = []
for loc in weed_locations:
    weed_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                               loc[1]:loc[1] + PATCH_SIZE])

# select some patches from ground areas of the image
ground_locations = [(10, 10), (540, 220), (200, 380), (540, 200)]
ground_patches = []
for loc in ground_locations:
    ground_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# validation patches
training_locations = [(220, 170), (15, 50), (500, 150), (320, 210)]
training_patches = []
for loc in training_locations:
    training_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

# compute some GLCM properties each patch
xs = []
ys = []
for patch in (weed_patches + ground_patches + training_patches):
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

# create the figure
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(image, cmap=plt.cm.gray,
          vmin=0, vmax=255)
for (y, x) in weed_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
for (y, x) in ground_locations:
    ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(weed_patches)], ys[:len(weed_patches)], 'go',
        label='Weed')
ax.plot(xs[len(weed_patches):], ys[len(weed_patches):], 'bo',
        label='Ground')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(weed_patches):
    ax = fig.add_subplot(3, len(weed_patches), len(weed_patches)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Weed %d' % (i + 1))

for i, patch in enumerate(ground_patches):
    ax = fig.add_subplot(3, len(ground_patches), len(ground_patches)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Ground %d' % (i + 1))


# display the patches and plot
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

X = [[xs[0], ys[0]], [xs[1], ys[1]], [xs[2], ys[2]], [xs[3], ys[3]], [xs[4], ys[4]], [xs[5], ys[5]], [xs[6], ys[6]], [xs[7], ys[7]]]
y = [0, 0, 0, 0, 1, 1, 1, 1]
clf = svm.SVC()
print(clf.fit(X, y))
print(clf.predict([[xs[8], ys[8]], [xs[9], ys[9]], [xs[10], ys[10]], [xs[11], ys[11]]]))
