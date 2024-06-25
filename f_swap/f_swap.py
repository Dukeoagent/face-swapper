import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import glob
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640,640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx',download=False,download_zip=False)

img1_path = 'johncena.jpg'
himymLocation = 'himym.jpg'
maxlocation = 'maxverstappen.jpg'

# Load the image using OpenCV
img = cv.imread(img1_path)
himym = cv.imread(himymLocation)
img3 = cv.imread(maxlocation)

# Convert the image from BGR to RGB
#OpenCV loads the image in BGR format, convert to RGB for matplotlib
johncena = cv.cvtColor(img, cv.COLOR_BGR2RGB)

max = cv.cvtColor(img3, cv.COLOR_BGR2RGB)

# Detected faces
john_face = app.get(johncena)
faces = app.get(himym)
max_face = app.get(max)
# len(john_face)
# max.shape

assert len(max_face) == 1
max_face = max_face[0]
bbox = max_face['bbox']
bbox = [int(b) for b in bbox]
# plt.imshow(max[bbox[1]:bbox[3],bbox[0]:bbox[2]])
# plt.show()

assert len(john_face) == 1
john_face = john_face[0]
bbox = john_face['bbox']
bbox = [int(b) for b in bbox]
# plt.imshow(johncena[bbox[1]:bbox[3],bbox[0]:bbox[2]])
# plt.show()

# Sorts left to right
faces = sorted(faces, key = lambda x : x.bbox[0])
res = img.copy()
assert len(faces)==5 # Confirm 5 faces found
# source_face = max_face[0]     # max_face is dict 
source_face = max_face


bbox = source_face['bbox']
bbox = [int(b) for b in bbox]
# plt.imshow(max[bbox[1]:bbox[3],bbox[0]:bbox[2]])
# plt.show()

# for swapping all faces
res = himym
for face in faces:
   res = swapper.get(res, face, john_face, paste_back=True)

## for swapping one character
# res = swapper.get(img3,source_face, john_face)

# plt.imshow(res)
# plt.axis('off')
# plt.show()

# saving the swapped image
output_path = 'swapped_image.jpg'
cv.imwrite(output_path, res)
print(f"Output image saved to {output_path}")




# CHANGES
# -> If the OpenCV version is >= 4.5.1, do not convert the image from RGB to BGR at the end. This version supports RGB images as well.
# -> Your code attempts to redefine the variable 'swapper' as a FaceAnalysis object, causing errors in face swapping. Instead, initialize the swapper model separately and use its 'get' method.
# -> The method 'swapper.get()' should be applied to the model, not to an instance of FaceAnalysis().
# -> Do not define source_face = max_face[0] because 'max_face' is a dictionary, not a list. Instead, set source_face = max_face
# -> The correct arguments for the final face swapping function should be accepted as 'swapper.get(source_img, source_face_to_be_swapped, target_face_required)'.
# -> In the code, you are using three images for swapping: 'swapper.get(res, face, source_face, paste_back=True)'. 'res' refers to John's image, 'face' refers to each character's face from HIMYM, and 'source_face' refers to 'max_face'. Ensure that 'res' and 'face' contain corresponding faces for the swapping to succeed.
# -> The first argument in 'swapper.get(res, face, source_face, paste_back=True)' should be the source image where the changes need to be applied.
# -> Even if you haven't made all the aforementioned errors, 'swapper.get(res, face, source_face, paste_back=True)' won't perform the swapping because 'res' and 'face' do not contain matching faces.