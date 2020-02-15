from pathlib import Path

import cv2

# Load a trained classifier to find faces
wd = Path('/home/ludwik/workspace/trials/opencv')
trained_classifier_path = wd / 'haarcascade_frontalface_default.xml'
# print(trained_classifier_path.exists())
trained_classifier = cv2.CascadeClassifier(str(trained_classifier_path))

# Load an image
image_path = wd / 'Magnus2.jpeg'
image = cv2.imread(str(image_path))
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print(image_grayscale)

# Find faces
faces = trained_classifier.detectMultiScale(image_grayscale, 1.1, 4)
# print(faces)


for (x, y, width, height) in faces:
    cv2.rectangle(img=image,
                  pt1=(x, y),
                  pt2=(x + width, y + height),
                  color=(255, 255, 255),
                  thickness=1
                 )

# Display the output
cv2.imshow('img', image)
cv2.waitKey()
