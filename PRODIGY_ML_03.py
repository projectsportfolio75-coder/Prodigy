import os
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 64
CATEGORIES = ['Cat', 'Dog']
DATA_DIR = 'dogs-vs-cats'

def load_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_flattened = img.flatten()  # Flatten for SVM
                data.append([img_flattened, class_num])
            except Exception as e:
                pass
    return data

dataset = load_data()
np.random.shuffle(dataset)
#print(dataset)

x = np.array([features for features, labels in dataset])
y = np.array([labels for features, labels in dataset])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train / 255.0
x_test = x_test / 255.0
svm = SVC(kernel='rbf', probability=True)
svm.fit(x_train, y_train)
import joblib
joblib.dump(svm, "svm_cat_dog_model.pkl")
y_predict = svm.predict(x_test)
print('Accuracy Score:', accuracy_score(y_test, y_predict))
print('\nClassification Report:\n', classification_report(y_test, y_predict, target_names=CATEGORIES))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_predict))


plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img.flatten().reshape(1, -1)
    prediction = svm.predict(img_flat)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {CATEGORIES[prediction[0]]}")
    plt.axis('off')
    plt.show()

predict_image("TestDog.jpg")
