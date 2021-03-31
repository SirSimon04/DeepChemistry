import numpy as np
import os
import cv2
import random
import pickle


y = []


def create_training_data():
    DATADIR = "trainingsdaten"
    CATEGORIES = ["2-Methylbutan", "2-Methylhexan", "2-Methylpentan", "2-Methylpropan", "3-Ethylpentan", "3-Methylhexan",
                  "3-Methylpentan", "22-Dimethylbutan", "22-Dimethylpentan", "22-Dimethylpropan", "23-Dimethylbutan",
                  "23-Dimethylpentan", "24-Dimethylpentan", "33-Dimethylpentan", "223-Trimethylbutan", "Butan", "Ethan",
                  "Heptan", "Hexan", "Pentan", "Propan"]
    IMG_SIZE = 64
    training_data = []
    X = []
    yn = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)
    for features, label in training_data:
        X.append(features)
        yn.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    yn = np.array(yn).reshape(-1)
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open("y.pickle", "wb")
    pickle.dump(yn, pickle_out)
    pickle_out.close()
    print("Dateien generiert")
    print(len(X), len(yn))
    print(X.shape[1:])




if __name__ == '__main__':
    create_training_data()



