from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

def pickleJar():
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    if not os.path.exists("face_detector"):
        os.makedirs("face_detector")
    if not os.path.exists("output/embeddings.pickle"):
        f = open("output/embeddings.pickle", "wb")
        f.close()
    if not os.path.exists("output/recognizer.pickle"):
        f = open("output/recognizer.pickle", "wb")
        f.close()
    if not os.path.exists("output/le.pickle"):
        f = open("output/le.pickle", "wb")
        f.close()
pickleJar()
BASE_DIR = os.path.dirname(__file__)
print("[INFO] BASE DIR: ", BASE_DIR)
print("[INFO] Loading face embeddings...")
data = pickle.loads(open((os.path.join(BASE_DIR, 'output/embeddings.pickle')), "rb").read())
print("[INFO] Encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
print("[INFO] Training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)
f = open((os.path.join(BASE_DIR, 'output/recognizer.pickle')), "wb")
f.write(pickle.dumps(recognizer))
f.close()
f = open((os.path.join(BASE_DIR, 'output/le.pickle')), "wb")
f.write(pickle.dumps(le))
f.close()
