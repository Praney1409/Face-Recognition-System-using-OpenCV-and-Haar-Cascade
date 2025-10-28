import streamlit as st
import cv2
import numpy as np
import os

# ---------------- PATHS ----------------
dataset_path = "dataset"
trainer_path = "trainer.yml"
labels_path = "labels.npy"

os.makedirs(dataset_path, exist_ok=True)

# Haarcascade & Recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ---------------- IMAGE ENHANCEMENT ----------------
def enhance_image(frame):
    # Convert to LAB and equalize histogram (improves low light)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced

# ---------------- CAPTURE FACES ----------------
def capture_faces(name):
    person_dir = os.path.join(dataset_path, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    st.info(f"📸 Capturing faces for {name}. Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible!")
            break

        frame = enhance_image(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Improved detection parameters for distance faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            count += 1
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{name}_{count}.jpg"), face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success(f"✅ Captured {count} images for {name}.")

# ---------------- TRAIN MODEL ----------------
def train_model():
    faces, labels = [], []
    label_dict = {}
    current_label = 0

    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            if person not in label_dict:
                label_dict[person] = current_label
                current_label += 1
            faces.append(img)
            labels.append(label_dict[person])

    if len(faces) == 0:
        st.warning("⚠ No faces found. Add persons first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)
    np.save(labels_path, label_dict)
    st.success("✅ Model trained successfully!")

# ---------------- RECOGNIZE FACES ----------------
def recognize_faces():
    if not os.path.exists(trainer_path):
        st.error("⚠ No trained model found! Train it first.")
        return

    recognizer.read(trainer_path)
    label_dict = np.load(labels_path, allow_pickle=True).item()
    reverse_dict = {v: k for k, v in label_dict.items()}

    cap = cv2.VideoCapture(0)
    st.info("🔍 Recognition started — Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = enhance_image(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label, confidence = recognizer.predict(face_img)

            if confidence < 80:
                name = reverse_dict[label]
                text = f"{name} ({int(confidence)}%)"
                color = (0, 255, 0)
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    st.success("✅ Recognition ended.")

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Efficient Face Recognition", layout="centered")
st.title("🧠 Advanced OpenCV Face Recognition")

menu = st.sidebar.radio("Select Option", ["Add New Person", "Train Model", "Recognize Faces", "View Dataset"])

if menu == "Add New Person":
    name = st.text_input("Enter person's name:")
    if st.button("Capture Faces"):
        if name.strip():
            capture_faces(name.strip())
            st.info("Training model after adding new data...")
            train_model()
        else:
            st.warning("⚠ Please enter a valid name.")

elif menu == "Train Model":
    if st.button("Train Now"):
        train_model()

elif menu == "Recognize Faces":
    if st.button("Start Recognition"):
        recognize_faces()

elif menu == "View Dataset":
    if os.path.exists(dataset_path):
        people = os.listdir(dataset_path)
        if people:
            st.subheader("📂 Saved Persons:")
            for person in people:
                person_dir = os.path.join(dataset_path, person)
                imgs = [os.path.join(person_dir, img) for img in os.listdir(person_dir)[:3]]
                st.write(f"👤 {person} ({len(os.listdir(person_dir))} images)")
                st.image(imgs, width=150)
        else:
            st.warning("⚠ Dataset empty.")
