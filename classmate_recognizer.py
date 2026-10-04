"""
Webcam face recognition with ONE reference photo per person.

Setup:
    pip install insightface onnxruntime opencv-python numpy

Folder layout:
    known_faces/
        Ali.jpg
        Leyla.jpg
        ...            (file name = person's name, one clear frontal photo each)

Run:
    python classmate_recognizer.py
Press "q" to quit.
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

KNOWN_DIR = "known_faces"
THRESHOLD = 0.40  # cosine similarity; raise to be stricter (fewer false matches)

# buffalo_l = RetinaFace detector + ArcFace embeddings (good one-shot accuracy)
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def load_known(folder):
    names, embeddings = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = cv2.imread(os.path.join(folder, fname))
        faces = app.get(img)
        if not faces:
            print(f"[warn] no face found in {fname}, skipping")
            continue
        if len(faces) != 1:
            print(
                f"[warn] expected exactly one face in {fname}, "
                f"found {len(faces)}, skipping"
            )
            continue

        face = faces[0]
        # if several faces, use the biggest one
        # face = max(
        #     faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        # )
        names.append(os.path.splitext(fname)[0])
        embeddings.append(face.normed_embedding)
    return names, np.array(embeddings)


names, known = load_known(KNOWN_DIR)
if len(known) == 0:
    raise RuntimeError(
        f"No valid reference faces found in '{KNOWN_DIR}'. "
        "Add at least one valid image containing exactly one face."
    )
print(f"Loaded {len(names)} people: {names}")

cap = cv2.VideoCapture(0)
frame_i = 0
results = []  # cache so we don't run the model on every frame

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # run recognition every 3rd frame for speed
    if frame_i % 3 == 0:
        results = []
        for face in app.get(frame):
            sims = (
                known @ face.normed_embedding
            )  # cosine similarity (embeddings are normalized)
            best = int(np.argmax(sims))
            label = names[best] if sims[best] >= THRESHOLD else "Unknown"
            results.append((face.bbox.astype(int), label, float(sims[best])))
    frame_i += 1

    for (x1, y1, x2, y2), label, score in results:
        color = (0, 200, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imshow("Classmates", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
