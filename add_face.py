from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import os

# Initialize models
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Prompt for name
name = input("Enter your name: ").strip()
if not name:
    print("Name cannot be empty.")
    exit()

# Create embeddings folder if it doesn't exist
os.makedirs('embeddings', exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(1)
print("[INFO] Press 's' to save when your face is clearly visible.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)

    # Draw box if a face is detected
    display_frame = frame.copy()
    if face_tensor is not None:
        cv2.putText(display_frame, "Face detected - Press 's' to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Register Face', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and face_tensor is not None:
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0)).squeeze()
        torch.save({'name': name, 'embedding': embedding}, f'embeddings/{name}.pt')
        print(f"[✅] Saved embedding for {name}")
        break
    elif key == ord('q'):
        print("[INFO] Quit without saving.")
        break

cap.release()
cv2.destroyAllWindows()
