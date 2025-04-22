from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def extract_face_embedding(image_path):
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        embedding = resnet(face).detach().cpu().numpy()[0]
        return embedding
    return None

def cosine_similarity(vec1, vec2):
    from scipy.spatial.distance import cosine
    return 1 - cosine(vec1, vec2)
