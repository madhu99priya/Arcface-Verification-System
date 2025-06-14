## responsible for face alignment, a critical preprocessing step that improves recognition accuracy.

from facenet_pytorch import MTCNN
import cv2, os

mtcnn = MTCNN(image_size=160, margin=10)

def align_face(img_path, save_path=None):
    img = cv2.imread(img_path)[:, :, ::-1]  # BGR→RGB
    face = mtcnn(img)
    if face is None:
        return None
    aligned = face.permute(1, 2, 0).int().numpy()[:, :, ::-1]  # back to BGR
    if save_path:
        cv2.imwrite(save_path, aligned)
    return aligned


