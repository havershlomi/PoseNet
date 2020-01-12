import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils import face_utils


class PoseVanila:
    def __init__(self, camera_matrix, model_3d, detector, predictor):
        self.camera_matrix = camera_matrix
        self.model_3d = model_3d
        self.detector = detector
        self.predictor = predictor
    
    def _show_image(self, img):
        plt.axis("off")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        
    def get_landmarks_2d(self, image):
        image = image.copy()
        # under the assumption that my photos are easy to detect i am not going to
        # resize the image case it affects the output one rotating landmarks
#         if(max(image.shape[:2]) < 500):
#             image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 1)

        # loop over the face detections
        shape = None
        face = None
        if(len(rects) > 1):
            return shape, face, image
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            p = 0 # padding
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = image[ y-p: y+h+p, x-p:x+w+p, :].copy()

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
#             for (x, y) in shape:
#                 cv2.circle(image, (x ,y), 1, (0, 0, 255), -1)
                
#             self._show_image(image)

        return shape, face, image
    
    def calc_6dofVector(self, image_points):
       
        if(image_points is None):
            return None, None
        image_points = image_points.astype(float)

        # 3D model points.
        model_points = self.model_3d

        # Camera internals

        camera_matrix = self.camera_matrix

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)

        if(rotation_vector is None or translation_vector is None):
            return None
        vector = [*rotation_vector, *translation_vector]

        return np.array(vector, dtype=np.float64).reshape(6)
    
    def predict(self, image, show=False):
        #2D image points. If you change the image, you need to change vector
        landmarks, face, original_image = self.get_landmarks_2d(image)
        
        vector = self.calc_6dofVector(landmarks)
        if(show):
            print("Rotation Vector:\n {0}".format(vector[:3]))
            print("Translation Vector:\n {0}".format(vector[3:]))
            self._show_image(face)
        
        return vector
