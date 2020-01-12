import cv2
import numpy as np


class ImageAugmenter:
    def __init__(self, model):
        self.model = model
        
    def generate_sample(self, image, angle, scale=1.0):
        aug_image, landmarks = self._augment(image, angle, scale)
        dof_6_vec = self.model.calc_6dofVector(landmarks)
        return aug_image, dof_6_vec
        
    def _augment(self, image, angle, scale=1.0):
        image = image.copy()
        height, width = image.shape[:2]
        tx, ty = np.array((width // 2, height // 2))

        #generate affine matrix
        affine_matrix = cv2.getRotationMatrix2D((tx, ty), angle, scale)
        # get landmarks
        shape, face, original_image = self.model.get_landmarks_2d(image)
        # warped the image with the landmarks
        warped = cv2.warpAffine(src= original_image, M=affine_matrix, dsize=(0,0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        #generate Humongous affine matrix
        h_affine_matrix = np.zeros((3,3))
        h_affine_matrix[:2,:] = affine_matrix

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        h_landmarks = np.ones((68, 3))
        h_landmarks[:,:2] = shape
        landmarks = []
        for i in range(68):
            x_tag = h_affine_matrix @ h_landmarks[i,:]
            landmarks.append(x_tag.tolist())

    #     for (x, y, z) in landmarks:
    #         cv2.circle(warped, (int(x), int(y)), 1, (0, 0, 255), -1)
    #     print(warped.shape)
    #     show_image(warped)
        landmarks = np.array(landmarks)[:,:2]
        return warped, landmarks
        
