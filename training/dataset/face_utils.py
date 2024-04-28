import cv2
import numpy as np
from skimage import transform as trans
# from mtcnn.mtcnn import MTCNN


def get_keypts(face):
    # get key points from the results of mtcnn

    if len(face['keypoints']) == 0:
        return []

    leye = np.array(face['keypoints']['left_eye'], dtype=np.int).reshape(-1, 2)
    reye = np.array(face['keypoints']['right_eye'],
                    dtype=np.int).reshape(-1, 2)
    nose = np.array(face['keypoints']['nose'], dtype=np.int).reshape(-1, 2)
    lmouth = np.array(face['keypoints']['mouth_left'],
                      dtype=np.int).reshape(-1, 2)
    rmouth = np.array(face['keypoints']['mouth_right'],
                      dtype=np.int).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
    """ align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
    """

    M = None

    target_size = [112, 112]

    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

    target_size = outsize

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.
    y_margin = target_size[1] * margin_rate / 2.

    # move
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    # resize
    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmark.astype(np.float32)

    # use skimage tranformation
    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    # M: use opencv
    # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

    img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

    if outsize is not None:
        img = cv2.resize(img, (outsize[1], outsize[0]))
    
    if mask is not None:
        mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
        mask = cv2.resize(mask, (outsize[1], outsize[0]))
        return img, mask
    else:
        return img


    


def expand_bbox(bbox, width, height, scale=1.3, minsize=None):
    """
    Expand original boundingbox by scale.
    :param bbx: original boundingbox
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: expanded bbox
    """
    x, y, w, h = bbox

    # box center
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    # expand by scale factor
    new_size = max(int(w * scale), int(h * scale))
    new_x = max(0, int(cx - new_size / 2))
    new_y = max(0, int(cy - new_size / 2))

    # Check for too big bbox for given x, y
    new_size = min(width - new_x, new_size)
    new_size = min(height - new_size, new_size)

    return new_x, new_y, new_size, new_size


def extract_face_MTCNN(face_detector, image, expand_scale=1.3, res=256):
    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector.detect_faces(rgb)
    if len(faces):
        # For now only take biggest face
        face = None
        bbox = None
        max_region = 0
        for ff in faces:
            if max_region == 0:
                face = ff
                bbox = face['box']
                max_region = bbox[2]*bbox[3]
            else:
                bb = ff['box']
                region = bb[2]*bb[3]
                if region > max_rigion:
                    max_rigion = region
                    face = ff
                    bbox = face['box']
        print(max_region)    
            #face = faces[0]

            #bbox = face['box']

        # --- Prediction ---------------------------------------------------
        # Face crop with MTCNN and bounding box scale enlargement
        x, y, w, h = expand_bbox(bbox, width, height, scale=expand_scale)
        cropped_face = rgb[y:y+h, x:x+w]

        cropped_face = cv2.resize(
            cropped_face, (res, res), interpolation=cv2.INTER_CUBIC)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        return cropped_face

    return None


def extract_aligned_face_MTCNN(face_detector, image, expand_scale=1.3, res=256, mask=None):
    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector.detect_faces(rgb)
    if len(faces):
        # For now only take biggest face
        face = None
        bbox = None
        max_region = 0
        for i, ff in enumerate(faces):
            if max_region == 0:
                face = ff
                bbox = face['box']
                max_region = bbox[2]*bbox[3]
            else:
                bb = ff['box']
                region = bb[2]*bb[3]
                if region > max_region:
                    max_region = region
                    face = ff
                    bbox = face['box']
            #print('face {}: {}'.format(i, max_region))
        #face = faces[0]

        landmarks = get_keypts(face)

        # --- Prediction ---------------------------------------------------
        # Face aligned crop with MTCNN and bounding box scale enlargement
        if mask is not None:
            cropped_face, cropped_mask = img_align_crop(rgb, landmarks, outsize=[
                                        res, res], scale=expand_scale, mask=mask)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2GRAY)
            return cropped_face, cropped_mask
        else:
            cropped_face = img_align_crop(rgb, landmarks, outsize=[
                                        res, res], scale=expand_scale)
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
            return cropped_face

    return None


def extract_face_DLIB(face_detector, image, expand_scale=1.3, res=256):
    # Image size
    height, width = image.shape[:2]

    # Convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect with dlib
    faces = face_detector(gray, 1)
    if len(faces):
        # For now only take biggest face
        face = faces[0]

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        bbox = (x1, y1, x2-x1, y2-y1)

        # --- Prediction ---------------------------------------------------
        # Face crop with dlib and bounding box scale enlargement
        x, y, w, h = expand_bbox(bbox, width, height, scale=expand_scale)
        cropped_face = image[y:y+h, x:x+w]

        cropped_face = cv2.resize(
            cropped_face, (res, res), interpolation=cv2.INTER_CUBIC)

        return cropped_face

    return None
