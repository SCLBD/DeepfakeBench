import dlib
from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from dataset.library.DeepFakeMask import dfl_full,facehull,components,extended
from dataset.utils.attribution_mask import *
import cv2
import tqdm

'''
from PIL import ImageDraw
# 创建一个可以在图像上绘制的对象
img_pil=Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)

# 在图像上绘制点
for i, point in enumerate(landmark):
    x, y = point
    radius = 1  # 点的半径
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill="red")
    draw.text((x+radius+2, y-radius), str(i), fill="black")  # 在点旁边添加标签
img_pil.show()
'''


def name_resolve(path):
    name = os.path.splitext(os.path.basename(path))[0]
    vid_id, frame_id = name.split('_')[0:2]
    return vid_id, frame_id 
    
def total_euclidean_distance(a,b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b,axis=1))

def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
    reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [ tuple(x.astype('int32')) for x in [
        leye_center,reye_center,nose,lmouth,rmouth,leye_left,leye_right,reye_left,reye_right
    ]]
    return out

def random_get_hull(landmark,img1,hull_type=None):
    if hull_type==None:
        hull_type = random.choice([0,1,2,3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask[:,:,0]/255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask[:,:,0]/255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask[:,:,0]/255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
        return mask[:,:,0]/255
    elif hull_type == 4:
        mask = remove_mouth(img1,get_five_key(landmark))
        return mask.astype(np.float32)
    elif hull_type == 5:
        mask = remove_eyes(img1,landmark)
        return mask.astype(np.float32)
    elif hull_type == 6:
        mask = remove_nose(img1,landmark)
        return mask.astype(np.float32)
    elif hull_type == 7:
        mask = remove_nose(img1,landmark) + remove_eyes(img1,landmark) + remove_mouth(img1,get_five_key(landmark))
        return mask.astype(np.float32)


def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
   
    maskIndices = np.where(mask != 0)
    
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

class BIOnlineGeneration():
    def __init__(self):
        with open('precomuted_landmarks.json', 'r') as f:
            self.landmarks_record =  json.load(f)
            for k,v in self.landmarks_record.items():
                self.landmarks_record[k] = np.array(v)
        # extract all frame from all video in the name of {videoid}_{frameid}
        self.data_list = [
                    '000_0000.png',
                    '001_0000.png'      
                    ] * 10000
        
        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
        
    def gen_one_datapoint(self):
        background_face_path = random.choice(self.data_list)
        data_type = 'real' if random.randint(0,1) else 'fake'
        if data_type == 'fake' :
            face_img,mask =  self.get_blended_face(background_face_path)
            mask = ( 1 - mask ) * mask * 4
        else:
            face_img = io.imread(background_face_path)
            mask = np.zeros((317, 317, 1))
        
        # randomly downsample after BI pipeline
        if random.randint(0,1):
            aug_size = random.randint(64, 317)
            face_img = Image.fromarray(face_img)
            if random.randint(0,1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
            face_img = face_img.resize((317, 317),Image.BILINEAR)
            face_img = np.array(face_img)
            
        # random jpeg compression after BI pipeline
        if random.randint(0,1):
            quality = random.randint(60, 100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
            face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
        
        face_img = face_img[60:317,30:287,:]
        mask = mask[60:317,30:287,:]
        
        # random flip
        if random.randint(0,1):
            face_img = np.flip(face_img,1)
            mask = np.flip(mask,1)
            
        return face_img,mask,data_type
        
    def get_blended_face(self,background_face_path):
        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_path]
        
        foreground_face_path = self.search_similar_face(background_landmark,background_face_path)
        foreground_face = io.imread(foreground_face_path)
        
        # down sample before blending
        aug_size = random.randint(128,317)
        background_landmark = background_landmark * (aug_size/317)
        foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
        
        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)
       
        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        
        # filte empty mask after deformation
        if np.sum(mask) == 0 :
            raise NotImplementedError

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask*255)
        
        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask*255)
        blended_face = blended_face.astype(np.uint8)
       
        # resize back to default resolution
        blended_face = sktransform.resize(blended_face,(317,317),preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask,(317,317),preserve_range=True)
        mask = mask[:,:,0:1]
        return blended_face,mask
    
    def search_similar_face(self,this_landmark,background_face_path):
        vid_id, frame_id = name_resolve(background_face_path)
        min_dist = 99999999
        
        # random sample 5000 frame from all frams:
        all_candidate_path = random.sample( self.data_list, k=5000) 
        
        # filter all frame that comes from the same video as background face
        all_candidate_path = filter(lambda k:name_resolve(k)[0] != vid_id, all_candidate_path)
        all_candidate_path = list(all_candidate_path)
        
        # loop throungh all candidates frame to get best match
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(np.float32)
            candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path

        return min_path
    
if __name__ == '__main__':
    ds = BIOnlineGeneration()
    from tqdm import tqdm
    all_imgs = []
    for _ in tqdm(range(50)):
        img,mask,label = ds.gen_one_datapoint()
        mask = np.repeat(mask,3,2)
        mask = (mask*255).astype(np.uint8)
        img_cat = np.concatenate([img,mask],1)
        all_imgs.append(img_cat)
    all_in_one = Image.new('RGB', (2570,2570))

    for x in range(5):
        for y in range(10):
            idx = x*10+y   
            im = Image.fromarray(all_imgs[idx])
            
            dx = x*514
            dy = y*257
            
            all_in_one.paste(im, (dx,dy))

    all_in_one.save("all_in_one.jpg")