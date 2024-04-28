'''
Create face mask and face boundary mask according to face landmarks,
so as to supervize the activation of Conv layer.
'''

import os
import numpy as np
import cv2
import dlib
import random
import argparse
from tqdm import tqdm
import time
from skimage import transform as trans
# from color_transfer import color_transfer
from .warp import gen_warp_params, warp_by_params, warp_mask


def crop_img_bbox(img, bbox, res, scale=1.3):
    x, y, w, h = bbox
    left, right = x, x+w
    top, bottom = y, y+h

    H, W, C = img.shape
    cx, cy = (left+right)//2, (top+bottom)//2
    w, h = (right-left)//2, (bottom-top)//2

    x1 = max(0, int(cx-w*scale))
    x2 = min(W, int(cx+w*scale))
    y1 = max(0, int(cy-h*scale))
    y2 = min(H, int(cy+h*scale))

    roi = img[y1:y2, x1:x2]
    roi = cv2.resize(roi, (res, res))

    return roi


def get_mask_center(mask):
    l, t, w, h = cv2.boundingRect(mask[:, :, 0:1].astype(np.uint8))
    center = int(l+w/2), int(t+h/2)
    return center


def get_5_keypoint(shape):
    def get_point(idx):
        # return [shape.part(idx).x, shape.part(idx).y]
        return shape[idx]

    def center(pt1, pt2):
        return [(pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2]

    leye = np.array(center(get_point(36), get_point(39)),
                    dtype=int).reshape(-1, 2)
    reye = np.array(center(get_point(45), get_point(42)),
                    dtype=int).reshape(-1, 2)
    nose = np.array(get_point(30), dtype=int).reshape(-1, 2)
    lmouth = np.array(get_point(48),
                      dtype=int).reshape(-1, 2)
    rmouth = np.array(get_point(54),
                      dtype=int).reshape(-1, 2)

    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts


def get_boundary(mask):
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    boundary = mask / 255.
    boundary = 4*boundary*(1.-boundary)
    return boundary


# def get_boundary(mask):
#     if len(mask.shape) == 3:
#         mask = mask[:, :, 0]
#     mask = cv2.GaussianBlur(mask, (3, 3), 0)
#     mask = mask.astype(np.uint8)

#     # Dilation and Erosion to find the boundary
#     dilated = cv2.dilate(mask, None, iterations=1)
#     boundary = cv2.subtract(dilated, mask)

#     # normalize the boundary to have values between 0 and 1
#     boundary = boundary / 255.

#     return boundary



def blur_mask(mask):
    blur_k = 2*np.random.randint(1, 10)-1
    
    #kernel = np.ones((blur_k+1, blur_k+1), np.uint8)
    #mask = cv2.erode(mask, kernel)
    
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)


    return mask


def random_deform(pt, tgt, scale=0.3):
    x1, y1 = pt
    x2, y2 = tgt

    x = x1+(x2-x1)*np.random.rand()*scale
    y = y1+(y2-y1)*np.random.rand()*scale
    #print('before:', pt, ' after:', [int(x), int(y)])
    return [int(x), int(y)]


def get_specific_mask(img, shape, mtype='mouth', random_side=False):
    if mtype == 'eyes':
        landmarks = shape[42:45] if random.choice([True, False]) else shape[36:39]

    elif mtype == 'nose':
        landmarks = shape[27:35]

    elif mtype == 'mouth':
        landmarks = shape[48:60]

    elif mtype == 'eyebrows':
        landmarks = shape[22:26] if random.choice([True, False]) else shape[17:21]

    else:
        raise ValueError(f"Invalid mtype. Choose from 'eyes', 'nose', 'mouth', or 'eyebrows', but got {mtype}")

    # find convex hull
    hull = cv2.convexHull(landmarks)
    hull = hull.astype(int)

    # mask
    hull_mask = np.zeros_like(img)
    cv2.fillPoly(hull_mask, [hull], (255, 255, 255))
    mask = hull_mask
    return mask


def get_hull_mask(img, shape, mtype='hull'):
    if mtype == 'normal-hull':
        landmarks = np.array(shape)

        # find convex hull
        hull = cv2.convexHull(landmarks)
        hull = hull.astype(int)

        # full face mask
        hull_mask = np.zeros_like(img)
        cv2.fillPoly(hull_mask, [hull], (255, 255, 255))
        mask = hull_mask

    elif mtype == 'inner-hull':
        landmarks = shape[17:]
        landmarks = np.array(landmarks)

        # find convex hull
        hull = cv2.convexHull(landmarks)
        hull = hull.astype(int)

        # full face mask
        hull_mask = np.zeros_like(img)
        cv2.fillPoly(hull_mask, [hull], (255, 255, 255))

        mask = hull_mask

    elif mtype == 'inner-hull-no-eyebrow':
        landmarks = shape[27:]
        landmarks = np.array(landmarks)
        # find convex hull
        hull = cv2.convexHull(landmarks)
        hull = hull.astype(int)

        # full face mask
        hull_mask = np.zeros_like(img)
        cv2.fillPoly(hull_mask, [hull], (255, 255, 255))

        mask = hull_mask

    elif mtype == 'mouth-hull':
        landmarks = shape[2:15]
        #landmarks.append(shape[29])
        landmarks = np.concatenate([landmarks, shape[29].reshape(1, -1)], axis=0)

        # find convex hull
        hull = cv2.convexHull(landmarks)
        hull = hull.astype(int)

        # full face mask
        hull_mask = np.zeros_like(img)
        cv2.fillPoly(hull_mask, [hull], (255, 255, 255))

        # kernel = np.ones((2, 2), np.uint8)
        # c_mask = cv2.dilate(hull_mask, kernel, iterations=1)
        mask = hull_mask

    elif mtype == 'whole-hull':
        face_height = shape[9][1] - shape[22][1]
        landmarks = []
        for i in range(27):
            lmk = shape[i]
            if i >= 5 and i <= 11:
                x, y = lmk[0], lmk[1]
                lmk = [x, max(0, y+15)]
            # lift the eyebrows to get a larger landmark convex hull
            if i >= 18 and i <= 27:
                x, y = lmk[0], lmk[1]
                lmk = [x, max(0, y-face_height//4)]

            landmarks.append(lmk)

        # find convex hull
        landmarks = np.array(landmarks, dtype=np.int32)
        hull = cv2.convexHull(landmarks)
        hull = np.reshape(hull, (1, -1, 2))

        # full face mask
        hull_mask = np.zeros_like(img)
        cv2.fillPoly(hull_mask, [hull], (255, 255, 255))

        # kernel = np.ones((2, 2), np.uint8)
        # c_mask = cv2.dilate(hull_mask, kernel, iterations=1)
        mask = hull_mask
    '''
    elif mtype == 'rect':
        cnt = []
        for idx in [5, 11, 17, 26]:
            cnt.append(shape[idx])
        x, y, w, h = cv2.boundingRect(np.array(cnt))
        rect_mask = np.zeros_like(img)
        cv2.rectangle(rect_mask, (x, y), (x+w, y+h),
                      (255, 255, 255), cv2.FILLED)
        mask = rect_mask
    '''
    return mask


def get_mask(shape, img, std=20, deform=True, restrict_mask=None):
    mask_type = [
        'normal-hull', 
        'inner-hull', 
        'inner-hull-no-eyebrow', 
        'mouth-hull', 
        'whole-hull'
    ]
    max_mask = get_hull_mask(img, shape, 'whole-hull')
    if deform:
        mtype = mask_type[np.random.randint(len(mask_type))]
        if mtype == 'rect':
            mask = get_hull_mask(img, shape, 'inner-hull-no-eyebrow')
            x, y, w, h = cv2.boundingRect(mask[:,:,0])
            for i in range(y, y+h):
                for j in range(x, x+w):
                    for k in range(mask.shape[2]):
                        mask[i, j, k] = 255
        else:
            mask = get_hull_mask(img, shape, mtype)

        # random deform
        if np.random.rand() < 0.9:
            mask = warp_mask(mask, std=std)

        # # random erode/dilate
        # prob = np.random.rand()
        # if prob < 0.3:
        #     erode_k = 2*np.random.randint(1, 10)+1
        #     kernel = np.ones((erode_k, erode_k), np.uint8)
        #     mask = cv2.erode(mask, kernel)
        # elif prob < 0.6:
        #     erode_k = 2*np.random.randint(1, 10)+1
        #     kernel = np.ones((erode_k, erode_k), np.uint8)
        #     mask = cv2.dilate(mask, kernel)
    else:
        mask = max_mask.copy()
    
    if restrict_mask is not None:
        mask = mask*(restrict_mask//255)

    # restrict mask range
    mask = mask *(max_mask//255)

    # random blur
    if deform and np.random.rand() < 0.9:
        mask = blur_mask(mask)

    return mask[:,:,0]

def mask_postprocess(mask):
    # random erode/dilate
    prob = np.random.rand()
    if prob < 0.3:
        erode_k = 2*np.random.randint(1, 10)+1
        kernel = np.ones((erode_k, erode_k), np.uint8)
        mask = cv2.erode(mask, kernel)
    elif prob < 0.6:
        erode_k = 2*np.random.randint(1, 10)+1
        kernel = np.ones((erode_k, erode_k), np.uint8)
        mask = cv2.dilate(mask, kernel)
    
    # random blur
    if np.random.rand() < 0.9:
        mask = blur_mask(mask)

    return mask


def get_affine_param(from_, to_):
        # use skimage tranformation
    tform = trans.SimilarityTransform()
    tform.estimate(from_.astype(np.float32), to_.astype(
        np.float32))  # tform.estimate(from_, to_)
    M = tform.params[0:2, :]

    return M


def random_sharpen_img(img):
    cand = ['bsharpen', 'gsharpen']  # , 'none']
    mode = cand[np.random.randint(len(cand))]
    # print('sharpen mode:', mode)
    if mode == "bsharpen":
        # Sharpening using filter2D
        kernel = np.ones((3, 3)) * (-1)
        kernel[1, 1] = 9
        #kernel /= 9.
        out = cv2.filter2D(img, -1, kernel)
    elif mode == "gsharpen":
        # Sharpening using Weighted Method
        gaussain_blur = cv2.GaussianBlur(img, (0, 0), 3.0)
        out = cv2.addWeighted(
            img, 1.5, gaussain_blur, -0.5, 0, img)
    else:
        out = img

    return out


def random_blur_img(img):
    cand = ['avg', 'gaussion', 'med']  # , 'none']
    mode = cand[np.random.randint(len(cand))]
    # print('blur mode:', mode)
    ksize = 2*np.random.randint(1, 5)+1

    if mode == 'avg':
        # Averaging
        out = cv2.blur(img, (ksize, ksize))
    elif mode == 'gaussion':
        # Gaussian Blurring
        out = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif mode == 'med':
        # Median blurring
        out = cv2.medianBlur(img, ksize)
    else:
        out = img
    # elif mode == 'bilateral'
    #     # Bilateral Filtering
    #     out = cv2.bilateralFilter(img,9,75,75)

    return out


def random_warp_img(img, prob=0.5):
    H, W, C = img.shape
    param = gen_warp_params(W, flip=False)
    choice = [True, False]

    out = warp_by_params(param, img,
                         can_flip=False,  # choice[np.random.randint(2)],
                         can_transform=False,  # choice[np.random.randint(2)],
                         can_warp=(np.random.randint(10) < int(prob*10)),
                         border_replicate=choice[np.random.randint(2)])
    return out


def main(args):
    np.random.seed(int(time.time()))
    detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(args.model)

    src_im = cv2.imread(args.src)
    tgt_im = cv2.imread(args.tgt)

    H, W, C = tgt_im.shape
    src_im = cv2.resize(src_im, (W, H))

    def get_shape(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(img, 1)
        det = dets[0]
        shape = landmark_predictor(img, det)

        return shape, det

    src_shape, src_det = get_shape(src_im)
    src_5_pts = get_5_keypoint(src_shape)
    src_mask = get_mask(src_shape, src_im, whole=True, deform=False)

    tgt_shape, tgt_det = get_shape(tgt_im)
    tgt_5_pts = get_5_keypoint(tgt_shape)
    tgt_mask = get_mask(tgt_shape, tgt_im, whole=False, deform=True)

    #aff_param = get_affine_param(src_5_pts, tgt_5_pts)

    # color transfer:
    mask = src_mask[:, :, 0:1]/255.
    ct_modes = ['lct', 'rct', 'idt', 'idt-m', 'mkl', 'mkl-m',
                'sot', 'sot-m', 'mix-m']  # , 'seamless-hist-match']
    for mode in ct_modes:
        colored_src = color_transfer(mode, src_im, tgt_im, mask)
        cv2.imwrite('{}_colored.png'.format(mode), colored_src)
    src_im = colored_src

    w1, h1 = src_det.right()-src_det.left(), src_det.bottom()-src_det.top()
    w2, h2 = tgt_det.right()-tgt_det.left(), tgt_det.bottom()-tgt_det.top()
    w_scale, h_scale = w2/w1, h2/h1

    scaled_src = cv2.resize(src_im, (int(W*w_scale), int(H*h_scale)))
    scaled_mask = cv2.resize(src_mask, (int(W*w_scale), int(H*h_scale)))

    src_5_pts[:, 0] = src_5_pts[:, 0]*w_scale
    src_5_pts[:, 1] = src_5_pts[:, 1]*h_scale
    aff_param = get_affine_param(src_5_pts, tgt_5_pts)

    aligned_src = cv2.warpAffine(
        scaled_src, aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    aligned_mask = cv2.warpAffine(
        scaled_mask, aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    center = get_mask_center(aligned_mask)
    print('mask center:', center)
    # colored_src = transfer_color(aligned_src, tgt_im)

    init_blend = cv2.seamlessClone(
        aligned_src, tgt_im, aligned_mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite('init_blended.png', init_blend)

    # aligned_blend = cv2.warpAffine(
    #    colored_blend, aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    b_mask = tgt_mask[:, :, 0:1]/255.
    out_blend = init_blend*b_mask + tgt_im*(1. - b_mask)
    cv2.imwrite('out_blended.png', out_blend)

    res = 256
    blend_crop = crop_img_bbox(out_blend, tgt_det, res, scale=1.5)
    mask_crop = crop_img_bbox(tgt_mask, tgt_det, res, scale=1.5)
    boundary = get_boundary(mask_crop)

    cv2.imwrite('crop_blend.png', blend_crop)
    cv2.imwrite('crop_mask.png', mask_crop)
    cv2.imwrite('crop_bound.png', boundary*255)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('-s', '--src', type=str,
                   help='src image')
    p.add_argument('-t', '--tgt', type=str,
                   help='tgt image')
    p.add_argument('--model', type=str, default='/data1/yuchen/download/face_landmark/shape_predictor_68_face_landmarks.dat',
                   help="path to downloaded detector")
    args = p.parse_args()
    print(args)

    main(args)
