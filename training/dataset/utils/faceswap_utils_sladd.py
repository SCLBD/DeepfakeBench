import numpy as np
import cv2

def AlphaBlend(foreground, background, alpha):
    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    if len(alpha.shape) < 3:
        alpha = np.expand_dims(alpha, 2)
    outImage = alpha * foreground + (1.-alpha) * background
    outImage = np.clip(outImage, 0, 255).astype(np.uint8)

    return outImage

def blendImages(src, dst, mask, featherAmount=0.1):
    maskIndices = np.where(mask != 0)
    maskPts = np.hstack(
        (maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    #hull = hull.astype(np.uint64)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        point = (int(maskPts[i, 0]), int(maskPts[i, 1]))
        """
        The third paprameter can be set as "True" for more visually diverse images.
        We use "False" to add imperceptible image patterns to synthesize new images.
        """
        dists[i] = cv2.pointPolygonTest(hull, point, False)

    weights = np.clip(dists / featherAmount, 0, 1)
    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * \
                                                  src[maskIndices[0], maskIndices[1]] + \
                                                  (1 - weights[:, np.newaxis]) * \
                                                  dst[maskIndices[0], maskIndices[1]]
    newMask = np.zeros_like(dst).astype(np.float32)
    newMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis]

    return composedImg, newMask


def colorTransfer(src_, dst_, mask):
    src = dst_
    dst = src_
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
