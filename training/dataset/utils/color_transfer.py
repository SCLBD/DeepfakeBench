import cv2
import numpy as np
from numpy import linalg as npla

import scipy as sp
import scipy.sparse
from scipy.sparse.linalg import spsolve


def color_transfer_sot(src, trg, steps=10, batch_size=5, reg_sigmaXY=16.0, reg_sigmaV=5.0):
    """
    Color Transform via Sliced Optimal Transfer
    ported by @iperov from https://github.com/dcoeurjo/OTColorTransfer

    src         - any float range any channel image
    dst         - any float range any channel image, same shape as src
    steps       - number of solver steps
    batch_size  - solver batch size
    reg_sigmaXY - apply regularization and sigmaXY of filter, otherwise set to 0.0
    reg_sigmaV  - sigmaV of filter

    return value - clip it manually
    """
    if not np.issubdtype(src.dtype, np.floating):
        raise ValueError("src value must be float")
    if not np.issubdtype(trg.dtype, np.floating):
        raise ValueError("trg value must be float")

    if len(src.shape) != 3:
        raise ValueError("src shape must have rank 3 (h,w,c)")

    if src.shape != trg.shape:
        raise ValueError("src and trg shapes must be equal")

    src_dtype = src.dtype
    h, w, c = src.shape
    new_src = src.copy()

    for step in range(steps):
        advect = np.zeros((h*w, c), dtype=src_dtype)
        for batch in range(batch_size):
            dir = np.random.normal(size=c).astype(src_dtype)
            dir /= npla.norm(dir)

            projsource = np.sum(new_src*dir, axis=-1).reshape((h*w))
            projtarget = np.sum(trg*dir, axis=-1).reshape((h*w))

            idSource = np.argsort(projsource)
            idTarget = np.argsort(projtarget)

            a = projtarget[idTarget]-projsource[idSource]
            for i_c in range(c):
                advect[idSource, i_c] += a * dir[i_c]
        new_src += advect.reshape((h, w, c)) / batch_size

    if reg_sigmaXY != 0.0:
        src_diff = new_src-src
        src_diff_filt = cv2.bilateralFilter(
            src_diff, 0, reg_sigmaV, reg_sigmaXY)
        if len(src_diff_filt.shape) == 2:
            src_diff_filt = src_diff_filt[..., None]
        new_src = src + src_diff_filt
    return new_src


def color_transfer_mkl(x0, x1):
    eps = np.finfo(float).eps

    h, w, c = x0.shape
    h1, w1, c1 = x1.shape

    x0 = x0.reshape((h*w, c))
    x1 = x1.reshape((h1*w1, c1))

    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None)))

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None)))

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(
        np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T)

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    result = np.dot(x0-mx0, t) + mx1
    return np.clip(result.reshape((h, w, c)).astype(x0.dtype), 0, 1)


def color_transfer_idt(i0, i1, bins=256, n_rot=20):
    relaxation = 1 / n_rot
    h, w, c = i0.shape
    h1, w1, c1 = i1.shape

    i0 = i0.reshape((h*w, c))
    i1 = i1.reshape((h1*w1, c1))

    n_dims = c

    d0 = i0.T
    d1 = i1.T

    for i in range(n_rot):

        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)

        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)

        for j in range(n_dims):

            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())

            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _ = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]

            f = np.interp(cp0r, cp1r, edges[1:])

            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)

        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0

    return np.clip(d0.T.reshape((h, w, c)).astype(i0.dtype), 0, 1)


def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    return mat_A


def seamless_clone(source, target, mask):
    h, w, c = target.shape
    result = []

    mat_A = laplacian_matrix(h, w)
    laplacian = mat_A.tocsc()

    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    q = np.argwhere(mask == 0)

    k = q[:, 1]+q[:, 0]*w
    mat_A[k, k] = 1
    mat_A[k, k + 1] = 0
    mat_A[k, k - 1] = 0
    mat_A[k, k + w] = 0
    mat_A[k, k - w] = 0

    mat_A = mat_A.tocsc()
    mask_flat = mask.flatten()
    for channel in range(c):

        source_flat = source[:, :, channel].flatten()
        target_flat = target[:, :, channel].flatten()

        mat_b = laplacian.dot(source_flat)*0.75
        mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

        x = spsolve(mat_A, mat_b).reshape((h, w))
        result.append(x)

    return np.clip(np.dstack(result), 0, 1)


def reinhard_color_transfer(target, source, clip=False, preserve_paper=False, source_mask=None, target_mask=None):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
            OpenCV image in BGR color space (the source image)
    target: NumPy array
            OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before
            converting back to BGR color space?
            If False then components will be min-max scaled appropriately.
            Clipping will keep target image brightness truer to the input.
            Scaling will adjust image brightness to avoid washed out portions
            in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
            layed out in original paper? The method does not always produce
            aesthetically pleasing results.
            If False then L*a*b* components will scaled using the reciprocal of
            the scaling factor proposed in the paper.  This method seems to produce
            more consistently aesthetically pleasing results

    Returns:
    -------
    transfer: NumPy array
            OpenCV image (w, h, 3) NumPy array (uint8)
    """

    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # compute color statistics for the source and target images
    src_input = source if source_mask is None else source*source_mask
    tgt_input = target if target_mask is None else target*target_mask
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc,
     bMeanSrc, bStdSrc) = lab_image_stats(src_input)
    (lMeanTar, lStdTar, aMeanTar, aStdTar,
     bMeanTar, bStdTar) = lab_image_stats(tgt_input)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
                # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def linear_color_transfer(target_img, source_img, mode='pca', eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c) and float in [0,1].
    Modes are chol, pca or sym for different choices of basis.
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2, 0, 1).reshape(t.shape[-1], -1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2, 0, 1).reshape(s.shape[-1], -1)
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])
    if mode == 'chol':
        chol_t = np.linalg.cholesky(Ct)
        chol_s = np.linalg.cholesky(Cs)
        ts = chol_s.dot(np.linalg.inv(chol_t)).dot(t)
    if mode == 'pca':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        eva_s, eve_s = np.linalg.eigh(Cs)
        Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)
        ts = Qs.dot(np.linalg.inv(Qt)).dot(t)
    if mode == 'sym':
        eva_t, eve_t = np.linalg.eigh(Ct)
        Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
        Qt_Cs_Qt = Qt.dot(Cs).dot(Qt)
        eva_QtCsQt, eve_QtCsQt = np.linalg.eigh(Qt_Cs_Qt)
        QtCsQt = eve_QtCsQt.dot(np.sqrt(np.diag(eva_QtCsQt))).dot(eve_QtCsQt.T)
        ts = np.linalg.inv(Qt).dot(QtCsQt).dot(np.linalg.inv(Qt)).dot(t)
    matched_img = ts.reshape(
        *target_img.transpose(2, 0, 1).shape).transpose(1, 2, 0)
    matched_img += mu_s
    matched_img[matched_img > 1] = 1
    matched_img[matched_img < 0] = 0
    return np.clip(matched_img.astype(source_img.dtype), 0, 1)


def lab_image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr


def channel_hist_match(source, template, hist_match_threshold=255, mask=None):
    # Code borrowed from:
    # https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
    masked_source = source
    masked_template = template

    if mask is not None:
        masked_source = source * mask
        masked_template = template * mask

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    masked_source = masked_source.ravel()
    masked_template = masked_template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles = hist_match_threshold * s_quantiles / s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles = 255 * t_quantiles / t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def color_hist_match(src_im, tar_im, hist_match_threshold=255, mask=None):
    h, w, c = src_im.shape
    matched_R = channel_hist_match(
        src_im[:, :, 0], tar_im[:, :, 0], hist_match_threshold, mask)
    matched_G = channel_hist_match(
        src_im[:, :, 1], tar_im[:, :, 1], hist_match_threshold, mask)
    matched_B = channel_hist_match(
        src_im[:, :, 2], tar_im[:, :, 2], hist_match_threshold, mask)

    to_stack = (matched_R, matched_G, matched_B)
    for i in range(3, c):
        to_stack += (src_im[:, :, i],)

    matched = np.stack(to_stack, axis=-1).astype(src_im.dtype)
    return matched


def color_transfer_mix(img_src, img_trg):
    img_src = np.clip(img_src*255.0, 0, 255).astype(np.uint8)
    img_trg = np.clip(img_trg*255.0, 0, 255).astype(np.uint8)

    img_src_lab = cv2.cvtColor(img_src, cv2.COLOR_BGR2LAB)
    img_trg_lab = cv2.cvtColor(img_trg, cv2.COLOR_BGR2LAB)

    rct_light = np.clip(linear_color_transfer(img_src_lab[..., 0:1].astype(np.float32)/255.0,
                                              img_trg_lab[..., 0:1].astype(np.float32)/255.0)[..., 0]*255.0,
                        0, 255).astype(np.uint8)

    img_src_lab[..., 0] = (np.ones_like(rct_light)*100).astype(np.uint8)
    img_src_lab = cv2.cvtColor(img_src_lab, cv2.COLOR_LAB2BGR)

    img_trg_lab[..., 0] = (np.ones_like(rct_light)*100).astype(np.uint8)
    img_trg_lab = cv2.cvtColor(img_trg_lab, cv2.COLOR_LAB2BGR)

    img_rct = color_transfer_sot(img_src_lab.astype(
        np.float32), img_trg_lab.astype(np.float32))
    img_rct = np.clip(img_rct, 0, 255).astype(np.uint8)

    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_BGR2LAB)
    img_rct[..., 0] = rct_light
    img_rct = cv2.cvtColor(img_rct, cv2.COLOR_LAB2BGR)

    return (img_rct / 255.0).astype(np.float32)


def colorTransfer_fs(src_, dst_, mask):
    src = dst_
    dst = src_
    transferredDst = np.copy(dst)
    # indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    # src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst
    return transferredDst

def colorTransfer_avg(img_src, img_tgt, mask=None):
    img_new = img_src.copy()
    img_old = img_tgt.copy()
    # print(mask)
    if mask is not None:
        img_new = (img_new*mask)#.astype(np.uint8)
        img_old = (img_old*mask)#.astype(np.uint8)
    # cv2.imshow('tgt', img_old)
    w,h,c = img_new.shape
    for i in range(img_new.shape[2]):
        old_avg = img_old[:, :, i].mean()
        new_avg = img_new[:, :, i].mean()
        diff_int = old_avg - new_avg
        # print(diff_int)
        for m in range(img_new.shape[0]):
            for n in range(img_new.shape[1]):
                temp = img_new[m,n,i] + diff_int
                temp = max(0., temp)
                temp = min(1., temp)
                # print(img_new[m,n,i], temp)
                img_new[m,n,i] = temp

    return img_new
                


def color_transfer(ct_mode, img_src, img_trg, mask):
    """
    color transfer for [0,1] float32 inputs
    """
    img_src = img_src.astype(dtype=np.float32) / 255.0
    img_trg = img_trg.astype(dtype=np.float32) / 255.0

    if ct_mode == 'lct':
        out = linear_color_transfer(img_src, img_trg)
    elif ct_mode == 'rct':
        out = reinhard_color_transfer(np.clip(img_src*255, 0, 255).astype(np.uint8),
                                      np.clip(img_trg*255, 0,
                                              255).astype(np.uint8),
                                      preserve_paper=np.random.rand() < 0.5,
                                      clip=np.random.rand() < 0.5)
        out = np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)
    elif ct_mode == 'rct-m':
        out = reinhard_color_transfer(np.clip(img_src*255, 0, 255).astype(np.uint8),
                                      np.clip(img_trg*255, 0,
                                              255).astype(np.uint8),
                                      source_mask=mask, target_mask=mask)
                                      #preserve_paper=np.random.rand() < 0.5,
                                      #clip=np.random.rand() < 0.5)
        out = np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)
    elif ct_mode == 'rct-fs':
        out = colorTransfer_fs(np.clip(img_src*255, 0, 255).astype(np.uint8),
                               np.clip(img_trg*255, 0, 255).astype(np.uint8), mask)
        out = np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)
    elif ct_mode == 'mkl':
        out = color_transfer_mkl(img_src, img_trg)
    elif ct_mode == 'mkl-m':
        out = color_transfer_mkl(img_src*mask, img_trg*mask)
    elif ct_mode == 'idt':
        out = color_transfer_idt(img_src, img_trg)
    elif ct_mode == 'idt-m':
        out = color_transfer_idt(img_src*mask, img_trg*mask)
    elif ct_mode == 'sot':
        out = color_transfer_sot(img_src, img_trg)
        out = np.clip(out, 0.0, 1.0)
    elif ct_mode == 'sot-m':
        out = color_transfer_sot(
            (img_src*mask).astype(np.float32), (img_trg*mask).astype(np.float32))
        out = np.clip(out, 0.0, 1.0)
    elif ct_mode == 'mix-m':
        out = color_transfer_mix(img_src*mask, img_trg*mask)
    elif ct_mode == 'seamless-hist-match':
        out = color_hist_match(img_src, img_trg)
    elif ct_mode == 'seamless-hist-match-m':
        out = color_hist_match(img_src, img_trg, mask=mask)
    elif ct_mode == 'avg-align':
        out = colorTransfer_avg(img_src, img_trg, mask=mask)
        out = np.clip(out, 0.0, 1.0)
    else:
        raise ValueError(f"unknown ct_mode {ct_mode}")

    out = np.clip(out*255, 0, 255).astype(np.uint8)
    return out