'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
from PIL import Image
import torchvision
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
import albumentations as A
from torchvision import transforms as T
from dataset.pair_dataset import pairDataset
import itertools


c=0
angel_mask=0
save_index=0


def data_shuffle(images,grid_g=-1,p=-1):
    if grid_g==-1:
        global c
        grid_g = c['Shu_GridShuffle']['grid_g']
        p=c['Shu_GridShuffle']['p']

#   先成pair，然后换，最后shuffle
    if p>random.random():
        tiles=A.RandomGridShuffle(grid=(grid_g,grid_g)).get_params_dependent_on_targets({'image':images[0]})['tiles']
        new_images = []
        for image in images:
            new_images.append(image.copy())

        for tile in tiles:
            arr=np.arange(len(images))
            np.random.shuffle(arr)
            for new_image,idx in zip(new_images,arr):
                new_image[tile[0] : tile[0] + tile[4], tile[1] : tile[1] + tile[5]] = images[idx][
                    tile[2] : tile[2] + tile[4], tile[3] : tile[3] + tile[5]
                ]
        return new_images
    else:
        return images

def getCutLineLoc(image_size,cut_type='half'):
    half=int(image_size/2)
    r=0
    if cut_type=='random_mg':
        r=random.uniform(-0.35, 0.35)
    if cut_type=='random':
        r=random.uniform(-0.5, 0.5)
    r_len = int(r*image_size)
    return half + r_len, r


def data_shuffle_v2(images, p, loc, cut_type='half', img_mix_num=2, full_angle=360,Mix_type='ClockMix'):
    if Mix_type=='ClockMix':
        return ClockMix(images, p, loc, cut_type, img_mix_num, full_angle)
    elif Mix_type=='CutMix':
        return CutMix(images,p)
    elif Mix_type=='Mixup':
        return Mixup(images,p)

def Mixup(images,p):
    global c
    rd_res = random.random()
    #   先成pair，然后换，最后shuffle
    if p > rd_res:
        arr = np.arange(len(images))
        arr = arr - 1
        new_images = []
        for image in images:
            new_images.append(image.copy())
        height, width, _ = images[0].shape
        ratio=random.random()
        for image, idx in zip(images, arr):
            new_images[idx]=new_images[idx]*ratio+image*(1-ratio)
        return new_images, 1, ratio
    else:
        return images,0, 0


def CutMix(images,p):
    global c
    rd_res = random.random()
    #   先成pair，然后换，最后shuffle
    if p > rd_res:
        arr = np.arange(len(images))
        arr = arr - 1
        new_images = []
        for image in images:
            new_images.append(image.copy())
        height, width, _ = images[0].shape
        #增加一个margin吧，保证至少换50？
        x_start = np.random.randint(0, width - 55)
        y_start = np.random.randint(0, height - 55)
        x_end = np.random.randint(x_start+50, width)
        y_end = np.random.randint(y_start+50, height)
        for image, idx in zip(images, arr):
            # 随机生成矩形区域的起点和大小

            new_images[idx][y_start:y_end, x_start:x_end] = image[y_start:y_end, x_start:x_end]
        swapped_ratio = ((x_end-x_start) * (y_end-y_start) ) / (width*height)
        return new_images,1,swapped_ratio
    else:
        return images,0,0

def ClockMix(images, p, loc, cut_type='half', img_mix_num=2, full_angle=360):

    if img_mix_num==1:
        return images,0,abs(full_angle/360-0.5)
    global c
    global angel_mask
    def getMask(images,loc):
        width, height = images[0].shape[0], images[0].shape[1]

        # 计算图片的中心点
        center_x, center_y = width // 2, height // 2

        # 创建坐标网格
        y, x = np.ogrid[:height, :width]

        # 计算每个像素与中心点的角度
        if loc =='angle':
            angel_mask = np.degrees(np.arctan2(center_y - y, x - center_x)) % 360
        elif loc =='angleR':
            angel_mask = (np.degrees(np.arctan2(x - center_x, center_y - y)) + 360) % 360
        return angel_mask

    if type(angel_mask) is int:
        angel_mask=getMask(images,loc)

    cut_loc,r=getCutLineLoc(image_size=images[0].shape[0],cut_type=cut_type)
    rd_angle=full_angle*(r+0.5)
    # 创建一个掩码，将指定角度范围内的像素设置为True
    mask = (0 <= angel_mask) & (angel_mask < rd_angle)
    if loc=='random':
        loc=np.random.choice(np.array(['right','top']))
    rd_res=random.random()
#   先成pair，然后换，最后shuffle
    if p>rd_res:
        new_images = []
        for image in images:
            new_images.append(image.copy())
        arr = np.arange(len(images))
        arr=arr-1
        for image,idx in zip(images,arr):
            if loc=="top":
                new_images[idx][:cut_loc, :, :]=image[:cut_loc, :, :]
            elif loc=="right":
                new_images[idx][:, cut_loc:,:]=image[:, cut_loc:,:]
            elif "angle" in loc:
                new_images[idx][mask] = image[mask]
        new_images,num,r=data_shuffle_v2(images=new_images, loc=loc, p=p, cut_type=cut_type, img_mix_num=img_mix_num - 1,
                        full_angle=rd_angle)
        return new_images,num+1,r
    else:
        return images,0,abs(r)


transform_two_trans = A.Compose([
    A.OneOf([
        A.RandomGridShuffle(grid=(
            3, 3),
            p=1),
        A.RandomGridShuffle(grid=(
            2, 2),
            p=1),
    ], p=0.5),
])


def position_shuffle_for_array(images, config):
    con_p=config['Con_GridShuffle']['p']
    con_g=config['Con_GridShuffle']['grid_g']
    two_trans=config['two_trans']
    transform = A.Compose([
        A.RandomGridShuffle(grid=(
            con_g, con_g),
            p=con_p)
    ])

    shuffled_images=[]
    shuffled_images_v3=[]
    for image in images:
        image_trans_v2 = transform(image=image)['image']
        shuffled_images.append(image_trans_v2)
        if two_trans:
            shuffled_images_v3.append(transform_two_trans(image=image)['image'])
    if two_trans:

        shuffled_images_v3=np.array(shuffled_images_v3)
    else:
        shuffled_images_v3=images
    return np.array(shuffled_images),shuffled_images_v3


def save_tuple(img_tuple,idx,type,type2):
    imgs=torch.cat(img_tuple).view(-1, 4, 256, 256).float()*0.5+0.5
    imgs=imgs[:,:3]
    for i,img in enumerate(imgs):
        pil_img=torchvision.transforms.ToPILImage()(img)
        pil_img.save(f"{type2}_img/{type}/{idx}-{i}.png")


def normalize_np(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img_normalized = (img - mean) / std
    return img_normalized


class ShuffleDataset(pairDataset):
    def __init__(self, config=None, mode='train'):
        global c
        c=config
        config['GridShuffle']['p']=0
        super().__init__(config, mode)


    def __getitem__(self, index):
        return super().__getitem__(index,norm=False)


    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                        the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        global c
        global save_index

        shape=c['resolution']
        std=c['std'].copy()
        mean=c['mean'].copy()
        channel=3
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])
        norm=True
        if norm:
            fake_images = tuple(normalize_np(img, mean, std) for img in fake_images)
            real_images = tuple(normalize_np(img, mean, std) for img in real_images)
        if c['Bimix_Shuffle']['mix_default']=='origin' or c['Bimix_Shuffle']['apply_mix']:
            mix_images = tuple(itertools.chain(*zip(real_images, fake_images)))
            mix_labels = np.dstack((real_labels, fake_labels)).ravel()
        else:
            mix_images = ()
            mix_labels = ()

        
        fake_images_shuffled,real_images_shuffled,mix_images_shuffled=fake_images,real_images,mix_images
        # fake & fake
        if c['Bimix_Shuffle']['apply_fake']:
            fake_images_shuffled,shuffled,_ = data_shuffle_v2(fake_images, p=c['Bimix_Shuffle']['p'],
                                                              loc=c['Bimix_Shuffle']['loc'],cut_type=c['Bimix_Shuffle']['fake_cut'],
                                                             img_mix_num=c['Bimix_Shuffle']['fake_num'],Mix_type=c['Bimix_Shuffle']['mix_type'])
        ps_fake_imgs,fake_images_shuffled = position_shuffle_for_array(fake_images_shuffled, c)
        #torchvision.transforms.ToPILImage()(torchvision.transforms.ToTensor()(fake_images_shuffled[0][:,:,:3])).show()

        fake_labels_shuffled = torch.LongTensor(fake_labels)





        # real & real
        if c['Bimix_Shuffle']['apply_real']:
            real_images_shuffled,shuffled,r = data_shuffle_v2(real_images, p=c['Bimix_Shuffle']['p'],
                                                              loc=c['Bimix_Shuffle']['loc'],cut_type=c['Bimix_Shuffle']['real_cut'],
                                                             img_mix_num=c['Bimix_Shuffle']['real_num'],Mix_type=c['Bimix_Shuffle']['mix_type'])
            if shuffled:
                real_labels = np.ones(len(real_images), dtype=int) * \
                              ((0.5-r) if not c['Bimix_Shuffle']['real_fix_label'] else c['Bimix_Shuffle']['mix_real_label'])
        ps_real_imgs,real_images_shuffled = position_shuffle_for_array(real_images_shuffled, c)

        real_labels_shuffled = torch.LongTensor(real_labels)





        # real & fake (r,f,r,f,...)
        if c['Bimix_Shuffle']['apply_mix']:
            mix_images_shuffled,shuffled,r = data_shuffle_v2(mix_images, p=c['Bimix_Shuffle']['p'],
                                                             loc=c['Bimix_Shuffle']['loc'],cut_type=c['Bimix_Shuffle']['mix_cut'],
                                                             img_mix_num=c['Bimix_Shuffle']['mix_num'],Mix_type=c['Bimix_Shuffle']['mix_type'])
            if shuffled:
                mix_labels = np.ones(len(mix_images), dtype=int)*c['Bimix_Shuffle']['mix_label'] if c['Bimix_Shuffle']['mix_fix_label'] else np.array([r, 1-r] * (len(mix_images)//2))
        ps_mix_imgs,mix_images_shuffled = position_shuffle_for_array(mix_images_shuffled, c)
        mix_labels_shuffled = torch.LongTensor(mix_labels)

        if norm:
            ps_fake_imgs = tuple(T.ToTensor()(img) for img in
                                 ps_fake_imgs)
            fake_images_shuffled = tuple(T.ToTensor()(img) for img in
                                 fake_images_shuffled)

            ps_real_imgs = tuple(T.ToTensor()(img) for img in
                                 ps_real_imgs)
            real_images_shuffled = tuple(T.ToTensor()(img) for img in
                                 real_images_shuffled)

            ps_mix_imgs = tuple(T.ToTensor()(img) for img in
                                 ps_mix_imgs)
            mix_images_shuffled = tuple(T.ToTensor()(img) for img in
                                 mix_images_shuffled)
        else:
            ps_fake_imgs = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                  ps_fake_imgs))
            fake_images_shuffled = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                          fake_images_shuffled))

            ps_real_imgs = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                  ps_real_imgs))
            real_images_shuffled = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                          real_images_shuffled))

            ps_mix_imgs = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                  ps_mix_imgs))
            mix_images_shuffled = tuple((T.Normalize(std=std, mean=mean)(T.ToTensor()(img)) for img in
                                          mix_images_shuffled))
        # Combine the fake and real tensors and create a dictionary of the tensors
        #, dim=1
        images = torch.cat(fake_images_shuffled+real_images_shuffled+mix_images_shuffled+ps_fake_imgs+ps_real_imgs+ps_mix_imgs).view(-1, channel,shape,shape).float()

        # save_tuple(fake_images_shuffled,type2='clockmix' ,type='fake',idx=save_index)
        # save_tuple(real_images_shuffled,type2='clockmix' ,type='real',idx=save_index)
        # save_tuple(mix_images_shuffled,type2='clockmix' ,type='mix',idx=save_index)
        # save_index+=1
        # save_tuple(ps_fake_imgs,type2='draw' ,type='fake',idx=save_index)
        # save_tuple(ps_real_imgs,type2='draw' ,type='real',idx=save_index)
        # save_tuple(ps_real_imgs,type2='draw' ,type='mix',idx=save_index)
        # save_index += 1
        # phase_without_amplitude(images[0].unsqueeze(0), dim=(-1, -2))
        # phase_without_amplitude(images, dim=(-1, -2))
        labels = torch.cat((fake_labels_shuffled, real_labels_shuffled,mix_labels_shuffled,fake_labels_shuffled, real_labels_shuffled,mix_labels_shuffled))


        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None
        }
        return data_dict



if __name__ == '__main__':

    img1 = np.array(Image.open(r'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/012_026/023.png'))
    img2 = np.array(Image.open(r'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/078_955/050.png'))
    img3 = np.array(Image.open(r'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/161_141/000.png'))
    img4 = np.array(Image.open(r'/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/003_000/019.png'))

    img5=np.array(Image.open(r"/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames/268/047.png"))
    img6=np.array(Image.open(r"/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames/269/149.png"))
    img7=np.array(Image.open(r"/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames/226/073.png"))
    img8=np.array(Image.open(r"/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/deepfakes_detection_datasets/FaceForensics++/original_sequences/youtube/c23/frames/451/172.png"))
    #p2 = phase_without_amplitude(torchvision.transforms.ToTensor()(img1).unsqueeze(0))
    # p1=phase_without_amplitude_np(img1)

    #set_angle_to_zero(img1,90)
    #(images, p, loc, cut_type='half'):
    images = (img1, img2, img3, img4)
    images2 = (img5, img6, img7, img8)
    images3 =(img1, img5, img2, img6,img3, img7, img4, img8)


    mix_num = 2
    mix_type='fake'
    for mix_num in [2]:
        for mix_type in ['fake']:

            if mix_type=='fake':
                random.seed(13)
                np.random.seed(13)
                new_imgs,_,r = data_shuffle_v2(images, p=1, loc='angleR', cut_type='random_mg', img_mix_num=mix_num)
                new_img1=Image.fromarray(new_imgs[0])
                new_img2=Image.fromarray(new_imgs[1])
                new_img3=Image.fromarray(new_imgs[2])
                new_img4=Image.fromarray(new_imgs[3])
                new_img1.save(f'../../draw_img/clockmix_instr_v3/mix/fake/{mix_num}/1.png')
                new_img2.save(f'../../draw_img/clockmix_instr_v3/mix/fake/{mix_num}/2.png')
                new_img3.save(f'../../draw_img/clockmix_instr_v3/mix/fake/{mix_num}/3.png')
                new_img4.save(f'../../draw_img/clockmix_instr_v3/mix/fake/{mix_num}/4.png')
                rd_angle = 360 * (-r+0.5) #注意这里的r用了负数
                # 创建一个掩码，将指定角度范围内的像素设置为True
                mask = (0 <= angel_mask) & (angel_mask < rd_angle)
                overlay = np.zeros((256, 256, 4), dtype=np.uint8)
                # 对于mask中为True的位置，设置颜色为透明度0.5的深绿
                overlay[mask] = [255, 100, 100,  191]

                # 对于mask中为False的位置，设置颜色为透明度0.5的浅绿
                overlay[~mask] = [251, 152, 152,  191]

                # 将numpy数组转换为PIL图像
                overlay = Image.fromarray(overlay, 'RGBA')
                overlay.save(f'../../draw_img/clockmix_instr_v2/mix/{mix_type}/{mix_num}/overlay.png')
            # #
            elif mix_type=='real':
                random.seed(20)
                np.random.seed(20)
                new_imgs2, _, r = data_shuffle_v2(images2, p=1, loc='angleR', cut_type='random_mg', img_mix_num=mix_num)
                new_img21=Image.fromarray(new_imgs2[0])
                new_img22=Image.fromarray(new_imgs2[1])
                new_img23=Image.fromarray(new_imgs2[2])
                new_img24=Image.fromarray(new_imgs2[3])
                new_img21.save(f'../../draw_img/clockmix_instr_v3/mix/real/{mix_num}/1.png')
                new_img22.save(f'../../draw_img/clockmix_instr_v3/mix/real/{mix_num}/2.png')
                new_img23.save(f'../../draw_img/clockmix_instr_v3/mix/real/{mix_num}/3.png')
                new_img24.save(f'../../draw_img/clockmix_instr_v3/mix/real/{mix_num}/4.png')

                rd_angle = 360 * (r + 0.5)
                # 创建一个掩码，将指定角度范围内的像素设置为True
                mask = (angel_mask < rd_angle)
                overlay = np.zeros((256, 256, 4), dtype=np.uint8)

                # 对于mask中为True的位置，设置颜色为透明度0.5的深绿
                overlay[mask] = [ 152, 251, 152, 191]

                # 对于mask中为False的位置，设置颜色为透明度0.5的浅绿
                overlay[~mask] = [100, 255, 100, 191]
                # 将numpy数组转换为PIL图像
                overlay = Image.fromarray(overlay, 'RGBA')
                overlay.save(f'../../draw_img/clockmix_instr_v2/mix/{mix_type}/{mix_num}/overlay.png')

            else:
                random.seed(0)
                np.random.seed(0)
                #
                new_imgs3, _, r = data_shuffle_v2(images3, p=1, loc='angleR', cut_type='random_mg', img_mix_num=mix_num)
                new_img31=Image.fromarray(new_imgs3[0])
                new_img32=Image.fromarray(new_imgs3[1])
                new_img33=Image.fromarray(new_imgs3[2])
                new_img34=Image.fromarray(new_imgs3[3])
                new_img35=Image.fromarray(new_imgs3[4])
                new_img36=Image.fromarray(new_imgs3[5])
                new_img37=Image.fromarray(new_imgs3[6])
                new_img38=Image.fromarray(new_imgs3[7])
                new_img31.save(f'../../draw_img/overall_arch/{mix_num}/1.png')
                new_img32.save(f'../../draw_img/overall_arch/{mix_num}/2.png')
                new_img33.save(f'../../draw_img/overall_arch/{mix_num}/3.png')
                new_img34.save(f'../../draw_img/overall_arch/{mix_num}/4.png')
                new_img35.save(f'../../draw_img/overall_arch/{mix_num}/5.png')
                new_img36.save(f'../../draw_img/overall_arch/{mix_num}/6.png')
                new_img37.save(f'../../draw_img/overall_arch/{mix_num}/7.png')
                new_img38.save(f'../../draw_img/overall_arch/{mix_num}/8.png')

                arr31 = [np.array(new_img31)]
                res=data_shuffle(arr31,3,1)


                rd_angle=360*(r+0.5)
                # 创建一个掩码，将指定角度范围内的像素设置为True
                mask = ((angel_mask-30)% 360 < rd_angle)
                overlay = np.zeros((256, 256, 4), dtype=np.uint8)

                # 对于mask中为True的位置，设置颜色为透明度0.5的深绿
                overlay[mask] = [255, 100, 100,  191]

                # 对于mask中为False的位置，设置颜色为透明度0.5的浅绿
                overlay[~mask] = [100, 255, 100,  191]
                # 将numpy数组转换为PIL图像
                overlay = Image.fromarray(overlay, 'RGBA')
                overlay.save(f'../../draw_img/clockmix_instr_v2/mix/{mix_type}/{mix_num}/overlay.png')
