# Face-X-ray
The author's unofficial PyTorch re-implementation of Face Xray 

This repo contains code for the BI data generation pipeline from  [Face X-ray for More General Face Forgery Detection](https://arxiv.org/abs/1912.13458) by Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, Baining Guo.

# Usage

Just run bi_online_generation.py and you can get the following result. which is describe at Figure.5 in the paper.

![demo](all_in_one.jpg)

To get the whole BI dataset, you will need crop all the face and compute the landmarks as describe in the code.