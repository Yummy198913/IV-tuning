<div align="center">
<h1>IV-tuning: Parameter-Efficient Transfer Learning <br>
for Infrared-Visible Tasks</h1>


The official implemention for IV-tuning: Parameter-Efficient Transfer Learning for Infrared-Visible Tasks

paper：https://arxiv.org/abs/2412.16654
</div>

## Introduction

IV-tuning is an efficient and effective Parameter-Efficient Transfer Learning (**PETL**) method for Infrared-Visible (**IR-VIS**) tasks. With approximately 3% of the backbone parameters trainable, IV-tuning achieves SOTA performance compared to previous IR-VIS methods, including Image Fusion methods and end-to-end IR-VIS methods in **Object Detection** and **Semantic Segmentation**.

## Framework

![overview](https://github.com/user-attachments/assets/9103a458-0b34-4ea3-acf5-8ad0d3740ccf)

## Main Results

### Semantic Segmentation (MSRS Dataset)

| Backbone-Head    | Method    | #TP               | mIoU              | Config | CKPT | Logs |
| ---------------- | --------- | ----------------- | ----------------- | ------ | ---- | ---- |
| ViT-L+SETR       | FFT       | 304.93M           | 75.08             |        |      |      |
| ViT-L+SETR       | IV-tuning | **8.90M(2.84%)** | **75.51**(+0.43)   |        |      |      |
| ViT-L+Segformer  | FFT       | 304.93M           | 75.42             |        |      |      |
| ViT-L+Segformer  | IV-tuning | **8.90M(2.84%)** | **76.98**(+1.56)   |        |      |      |
| Swin-L+Segformer | FFT       | 192.50M           | 78.23             |        |      |      |
| Swin-L+Segformer | IV-tuning | **6.01M(3.03%)** | **78.60**(+0.37)   |        |      |      |

### Object Detection (M3FD Dataset)

| Backbone-Head  | Method    | #TP               | mAP             | mAP50           | mAP75           | Config | CKPT | Logs |
| ---------------| --------- | ----------------- | --------------- | --------------- | --------------- | ------ | ---- | ---- |
| Swin-L+CO-DETR | FFT       | 192.50M           | 59.5            | 90.2            | 62.5            |        |      |      |
| Swin-L+CO-DETR | IV-tuning | **6.06M(3.01%)** | **61.3**(+1.8)   | **90.6**(+0.4)  | **65.2**(+2.7)  |        |      |      |
| Swin-L+DINO    | FFT       | 192.50M           | 60.2            | 91.1            | 64.1            |        |      |      |
| Swin-L+DINO    | IV-tuning | **6.06M(3.01%)** | **61.1**(+0.9)   | **91.9**(+0.8)  | **66.0**(+1.9)  |        |      |      |
| ViTDet-B       | FFT       | 85.89M            | 44.1            | -               | 48.9            |        |      |      |
| ViTDet-B       | IV-tuning | **3.66M(4.09%)** | **45.0**(+0.9)   | -               | **50.2**(+1.3)  |        |      |      |



## TODO
- [ ] Upload the code, config file, checkpoint model and logs.

## Citation
If you find our work helpful, please cite our paper:

```
@article{zhang2024iv,
  title={IV-tuning: Parameter-Efficient Transfer Learning for Infrared-Visible Tasks},
  author={Zhang, Yaming and Gao, Chenqiang and Liu, Fangcen and Guo, Junjie and Wang, Lan and Peng, Xinggan and Meng, Deyu},
  journal={arXiv preprint arXiv:2412.16654},
  year={2024}
}
```


