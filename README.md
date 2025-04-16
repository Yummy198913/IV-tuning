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

### Salient Object Detection

#### VT821 Dataset

| Backbone-Head     | Method    | #TP               | S_alpha           | F_beta_weight     | E_m               | MAE               | Config | CKPT | Logs |
| ----------------  | --------- | ----------------- | ----------------- | --------------    | ----------------- | ----------------- | ------ | ---- | ---- |
| Swin-L+CPNet      | FFT       | 192.50M           | 0.890             | 0.831             | 0.917             | 0.033             |        |      |      |
| Swin-L+CPNet      | IV-tuning | **6.06M(3.15%)**  | **0.909**(+0.019) | **0.859**(+0.028) | **0.938**(+0.021) | **0.027**(-0.006) |        |      |      |
| EVA02-L+Segformer | FFT       | 304.24M           | 0.890             | 0.832             | 0.915             | 0.034             |        |      |      |
| EVA02-L+Segformer | IV-tuning | **8.90M(2.93%)**  | **0.923**(+0.033) | **0.889**(+0.057) | **0.952**(+0.037) | **0.022**(-0.012) |        |      |      |

#### VT1000 Dataset

| Backbone-Head     | Method    | #TP               | S_alpha           | F_beta_weight     | E_m               | MAE               | Config | CKPT | Logs |
| ----------------  | --------- | ----------------- | ----------------- | --------------    | ----------------- | ----------------- | ------ | ---- | ---- |
| Swin-L+CPNet      | FFT       | 192.50M           | 0.939             | 0.918             | 0.966             | 0.015             |        |      |      |
| Swin-L+CPNet      | IV-tuning | **6.06M(3.15%)**  | **0.941**(+0.002) | **0.918**(+0.000) | **0.965**(-0.001) | **0.015**(-0.000) |        |      |      |
| EVA02-L+Segformer | FFT       | 304.24M           | 0.938             | 0.915             | 0.959             | 0.017             |        |      |      |
| EVA02-L+Segformer | IV-tuning | **8.90M(2.93%)**  | **0.948**(+0.010) | **0.931**(+0.016) | **0.975**(+0.016) | **0.013**(-0.004) |        |      |      |

#### VT5000 Dataset

| Backbone-Head     | Method    | #TP               | S_alpha           | F_beta_weight     | E_m               | MAE               | Config | CKPT | Logs |
| ----------------  | --------- | ----------------- | ----------------- | --------------    | ----------------- | ----------------- | ------ | ---- | ---- |
| Swin-L+CPNet      | FFT       | 192.50M           | 0.911             | 0.873             | 0.949             | 0.025             |        |      |      |
| Swin-L+CPNet      | IV-tuning | **6.06M(3.15%)**  | **0.919**(+0.008) | **0.887**(+0.014) | **0.959**(+0.010) | **0.021**(-0.004) |        |      |      |
| EVA02-L+Segformer | FFT       | 304.24M           | 0.907             | 0.866             | 0.943             | 0.026             |        |      |      |
| EVA02-L+Segformer | IV-tuning | **8.90M(2.93%)**  | **0.930**(+0.023) | **0.904**(+0.038) | **0.966**(+0.023) | **0.019**(-0.007) |        |      |      |

### Semantic Segmentation (MFNet Dataset)

| Backbone-Head     | Method    | #TP               | mIoU              | Config | CKPT | Logs |
| ----------------  | --------- | ----------------- | ----------------- | ------ | ---- | ---- |
| EVA02-L+Segformer | FFT       | 304.24M           | 54.53             |        |      |      |
| EVA02-L+Segformer | IV-tuning | **8.90M(2.93%)**  | **59.56**(+5.03)  |        |      |      |
| Swin-L+Segformer  | FFT       | 192.50M           | 56.78             |        |      |      |
| Swin-L+Segformer  | IV-tuning | **6.01M(3.15%)**  | **60.07**(+3.29)  |        |      |      |

### Object Detection (M3FD Dataset)

| Backbone-Head  | Method    | #TP               | mAP             | mAP50           | mAP75           | Config | CKPT | Logs |
| ---------------| --------- | ----------------- | --------------- | --------------- | --------------- | ------ | ---- | ---- |
| Swin-L+CO-DETR | FFT       | 192.50M           | 59.5            | 90.2            | 62.5            |        |      |      |
| Swin-L+CO-DETR | IV-tuning | **6.06M(3.15%)**  | **61.3**(+1.8)  | **90.6**(+0.4)  | **65.2**(+2.7)  |        |      |      |
| Swin-L+DINO    | FFT       | 192.50M           | 60.2            | 91.1            | 64.1            |        |      |      |
| Swin-L+DINO    | IV-tuning | **6.06M(3.15%)**  | **61.1**(+0.9)  | **91.9**(+0.8)  | **66.0**(+1.9)  |        |      |      |




## TODO
- [ ] Upload the code, config file, checkpoint model and logs once the work is accepted.

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


