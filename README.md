# JigMark: Enhanced Robust Image Watermark against Diffusion Models via Contrastive Learning


<img width="100%" src="https://github.com/pmzzs/JigMark/assets/77789132/5721af1f-83e5-4db1-96cc-3888f56757d6">

This repository is the official implementation of "JigMark: Enhanced Robust Image Watermark against Diffusion Models via Contrastive Learning." Our work introduces a groundbreaking shift in image watermarking, directly addressing the challenges posed by advanced diffusion models. By introducing the HAV score and developing JigMark, we offer a novel solution to protect IP rights in the digital era. This initiative not only paves the way for further research in this field but also underscores the importance of continuous innovation and adaptability in the era of AI and machine learning. JigMark, known for its superior performance and practicality, sets a new standard for watermarking amid evolving digital challenges. We invite the community to explore, contribute, and apply our approach in various contexts to safeguard digital media's integrity and ownership. This repository includes the code, datasets, and additional resources, fostering collaborative advancements in this crucial research area.

# Features
1. **Human Aligned Variation (HAV) Score**: Quantifies human perception of image variations post-diffusion model transformations.
2. **Contrastive Learning**: Boosts watermark adaptability and robustness through contrastive learning.
3. **Jigsaw Puzzle Embedding**: A novel, flexible watermarking technique utilizing a 'Jigsaw' puzzle approach.
4. **High Robustness and Adaptability**: Demonstrates exceptional performance against sophisticated image perturbations and various transformations.

# Requirements
+ Python >= 3.10
+ PyTorch >= 2.0.0
+ diffusers >= 0.14.0
+ accelerate >= 0.21.0

## Training
### Dataset Preparation
Download the ImageNet-1k dataset and organize it in the datasets folder as follows:


```
├── datasets
│   ├── test
│   │   ├── test
│   │       ├── xxx.JPEG
│   │       │
│   │       ├── ...
│   ├── val
│   │   ├── n01440764
│   │   │  ├── xxx.JPEG
│   │   │  │
│   │   │  ├── ...
│   │   ├── ...
```


### Setup Accelerate
Our code utilizes 'accelerate' for multi-GPU training. Set up the accelerate configuration with:

```
accelerate config
```


### Train
Initiate training with:

```
accelerate launch train.py --train_path "Imagenet Path"
```
Trained models will be saved in "./checkpoints/".

## Evaluate

### Download Pretrained Models
Download the following pretrained models:
- Zero 1-to-3: [Official Repository Link](https://cv.cs.columbia.edu/zero123/assets/10500.ckpt) (Place in "./checkpoints/")
- HAV Model: [Download Link]
- JigMark Watermark Model: [Download Link]

After downloading, run `eval.ipynb` for model evaluation.

# Acknowledgement
This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Zero 1-to-3](https://github.com/cvlab-columbia/zero123), and [Diffusers](https://huggingface.co/docs/diffusers/index). We express our gratitude to the authors of these projects for making their code publicly available.
