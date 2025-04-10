# ECE57000 Artificial Intelligence Course Project - Denoising Diffusion Probabilistic Models (DDPM)
**ECE57000 Artificial Intelligence Final Project on Denoising Diffusion Probabilistic Model implementation with PyTorch**

This project explores the use of Denoising Diffusion Probabilistic Models (DDPM) for image restoration and generative tasks. It implements a U-Net based architecture along with a noise scheduling mechanism to progressively denoise images, recovering clean outputs from noisy inputs. The project focuses on two popular image datasets—MNIST and CIFAR-10—demonstrating how diffusion models can be applied for both image generation and enhancement. By leveraging recent advances in self-supervised learning and probabilistic modeling, the project provides insights into state-of-the-art techniques for synthesizing high-quality images while also evaluating performance through quantitative metrics.
## Table of Contents

- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Code Structure](#code-structure)
- [Experimental Results](#experimental-results)
- [References](#references)

## Overview

This project focuses on implementing Denoising Diffusion Probabilistic Models (DDPM) to perform image denoising and generation tasks. By leveraging a U-Net architecture with time-conditioned residual blocks, the system progressively adds and then removes noise from images to recover clean images from noisy inputs or to generate new samples from random noise. The implementation is applied to popular benchmark datasets such as MNIST (grayscale) and CIFAR-10 (colored) to demonstrate versatility across different image types.

Key aspects of this project include:

- **Self-Supervised Denoising:**  
  The model is trained in a self-supervised manner where noisy images serve both as input and target for denoising, eliminating the need for paired clean images.

- **U-Net Based Architecture:**  
  A U-Net architecture forms the backbone of the denoising model, featuring encoder and decoder paths with skip connections that help preserve fine details during the noise removal process.

- **Noise Scheduling:**  
  A robust noise scheduler is used to control the diffusion process by configuring parameters such as the number of timesteps, beta start, and beta end values. This process governs how noise is incrementally added to, and subsequently removed from, the images.

- **Training and Sampling Pipelines:**  
  The project includes a comprehensive training loop that optimizes the model through iterative loss minimization. Additionally, the sampling (or inference) routine allows for generating high-quality images from random noise, showcasing the reverse diffusion process.

- **Evaluation Metrics:**  
  Quantitative evaluation methods, such as the Frechet Inception Distance (FID), are implemented to assess the performance and quality of the generated images.

This comprehensive framework not only demonstrates the practical application of diffusion models for image restoration and synthesis but also provides a strong foundation for further exploration and development in the realm of generative modeling.
## Setup and Installation
1. Clone the repository and extract the ZIP File.
    
    ```git clone https://anonymous.4open.science/r/ECE570-DDPM-0B23.git```
2. Upload the desired Notebook (**.ipynb** file) on Google Drive.
3. Go to the toolbar, **Runtime > Change Runtime Type > Switch to V2-8 TPU > Save**.
4. Run through the configuration cells.

---
**Optional**
> If you wish to train the model faster, you can choose **A100 GPU** or **T4 GPU** as your runtime type instead. However, that would require you to purchase membership.
## Code Structure
1. Dataset Class:

+ Provides implementations for loading and pre-processing datasets (e.g., MNIST and CIFAR-10).

+ Implements standard PyTorch Dataset functionality to facilitate batch loading and data augmentation.

2. U-Net Model:

+ Contains the network architecture definition for the U-Net used within the DDPM framework.

+ Implements residual blocks with time conditioning and handles both downsampling (encoder) and upsampling (decoder) operations.

3. Noise Scheduler:

+ Defines the noise scheduling parameters (e.g., number of timesteps, beta start/end values).

+ Includes functions for computing and applying the diffusion process over a sequence of timesteps.

4. Training:

+ Implements the training loop which optimizes the model over batches of noisy images.

+ Contains loss calculations, backpropagation, and checkpointing for model weights.

5. Sampling:

+ Provides routines to generate new images by iteratively denoising a sample starting from pure noise.

+ Implements methods to visualize and save the generated samples during and after training.

6. Inference:

+ Offers scripts to load trained models and perform image reconstruction or generation on new input data.

+ Includes commands and functions to deploy the model for real-time or batch inference.

7. Evaluation:

+ Implements evaluation metrics (e.g., Frechet Inception Distance) to assess the quality of the generated images.

+ Contains routines for quantitative and qualitative analysis of model performance on benchmark datasets.
## Experimental Results

|Dataset        | FID Score     |
| ------------- |:-------------:|
| **MNIST**         | `58.8`          |
| **CIFAR-10**      | `91.6`          |

The Frechet Inception Distance (FID) score is employed as the primary metric to assess the similarity between the distribution of generated images and that of real images. FID calculates the distance between the feature representations extracted from an Inception network for both sets of images, capturing aspects of image quality and diversity.

For the MNIST dataset, our model produced an FID score of **58.8**, which is higher (worse) than expected for this relatively simple dataset. For the CIFAR-10 dataset, the model achieved an FID score of **91.6**, demonstrating poor performance and unideal outcome. These results suggest that while our DDPM framework is capable of generating diverse samples for both datasets, the challenges posed by CIFAR-10 (e.g. increased image complexity and color diversity) necessitate further optimization.
## References


- **Xie, Yaochen; Wang, Zhengyang; Ji, Shuiwang. (2020).**  
  [Noise2Same: Optimizing A Self-Supervised Bound for Image Denoising](https://arxiv.org/abs/2010.11971)  
  In *Advances in Neural Information Processing Systems*.

- **Ho, Jonathan; Jain, Ajay; Abbeel, Pieter. (2020).**  
  [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
  In *Advances in Neural Information Processing Systems*.

- **Kawar, Bahjat; Elad, Michael; Ermon, Stefano; Song, Jiaming. (2022).**  
  [Denoising Diffusion Restoration Models](https://arxiv.org/abs/2201.11793)  
  In *Advances in Neural Information Processing Systems*.

- **Nichol, Alex; Dhariwal, Prafulla. (2021).**  
  [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)  


- **Kumar, Tushar. (2023).**  
  [Denoising Diffusion Probabilistic Models | DDPM Explained](https://www.youtube.com/watch?v=H45lF4sUgiE)  
  Video available on YouTube.

- **Ronneberger, Olaf; Fischer, Philipp; Brox, Thomas. (2015).**  
  [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)  

- **Gao, Ruiqi; Nijkamp, Erik; Kingma, Diederik P.; Xu, Zhen; Dai, Andrew M.; Wu, Ying Nian. (2020).**  
  [Flow Contrastive Estimation of Energy-Based Models](https://arxiv.org/abs/1912.00589)
