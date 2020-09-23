# Conditional Sequential Modulation for Efficient Global Image Retouching [Paper Link](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580664.pdf)
By Jingwen He*, Yihao Liu*, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/), and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=en) (* indicates equal contribution)


## Datasets

Here, we provide the preprocessed datasets: [MIT-Adobe FiveK dataset](https://drive.google.com/drive/folders/1qrGLFzW7RBlBO1FqgrLPrq9p2_p11ZFs?usp=sharing), which contains both training pairs and testing pairs.

## Abstract

Photo retouching aims at improving the aesthetic visual quality of images that suffer from photographic defects such as over/under exposure, poor contrast, inharmonious saturation. In practice, photo retouching can be accomplished by a series of image processing operations. As most commonly-used retouching operations are pixelindependent, we can take advantage of this property and design a specialized algorithm for efficient global photo retouching. We first analyze these global operations and find that they can be mathematically formulated by a Multi-Layer Perceptron (MLP). Based on this observation, we propose an extremely light-weight framework - Conditional Sequential Retouching Network (CSRNet). CSRNet consists of a base network and a condition network. The base network acts like an MLP that processes each pixel independently, while the condition network extracts the global features of the input image to generate a condition vector. To realize retouching operations, we modulate the intermediate features using Global Feature
Modulation (GFM), of which the parameters are transformed by the condition vector. Benefiting from the utilization of 1 × 1 convolution, CSRNet only contains less than 37k trainable parameters, which are orders of magnitude smaller than existing learning-based methods. Extensive experiments show that our method achieves state-of-the-art performance on the benchmark MIT-Adobe FiveK dataset quantitively and qualitatively.
