<div id="top"></div>

# Segformer-Keras
Implementation of Segformer in Keras

<!-- ABOUT THE PROJECT -->
## About The Project

  This repository implements Segformer, introduced in the paper: [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) using Keras. The original implementation is in Pytorch and is available on the open-mmlab/mmsegmentation repo. The implementation is heavily guided by [Implementing SegFormer in PyTorch](https://towardsdatascience.com/implementing-segformer-in-pytorch-8f4705e2ed0e) where the author goes thru the implementation step by and step and in great detail.

Pretrained weights is only available for:
1. People segmentation (trained on 64k images from COCO on only a single class, person)


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Using the Repo -->

## Using the Repo

### Getting Started

1. Clone this repository 
2. Install the required packages using the requirements.txt. 
3. Accepted ground truth is in the format of a mask png, where the pixel values corresponds to the class. The size of the mask will have to be W/4;H/4, where W and H are the width and the height of the input shape respectively. (A helper notebook to convert COCO anns to mask is provided)
4. The model, dataset and training parameters are controlled by the config file

<br/>

### Training

In the root folder, run

```
python train.py 
```
- Tensorboard logs will be saved in the logs directory. 
- Model weights will be saved in the weights directory

<br/>

### Evaluation

Only validation accuracy is being evaluated in this repo. mIOU is not implemented yet.

<p align="right">(<a href="#top">back to top</a>)</p>

### Inferencing

A helper notebook is provided for visualization of the results.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Trained Weights -->

## Trained Weights

[Segmentation_B0_1_class](https://drive.google.com/file/d/1ejXOjzURQOheZGAQJyssK86FB-VKnSdv/view?usp=share_link)

<p align="right">(<a href="#top">back to top</a>)</p>

## Issues and TO DO

Segmentation annotations in COCO have overlapping annotations which causes some issues in some images when deciding which classes should preside over which classes. Nevertheless, I have tried training the model on the full 90 classes with COCO but due to computational limitations, I have stopped at 10 epochs. The model performance leaves much to be desired. If anyone could suggest anything (optimizer, LR, etc.) to improve training please drop me a message.

I have requested for cityscapes and ADE20k datasets but still pending... (as of 3 Dec 2022)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

The code is published under the Apache License 2.0.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- References -->
## References

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [Implementing SegFormer in PyTorch](https://towardsdatascience.com/implementing-segformer-in-pytorch-8f4705e2ed0e)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

<p align="right">(<a href="#top">back to top</a>)</p>
