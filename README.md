# This code is the source code implementation for the paper "Unsupervised Domain Adaptation with Differentially Private Gradient Projection"
## Abstract
Domain adaptation is a viable solution for handling small datasets in deep learning. However, domain adaptation models trained on data containing sensitive information may violate personal privacy. In this paper, we propose a solution for unsupervised domain adaptation, called DP-CUDA, which is based on differentially private gradient projection and contradistinguisher features. Compared with traditional domain adaptation processes, DP-CUDA first searches for domain-invariant features between the source and target domains and then transfers knowledge. Specifically, the model is trained in the source domain by supervised learning from labeled data. During the target model's training, feature learning is directly used in an end-to-end manner using unlabeled data, and differentially private noise is injected into the gradients. We conducted extensive experiments on various benchmark datasets including MNIST, USPS, SVHN, VisDA-2017, Office-31, and Amazon Review to demonstrate the utility and privacy-preserving properties of our proposed method. Paper access: https://doi.org/10.1155/2023/8426839
![DP-CUDA框架图](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/1.png)

# Reference
```
@article{zheng2023unsupervised,
  title={Unsupervised domain adaptation with differentially private gradient projection},
  author={Zheng, Maobo and Zhang, Xiaojian and Ma, Xuebin},
  journal={International Journal of Intelligent Systems},
  volume={2023},
  number={1},
  pages={8426839},
  year={2023},
  publisher={Wiley Online Library}
}
```


# Experimental Environment

```
- Python 3.6.8 (Anaconda Python recommended)
- PyTorch 1.4.0
- torchvision 0.5.0
- nltk 3.4.5
- pandas 1.0.1
- scipy 1.4.1
- tqdm 4.42.1
- six 1.14.0
- backpack-for-pytorch 1.0.0
- scikit-image 0.16.2
- scikit-learn 0.22.1
- tensorboardX 2.0
- tensorflow 1.13.1 (specifically for tensorboard visualizations)
```

## Datasets
`MNIST、USPS、SVHN、VisDA-2017、Office-31、Amazon Review`
![数据集详情](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/2.png))
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/3.png)

## Experimental Setup

-   **Visual Domain Task Settings**:
    -   **Baseline Methods**: Include DPOT, DPSWD, and CUDA (non-private version of DP-CUDA).
    -   **Specific Techniques**:
        -   **DPOT**: Processes source domain data through Johnson-Lindenstrauss transformation and releases a random matrix treated with Laplace noise to the target domain.
        -   **DPSWD**: Calculates the sliced Wasserstein distance between the source and target domains, similar to MMD, to achieve domain adaptation.
    -   **Model Architecture**: Uses ResNet-152 as the base model with ReLU activation.
    -   **Experimental Environment**: Uses the PyTorch framework, running on a Tesla A100 GPU with 40 GB of video memory.
-   **Language Domain Task Settings**:
    -   **Baseline Methods**: Include DPDA, G-DPDA, and CUDA.
    -   **Specific Techniques**:
        -   **DPDA**: Adds Gaussian noise to specific layers of the feature extractor and domain classifier, which determines whether the data comes from the source or target domain.
        -   **G-DPDA**: An enhanced version of DPDA that adds noise perturbation to protect both the source and target domain data.
    -   **Model Architecture**: The encoder contains a hidden layer of 50 neurons for feature extraction, while the classifier uses a fully connected layer with sigmoid activation.
    -   **Dataset**: Uses the Amazon Reviews dataset for experiments.
## Python Files
Here are the English translations for the descriptions of the Python files you provided:
1. **lr_schedule.py** - Responsible for learning rate scheduling, adjusting the learning rate during the training process to improve model convergence.
2. **main.py** - The entry point of the program, controlling the entire execution flow, including data loading, training process, model evaluation, etc.
3. **mmd.py** - May contain implementations of Maximum Mean Discrepancy (MMD), used to compare the similarity of data distributions between source and target domains.
4. **rdp_accountant.py** - Manages and accounts for the privacy budget in differential privacy applications, especially using Rényi Differential Privacy (RDP) methods.
5. **util.py** - Contains helper functions used in the project, such as data handling, metric calculations, etc.
6. **basis_matching.py** - May involve methods for matching basis vectors in different spaces, used in applications where data representation alignment is required.
7. **cmdline_helpers.py** - Contains helper functions for command line interface operations, such as parameter parsing and command logging.

##  Experimental Results
Figures 4 and Tables 4 show the test accuracy of four methods in eight groups of visual tasks.

![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/4.png)
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/5.png)
Figures 5 and Tables 5 show the test accuracy of four methods in language tasks.
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/6.png)
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/7.png)

## Update log

```
- {24.06.13} Uploaded overall framework code and readme file

```


