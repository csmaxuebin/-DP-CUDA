# 本代码是论文《Unsupervised Domain Adaptation with Differentially Private Gradient Projection》的源码实现



## 论文简介
![DP-CUDA框架图](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/1.png)
领域适应是深度学习中处理小数据集的一个可行解决方案。然而，基于包含敏感信息数据训练的领域适应模型可能会侵犯个人隐私。在本文中，我们提出了一种无监督领域适应的解决方案，称为DP-CUDA，该方案基于差分隐私梯度投影和对立特征识别。与传统的领域适应过程相比，DP-CUDA首先在源域和目标域之间寻找域不变特征，然后进行知识转移。具体来说，模型在源域通过有标签数据的监督学习进行训练。在目标模型的训练过程中，使用未标记数据直接以端到端的方式进行特征学习解决分类任务，并在梯度中注入差分隐私噪声。我们在多个基准数据集上进行了广泛的实验，包括MNIST、USPS、SVHN、VisDA-2017、Office-31和Amazon Review，以展示我们提出的方法的效用和隐私保护特性。论文访问地址：(https://doi.org/10.1155/2023/8426839)
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


# 实验环境

- Python 3.6 (Anaconda Python recommended)

- PyTorch

- torchvision

- nltk

- pandas

- scipy

- tqdm

- six

- backpack-for-pytorch

- scikit-image

- scikit-learn

- tensorboardX

- tensorflow==1.13.1 (for tensorboard visualizations)

## 数据集
`本代码所使用的数据集如下，均为公开数据集
MNIST、USPS、SVHN、VisDA-2017、Office-31、Amazon Review`
![数据集详情](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/2.png))
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/3.png)

## 实验设置

-   **视觉域任务设置**：
    
    -   **基线方法**：包括DPOT、DPSWD以及CUDA（非隐私保护版本的DP-CUDA）。
    -   **具体技术**：
        -   **DPOT**：通过Johnson-Lindenstrauss变换处理源域数据，并在目标域释放经过拉普拉斯噪声处理的随机矩阵。
        -   **DPSWD**：计算源域和目标域之间的切片Wasserstein距离，通过最小化该距离实现域自适应。
    -   **模型架构**：使用ResNet-152作为基础模型，采用ReLU激活函数。
    -   **实验环境**：使用PyTorch框架，运行在配备有40 GB视频内存的Tesla A100 GPU上。
-   **语言域任务设置**：
    
    -   **基线方法**：包括DPDA、G-DPDA以及CUDA。
    -   **具体技术**：
        -   **DPDA**：在特征提取器和领域分类器的特定层中添加高斯噪声，通过领域分类器判断数据来自源域还是目标域。
        -   **G-DPDA**：DPDA的增强版本，增加了噪声扰动，同时保护源域和目标域数据。
    -   **模型架构**：编码器包含50个神经元的隐藏层用于特征提取，分类器使用全连接层和Sigmoid激活函数。
    -   **数据集**：使用Amazon Reviews数据集进行实验。
-   **参数设置**：
    
    -   **学习率**：使用Adam优化器，初始学乃率设为0.001，每30个epoch衰减率为0.6。
    -   **隐私预算**：探索隐私预算k的不同设置（k = 2，5，8，10）对模型性能的影响。
    -   **基向量数量**：设置为k = 1000，用于构建投影子空间。

##  实验结果
图4和表4展示四种方法在八组视觉任务中的测试准确度

![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/4.png)
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/5.png)
图5和表5展示四种方法在语言任务中的测试准确度
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/6.png)
![输入图片说明](https://github.com/csmaxuebin/-DP-CUDA/blob/main/pic/pic/7.png)

## Update log

```
 - {24.06.13}上传整体框架代码和reademe文件
```


