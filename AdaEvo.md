# AdaEvo——面向多终端的及时、高效边缘辅助模型演化（自适应）系统

Organization: Northwestern Polytechnical University

Author: Lehao Wang, Zhiwen Yu, Haoyi Yu, Sicong Liu, Yaxiong Xie, Bin Guo, Yunxin Liu

To appear at TMC 2023

[[`Paper`](https://facebookresearch.github.io/ImageBind/paper)] [[`CSDN`](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)] [[`github`](https://imagebind.metademolab.com/)] [[`gitee`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`开源中国`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`CrowdHMT`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`BibTex`](#citing-imagebind)]

## 问题

### 概述

基于深度学习的视频分析不可避免地会面临数据漂移问题，即真实场景的视频数据与模型训练数据不相符，数据漂移会降低模型的推理精度，对于部署在终端的模型尤为严重，这其中有两方面原因：

- 在移动端捕捉到的数据中，目标类别的分布变化大

- 部署在终端的轻量化模型结构简单，泛化能力差，对新场景下的目标识别精度低，而边缘辅助的模型持续演化是解决数据漂移问题的一种有效方法。

*AdaEvo* 提出了一个端到端的公平共享边缘有限资源的多终端深度模型整体服务质量优化框架，一方面针对不同的数据漂移类型采用自适应的视频数据筛选策略以实现演化收益最大化，另一方面基于演化任务的性能指标预测进行动态的异步演化任务调度和自适应的边缘资源分配以获得最优整体性能。



## 动机

### 概述

*AdaEvo* 的动机是基于两点发现：（i）不同视频帧对于模型的精度增益具有差异性，少部分的数据即可实现快速的理想的精度增益（ii）任务的合理调度和动态资源分配可以在保证精度增益的同时，大幅度缩减多任务系统的总延迟。

#### 关键挑战

尽管研究者对边缘辅助的模型持续演化进行了广泛的研究，但将其仍存在三个**复杂性的挑战**：

- 现有终端深度模型演化通常采用定期或基于历史数据的演化触发方式，导致冗余或不及时的模型演化，从而加大了系统开销并降低了长生命周期内的终端模型推理精度；
- 现有演化系统的终端筛选视频帧通常是基于动态采样率或帧差法，并不能选出此场景输入下具有代表性的视频数据，存在精度优化空间；
- 已有演化系统通常关注单演化任务的演化效果，而多终端异步演化任务在资源有限的边缘服务器会发生激烈的共享资源争夺，影响系统的整体性能。

## 系统设计

### 系统概述

*AdaEvo* 系统图如图1所示，其创新设计包括三个模块：

- Adaptive mobile DNN evolution trigger: 预测在移动场景中模型的检测精度来检测数据漂移的发生，确定模型演化的触发时间；
- Data drift-aware video frame sampling: 检测终端模型在移动场景中遇到的**数据漂移类型**，并根据漂移类型选择用于模型演化的视频帧
- Mobile DNNN evolution task profiler & asynchronous evolution task scheduler: 预测来自移动终端所有演化任务的显存需求、精度增益和训练时间并进行同一GPU资源池内的服务任务组合的选择和资源分配。

![image](https://github.com/user-attachments/assets/2f234cb0-5dee-4077-890e-771bdb3998a8)
图1 Adaevo系统示意图

### 模块功能

#### Adaptive mobile DNN evolution trigger

该模块设计一个可以由终端快速计算得到的目标检测精度预测指标，辅助进行演化触发。该模块采用了检测置信分数，即分类置信分数与定位置信分数结合，来作为预测指标。

分类置信分数可以在模型向前传播过程中直接得到 $CC_{i}=getCC(f_{θ_{1}},z_{i})=MAX(softmax(z_{i}))$，也是最常使用的置信分数，但是在错误的检测结果中也有可能输出较高的分类置信分数。

定位置信分数则是衡量目标检测定位框位置准确性的衡量指标，我们将其视作一个回归问题，并使用两个全连接层进行预测，我们利用COCO数据集中真实边界框进行伸缩变化产生几个候选框，并以数据特征向量以及真实框和候选框之间的IOU作为数据训练此回归模型，即$LC_{i}=getLC(f_{θ_{2}},z_{i})=Linear(z_{i})$。

而我们采用的目标检测精度预测指标是二者的乘积，$CLC=\frac{1}{N}x\sum_{i=1}^{N} CC_{i}xLC_{i}$。通过实验发现，分类置信分数和定位置信分数都无法单独地衡量目标检测的精度变化，而检测置信分数则可以较为准确的反应其变化趋势。

之后，我们通过滑动窗口，如图2所示，我们定义了精度下降率 $ROD=CLC_{1}−CLC_{2}/CLC_{1}$ ，通过窗口内子窗口之间的CLC的方差比对来确定数据漂移结束的时间。在数据漂移结束后，系统触发演化。

![image](https://github.com/user-attachments/assets/383f0636-8fe3-45db-a94d-13f503dbe39d)
图2.基于滑动窗口的演化触发检测

#### Data drift-aware video frame sampling

该模块旨在筛选出具有代表性的视频帧数据以进行高效的模型演化。而在真实移动场景中，存在不同的漂移类型，主要分为突发性漂移、增量性漂移和渐进性漂移，不同漂移类型的数据分布特征不同，针对不同漂移类型的数据特点采用不同的合适的视频帧筛选策略是必要的。利用上个模块的滑动窗口的伴生结果，可以进行漂移类型的检测：

- 突发性漂移：当滑动窗口的滑动步长小于设定阈值时，说明新数据分布在短时间内达到平稳状态，数据漂移结束，我们视为突发性漂移，此时t1-t3之间均为新数据分布下的数据。因此我们采用固定采样率。

- 增量性漂移和渐进性漂移：此时，我们引入中间概念，即漂移过程中的数据分布与新旧数据之间的差异 $frameDiffer(D_{win_{1}},D_{[t_{1},t_{2}]/2)}$。若d小于一定阈值，则中间概念是旧概念和新概念中的一个，此时的漂移类型为增量性漂移，否则就是渐进性漂移。而对增量性漂移，由于其数据越靠后越接近新数据分布，因此我们采用一个线性增加的采样率，而渐进性漂移我们则采用逐帧筛选，利用欧氏距离筛选出与旧数据特征差异大的视频帧作为代表帧上传。



#### Mobile DNNN evolution task profiler & asynchronous evolution task scheduler: 

该模块旨从多个异步演化中选择最优的任务组合以减少系统的总延迟，如图3所示。为了更好地做出任务调度决策，我们需要得到演化任务所需的显存占用、精度增益和训练所需时间。

- 显存占用：演化任务的显存占用由模型自身参数、中间结果也是feature map、反向传播的梯度、优化器参数以及workspace五个部分组成，通过这五个部分的分析计算可以得到演化任务占用显存。

- 精度增益：通过小部分视频数据的少轮次训练，可以得到精度-周期关系，我们利用非负最小二乘求解器来拟合精度与周期之间的非线性曲线，通过该曲线可以预测得到每个演化任务的精度增益。

- 训练时间：我们利用轻量化的两层神经网络，并将与训练时间相关的因素作为输入进行训练时间的预测。

之后，我们就进行任务调度，我们将任务调度归类为背包问题，并通过精简搜索空间下的动态规划算法选择最优的任务进行服务，以最小化系统的总延迟，伪代码如Algorithm 1所示。

![image](https://github.com/user-attachments/assets/4ddccc3d-d73f-482d-9ab5-e6935b4210ab)
图3.异步任务调度

![image](https://github.com/user-attachments/assets/f4aa9c86-0501-4b06-96c9-349802e0b6a8)


## 实验结果

将AdaEvo与三种基线，即*无适应（A1）*、*域适应（A2）*、*固定采样率（A3）* ，在网络带宽为 10.65 MB/s的环境下基于四个真实移动视频（D1~D4）进行对比。这些视频都是车载摄像头拍摄的三十分钟视频。不同方法下的移动端均部署了压缩后的Faster-RCNN模型。表 5 显示了不同交并比 (IoU) 阈值下 A1、A2、A3 和 AdaEvo 的平均精度 (mAP)。与原始模型（A1）相比，AdaEvo 在不同的 IoU 阈值（IoU=0.50:0.05:0.95、IoU=0.50、IoU=0.75）下，mAP 分别提升了 22.9%、34% 和 29.3%。

![image](https://github.com/user-attachments/assets/4f8e0908-098b-4e32-afab-b464406b73e6)




## Citing AdaEvo

If you find this repository useful, please consider giving a citation.

```
@article{wang2023adaevo,
  title={AdaEvo: Edge-Assisted Continuous and Timely DNN Model Evolution for Mobile Devices},
  author={Wang, Lehao and Yu, Zhiwen and Yu, Haoyi and Liu, Sicong and Xie, Yaxiong and Guo, Bin and Liu, Yunxin},
  journal={IEEE Transactions on Mobile Computing},
  year={2023},
  publisher={IEEE}
}  
```



