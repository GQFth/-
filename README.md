[中文版](README.md) | [English](README_EN.md)


<div align="center">
  <img src="assets/log1.png" alt="UprmT_T AI" style="height: 150px; width: auto; max-width: 50%;" />
</div>
<hr>

<div align="center">
  <a href="https://huggingface.co/GQFth/Uprm-i1" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Uprm--i1-ffc107?style=flat" />
  </a>
  <a href="https://swanlab.cn/@020202/multimodal-object-detection/runs/u2nvr8dtqnfs7iv86r7xs/chart" target="_blank">
    <img alt="SwanLab Experiment"  src="https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg" />
  </a>
  <a href="https://github.com/GQFth/UprmT_T" target="_blank">
    <img alt="Contributions Welcome" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" />
  </a>
</div>
<h1 align="center">UprmT_T</h1>

## Usage

CIFAR-10训练集 的 多模态模型， 用于识别图片，简单的模型架构， 让你快速入门机器学习

## Why

当我们学习 机器学习 的时候，我们可能会无从下手，但是官方文档太多太冗杂，很多逻辑会很难运用，是不利于我们学习的。

我们应该以实践为主，理论为辅，而这个库的目的就是从最基础的模型架构开始，只留下核心逻辑，供大家学习。

## How

基于每个函数的功能点，一点一点的拆分出来。

## task decomposition

#### runtime-core-1

- [ ] 简单的CNN模型(用于物体识别，接机械臂控制)
- [ ] 2
- [ ] 3
- [ ] 4
- [ ] 5
- [ ] 6
- [ ] 7
- [ ] 8
- [ ] 9

### runtime-question

- [ ] 完成项目一
  - [x] 构建基于CIFAR-10训练集的CNN模型 
  - [x] 模型太依赖于文本提示，加入噪声干扰，以此来迫使模型学习图像特征
  - [x] 设计的cnn结构太简单, 导致还有模型无法学习图像特征
  - [x] 将图像特征提取层数从 2 -> 3 层， 增加了bn层(增加了其泛化能力, 加快了训练速度等等)， 增加bert文本解码预训练模型。
  - [ ] 发现问题： CIFAR-10训练集的太少，难以承担多模态模型的训练。在测试时难以正确识别图片。方向：使用新的预训练模型，使用新的训练集，训练模型的逻辑能力，训练模型的文本表达能力。
