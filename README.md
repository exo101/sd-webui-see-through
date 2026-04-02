# See-Through WebUI Plugin

## 功能介绍

See-Through是一个基于AI的动漫图像分解插件，可以将单张动漫图像转换为多层2.5D模型。这是基于[See-Through: Single-image Layer Decomposition for Anime Characters](https://github.com/shitagaki-lab/see-through)项目的研究成果。

### 核心功能

- **图像分解** - 将单张动漫图像分解为最多23个语义层
- **LayerDiff 3D** - 基于扩散的透明层生成
- **Marigold深度估计** - 专为动漫优化的伪深度估计
- **SAM分割** - 语义身体部分分割
- **PSD输出** - 导出为可编辑的Photoshop文件

### 分解的图层包括

头发、脸部、眼睛、服装、配饰等最多23个语义层，每个图层都经过完整的修复和排序。

## 安装方法

1. 将插件目录复制到WebUI的extensions文件夹中
2. 重启WebUI
3. 在设置中启用See-Through插件

## 使用方法

### 基本使用

1. 在txt2img或img2img界面中找到"See-Through Layer Decomposition"折叠面板
2. 勾选"启用See-Through"选项
3. 选择输入模式：
   - **使用生成的图像** - 处理WebUI生成的图像
   - **上传图像** - 上传本地图像进行处理
   - **指定图像路径** - 输入图像文件路径
4. 配置处理选项：
   - 保存为PSD文件
   - 保存深度图
   - 保存分割掩码
5. 点击"开始处理"按钮

### 高级选项

- **使用LayerDiff 3D** - 启用基于扩散的透明层生成
- **使用Marigold深度估计** - 启用深度估计
- **使用SAM分割** - 启用语义分割

### 模型下载

- 模型文件会保存在 `extensions\sd-webui-see-through\see-through\models` 目录中
- seethroughv0.0.1_marigold
- seethroughv0.0.2_layerdiff3d


### 输出文件

处理完成后，输出文件保存在 `extensions\sd-webui-see-through\see-through\workspace\layerdiff_output` 目录中，包括：

- 分层的PSD文件
- 深度图（如果启用）
- 分割掩码（如果启用）

## 依赖项

# See-Through Plugin Dependencies
# 基础依赖
numpy
opencv-python
Pillow
torchvision
transformers

# See-Through核心依赖
segment-anything
groundingdino-py
diffusers
accelerate
omegaconf
einops
pytorch-lightning

# 图像处理
opencv-contrib-python
scikit-image
imageio

# COCO数据集工具
pycocotools

# PSD文件处理
psd-tools

# 可选依赖（根据需要安装）
# detectron2  # 用于身体解析
# mmcv       # 用于动漫实例分割
# mmdet      # 用于动漫实例分割

## 注意事项

- 处理时间会根据图像大小和硬件性能而变化
- 建议使用GPU以获得最佳性能
- 输出目录需要在See-Through项目目录中

## 项目参考

本插件基于以下研究项目：
- [See-Through GitHub](https://github.com/shitagaki-lab/see-through)
- 论文：See-through: Single-image Layer Decomposition for Anime Characters
- 发表于：ACM SIGGRAPH 2026

## 常见问题

**Q: 处理失败怎么办？**
A: 检查See-Through项目是否正确安装，确保所有依赖项都已安装。

**Q: 可以批量处理图像吗？**
A: 目前插件支持单张图像处理，如需批量处理，请直接使用See-Through项目的命令行工具。

**Q: 输出的PSD文件如何使用？**
A: 可以在Photoshop或其他支持PSD的软件中打开，每个图层都是独立的，可以进行编辑和动画制作。

## 技术支持

如有问题，请参考：
- See-Through项目的GitHub仓库
- WebUI的扩展文档
