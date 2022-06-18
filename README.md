# VGGCmp
 基于VGGFace实现的人脸识别认证网络

# 数据集
本项目使用哥伦比亚大学PibFig公众人物人脸数据集进行训练与测试。本节将主要介绍数据集的基本情况及获取方法。

## 数据集简介
PibFig数据集的简介如下：
> The PubFig database is a large, real-world face dataset consisting of 58,797 images of 200 people collected from the internet. Unlike most other existing face datasets, these images are taken in completely uncontrolled situations with non-cooperative subjects. 

PibFig数据集的具体指标如下：
- 数据集用途：人脸识别、人脸目标检测等
- 数据集呈现形式：图片来源url
- 数据集表项：
  - person/人名
  - imagenum/（同一人名下的）图片序号
  - url/图片来源url
  - rect/人脸在图片中的位置
  - md5sum/图片MD5校验码
- 数据集大小：
  - 训练集：60人-共16336张图片
  - 测试集：140人-共42461张图片
原生数据集相关文件详见[原生数据文件夹`./dataset/PubFig`](./dataset/PubFig)及[原数据集官网](https://www.cs.columbia.edu/CAVE/databases/pubfig/download/)

## 数据集压缩
针对于本项目所要完成的人脸认证任务，我们对数据集进行了如下筛选：
- 由于数据集中的url大量失效，以及考虑到本项目所采用的网络规模并不大，故对于每一识别对象仅采用前10张url有效的图片
- 由于本项目不涉及目标识别，故预先根据数据集内的位置标识对原图像进行裁剪，并统一调整为128×128大小（VGGFace输入大小）

## 数据集获取：原url下载
可通过在项目文件夹下运行如下命令，以手动按照前述数据集压缩设置下载并裁剪数据集图片至`./dataset/train`与`./dataset/eval`:
1. 切换至`./dataset`文件夹
   ```
   cd dataset
   ```
2. 将`./dataset/PubFig`下的原数据集txt文件转换为csv文件并保存至`./dataset/csvs`以便后续处理
   ```
   python txt2csv.py
   ```
3. 开始下载前，建议启动VPN，否则大部分图片将无法获取
4. 下载图片至`./dataset/train`与`./dataset/eval`
   ```
   python im_download.py
   ```
当然，您可以通过修改`./dataset/im_download.py`中的参数，以对下载设置进行调整

## 数据集获取：预下载范例数据集
由于原数据集提供的url连接并不稳定，故在`./VGGCmp_dataset.zip`内提供了已经按照“数据集压缩”中要求下载并处理好的数据集图片，将其解压并将对应图片复制到前述相应文件夹即可使用。