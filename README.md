## Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention

This repository is the code for [*Deep Fashion Analysis with Feature Map Upsampling and Landmark-driven Attention*](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view) in the First Workshop on Computer Vision for Fashion, Art and Design (Fashion) of ECCV 2018.

![network](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018/blob/master/images/network.png)

### Requirements

Python 3, PyTorch >= 0.4.0, and make sure you have installed TensorboardX:

```
pip install tensorboardX
```

### Quick Start

__1\. Prepare the Dataset__

Download the "Category and Attribute Prediction Benchmark" of the DeepFashion dataset from http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html . Extract all the files to a folder and put all the images in a folder named "img".

For example, if you choose to put the dataset to /home/user/datasets/benchmark1/, the structure of this folder will be:

```
benchmark1/
    Anno/
    Eval/
    img/
    README.txt
```

Please modify the variable "base_path" in src/const.py correspondingly:

```
# in src/const.py
base_path = "/home/user/datasets/benchmark1/"
```


__2\. Create info.csv__

```
python -m src.create_info
```

Please make sure you have modified the variable "base_path" in src/const.py, otherwise you may encounter a FileNotFound error. After the script finishes, you will find a file named "info.csv" in your "base_path".

__3. Train the model__

To train the landmark branch solely, run:

```
python -m src.train --conf src.conf.lm
```


To train the landmark branch and the category/attribute prediction network jointly, run:

```
python -m src.train --conf src.conf.whole
```


### Monitor your training

You can monitor all the training losses and evaluation metrics via tensorboard. Please run:

```
tensorboard --logdir runs/
```

Then visit localhost:6006 for detailed information.

### Results

The following table shows the landmark localization results on the DeepFashion dataset. Numbers stands for normalized distances between prediction and the ground truth. Best results are marked in bold.

| Methods     | L.Collar   | R.Collar   | L.Sleeve | R.Sleeve | L.Waistline | R.Waistline | L.Hem      | R.Hem      | Avg.   |
|-------------|------------|------------|----------|----------|-------------|-------------|------------|------------|--------|
| [FashionNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)  | 0.0854     | 0.0902     | 0.0973   | 0.0935   | 0.0854      | 0.0845      | 0.0812     | 0.0823     | 0.0872 |
| [DFA](https://arxiv.org/pdf/1608.03049)         | 0.0628     | 0.0637     | 0.0658   | 0.0621   | 0.0726      | 0.0702      | 0.0658     | 0.0663     | 0.0660 |
| [DLAN](https://arxiv.org/pdf/1708.02044)        | 0.0570     | 0.0611     | 0.0672   | 0.0647   | 0.0703      | 0.0694      | 0.0624     | 0.0627     | 0.0643 |
| [Wang et al.](http://web.cs.ucla.edu/~yuanluxu/publications/fashion_grammar_cvpr18.pdf) | 0.0415     | 0.0404     | 0.0496   | **0.0449**   | 0.0502      | 0.0523      | **0.0537** | **0.0551** | 0.0484 |
| Ours        | **0.0332** | **0.0346** | **0.0487**   | 0.0519   | **0.0422**      | **0.0429**  | 0.0620     | 0.0639     | **0.0474** |

The following table shows the category classification and attribute prediction results on the DeepFashion dataset. The two numbers in each cell stands for top-3 and top-5 accuracy. Best results are marked in bold.

| Methods         | Category               | Texture                | Fabric         | Shape                  | Part                    | Style              | All                |
|:---------------:|:----------------------:|:----------------------:|:--------------:|:----------------------:|:-----------------------:|:------------------:|:------------------:|
| [WTBI](https://pdfs.semanticscholar.org/b185/f0a39384ceb3c4923196aeed6d68830a069f.pdf)            |  43.73 \| 66.25        | 24.21 \| 32.65         | 25.38 \| 36.06 | 23.39 \| 31.26         | 26.31 \| 33.24          | 49.85 \| 58.68     | 27.46 \| 35.37     |
| [DARN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_Cross-Domain_Image_Retrieval_ICCV_2015_paper.pdf)            | 59.48 \| 79.58         | 36.15 \| 48.15         | 36.64 \| 48.52 | 35.89 \| 46.93         | 39.17 \| 50.14          | 66.11 \| 71.36     | 42.35 \| 51.95     |
| [FashionNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)      | 82.58 \| 90.17         | 37.46 \| 49.52         | 39/30 \| 49.84 | 39.47 \| 48.59         | 44.13 \| 54.02          | 66.43 \| 73.16     | 45.52 \| 54.61     |
| [Lu et al.](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lu_Fully-Adaptive_Feature_Sharing_CVPR_2017_paper.pdf)       | 86.72 \| 92.51         |        -              |     -         |     -                  |     -                   |     -              |     -              |
| [Corbiere et al.](https://arxiv.org/pdf/1709.09426) | 86.30 \| 92.80         | 53.60 \| 63.20         | 39.10 \| 48.80 | 50.10 \| 59.50         | 38.80 \| 48.90          | 30.50 \| 38.30     | 23.10 \| 30.40     |
| [Wang et al.](http://web.cs.ucla.edu/~yuanluxu/publications/fashion_grammar_cvpr18.pdf)       | 90.99 \| 95.78         | 50.31 \| 65.48         | 40.31 \| 48.23 | 53.32 \| 61.05         | 40.65 \| 56.32          | 68.70 \| **74.25** | 51.53 \| 60.95     |
| Ours          | **91.16** \| **96.12** | **56.17** \| **65.83** | **43.20** \| **53.52** | **58.28** \| **67.80** | **46.97** \| **57.42**  | **68.82** \| 74.13 | **54.69** \| **63.74** |

### Citation

The paper are going to be published soon. You can find the full text [here](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view).