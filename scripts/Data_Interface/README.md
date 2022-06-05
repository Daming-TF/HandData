# 目录
- [功能介绍](#功能介绍)
- [文件结构](#文件结构)
- [数据集介绍](#数据集介绍)  


## 功能介绍
![image](https://github.com/Daming-TF/HandData/blob/master/material/%E5%8A%9F%E8%83%BD%E4%BB%8B%E7%BB%8D.jpg)  
Data_Interface文件夹主要用于处理不同的开源数据集，转化为训练所需要的coco格式json文件；主要功能如下：  
- [x] 生成训练数据集  
  - 全图模式
  - 剪裁模式（预处理）
- [x] 重新标注数据处理 
  - 数据上传
  - 检查重标数据质量
  - 数据覆盖更新
- [x] 原始数据可视化
- [x] 其他工具
  - 检查分辨率分布
  - 全图和剪裁图片badcase对应关系


## 文件结构
![image](https://github.com/Daming-TF/HandData/blob/master/material/%E6%96%87%E4%BB%B6%E7%BB%93%E6%9E%84.jpg)  
Data_Interface 的每一个子目录对应一个开源数据的处理接口
```Note```: cmu目录下有三个开源数据集的子目录  
以上图的coco_wholebody作为例子：
- check_mark_diff.py  
```Note```:  
主要用于检查标注团队反馈回来的数据质量如何，便于作出反馈
- concurrent2_weed_out_badcase_from_whole_body.py  
```Note```:  
由于开始使用预处理数据crop image作为训练数据，前面做了一些数据清洗工作都是基于crop image展开， 后面为了更好保留数据信息希望把所有信息转移到全图上，后面运用到其他任务或者想做其他数据处理比较方便， 所以需要把crop image badcase和whole image的联系起来找到whole image中的badcase
- convert_coco_format.py  
```Note```:  
原始数据预处理得到训练所需要的crop image以及json文件
- convert_coco_format_from_whole_body-v2_6.py  
```Note```:  
原始数据预处理得到训练所需要的whole image以及json文件
- data_update  
```Note```:  
重标数据上传到标注平台  
- update_json--batch_data_coverage.py  
```Note```:  
重标数据经由标注平台重新标注返回批次数据以及无效数据，该程序是根据批次数据对训练数据进行覆盖操作
- update_json--invalid_data_coverage.py
```Note```:  
去除标注团队反馈的无效样本
- json_update_integration_v2-6.py  
```Note```:  
重标数据一体化更新程序包括：1.根据txt文件去除自检的无效样本数据；2.根据标注团队反馈的批次数据对数据进行覆盖；3.去除标注团队反馈回来的无效样本


## 数据集介绍
- [You Tube 3D Hands](##YouTube3DHands)
- [Halpe Full-Body Human Keypoints and HOI-Det dataset](#2.Halpe Full-Body Human Keypoints and HOI-Det dataset)
- [handpose_x_gesture_v1](#3.handpose_x_gesture_v1(静态手势识别数据集))
- [FreiHand Dataset](#FreiHandDataset)
- [MPII Human Pose Dataset](##5.MPII Human Pose Dataset)



### YouTube3DHands
`Homepage`: https://www.arielai.com/mesh_hands/  

`Git`: https://github.com/arielai/youtube_3d_hands/  

`Paper`: Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild,CVPR2020  

`Introduction`:  
        The dataset contains 3D vertex coordinates of 50,175 hand meshes aligned with in the wild images comprising hundreds of subjects performing a wide variety of tasks.
The training set was generated from 102 videos resulting in 47,125 hand annotations. The validation and test sets cover 7 videos with an empty intersection of subjects with the training set and contain 1,525 samples each.
The dataset has been collected in a fully automated manner. Please, refer to our paper for the details.https://github.com/arielai/youtube_3d_hands.git  
![image](https://github.com/Daming-TF/HandData/blob/master/material/YT3D.jpg)


### 2.Halpe Full-Body Human Keypoints and HOI-Det dataset
`Git`: https://github.com/Fang-Haoshu/Halpe-FullBody  

`Paper`: 
- RMPE: Regional Multi-person Pose Estimation, ICCV2017; 
- PaStaNet: Toward Human Activity Enowledge Engine, CVPR2020;  

`Introduction`:  
        Halpe is a joint project under AlphaPose and HAKE. It aims at pushing Human Understanding to the extreme. We provide detailed annotation of human keypoints, together with the human-object interaction trplets from HICO-DET. For each person, we annotate 136 keypoints in total, including head,face,body,hand and foot. Below we provide some samples of Halpe dataset.  
![image](https://github.com/Daming-TF/HandData/blob/master/material/HFB.jpg)


## 3.handpose_x_gesture_v1(静态手势识别数据集)
`Homepage`: https://codechina.csdn.net/EricLee/classification  

`Paper`:
- Hand gesture recognition with Leap Motion and Kinect devices, ICIP2014;
- Hand Gesture Recognition with Jointly Calibrated Leap Motion and Depth Sensor,  

`Introduction`:  
        数据集来源3部分，且网上数据占绝大多数，具体：1）来源于网上数据并自制；2）来源于自己相机采集并自制；3）来源于Kinect_leap_dataset数据集并自制，其官方网址为: https://lttm.dei.unipd.it/downloads/gesture/  
![image](https://github.com/Daming-TF/HandData/blob/master/material/HXG.jpg)


### FreiHandDataset
`Homepage`: https://lmb.informatik.uni-freiburg.de/projects/freihand/  

`Git`: https://github.com/lmb-freiburg/freihand  

`Paper`: FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images, ICCV2019  

`Introduction`:  
        In our recent publication we presented the challenging FreiHAND dataset, a dataset for hand pose and shape estimation from single color image, which can serve both as training and benchmarking dataset for deep learning algorithms. It contains 4*32,560 = 130,240 training and 3,960 evaluation samples. Each training sample provides:
- RGB image (224x224 pixels)
- Hand segmentation mask (224x224 pixels)
- Intrinsic camera matrix K
- Hand scale (metric length of a reference bone)
- 3D keypoint annotation for 21 Hand Keypoints
- 3D shape annotation  

    The training set contains 32,560 unique samples post processed in 4 different ways to remove the green screen background. Each evaluation sample provides an RGB image, Hand scale and intrinsic camera matrix. The keypoint and shape annotation is withhold and scoring of algorithms is handled through our Codalab evaluation server. For additional information please visit our project page.
![image](https://github.com/Daming-TF/HandData/blob/master/material/FH.jpg)


### 5.MPII Human Pose Dataset
`Homepage`: http://domedb.perception.cs.cmu.edu/handdb.html
`Paper`：Hand Keypoint Detection in Single Images using Multiview Bootstrapping, CVPR2017
`Introduction`: 该数据集主要有三部分组成：
1. CMUreal：Hands with Manual Keypoint Annotations (Training: 1912 annotations, Testing: 846 annotations)
2. CMUsynth: Hands from Synthetic Data (6546 + 3243 + 2348 + 2124 = 14261 annotations)
3. CMUpanopticDB: Hands from Panoptic Studio by Multiview Bootstrapping (14817 annotations) 

![image](https://github.com/Daming-TF/HandData/blob/master/material/CMU1.jpg)  
![image](https://github.com/Daming-TF/HandData/blob/master/material/CMU2.jpg)  
![image](https://github.com/Daming-TF/HandData/blob/master/material/CMU3.jpg)


### 6) Rendered Handpose Dataset (RHD)
`Homepage`: https://lmb.informatik.uni-freiburg.de/projects/hand3d/

`Git`: https://github.com/lmb-freiburg/hand3d

`Paper`: Learning to Estimate 3D Hand Pose from Single RGB Images, ICCV2017

`Introduction`:  
        It contains 41,258 training and 2,728 testing samples. Each sample provides:
- RGB image (320x320 pixels);
- Depth map (320x320 pixels);
- Segmentaion masks (320x320 pixels) for the classes: background, person, three classess for each finger and one for each palm;
- 21 Keypoints for each hand with their uv coordinates in the image frame, xyz coordinates in the world frame and a visibility indicator;
- Intrinsic Camera Matrix K  

    It was created with freely available characters from www.mixamo.com and rendered with www.blender.org. For more details on how the dataset was created please see the mentioned paper.  
![image](https://github.com/Daming-TF/HandData/blob/master/material/RHD.jpg)


### 7) COCO-WholeBody
`Homepage` :  https://github.com/jin-s13/COCO-WholeBody

`Parper` : Whole-Body Human Pose Estimation in the Wild, ECCV2020

`Introduction` :  
        COCO-WholeBody dataset is the first large-scale benchmark for whole-body pose estimation. It is an extension of COCO 2017 dataset with the same train/val split as COCO.
For each person, we annotate 4 types of bounding boxes (person box, face box, left-hand box, and right-hand box) and 133 keypoints (17 for body, 6 for feet, 68 for face and 42 for hands). The face/hand box is defined as the minimal bounding rectangle of the keypoints.   
![image](https://github.com/Daming-TF/HandData/blob/master/material/COCO.jpg)  
![image](https://github.com/Daming-TF/HandData/blob/master/material/COCO2.jpg)


### 8) Multiview Hand Pose
`Homepage` : https://www.kaggle.com/kmader/multiview-hand-pose

`Dataset` : http://www.rovit.ua.es/dataset/mhpdataset/

`Introduction` :  
        The dataset is structured by sequences. Inside each sequence you'll find the frames that compose it. A frame is composed of 4 color images, 4 sets of 2D joints as projected in each of the image planes, 4 bounding boxes, 1 set of 3D points as provided by the Leap Motion Controller and 4 sets of 3D points as reprojected to each camera coordinate frame.
Along with the dataset itself, it is provided also a set of python scripts (in the utils folder) that will allow you to compute locally the aformentioned data using the calibration files that now we provide (in the calibrations folder). The folder utils contains the following scripts:
- generate2Dpoints.py: It will create the 2D joints projections for each image
- generate3Dpoints.py: It will create the 3D joints projections for each camera
- generateBBoxes.py: It will create the bounding box of the hands for each image  

        As earlier explained, it is required calibration data to generate these annotations. The calibration data we provide consists of rotation and translation matrices. We released both R and T matrices for each camera for each sequence. You will find rvec.pkl and tvec.pkl which are pickle serialized files of numpy matrices that contains the calibration data.  
        ![image](https://github.com/Daming-TF/HandData/blob/master/material/MHP.jpg)


### 9) NYU Hand
`Homepage` : https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm#download  

`Paper` : Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks.Jonathan Tompson, Murphy Stein, Yann Lecun and Ken Perlin.
TOG'14 (Presented at SIGGRAPH'14)  

`Introduction` :  
        The NYU Hand pose dataset contains 8252 test-set and 72757 training-set frames of captured RGBD data with ground-truth hand-pose information. For each frame, the RGBD data from 3 Kinects is provided: a frontal view and 2 side views. The training set contains samples from a single user only (Jonathan Tompson), while the test set contains samples from two users (Murphy Stein and Jonathan Tompson). A synthetic re-creation (rendering) of the hand pose is also provided for each view.（216 篇学术文章引用了此数据集） They also provide the predicted joint locations from our ConvNet (for the test-set) so you can compare performance. Note: for real-time prediction we used only the depth image from Kinect 1. The source code to fit the hand-model to the depth frames here can be found：https://github.com/jonathantompson/ModelFit.git  
![image](https://github.com/Daming-TF/HandData/blob/master/material/NYU.jpg)


### 10) InterHand2.6M dataset
`Homepage` : https://mks0601.github.io/InterHand2.6M/  

`Note` : When downloading, you must see clearly whether it is 5fps or 30fps, and download according to your needs  

`Paper` : A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image (ECCV 2020)  

`Introducyion` :  
        The InterHand2.6M dataset is a large-scale real-captured dataset with accurate GT 3D interacting hand poses, used for 3D hand pose estimation The dataset contains 2.6M labeled single and interacting hand frames.  
![image](https://github.com/Daming-TF/HandData/blob/master/material/InterHand2.6M.jpg)


### 11) HO3D
`Homepage` : https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/  

`Paper` : 
- HOnnotate: A method for 3D Annotation of Hand and Objects Poses，CVPR2020
- Learning Joint Reconstruction of Hands and Manipulated Objects. CVPR2019  

`Introduction` :  
        HO-3D is a dataset with 3D pose annotations for hand and object under severe occlusions from each other. The 68 sequences in the dataset contain 10 different persons manipulating 10 different objects, which are taken from YCB objects dataset. The dataset currently contains annotations for 77,558 images which are split into 66,034 training images (from 55 sequences) and 11,524 evaluation images (from 13 sequences). The evaluation sequences are carefully selected to address the following scenarios:Seen object and seen hand: Sequences SM1, SB11 and SB13 contain hands and objects which are also used in the training set.
- Unseen object and seen hand: Sequences AP10, AP11, AP12, AP13 and AP14 contain 019_pitcher_base object which is not used in the training set.
- Seen object and unseen hand: Sequences MPM10, MPM11, MPM12, MPM13 and MPM14  

        contain a subject with different hand shape and color andis not part of the training set.  
![image](https://github.com/Daming-TF/HandData/blob/master/material/HO3D1.jpg)  
![image](https://github.com/Daming-TF/HandData/blob/master/material/HO3D2.jpg)


### 12) RGB2Hands
`Homepage` : https://handtracker.mpi-inf.mpg.de/projects/RGB2Hands/Benchmark/RGB2HandsBenchmark.htm

`Paper` : RGB2Hands: Real-Time Tracking of 3D Hand Interactions from Monocular RGB Video, ACM Transactions on Graphics (ToG) 2020

`Introduction`: 
        RGB2Hands is an RGB dataset for evaluating algorithms that tracks two hands during interactions. It consists of 4 sequences with varying types of hand-hand interactions. Hand joints locations were manually annotated on paired RGB and Depth images to provided 3D and 2D annotations. 
![image](https://github.com/Daming-TF/HandData/blob/master/material/RGB2Hands.jpg)


### 13) EgoHands
`Homepage` : http://vision.soic.indiana.edu/projects/egohands/

`Paper` : Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions, ICCV2015

`Introduction`:  
        The EgoHands dataset contains 48 Google Glass videos of complex, first-person interactions between two people. The main intention of this dataset is to enable better, data-driven approaches to understanding hands in first-person computer vision. The dataset offers
high quality, pixel-level segmentations of hands the possibility to semantically distinguish between the observer’s hands and someone else’s hands, as well as left and right hands
virtually unconstrained hand poses as actors freely engage in a set of joint activities
lots of data with 15,053 ground-truth labeled hands  
![image](https://github.com/Daming-TF/HandData/blob/master/material/EgoHands.jpg)


