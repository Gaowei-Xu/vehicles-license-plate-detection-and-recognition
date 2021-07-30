# 基于无服务架构的车牌检测与识别解决方案

## 目录
* [方案介绍](#方案介绍)
* [方案部署](#方案部署)
* [算法实现细节](#算法实现细节)
  * [训练yolov4车牌检测模型](#训练yolov4车牌检测模型)
  * [训练车牌识别模型](#训练车牌识别模型)
* [测试](#测试)
* [许可](#许可)


## 方案介绍
该解决方案基于`Amazon S3`，`Amazon Lambda`，`Amazon Elastic Container Registry (ECR)`
组件实现。用户基于该解决方案架构可以实现车牌的自动检测和识别，该方案基于无服务架构实现，用户设备侧上传视频片段到
云端S3桶中，会自动触发`Lambda`函数进行车牌检测（检测网络为`yolo-v4`）和识别， 识别结果会被存储到`Amazon S3`桶中进行存储。该方案中
针对检测和识别的是中国城市车牌，所用到的训练数据集为[CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD) ，
数据集中图像基本均为各种停车场景下进行捕捉的，视角基本是平视。迁移至其他地域其他应用场景下的车牌检测和识别需要收集当地的数据集，同时需要用相应
视角场景下的数据来训练算法模型。

架构图如下所示:
![license_plate_detection_and_recognition_serverless_architecture](architecture.png)

架构图中各个组件的功能描述如下:
1. `Amazon S3`: 用来存储用户设备端的视频片段，和用来存储车牌识别和检测的推理结果；
1. `Amazon Lambda`: 完成视频片段的抽帧，以及对图像帧中的车牌进行检测和识别；
1. `Amazon Elastic Container Registry (ECR)`: 存储着车牌检测和识别处理镜像；


## 方案部署

整个部署过程包括编译镜像和资源部署，大概需要10-20分钟。

该解决方案支持`aws-cdk`进行部署，部署区域的`lambda`服务需要支持从容器镜像启动，部署示例如下：

1. 启动`Amazon EC2`实例，如`us-east-1`区域选择系统镜像`Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0747bdcabd34c712a (64-bit x86) / ami-08353a25e80beea3e (64-bit Arm)
`，机型选择`t2.large`，添加存储`128 GiB`，启动机器；


2. 通过`ssh`登录至上述实例，安装依赖项，如下所示：
```angular2html
sudo apt-get update
sudo apt-get install -y cmake git zip awscli
sudo apt install nodejs npm
sudo npm install -g n
sudo n stable
PATH="$PATH"
sudo n 16.2.0
sudo npm install -g npm@7.19.0
sudo npm install -g aws-cdk@1.115.0

sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo groupadd docker
sudo usermod -aG docker ${USER}
```
上述命令执行完成之后需要登出（命令行输入`logout`）后再次登入（`ssh`登录至实例）使得`Docker`生效。


3. 用户配置`IAM User`，在命令行输入`aws configure`后，输入`AWS Access Key ID`, `AWS Secret Access Key`，`Default region name`和
`Default output format`，该`IAM User`需要具有创建该解决方案中所有资源的权限（可以为其赋予`AdministratorAccess`权限）。


4. 部署方案
```angular2html
git clone https://github.com/Gaowei-Xu/vehicles-license-plate-detection-and-recognition.git
cd vehicles-license-plate-detection-and-recognition/
npm install
cdk deploy
```






## 算法实现细节
该小节阐述车牌检测算法yolov4的训练过程，以及车牌识别算法的模型定义和训练，主要是为了分别获得检测和识别的模型参数，用来将其封装到ECS推理镜像中。

预训练好的车牌检测和车牌识别模型为：

| 任务         |  模型                 |
|--------------|----------------------|
| 车牌检测       |   [detector-tensorflow-2.5.0](https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_detection/detector.zip)   |
| 车牌识别       |   [recognizer-tensorflow-2.5.0](https://ip-camera-ai-saas.s3.amazonaws.com/models/license_plates_recognition/recognizer.zip)   |



### 训练yolov4车牌检测模型
#### 1. EC2训练环境准备
在`us-east-1`区域中选择`Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-0747bdcabd34c712a (64-bit x86) / ami-08353a25e80beea3e (64-bit Arm)`系统镜像，机型选择
`g4.dn.xlarge`，存储添加`512 GiB`，开启机器，机器启动后通过`ssh`登录至该实例安装`CUDA`，`cudnn`依赖项，如下所示：

```angular2html
sudo apt-get update
sudo apt-get install -y cmake git zip wget
sudo apt-get install -y python3 python3-pip python3-opencv libopencv-dev
python3 -m pip install --upgrade pip

# install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu1804-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-3-local_11.3.0-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# install cudnn
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/software/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -zxvf cudnn-11.3-linux-x64-v8.2.1.32.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```


#### 2. `darknet`编译
由于`yolov4`是基于[Darknet](https://github.com/AlexeyAB/darknet) 进行训练，我们需要在训练环境中编译好`darknet`，编译命令如下所示：

```angular2html
git clone https://github.com/AlexeyAB/darknet.git
cd darknet

# 更改编译选项 Makefile，使其支持 GPU 训练，将如下FLAG设置为1
# GPU=1
# CUDNN=1
# CUDNN_HALF=1
# OPENCV=1
# AVX=1
# OPENMP=1
# LIBSO=1

# 开始编译
make -j4
```

编译成功会生成一个可执行文件，如下所示：
```angular2html
ubuntu@ip-11-0-1-195:~/darknet$ ls -al darknet
-rwxrwxr-x 1 ubuntu ubuntu 6705768 Jul 14 03:56 darknet
```

#### 3. 创建`darknet`训练配置文件
车牌识别中共有一个检测类别（车牌），即目标检测类别为`1`，以下命令创建训练配置文件
```angular2html
cd darknet/cfg/
cp yolov4-custom.cfg yolov4-license-plates.cfg

# 手动更改yolov4-license-plates.cfg配置：
# 输入宽高改为512；
# max_batches改为10000；
# steps改为max_batches的0.8，0.9倍；
# classes改为1（不含背景类）；
# filters=255改为filters=(类别数 + 5)x3)
# 更多细节参考：https://github.com/AlexeyAB/darknet

cd ../data/
echo "license plate" >> license_plates.names

# 在目录data/下创建 license_plates.data，里面共五行信息，分别配置
# 类别数量，训练数据，验证数据，类别名称，模型存储目录等信息，如下所示：
# classes = 1
# train = data/license_plates/train.txt
# valid = data/license_plates/val.txt
# names = data/license_plates.names
# backup = backup/

# 下载训练数据
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/dataset/vehicle-plate-detection-recognition/IPCVehicleLicenseDetectionDataset.zip
unzip IPCVehicleLicenseDetectionDataset.zip
mv IPCVehicleLicenseDetectionDataset license_plates
rm IPCVehicleLicenseDetectionDataset.zip

# 下载pre-trained模型
cd ../
mkdir models
cd models/
wget -c https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

#### 4. 开始车牌检测模型训练
执行如下脚本启动后台训练：
```angular2html
nohup ./darknet detector train data/license_plates.data cfg/yolov4-license-plates.cfg models/yolov4.conv.137 -dont_show -mjpeg_port 8090 -map 2>&1 > train.log &
```
   
### 训练车牌识别模型
#### 1. 车牌识别网络设计
车牌识别是建立在车牌检测之后的基础之上进行的，首先车牌检测出车牌的bounding boxes，在原始输入图像中将该
区域（Re gion of Interest, ROI）扣取出来，进行图像resize，在输入到车牌识别网络进行识别，
识别网络的大小固定为`40x116x3`，中国车牌共有七位，我们构建了一个共有卷积层后级联7个分类网络，
分别对每一个字符进行分类。车牌识别网络的具体架构如下图所示：

![license_plate_recognition_model](./tensorflow/recognizer/license_plate_recognition_model.png)

#### 2. 开始车牌识别模型训练
```angular2html
# 安装依赖项
pip3 install tensorflow-gpu==2.5.0
pip3 install pydot==1.4.2
sudo apt-get install -y graphviz

# clone代码
git clone https://github.com/Gaowei-Xu/vehicles-license-plate-detection-and-recognition.git
cd vehicles-license-plate-detection-and-recognition/tensorflow

# 下载数据
wget -c https://ip-camera-ai-saas.s3.amazonaws.com/dataset/vehicle-plate-detection-recognition/IPCVehicleLicenseRoIForRecognition.zip
unzip IPCVehicleLicenseRoIForRecognition.zip
rm IPCVehicleLicenseRoIForRecognition.zip

# 开始训练
cd recognizer
nohup python3 license_plate_recognizer.py > train.log 2>&1 &
```


## 测试
在解决方案部署成功之后，即可上传`.ts`格式或者`.mp4`格式的视频片段至`S3`桶（桶的名称前缀为`vehicleslicenseplatedetection-videosasset`）进行测试，
建议视频片段的时间长度小于1分钟，测试视频样本


## 许可
该解决方案遵从MIT-0 许可，更多信息请参阅 LICENSE 文件.
