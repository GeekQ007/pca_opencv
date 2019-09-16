Project pca_opencv

gcc (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
基于opencv4.1.1（项目已包含），依赖的第三方库安装方式

sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python-dev python-numpy python3-dev python3-numpy
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev


项目结构说明
-----------------------------------------------------------
├── .vscode
├── include
├── src
├── CMakeLists.txt
├── main.cpp
└── README.md

include文件夹： 是用于存放头文件;

src文件夹：存放着一些基础的源文件，不依赖于项目中的其它文件；

CMakeLists.txt: 是整个项目的编译规则；

main.cpp: 是整个项目的入口；

README.md: 是关于这个项目的介绍；