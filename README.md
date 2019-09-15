# Project pca_opencv

## 项目结构说明

├── src
├── build
├── CMakeLists.txt
├── main.cpp
├── README.md
├── include
└── thirdParty

+ src文件夹：存放着一些基础的源文件，不依赖于项目中的其它文件；

+ build文件夹：存放着编译生成的makefile文件，库文件，可执行文件等文件，这个文件夹不被git跟踪;

+ CMakeLists.txt: 是整个项目的编译规则；

+ main.cpp: 是整个项目的入口；

+ README.md: 是关于这个项目的介绍；

+ include文件夹： 是用于存放头文件。

+ thirdparty文件夹： 存放着项目引用的第三方库（opencv），目的是实现与平台的低耦合。
