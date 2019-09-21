# CMake 最低的版本要求
cmake_minimum_required(VERSION 3.0)

# 定义项目名称变量PROJECT_NAME, 默认值为demo
set(PROJECT_NAME demo)

# 项目名称
project(${PROJECT_NAME})

# 设置C++的版本
set(CMAKE_CXX_STANDARD 14)

# 指定模式
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# 当前目录下除了测试文件外所有的源文件的名称保存到 DIR_SRCS 变量
set(DIR_SRCS main.cpp)

# 添加包含源文件的子目录(工作目录的次一级子目录)
aux_source_directory(src DIR_SUB_SRC)

# 包含头文件目录
include_directories(include)
include_directories(/usr/local/include/opencv4)

# 第三方链接库的目录
link_directories(/usr/local/lib)

# 指定生成目标
add_executable(${PROJECT_NAME} ${DIR_SRCS})

# 链接第三方库
target_link_libraries(${PROJECT_NAME} -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs)