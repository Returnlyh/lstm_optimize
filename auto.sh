#!/bin/bash
###
 # @Descripttion: 
 # @version: 1.0.0
 # @Author: Gager
 # @Date: 2022-11-23 15:43:29
 # @LastEditors: Gager
### 

if [ ! -d "./build/" ];then
  echo "[INFO]>>> 创建build文件夹"
  mkdir ./build
else
  echo "[INFO]>>> 清空build下内容"
  rm -rf build/*
fi

cd build
cmake ..
make -j8
mv lstm_opt ..
cd ..
