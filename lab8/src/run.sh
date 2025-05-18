#!/bin/bash

# 编译 main.cpp
g++ -fopenmp -O2 -std=c++11 main.cpp -o main

# 执行程序，参数是邻接表文件 和 测试文件
echo "===== 测试 updated_mouse.csv → updated_flower.csv ====="
./main ../data/updated_mouse.csv ../data/updated_flower.csv

echo "===== 测试 updated_flower.csv → updated_mouse.csv ====="
./main ../data/updated_flower.csv ../data/updated_mouse.csv
