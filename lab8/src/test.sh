#!/bin/bash

make

python3 test.py ../data/updated_flower.csv ../data/updated_flower.csv
python3 test.py ../data/updated_mouse.csv ../data/updated_mouse.csv
