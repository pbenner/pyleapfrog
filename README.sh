#! /bin/sh

# conda create --name nbconvert
# conda activate nbconvert
# conda install nbconvert==5.6.1

conda activate nbconvert

jupyter nbconvert --template "README.tpl" --to markdown README.ipynb

rm README_files/README_*_*.png
