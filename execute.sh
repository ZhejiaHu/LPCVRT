#!/bin/bash


filePath="../data/LPCVC_Val/LPCVC_Val/IMG/val"
predPath="../Oreki/pred1"
modelPath="../LPCV2023/model1.pth"
evalPath="../Oreki/eval1"

rm -rf ../Oreki/pred1 ../Oreki/eval1 ../Oreki/pred1.zip ../Oreki/eval1.zip

echo "Begin executing"
python3 main.py --model_path $modelPath --file_path $filePath --pred_path $predPath --eval_path $evalPath
