# Lab1 - Pneumonia Classification
Chest X-ray Pneumonia Classification Using ResNet

原始最佳權重為 ResNet18 搭配隨機旋轉 ( 15度 ) 在第 29 個 Epoch 的權重 (random seed = 2026)。


## Download the dataset if needed
```bash
curl -L -o ./chest-xray-pneumonia.zip\
  https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip 
```

## Evaluate the best model weight on a labeled test set

```bash

# 評估原始資料集
------------------------------------------------------
|chest_xray/test/NORMAL/normal_001.jpeg
|chest_xray/test/NORMAL/normal_002.jpeg
|chest_xray/test/PNEUMONIA/pneumonia_001.jpeg
|chest_xray/test/PNEUMONIA/pneumonia_002.jpeg
------------------------------------------------------
python inference.py --dataset chest_xray


# 評估新資料集
------------------------------------------------------
|new_data_set/test/NORMAL/normal_001.jpeg
|new_data_set/test/NORMAL/normal_002.jpeg
|new_data_set/test/PNEUMONIA/pneumonia_001.jpeg
|new_data_set/test/PNEUMONIA/pneumonia_002.jpeg
------------------------------------------------------

python inference.py --dataset new_data_set

```


## Predict labels for new images

```bash
# 單張圖片
python inference.py --task predict --input image.jpeg

# 多張圖片或整個資料夾
python inference.py --task predict --input image1.jpeg image2.jpeg
python inference.py --task predict --input ./images


```
## Reproduce 
```bash
## 重現 best model 訓練權重

bash reproduce_best.sh

## 重現所有訓練實驗

bash reproduce_all.sh
```



# Environment setup (WSL)

If you encounter dependency issues, create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-wsl.txt
python -m pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu126
```




exec
```source .venv/bin/activate```
