## 1.create conda environment

```
conda create -n pipe_transformer python=3.8.0
conda activate pipe_transformer
```

## 2.install the latest nightly Pytorch, in which DPipe (torchpipe) is supported.

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Note: please match the CUDA version with your own environment.

## 3.install other packages
```
pip install -r requirements.txt 
```
Note we modified huggingface transformer==3.5.0 slightly and maintained the source code at `transformers` under the root folder.


## 4.Experimental Tracker (WandB.com)
``` 
# Please change to your own ID
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```

## 5.prepare dataset and pretrained weights

(1) prepare ViT model pretrained weights (for fine-tuning)

```
cd model/cv/pretrained/
sh download_pretrained_weights.sh
cd ../../../
```

(2) download datasets at `data` folder
```
cd ./data/cifar10/
sh download_cifar10.sh
cd ../../
cd ./data/cifar100/
sh download_cifar100.sh
cd ../../
# the source code already includes SQuAD_1.1 and SST-2 datasets.

```