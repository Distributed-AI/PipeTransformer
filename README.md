

## Installation

1.create conda environment
```
conda create -n pipe_transformer python=3.8.0
conda activate pipe_transformer
```

2.install the latest nightly Pytorch, in which DPipe (torchpipe) is supported.

```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

3.install other packages
```
pip install -r requirements.txt 
```

4. WandB.com
```
wandb login ee0b5f53d949c84cee7decbe7a629e63fb2f8408
```
## Experiments
check README.md at 

examples/image_classification

examples/question_answering

examples/text_classification