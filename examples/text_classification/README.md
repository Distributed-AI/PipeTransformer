# Reference Accuracy

According to the original BERT paper, SST-2 test result it 93.5 (BERT-base) and 94.9 (BERT-large).

The optimal parameters are:

1. pretraining:

Batch size 256 

Adam, lr = 1e-4, beta1 = 0.9, beta2 = 0.999, L2 weight decay = 0.01

dropout = 0.1

2. fine-tuning:

Batch size: 16, 32

Number of epochs: 3

Learning rate (Adam): 5e-5, 3e-5, 2e-5



# HPO
3e5: best_accuracy = 0.924217

5e5: best_accuracy = 0.929160