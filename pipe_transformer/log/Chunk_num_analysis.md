# ImageNet, Batch size = 400

##  Pipe len = 8

4 * pipe len: 180 images/second
3 * pipe len: 210 images/second   
2 * pipe len: 210 images/second (***)
1 * pipe len: 200 images/second
6 (less than 1 pipe len): 188 images/second

# Pipe len = 4

1 pipe len: 264 images/second

2 pipe len: 288 images/second

3 pipe len: 300 images/second

4 pipe len: 296 images/second (***)

6 pipe len: 300 images/second


# Pipe len = 2
1 pipe len: 520 images/second

2 pipe len: 528 images/second

3 pipe len: 520 images/second

4 pipe len: 528 images/second

6 pipe len: 536 images/second

8 pipe len: 544 images/second (***)

# CIFAR-100, Batch size = 320

##  Pipe len = 8
chunk num = 2 * Pipe len

# Pipe len = 4
chunk num = 4 * Pipe len

# Pipe len = 2
chunk num = 4 * Pipe len
