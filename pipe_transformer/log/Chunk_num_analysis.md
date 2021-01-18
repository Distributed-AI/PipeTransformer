# ImageNet, Batch size = 400

##  Pipe len = 8

4 * pipe len: 180 images/second
3 * pipe len: 210 images/second   
2 * pipe len: 210 images/second (***)
1 * pipe len: 200 images/second
6 (less than 1 pipe len): 188 images/second

# Pipe len = 4

6 (less than 1 pipe len): 328 images/second

1 pipe len: 300 images/second

2 pipe len: 312 images/second  (***)

3 pipe len: 308 images/second

4 pipe len: 304 images/second

# Pipe len = 2



# CIFAR-100, Batch size = 320

##  Pipe len = 8
chunk num = 2 * Pipe len

# Pipe len = 4
chunk num = 4 * Pipe len

# Pipe len = 2
chunk num = 4 * Pipe len
