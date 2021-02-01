import numpy as np

# ImageNet Freeze Training
imagenet_acc_freeze = [82.97, 81.72, 81.84]
imagenet_acc_freeze_mean = np.mean(imagenet_acc_freeze)
imagenet_acc_freeze_var = np.var(imagenet_acc_freeze)
print("imagenet_acc_freeze_mean = %f" % imagenet_acc_freeze_mean)
print("imagenet_acc_freeze_var = %f" % imagenet_acc_freeze_var)


# ImageNet normal Training (no freeze)
imagenet_acc_no_freeze = [80.69, 80.96]
imagenet_acc_no_freeze_mean = np.mean(imagenet_acc_no_freeze)
imagenet_acc_no_freeze_var = np.var(imagenet_acc_no_freeze)
print("imagenet_acc_no_freeze_mean = %f" % imagenet_acc_no_freeze_mean)
print("imagenet_acc_no_freeze_var = %f" % imagenet_acc_no_freeze_var)


def translate_to_second(hours, minutes, seconds):
    return hours*3600 + minutes*60 + seconds

# Training Time ImageNet
# No AutoCache: 11h 3m 27
time_imagenet_no_cache = translate_to_second(11, 3, 27)
# Baseline: 20h 36m 4
time_imagenet_baseline = translate_to_second(20, 36, 4)
speedup_imagenet = time_imagenet_baseline/time_imagenet_no_cache
print("speedup_imagenet = %f" % speedup_imagenet)

# Training Time CIFAR-10
# No AutoCache: "20m 12s"