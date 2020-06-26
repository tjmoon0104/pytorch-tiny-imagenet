# Pytorch-Tiny-ImageNet

### Requirements

```
torch, torchvision, cv2, livelossplot, unzip
```

Use run.sh to format tiny-imagenet into pytorch dataset format.

cv2 must be installed before executing ./run.sh

**Trouble shooting** with OpenCV [here](https://github.com/NVIDIA/nvidia-docker/issues/864#issuecomment-452023152)

### Summary

Train tiny-imagenet dataset on ResNet18 using pretrained weight

Resize tiny-imagenet dataset to 224x224 and train  on ResNet18 using pretrained weight

Finetune few layers, and use pretrained weight from 224x224 trained model to retrain 64x64 image on ResNet18



### Test Result

| Model    | Test Result | Input size | pretrained weight |
| -------- | ----------- | ---------- | ----------------- |
| AlexNet  | 35.88%      | 64x64      | ImageNet          |
| ResNet18 | 53.58%      | 64x64      | ImageNet          |
| ResNet18 | 69.62%      | 224x224    | ImageNet          |
| ResNet18 | 69.80%      | 64x64      | Model Above       |

### Capstone Proposal Review
https://review.udacity.com/#!/reviews/1541377
