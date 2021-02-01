import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5#255.
    target_ = target > 0.5#255.
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    print("The value of FP is :",FP)
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN

def Acc(output, target):
    """Getting the pixel accuracy of the model 准确率
    """
    """#成功了的代码
    output = output.contiguous().view(-1, 1)
    # print("The shape of output after change:", np.shape(output))
    target = target.contiguous().view(-1, 1)
    # print("The shape of output after change:", np.shape(target))
    total_pixel = 0
    total_pixel += target.nelement()
    print("####The total_pixel is####:", total_pixel)
    for i in range(total_pixel):
        if output[i] > 0.5:
            # output_[i] = output[i]
            output[i] == 1
        else:
            # output_[i] = output[i]
            output[i] == 0
        if target[i] > 0.5:
            # target_[i] = target[i]
            target[i] == 1
        else:
            # target_[i] = target[i]
            target[i] == 0
    _, predicted = torch.max(output, 0)
    print("The predicted is :", predicted)

    correct_pixel = 0
    correct_pixel += predicted.eq(target.long()).sum().item()
    print("****The predicted.eq(target.long()) is *****",predicted.eq(target.long()))
    print("####The correct_pixel is####:", correct_pixel)
    acc = 100 * correct_pixel / total_pixel
    print("####The acc is####:", acc)

    return acc"""
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    print("The value of the output is:",output)
    print("The value of the output shape is:", np.shape(output))
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    print("The value of the target is:", target)
    print("The value of the target shape is:", np.shape(target))
    total_pixel = 0
    total_pixel += target.nelement()
    print("####The total_pixel is####:", total_pixel)

    _, predicted = torch.max(output, 0)
    print("The predicted is :", predicted)

    correct_pixel = 0
    correct_pixel += predicted.eq(target.long()).sum().item()
    print("****The predicted.eq(target.long()) is *****", predicted.eq(target.long()))
    print("####The correct_pixel is####:", correct_pixel)
    acc = 100 * correct_pixel / total_pixel
    print("####The acc is####:", acc)


    return acc
