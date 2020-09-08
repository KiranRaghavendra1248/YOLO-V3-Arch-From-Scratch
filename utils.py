from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

# Note origin is top-left corner of the image
# Formula for bx,by,bh,w

# bx=sigmoid(tx)+cx , by=sigmoid(ty)+cy
# where tx,ty is prediction of NN and cx,cy is the origin
# i.e top left corner of present grid cell

#bw=pw*exp(tw), bh=ph*exp(th)
# where tw,th is prediction of NN
# pw,ph is height and width of predefined bounding box


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w,h = inp_dim
    new_w = int(img_w * min(torch.true_divide(w ,img_w), torch.true_divide(h, img_h)))
    new_h = int(img_h * min(torch.true_divide(w ,img_w), torch.true_divide(h, img_h)))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Here img , in_dim  both are single values which are fed one by one to this fun
    by the map function
    Returns a Variable
    """

    img = letterbox_image(img, inp_dim)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    Note:
        box1 contains bbx attributes of 1 bbx
        box2 contains bbx attributes of all other bbx


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    # predict_transform takes in 5 parameters;
    # prediction (our output), inp_dim (input image dimension),
    # anchors, num_classes, and an optional CUDA flag
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes  # 5 + 80
    num_anchors = len(anchors)    # 3
    # There is [13,13] grid and each grid, at a particular scale
    # Outputs 3 bounding boxes.
    # Total bounding boxes= 13 * 13 * 3 = 507
    # Bounding box attributes= 5 + C = 85
    #[1, 255, 13, 13]
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    #[1, 255, 169]
    prediction = prediction.transpose(1, 2).contiguous()
    #[1, 169, 255]
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    #[1, 507, 85]

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)   #[13,13]

    x_offset = torch.FloatTensor(a).view(-1,1)  #[169,1]
    y_offset = torch.FloatTensor(b).view(-1,1)  #[169,1]

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # Here this is what happens
    # [169,1],[169,1]--cat--> [169,2]--repeat-->[169,6]--view-->
    # [507,2]--unsqueeze-->[1,507,2]

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    # Log space transform height and the width

    # Anchors are a tensor where each element is a tuple h,w
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    # This is what happens here
    # [3, 2]--repeat-->[507, 2]--unsqueeze-->[1,507,2]
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Apply sigmoid activation to the the class scores
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # Resize the detections map to the size of the input image
    prediction[:, :, :4] *= stride

    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    # Object Confidence Thresholding
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # Performing Non-maximum Suppression
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        #image_pred is now 2D tensor
           #confidence threshholding
           #NMS
        #  At this point, we're only concerned with the class score
        #  having the maximum value. So, we remove the 80 class scores
        #  from each row, and instead add the index of the class having
        #  the maximum values, as well the class score of that class.
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # torch.max() returns max value and the index where max exists
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        # concatinating index values and max probability with box coordinates as columns
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        # Now image_pred dim is [num_bbox,7]

        # We have set bounding box attributes of those boxes with objectness
        # score less than the threshold as zero. Now we will remove them.
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # For PyTorch 0.4 compatibility
        # Since the above code with not raise exception for no detection
        # as scalars are supported in PyTorch 0.4
        if image_pred_.shape[0] == 0:
            continue
        #  The try-except block is there to handle situations where
        #  we get no detections. In that case, we use continue to skip
        #  the rest of the loop body for this image.
        # Get the various classes detected in the image
        try:
            img_classes = torch.unique(image_pred_[:, -1])
            # -1 index holds the class index
        except:
            continue
        for cls in img_classes:
        # perform NMS
        # get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            # Unsqueeze is used as broadcasting b/w vector and matrix is not possible
            # Hence we make the vector a matrix using unsqueeze
            # Tensors are broadcastable if
            # When iterating over the dimension sizes, starting at the trailing dimension,
            # the dimension sizes must either be equal, one of them is 1,
            # or one of them does not exist.
            # Now cls_mask dim=[num_bbx,7]
            # Basically now all bbx attributes are zero for those bbxs which
            # doesnt contain present class
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            sorted_, conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0) # Number of detections

            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)


                # We use the write flag to indicate whether the tensor has been initialized
                # or not. Once it has been initialized, we concatenate subsequent detections to it.
                # At the end of loop that iterates over classes, we add the resultant
                # detections to the tensor output.


            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to
            #We use a linear structure to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

            # At the end of the function, we check whether output has been initialized at all or not.
            # If it hasn't been means there's hasn't been a single detection in any images of the batch.
            # In that case, we return 0.

    try:
        return output
    except:
        return 0

