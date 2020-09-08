from __future__ import division
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    # img_ = Variable(img_)                   # Convert to Variable->Depreciated
    return img_

# Empty layer for concatenating feature maps in route layer
# Also used for skip connections
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Detection layer for yolo
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  # store the lines in a list
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)  # add it the blocks list
                block = {}  # re-init the block
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

# Every iteration of this loop creates just one block
# That block is put in nn.Sequential() container
# Layers are added to each block i.e module thru add_module method
# As that block will not have a single layer eg CNN
# It'll be a block of CNN,BN,Leaky ReLU together
# Which is then added to nn.ModuleList()
# prev_filters is updated at the end of every iteration
# output_filters contains number of output channels of all layers till now
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append to module_list

        if (x["type"] == "convolutional"):
            # Get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the Batch Norm Layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            #Check the activation.
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        #If it's an upsampling layer
        #We use Bilinear2dUpsampling
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)

# If it is a route/shortcut layer, we dont do anything here
# We just add a dummy layer. The concatenation in route or the skip connection
# in the shortcut layer is implemented in forward pass of Darknet Network
# Here what we're basically doing is collecting all the output channels
# of all layers, be it conv/route/shortcut, so that it may be needed next
# i.e if a CONV comes after a route layer, the route layer concatenates
# the channels of 1 or 2 previous channels
# This new number of filters is set as prev_filter
# Which is next used by the conv layer in its in_channels argument


        #If it is a route layer
        elif (x["type"] == "route"):
            # start and end is given in format (eg:-1 36 so we will find layer number from it.
            # we will find layer number in negative format
            # so that we can get the number of filters in that layer
            x["layers"] = x["layers"].split(',')
            #Start  of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive anotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]

        #shortcut corresponds to skip connection
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

# Test if code works
# blocks = parse_cfg("cfg/yolov3.cfg.txt")
# print(create_modules(blocks))


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0  # This is explained a bit later
        for i, module in enumerate(modules):
            module_type = (module["type"])
            # Conv
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            # Route and Shortcut layer does no operation on tensor
            # If route, torch.cat() is used on functions
            # If shortcut, we just bring feature maps from previous layer forward
            # Route
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)
            # Shortcut
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]
            # Yolo Layer
            # If you have to access the second bounding of cell at (5,6),
            # Then you will have to index it by map[5,6, (5+C): 2*(5+C)]
            elif module_type == 'yolo':

                # Accessing the anchors inside instance of Detection Layer class
                # inside yolo module which is inside the Module list
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:  # if no collector has been intialised.
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections
# What happens in the predict_transform function
# Before predict_transform
# Predictions [batch_size,255,13,13],[batch_size,255,26,26],[batch_size,255,52,52]
# i.e at 3 different scales cannot be concatenated together
# After predict_transfrom
# Predictions [batch_size,507,85], [batch_size,2028,85], [batch_size,8112,85]
# Now class confidence, bbox centre,height and width is easily accessible
# And at three scales the detections are concatenated along axis=1
# So total bboxes =bboxes@scale1+ bboxes@scale2+ bboxes@scale3
# Total bbox = 13*13*3+ 26*26*3+ 52*52*3 = 507+ 2028+ 8112 =  10647



# model = Darknet("cfg/yolov3.cfg.txt")
# inp = get_test_input()
# pred = model(inp, torch.cuda.is_available())
# print(pred.shape) output=torch.Size([1, 10647, 85])
# 13*13*3+ 26*26*3+ 52*52*3= 507+ 2028+ 8112=  10647


    def load_weights(self, weightfile):
    #Open the weights file
        fp = open(weightfile, "rb")

        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            #If module_type is convolutional load weights
            #Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]
                # So basically if you dont use weight.data while updating the weights/biases
                # We end up getting a runtime error cuz we cant update weight with gradient tracking on
                # Hence , if we dont use weight.data we have to perform the updation
                # of weights/biases within a torch.no_grad():

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    with torch.no_grad():
                        bn.bias.copy_(bn_biases)
                        bn.weight.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                # If batch_norm is not true, simply load the biases
                # of the convolutional layer.
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    with torch.no_grad():
                        conv.bias.copy_(conv_biases)

                # Finally, we load the convolutional layer's weights at last.
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight)
                with torch.no_grad(): # Commenting this line gives an error if we dont use weight.data
                    conv.weight.copy_(conv_weights)


model = Darknet("cfg/yolov3.cfg.txt")
model.load_weights("yolov3.weights")



