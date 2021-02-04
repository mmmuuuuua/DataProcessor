from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
np.set_printoptions(threshold=np.inf)


# # settings for LBP
# radius = 2	# LBP算法中范围半径的取值
# n_points = 8 * radius # 领域像素点数
#
#
# def nothing(x):
#     pass
#
#
# def canny(path):
#     # original_img = cv2.imread(path, 0)
#     #
#     # # canny(): 边缘检测
#     # img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
#     # canny = cv2.Canny(img1, 50, 150)
#     #
#     # # 形态学：边缘检测
#     # _, Thr_img = cv2.threshold(original_img, 210, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
#     # gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度
#     #
#     # cv2.imshow("original_img", original_img)
#     # cv2.imshow("gradient", gradient)
#     # cv2.imshow('Canny', canny)
#     #
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     cv2.namedWindow('res')
#
#     cv2.createTrackbar('max', 'res', 0, 255, nothing)
#     cv2.createTrackbar('min', 'res', 0, 255, nothing)
#
#     img = cv2.imread(path, 0)
#
#     max_val = 200
#     min_val = 100
#
#     while (1):
#         if cv2.waitKey(20) & 0xFF == 27:
#             break
#         max_val = cv2.getTrackbarPos('min', 'res')
#         min_val = cv2.getTrackbarPos('max', 'res')
#         if min_val < max_val:
#             edge = cv2.Canny(img, 100, 200)
#             cv2.imshow('res', edge)
#         else:
#             edge = cv2.Canny(img, min_val, max_val)
#             cv2.imshow('res', edge)
#
#     cv2.destoryAllWindows()
#
#
# def lbp(path):
#     image = cv2.imread((path), 0)
#     blur = cv2.bilateralFilter(image, 9, 75, 75)
#     lbp = local_binary_pattern(blur, n_points, radius)
#     cv2.imshow("lbp", lbp)
#     cv2.waitKey(0)
#
#
# def sobel(input):
#     overlay_img = np.zeros((1216, 1824), dtype=np.int32)
#     for subfile in os.listdir(input):
#         if subfile.find('A') == -1:
#             image = cv2.imread(os.path.join(input, subfile), 0)
#             # print(image.shape)
#             # print(image.dtype)
#             # blur = cv2.GaussianBlur(image, (5, 5), 0)
#             blur = cv2.bilateralFilter(image, 9, 75, 75)
#
#             # cv2.imshow('GaussianBlur', blur)
#             # 显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
#             # image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             # plt.subplot(111)
#             # plt.imshow(image)
#             # lbp = local_binary_pattern(blur, n_points, radius)
#             # cv2.imshow("lbp", lbp)
#             # cv2.waitKey(0)
#
#             # plt.subplot(111)
#             # plt.imshow(lbp, plt.cm.gray)
#             edges = filters.sobel(blur)
#             # cv2.imshow("edges", edges)
#             # cv2.waitKey(0)
#             overlay_img = overlay_img + edges
#             # print(edges.shape)
#             # plt.subplot(111)
#             # plt.imshow(edges, plt.cm.gray)
#
#     # the = cv2.getTrackbarPos('max', 'res')
#     # maxval = 255
#     # the, dst = cv2.threshold(overlay_img, the, maxval, cv2.THRESH_BINARY_INV)
#
#     resize_width = int(456)
#     resize_height = int(304)
#
#     windowname = "Image"  # 使用中文会乱码
#     cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(windowname, resize_width, resize_height)
#
#     cv2.imshow(windowname, overlay_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def kmeans(path):
#     img = cv2.imread(path)
#     Z = img.reshape((-1, 3))
#
#     # convert to np.float32
#     Z = np.float32(Z)
#
#     # define criteria, number of clusters(K) and apply kmeans()
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     K = 2
#     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#
#     # Now convert back into uint8, and make original image
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     res2 = res.reshape((img.shape))
#
#     cv2.imshow('res2', res2)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def watershed(threth_path, original_path):
#     img1 = cv2.imread(original_path)
#
#     thresh = cv2.imread(threth_path, 0)
#
#     # nosing removoal迭代两次
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#
#     # sure background area
#     sure_bg = cv2.dilate(opening, kernel, iterations=3)
#
#     dist_transform = cv2.distanceTransform(opening, 1, 5)
#     ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
#
#     sure_fg = np.uint8(sure_fg)
#     unknow = cv2.subtract(sure_bg, sure_fg)
#
#     cv2.imshow('sure_bg', sure_bg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     cv2.imshow('unknow', unknow)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # Marker labeling
#     ret, makers1 = cv2.connectedComponents(sure_fg)
#
#     # Add one to all labels so that sure background is not 0 but 1;
#     markers = makers1 + 1
#
#     # Now mark the region of unknow with zero;
#     markers[unknow == 255] = 0
#     markers3 = cv2.watershed(img1, markers)
#
#     img1[markers3 == -1] = [255, 0, 0]
#
#     # cv2.imshow('makers1', makers1)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # cv2.imshow('markers3', markers3)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     cv2.imshow('img1', img1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # plt.subplot(1, 3, 1),
#     # plt.imshow(sure_bg),
#     # plt.title('Black region\n must be background')
#     #
#     # plt.subplot(1, 3, 3),
#     # plt.imshow(unknow),
#     # plt.title('Yellos region\n must be foregroun'),
#
#
# def process(path):
#     mask = np.array(Image.open(path))
#     print(mask)
#
#
# if __name__ == '__main__':
#     # threth_path = "C:\\zhulei\\gsn\\Pytorch-UNet-Sequence\\data\\test-1_result_2020_11_12\\1_SeqUnet50_30ch_sasc_epoch3000.BMP"
#     # original_path = 'C:\\zhulei\\gsn\\Pytorch-UNet-Sequence\\data\\temp\\1\\C2.jpg'
#     # dir_path = 'C:\\zhulei\\gsn\\Pytorch-UNet-Sequence\\data\\temp\\1'
#     # sobel(dir_path)
#     # kmeans(input)
#     # watershed(threth_path, original_path)
#     # canny(original_path)
#     path = "D:\\zl\\GraduationThesis\\material\\test\\convlstm_999\\10.BMP"
#     process(path)

'''FPN in PyTorch.

See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import os
from torch.autograd import Variable


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal(self.reset_gate.weight)
        init.orthogonal(self.update_gate.weight)
        init.orthogonal(self.out_gate.weight)
        init.constant(self.reset_gate.bias, 0.)
        init.constant(self.update_gate.bias, 0.)
        init.constant(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class FPN(nn.Module):
#     def __init__(self, block, num_blocks, zero_init_residual=False,
#                  groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                  norm_layer=None):
#         super(FPN, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.dilation = 1
#         self.inplanes = 64
#         self.groups = groups
#         self.base_width = width_per_group
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         # Bottom-up layers
#         self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#
#         # Top layer
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         return F.upsample(x, size=(H,W), mode='bilinear') + y
#
#     def forward(self, x):
#         # Bottom-up
#         c1 = F.relu(self.bn1(self.conv1(x)))
#         c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
#         print("c1's shape is {}".format(c1.shape))
#         c2 = self.layer1(c1)
#         print("c2's shape is {}".format(c2.shape))
#         c3 = self.layer2(c2)
#         print("c3's shape is {}".format(c3.shape))
#         c4 = self.layer3(c3)
#         print("c4's shape is {}".format(c4.shape))
#         c5 = self.layer4(c4)
#         print("c5's shape is {}".format(c5.shape))
#         # Top-down
#         p5 = self.toplayer(c5)
#         print("p5's shape is {}".format(p5.shape))
#         p4 = self._upsample_add(p5, self.latlayer1(c4))
#         print("p4's shape is {}".format(p4.shape))
#         p3 = self._upsample_add(p4, self.latlayer2(c3))
#         print("p3's shape is {}".format(p3.shape))
#         p2 = self._upsample_add(p3, self.latlayer3(c2))
#         print("p2's shape is {}".format(p2.shape))
#         # Smooth
#         p4 = self.smooth1(p4)
#         p3 = self.smooth2(p3)
#         p2 = self.smooth3(p2)
#         return p2, p3, p4, p5


class FPN(nn.Module):
    def __init__(self, block, num_blocks, ctx, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(FPN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.out_channels = 256

        if ctx == '34':
            channels = [512, 256, 128, 64]
        elif ctx == '18':
            channels = [512, 256, 128, 64]
        else:
            channels = [2048, 1024, 512, 256]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(channels[0], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # LSTM layers
        self.lstm1 = ConvLSTM(channels[0], 256, (3, 3), 1, True, True, True)
        self.lstm3 = ConvLSTM(channels[2], 256, (3, 3), 1, True, True, True)
        self.lstm2 = ConvLSTM(channels[1], 256, (3, 3), 1, True, True, True)
        self.lstm4 = ConvLSTM(channels[3], 256, (3, 3), 1, True, True, True)

        # GRU layers
        self.gru1 = ConvGRU(channels[0], 256, 3, 1)
        self.gru2 = ConvGRU(channels[1], 256, 3, 1)
        self.gru3 = ConvGRU(channels[2], 256, 3, 1)
        self.gru4 = ConvGRU(channels[3], 256, 3, 1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        return_list = []
        for i in range(x.shape[1]):
            tmp_x = x[:, i, :, :, :]
            tmp_y = y[:, i, :, :, :]
            _, _, H, W = tmp_y.size()
            return_list.append(F.upsample(tmp_x, size=(H, W), mode='bilinear') + tmp_y)

        return_list = torch.stack(return_list)
        return_list = return_list.permute(1, 0, 2, 3, 4)

        return return_list
        # _,_,H,W = y.size()
        # return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # print(x.shape)
        x1 = []
        x2 = []
        x3 = []
        x4 = []
        x5 = []

        for i in range(x.shape[1]):
            tmp_x = x[:, i, :, :, :]
            tmp_x = tmp_x.reshape((tmp_x.shape[0], -1, tmp_x.shape[-2], tmp_x.shape[-1]))

            c1 = F.relu(self.bn1(self.conv1(tmp_x)))
            c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
            # print("c1's shape is {}".format(c1.shape))
            c2 = self.layer1(c1)
            # print("c2's shape is {}".format(c2.shape))
            c3 = self.layer2(c2)
            # print("c3's shape is {}".format(c3.shape))
            c4 = self.layer3(c3)
            # print("c4's shape is {}".format(c4.shape))
            c5 = self.layer4(c4)
            # print("c5's shape is {}".format(c5.shape))

            x1.append(c1)
            x2.append(c2)
            x3.append(c3)
            x4.append(c4)
            x5.append(c5)

        x1 = torch.stack(x1)  # ([9, 2, 64, 256, 256])
        # print("x1'shape is:{}".format(x1.shape))
        x2 = torch.stack(x2)  # ([9, 2, 128, 128, 128])
        # print("x2'shape is:{}".format(x2.shape))
        x3 = torch.stack(x3)  # ([9, 2, 256, 64, 64])
        # print("x3'shape is:{}".format(x3.shape))
        x4 = torch.stack(x4)  # ([9, 2, 512, 32, 32])
        # print("x4'shape is:{}".format(x4.shape))
        x5 = torch.stack(x5)  # ([9, 2, 1024, 16, 16])
        # print("x5'shape is:{}".format(x5.shape))  # [2, 9, 1024, 16, 16]

        x1 = x1.permute(1, 0, 2, 3, 4)
        x3 = x3.permute(1, 0, 2, 3, 4)
        x2 = x2.permute(1, 0, 2, 3, 4)
        x4 = x4.permute(1, 0, 2, 3, 4)
        x5 = x5.permute(1, 0, 2, 3, 4)

        # p5, last_states = self.lstm1(x5)  # [2, 9, 512, 16, 16]
        p5 = self.gru1(x5)
        # p5 = p5[0]
        # print("p5'shape is:{}".format(p5.shape))  # [2, 9, 1024, 16, 16]

        # x4, last_states = self.lstm2(x4)  # [2, 9, 256, 32, 32]
        x4 = self.gru2(x4)
        p4 = self._upsample_add(p5, x4)
        # # print("p4 s'shape is:{}".format(p4.shape))  # [2, 9, 512, 32, 32]
        #
        # x3, last_states = self.lstm3(x3)  # [2, 9, 128, 64, 64]
        x3 = self.gru3(x3)
        p3 = self._upsample_add(p4, x3)
        # # print("p3 s'shape is:{}".format(p3.shape))  # [2, 9, 256, 64, 64]
        #
        # x2, last_states = self.lstm4(x2)  # [2, 9, 128, 64, 64]
        x2 = self.gru4(x2)
        p2 = self._upsample_add(p3, x2)
        # # print("p2 s'shape is:{}".format(p2.shape))  # [2, 9, 128, 128, 128]

        # x4 = x4.reshape(x4[0], )

        p5 = p5[:, -1, :, :, :]
        p4 = p4[:, -1, :, :, :]
        p3 = p3[:, -1, :, :, :]
        p2 = p2[:, -1, :, :, :]

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        d = collections.OrderedDict()
        d[0] = p2
        d[1] = p3
        d[2] = p4
        d[3] = p5
        d['pool'] = F.max_pool2d(p5, 1, 2, 0)

        return d


def FPN50():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [3, 4, 6, 3], '50')
# self.build_graph = {'resnet34':[3, 4, 6, 3], 'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}


def FPN34():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(BasicBlock, [3, 4, 6, 3], '34')


def FPN18():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(BasicBlock, [2, 2, 2, 2], '18')


def test():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = FPN50()
    fms = net(Variable(torch.randn(1, 9, 3, 600, 900)))
    # for fm in fms:
    #     print(fm.size())
    print(fms[0].shape)
    print(fms[1].shape)
    print(fms[2].shape)
    print(fms[3].shape)
    print(fms['pool'].shape)


if __name__ == '__main__':
    # test()
    a = np.rand

# '''FPN in PyTorch.
#
# See the paper "Feature Pyramid Networks for Object Detection" for more details.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch.autograd import Variable
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class FPN(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(FPN, self).__init__()
#         self.in_planes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         # Bottom-up layers
#         self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#
#         # Top layer
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
#
#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         return F.upsample(x, size=(H,W), mode='bilinear') + y
#
#     def forward(self, x):
#         # Bottom-up
#         c1 = F.relu(self.bn1(self.conv1(x)))
#         c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
#         c2 = self.layer1(c1)
#         c3 = self.layer2(c2)
#         c4 = self.layer3(c3)
#         c5 = self.layer4(c4)
#         # Top-down
#         p5 = self.toplayer(c5)
#         p4 = self._upsample_add(p5, self.latlayer1(c4))
#         p3 = self._upsample_add(p4, self.latlayer2(c3))
#         p2 = self._upsample_add(p3, self.latlayer3(c2))
#         # Smooth
#         p4 = self.smooth1(p4)
#         p3 = self.smooth2(p3)
#         p2 = self.smooth3(p2)
#         return p2, p3, p4, p5
#
#
# def FPN101():
#     # return FPN(Bottleneck, [2,4,23,3])
#     return FPN(Bottleneck, [2,2,2,2])
#
#
# def test():
#     net = FPN101()
#     fms = net(Variable(torch.randn(1,3,600,900)))
#     for fm in fms:
#         print(fm.size())
#
# test()
