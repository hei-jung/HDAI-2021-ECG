import torch
from torch import nn

"""
소스코드는 아래 논문의 모델 구조를 참고하여 구현했습니다.
Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network.
Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4
"""

res_blocks = [2, 4, 8, 16]
kernel_sizes = [8, 16, 32]
dropout_rates = [0, 0.5, 0.8]


class ResidualBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, n_samples_out, kernel_size=17, dropout_keep_prob=0.8,
                 pre_act=True, post_act_bn=False):
        super(ResidualBlock, self).__init__()
        self.n_filters_in = n_filters_in
        self.n_filters_out = n_filters_out
        self.n_samples_out = n_samples_out
        self.kernel_size = kernel_size
        self.dropout_rate = 1 - dropout_keep_prob
        self.pre_act = pre_act
        self.post_act_bn = post_act_bn

        self.relu = nn.ReLU(inplace=True)
        self.conv_skip = nn.Conv1d(self.n_filters_in, self.n_filters_out, kernel_size=1, padding='same', bias=False)
        self.conv_1 = nn.Conv1d(self.n_filters_in, self.n_filters_out, self.kernel_size, padding='same', bias=False)
        self.conv_2 = nn.Conv1d(self.n_filters_out, self.n_filters_out, self.kernel_size, padding=7, bias=False)
        self.bn_act = self._batch_norm_plus_activation()
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def _batch_norm_plus_activation(self):
        if self.post_act_bn:
            return nn.Sequential(
                self.relu,
                nn.BatchNorm1d(self.n_filters_out, eps=0.001, momentum=0.99))
        else:
            return nn.Sequential(
                nn.BatchNorm1d(self.n_filters_out, eps=0.001, momentum=0.99, affine=False),
                self.relu)

    def forward(self, x):
        x0, x1 = x
        n_samples_in = x1.shape[2]
        down_sample = n_samples_in // self.n_samples_out

        # skip connection
        if down_sample > 1:
            x1 = nn.MaxPool1d(1, stride=down_sample)(x1)  # (N, 64, 1024)
        elif self.down_sample == 1:
            x1 = x1
        else:
            raise ValueError("Number of samples should always decrease.")
        if self.n_filters_in != self.n_filters_out:
            x1 = self.conv_skip(x1)

        # 1st layer
        x0 = self.conv_1(x0)
        x0 = self.bn_act(x0)
        if self.dropout_rate > 0:
            x0 = self.dropout(x0)

        # 2nd layer
        self.conv_2.stride = down_sample
        x0 = self.conv_2(x0)

        if self.pre_act:
            x0 = x0 + x1
            x1 = x0
            x0 = self.bn_act(x0)
            if self.dropout_rate > 0:
                x0 = self.dropout(x0)
        else:
            x0 = nn.BatchNorm1d(self.n_filters_out, eps=0.001, momentum=0.99, affine=False)(x0)
            x0 = x0 + x1
            x0 = self.relu(x0)
            if self.dropout_rate > 0:
                x0 = self.dropout(x0)
            x1 = x0
        return x0, x1


class DNN(nn.Module):
    def __init__(self, input_channels=12, in_planes=64, n_classes=1):
        super(DNN, self).__init__()
        kernel_size = 16
        self.conv = nn.Conv1d(input_channels, in_planes, kernel_size, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(in_planes, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.res_blk1 = ResidualBlock(in_planes, 128, 1024)
        self.res_blk2 = ResidualBlock(128, 196, 256)
        self.res_blk3 = ResidualBlock(196, 256, 64)
        # self.res_blk4 = ResidualBlock(256, 320, 16)
        self.fc = nn.Linear(256 * 64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # input shape: (N, 12, 4096)
        x = self.conv(x)  # (N, 64, 4096)
        x = self.bn(x)  # (N, 64, 4096)
        x = self.relu(x)  # (N, 64, 4096)
        x0, x1 = self.res_blk1((x, x))  # (N, 128, 1024)
        x0, x1 = self.res_blk2((x0, x1))  # (N, 196, 256)
        x0, _ = self.res_blk3((x0, x1))  # (N, 256, 64)
        # x0, _ = self.res_blk4((x0, x1))  # (N, 320, 16)
        x = x0.view(x0.size(0), -1)  # (N, 5120)
        x = self.fc(x)  # (N, 1)
        x = self.sigmoid(x)
        return x


def dnn(**kwargs):
    model = DNN(**kwargs)
    return model