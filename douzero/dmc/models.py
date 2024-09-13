"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""
import math
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlockM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockM, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.mish = nn.Mish(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = ChannelAttention(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = self.mish(out)
        return out


class GeneralModelResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 72
        self.layer1 = self._make_layer(BasicBlockM, 72, 3, stride=2)  # 1*27*72
        self.layer2 = self._make_layer(BasicBlockM, 144, 3, stride=2)  # 1*14*146
        self.layer3 = self._make_layer(BasicBlockM, 288, 3, stride=2)  # 1*7*292
        self.linear1 = nn.Linear(288 * BasicBlockM.expansion * 7 + 18 * 4, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def check_no_bombs(self, a):
        reshaped_a = a[:52].view(-1, 4)
        sums = reshaped_a.sum(dim=1)
        if torch.all(sums < 4) and a[52] + a[53] < 2:
            return True
        else:
            return False

    def forward(self, z, x, return_value=False, flags=None, debug=False):

        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)
        out = torch.cat([x, x, x, x, out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)
        win_rate, win, lose = torch.split(out, (1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        bombs = True
        if self.check_no_bombs(z[0, 2]) and self.check_no_bombs(z[0, 3]) and (0 in z[0, 11]):
            bombs = False
            out = _win_rate
        else:
            out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            elif flags is not None and flags.action_threshold > 0 and bombs:
                max_adp = torch.max(out)
                if max_adp >= 0:
                    min_threshold = max_adp * (1 - flags.action_threshold)
                else:
                    min_threshold = max_adp * (1 + flags.action_threshold)
                valid_indices = torch.where(out >= min_threshold)[0]
                action = valid_indices[torch.argmax(_win_rate[valid_indices])]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


class GeneralModelBid(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 5
        # input 1*54*22
        self.layer1 = self._make_layer(BasicBlockM, 5, 3, stride=2)  # 1*14*12
        self.layer2 = self._make_layer(BasicBlockM, 10, 3, stride=2)  # 1*7*24
        self.layer3 = self._make_layer(BasicBlockM, 20, 3, stride=2)  # 1*4*48
        self.linear1 = nn.Linear(20 * BasicBlockM.expansion * 7, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 5)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z, x, return_value=False, flags=None):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)
        win_rate, win, lose = torch.split(out, (3, 1, 1), dim=-1)
        win_rate = torch.softmax(win_rate, dim=-1)
        out = win_rate[:, 0] * win + win_rate[:, 1] * lose

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class GeneralModelTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6):
        super(GeneralModelTransformer, self).__init__()

        self.in_planes = 12

        self.layer1 = self._make_layer(BasicBlockM, 12, 3, stride=2)
        self.layer2 = self._make_layer(BasicBlockM, 24, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlockM, 48, 3, stride=2)

        self.fc1 = nn.Linear(54, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=60)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=758, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.conv = nn.Conv1d(60, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(4)

        self.linear1 = nn.Linear(d_model * 4 + 18 * 2 + 48 * BasicBlockM.expansion * 7, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 3)

        self.mish = nn.Mish(inplace=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, src1, src2, return_value=False, flags=None):
        out1 = self.fc1(src1[:, -60:])
        out1 = self.pos_encoder(out1)
        out1 = self.transformer_encoder(out1)
        out1 = self.mish(self.bn1(self.conv(out1)))
        out1 = out1.flatten(1, 2)

        out = self.layer1(src1[:, :-60])
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)

        out = torch.cat([src2, src2, out1, out], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)

        win_rate, win, lose = torch.split(out, (1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


GeneralModel = GeneralModelTransformer


model_dict = {
    "first": GeneralModelBid,
    "second": GeneralModelBid,
    'third': GeneralModelBid,
    "landlord": GeneralModelResnet,
    "landlord_down": GeneralModelResnet,
    "landlord_up": GeneralModelResnet,
}


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, device=0):
        if not device == "cpu":
            device = 'cuda:' + str(device)

        self.models = {
            'first': GeneralModelBid().to(torch.device(device)),
            'second': GeneralModelBid().to(torch.device(device)),
            'third': GeneralModelBid().to(torch.device(device)),
            'landlord': GeneralModelResnet().to(torch.device(device)),
            'landlord_down': GeneralModelResnet().to(torch.device(device)),
            'landlord_up': GeneralModelResnet().to(torch.device(device)),
        }

    def forward(self, position, z, x, training=False, flags=None, debug=False):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['first'].share_memory()
        self.models['second'].share_memory()
        self.models['third'].share_memory()
        self.models['landlord'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['landlord_up'].share_memory()

    def eval(self):
        self.models['first'].eval()
        self.models['second'].eval()
        self.models['third'].eval()
        self.models['landlord'].eval()
        self.models['landlord_down'].eval()
        self.models['landlord_up'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


# Model dict is only used in evaluation but not training
model_dict_douzero = {'landlord': LandlordLstmModel, 'landlord_up': FarmerLstmModel, 'landlord_down': FarmerLstmModel}


