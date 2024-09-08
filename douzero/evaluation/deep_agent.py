import torch
import numpy as np
import os
from douzero.env.env import get_obs
from douzero.env.env_douzero import get_obs_douzero
from douzero.env.env_res import _get_obs_resnet
from baseline.SLModel.BidModel import Net2 as Net


def _load_model(position, model_path, model_type):
    from douzero.dmc.models import model_dict, model_dict_douzero
    if model_type == "test":
        model = model_dict_douzero[position]()
    elif model_type == "best":
        from douzero.dmc.models_res import model_dict_resnet
        model = model_dict_resnet[position]()
    else:
        model = model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class DeepAgent:

    def __init__(self, position, model_path):
        if "test" in model_path:
            self.model_type = "test"
        elif "best" in model_path:
            self.model_type = "best"
        else:
            self.model_type = "new"
        self.model = _load_model(position, model_path, self.model_type)
        self.EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                            8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                            13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

    def act(self, infoset):
        if self.model_type == "test":
            obs = get_obs_douzero(infoset)
        elif self.model_type == "best":
            obs = _get_obs_resnet(infoset, infoset.player_position)
        else:
            obs = get_obs(infoset, bid_over=infoset.bid_over, new_model=True)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        if self.model_type != 'new':
            y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        else:
            win_rate, win, lose = self.model.forward(z_batch, x_batch, return_value=True)['values']
            if infoset.player_position in ["landlord", "landlord_up", "landlord_down"]:
                _win_rate = (win_rate + 1) / 2
                y_pred = _win_rate * win + (1. - _win_rate) * lose
            else:
                y_pred = win_rate[:, 0] * win + win_rate[:, 1] * lose
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]
        return best_action


class SupervisedModel:

    def __init__(self):
        self.net = Net()
        self.net.eval()
        if torch.cuda.is_available():
            self.gpu = True
        else:
            self.gpu = False
        if self.gpu:
            self.net = self.net.cuda()
        if os.path.exists("baseline/SLModel/bid_weights_new.pkl"):
            if torch.cuda.is_available():
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl'))
            else:
                self.net.load_state_dict(torch.load('baseline/SLModel/bid_weights_new.pkl', map_location=torch.device("cpu")))

    def RealToOnehot(self, cards):
        Onehot = torch.zeros((4, 15))
        m = 0
        for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]:
            Onehot[:cards.count(i), m] = 1
            m += 1
        return Onehot

    def predict_score(self, cards):
        input = RealToOnehot(cards)
        input = torch.flatten(input)
        input = input.unsqueeze(0)
        result = self.net(input)
        return result[0].item()

    def act(self, infoset):
        legal_action = infoset.legal_actions
        obs = torch.flatten(self.RealToOnehot(infoset.player_hand_cards))
        if self.gpu:
            obs = obs.cuda()
        predict = self.net.forward(obs.unsqueeze(0))
        one = -0.1
        two = 0
        three = 0.1
        if predict > three and ([3] in legal_action):
            return [3]
        elif predict > two and ([2] in legal_action):
            return [2]
        elif predict > one and ([1] in legal_action):
            return [1]
        else:
            return [0]

def RealToOnehot(cards):
    RealCard2EnvCard = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                        '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                        'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14}
    cards = [RealCard2EnvCard[c] for c in cards]
    Onehot = torch.zeros((4,15))
    for i in range(0, 15):
        Onehot[:cards.count(i),i] = 1
    return Onehot