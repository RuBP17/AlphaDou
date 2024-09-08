from collections import Counter
import numpy as np

from douzero.env.game import GameEnv


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:

    def __init__(self, objective):
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['first', 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            self.players[position] = DummyAgent(position)

        # Initialize the internal environment
        self._env = GameEnv(self.players)
        self.total_round = 0
        self.infoset = None

    def reset(self, model, device, flags=None):
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'first': _deck[:17],
                          'second': _deck[20:37],
                          'third': _deck[37:],
                          'three_landlord_cards': _deck[17:20]
                          }
        for key in card_play_data:
            card_play_data[key].sort()
        # 重置叫牌
        self._env.bid_init(card_play_data)

        bid_over = self._bid_over
        self.infoset = self._bid_infoset
        return get_obs(self.infoset, bid_over)

    def step(self, action):
        if not self._draw:
            if self._bid_over:
                pos = self._acting_player_position
            else:
                pos = self._bidding_player_position

            self.players[pos].set_action(action)
            self._env.step()
        if not self._draw:
            if self._bid_over:
                self.infoset = self._game_infoset
            else:
                self.infoset = self._bid_infoset

        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = {
                "play": {
                    "landlord": self._get_reward("landlord") / 2,
                    "landlord_down": self._get_reward("landlord_down"),
                    "landlord_up": self._get_reward("landlord_up"),
                    "first": self._get_reward_bidding("first"),
                    "second": self._get_reward_bidding("second"),
                    "third": self._get_reward_bidding("third"),
                }
            }
            obs = None
        elif self._draw:
            obs = None
        else:
            obs = get_obs(self.infoset, self._bid_over)
        return obs, reward, done, self._draw, {}

    def _get_reward(self, pos):
        winner = self._game_winner

        bomb_num = self._game_bomb_num + 1 if self._env.spring else self._game_bomb_num
        bid_count = self._env.bid_count

        multiply = 2
        if pos != 'landlord':
            pos = 'farmer'
            multiply = 1

        if bomb_num == 0:
            r = bid_count
        elif bomb_num == 1:
            r = bid_count * (1 + bomb_num)
        else:
            r = bid_count * (2 + 2 * (bomb_num - 1))

        if pos == winner:
            return r * multiply / 24
        else:
            return -r * multiply / 24

    def _get_reward_bidding(self, pos):
        if self._draw:
            return 0
        winner = self._bid_winner
        bomb_num = self._game_bomb_num + 1 if self._env.spring else self._game_bomb_num
        bid_count = self._env.bid_count
        multiply = 1 if '&' in winner else 2
        _multiply = 2 if multiply == 1 else 1
        if pos in winner:
            if bomb_num == 0:
                r = bid_count
            elif bomb_num == 1:
                r = bid_count * (1 + bomb_num)
            else:
                r = bid_count * (2 + 2 * (bomb_num - 1))
            return r * multiply / 24
        else:
            if bomb_num == 0:
                r = bid_count
            elif bomb_num == 1:
                r = bid_count * (1 + bomb_num)
            else:
                r = bid_count * (2 + 2 * (bomb_num - 1))
            return -r * _multiply / 24

    @property
    def _game_infoset(self):
        return self._env.game_infoset

    @property
    def _bid_infoset(self):
        return self._env.bid_infoset

    @property
    def _game_bomb_num(self):
        return self._env.get_bomb_num()


    @property
    def _acting_player_position(self):
        return self._env.acting_player_position

    @property
    def _bidding_player_position(self):
        return self._env.bidding_player_position

    @property
    def _game_over(self):
        return self._env.game_over

    @property
    def _bid_over(self):
        return self._env.bid_over

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _bid_winner(self):
        return self._env.get_winner_bid()

    @property
    def _draw(self):
        return self._env.draw


class DummyAgent(object):
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        self.action = action


def get_obs(infoset, bid_over, new_model=True):
    if new_model:
        if infoset.player_position not in ["first", 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            raise ValueError('')
        if bid_over:
            return _get_obs_resnet(infoset)
        if infoset.player_position in ["first", 'second', 'third']:
            return _get_bid_obs_resnet(infoset)
    else:
        return _get_obs_general(infoset, infoset.player_position)


def _get_one_hot_array(num_left_cards, max_num_cards):
    one_hot = np.zeros(max_num_cards)
    if num_left_cards > 0:
        one_hot[num_left_cards - 1] = 1

    return one_hot


def _cards2array(list_cards):
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.ones((len(action_seq_list), 54)) * -1  # Default Value -1 for not using area
    for row, list_cards in enumerate(action_seq_list):
        if list_cards != []:
            action_seq_array[row, :54] = _cards2array(list_cards[1])
    return action_seq_array


def _action_seq_list2array_lstm(action_seq_list):
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array


def _process_action_seq(sequence, length=15, new_model=True):
    sequence = sequence[-length:].copy()
    sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        if new_model:
            sequence.extend(empty_sequence)
        else:
            empty_sequence.extend(sequence)
            sequence = empty_sequence
    return sequence


def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot


def _get_obs_resnet(infoset):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)

    spring = np.array([1]) if infoset.spring else np.array([0])
    # spring = np.array([0])
    spring = np.multiply(spring, np.ones((54, 1))).transpose((1, 0))

    bid_info = np.array(infoset.bid_info).flatten()

    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])

    bid_info_z = np.multiply(bid_info, np.ones((54, 3))).transpose((1, 0))

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    num_cards_left = np.hstack((
                         landlord_num_cards_left,  # 20
                         landlord_up_num_cards_left,  # 17
                         landlord_down_num_cards_left))

    x_batch = np.hstack((
                         bid_info_batch,  # 3
                         bomb_num_batch,  # 15
                         ))
    x_no_action = np.hstack((
                             bid_info,
                             bomb_num,
                             ))

    z = np.vstack((
                  num_cards_left,  # 54
                  my_handcards,  # 54
                  other_handcards,  # 54
                  three_landlord_cards,  # 54
                  landlord_played_cards,  # 54
                  landlord_up_played_cards,  # 54
                  landlord_down_played_cards,  # 54
                  bid_info_z,
                  spring,
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 60))
                  ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_bid_obs_resnet(infoset):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info)
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)
    bid_info_z = np.multiply(bid_info, np.ones((54, 3))).transpose((1, 0))

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = np.multiply(action, np.ones((54, 1))).transpose((1, 0))

    x_batch = np.hstack((
        bid_info_batch,  # 5
    ))
    x_no_action = np.hstack((
        bid_info,
    ))
    z = np.vstack((
        my_handcards,  # 54
        bid_info_z
    ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)

    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs


def _get_obs_general(infoset, position):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    position_map = {
        "landlord": [1, 0, 0],
        "landlord_up": [0, 1, 0],
        "landlord_down": [0, 0, 1]
    }
    position_info = np.array(position_map[position])
    position_info_batch = np.repeat(position_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info).flatten()
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    multiply_info = np.array(infoset.multiply_info)
    multiply_info_batch = np.repeat(multiply_info[np.newaxis, :],
                                    num_legal_actions, axis=0)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)
    three_landlord_cards_batch = np.repeat(three_landlord_cards[np.newaxis, :],
                                           num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    other_handcards_left_list = []
    for pos in ["landlord", "landlord_up", "landlord_up"]:
        if pos != position:
            other_handcards_left_list.extend(infoset.all_handcards[pos])

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((position_info_batch,  # 3
                         my_handcards_batch,  # 54
                         other_handcards_batch,  # 54
                         three_landlord_cards_batch,  # 54
                         last_action_batch,  # 54
                         landlord_played_cards_batch,  # 54
                         landlord_up_played_cards_batch,  # 54
                         landlord_down_played_cards_batch,  # 54
                         landlord_num_cards_left_batch,  # 20
                         landlord_up_num_cards_left_batch,  # 17
                         landlord_down_num_cards_left_batch,  # 17
                         bomb_num_batch,  # 15
                         bid_info_batch,  # 12
                         multiply_info_batch, # 3
                         my_action_batch))  # 54
    x_no_action = np.hstack((position_info,
                             my_handcards,
                             other_handcards,
                             three_landlord_cards,
                             last_action,
                             landlord_played_cards,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_num_cards_left,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num,
                             bid_info,
                             multiply_info))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq, 32), "general")
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
        'position': position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs
