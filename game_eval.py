from copy import deepcopy
from douzero.env import move_detector as md, move_selector as ms
from douzero.env.move_generator import MovesGener
import random

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: '10', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

bombs = [[3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6],
         [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10],
         [11, 11, 11, 11], [12, 12, 12, 12], [13, 13, 13, 13], [14, 14, 14, 14],
         [17, 17, 17, 17], [20, 30]]

class GameEnv(object):

    def __init__(self, players):
        self.players = players

        self.bid_over = False

        self.bidding_player_position = None

        self.bid_info_sets = {'first': InfoSet('first'),
                              'second': InfoSet('second'),
                              'third': InfoSet('third')}

        self.bid_action_seq = []

        self.bid_info = [-1, -1, -1]

        self.bid_count = 0

        self.position = ['landlord', 'landlord_down', 'landlord_up']

        self.bid_infoset = None

        self.draw = False

        self.bid_step_count = 0

        self.card_play_action_seq = []

        self.three_landlord_cards = None

        self.game_over = False

        self.acting_player_position = None

        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []

        self.last_two_moves = []

        self.num_landlord = {
             'first': 0,
             'second': 0,
             'third': 0
        }

        self.num_wins = {'landlord': 0,
                         'farmer': 0,
                         'first': 0,
                         'second': 0,
                         'third': 0,
                         'draw': 0}

        self.num_scores = {'landlord': 0,
                           'farmer': 0,
                           'first': 0,
                           'second': 0,
                           'third': 0}

        self.info_sets = {'landlord': InfoSet('landlord'),
                          'landlord_up': InfoSet('landlord_up'),
                          'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0

        self.pos_bomb_num = {
            "landlord": 0,
            "landlord_up": 0,
            "landlord_down": 0
        }
        self.last_pid = 'landlord'

        self.step_count = 0

        self.game_infoset = None

        self.spring = True

        self.spring_count = {'landlord': 0,
                             'farmer': 0}

        self.winner = None

        self.bid_winner = None

    def bid_init(self, card_play_data):
        self.bid_info_sets['first'].player_hand_cards = \
            card_play_data['first']
        self.bid_info_sets["first"].player_hand_cards.sort()
        self.bid_info_sets['second'].player_hand_cards = \
            card_play_data['second']
        self.bid_info_sets["second"].player_hand_cards.sort()
        self.bid_info_sets['third'].player_hand_cards = \
            card_play_data['third']
        self.bid_info_sets["third"].player_hand_cards.sort()
        self.three_landlord_cards = card_play_data['three_landlord_cards']
        self.get_bidding_player_position()
        self.bid_infoset = self.get_bid_infoset()

    def judge_landlord(self):
        max_index = self.bid_info.index(max(self.bid_info))
        if max_index == 0:
            self.position = ['landlord', 'landlord_down', 'landlord_up']
        elif max_index == 1:
            self.position = ['landlord_up', 'landlord', 'landlord_down']
        elif max_index == 2:
            self.position = ['landlord_down', 'landlord_up', 'landlord']

    def bid_done(self):
        if self.bid_step_count == 3:
            self.bid_over = True
            if self.bid_info == [0, 0, 0]:
                self.draw = True
            else:
                self.judge_landlord()
        elif self.bid_count == 3:
            self.bid_over = True
            self.judge_landlord()

        if self.bid_over:
            self.bid_info_sets['first'].play_card_position = self.position[0]
            self.bid_info_sets['second'].play_card_position = self.position[1]
            self.bid_info_sets['third'].play_card_position = self.position[2]
            # 地主牌加入手中
            if self.bid_info_sets["first"].play_card_position == "landlord":
                self.bid_info_sets["first"].player_hand_cards += \
                    self.bid_info_sets["first"].three_landlord_cards
                self.bid_info_sets["first"].player_hand_cards.sort()
            elif self.bid_info_sets["second"].play_card_position == "landlord":
                self.bid_info_sets['second'].player_hand_cards += \
                    self.bid_info_sets['second'].three_landlord_cards
                self.bid_info_sets['second'].player_hand_cards.sort()
            else:
                self.bid_info_sets['third'].player_hand_cards += \
                    self.bid_info_sets['third'].three_landlord_cards
                self.bid_info_sets['third'].player_hand_cards.sort()
            if not self.draw:
                self.update_num_landlord()
            self.card_play_init()

    def update_num_landlord(self):
        if self.position[0] == "landlord":
            self.num_landlord['first'] += 1
        elif self.position[0] == "landlord_up":
            self.num_landlord['second'] += 1
        else:
            self.num_landlord['third'] += 1

    def bid_step(self):
        if not isinstance(self.players[self.bidding_player_position], dict):
            action = self.players[self.bidding_player_position].act(self.bid_info_sets[self.bidding_player_position])
        else:
            action = self.players[self.bidding_player_position][self.bidding_player_position].act(self.bid_info_sets[self.bidding_player_position])
        action_list = None

        self.bid_info[self.bid_step_count] = int(action[0])

        self.bid_step_count += 1

        if action[0] > 0:
            self.bid_count = action[0]

        self.bid_action_seq.append((self.bidding_player_position, action))

        self.bid_done()

        if (not self.bid_over) and (not self.draw):
            self.get_bidding_player_position()
            self.bid_infoset = self.get_bid_infoset()
        # 流局的情况
        elif self.draw:
            self.compute_player_utility()
            self.update_num_wins_scores()
        # 叫牌正常结束
        else:
            self.bid_infoset = self.get_bid_infoset()
        return action, action_list

    def get_bidding_player_position(self):
        if self.bidding_player_position is None:
            self.bidding_player_position = 'first'
        else:
            if self.bidding_player_position == 'first':
                self.bidding_player_position = 'second'
            elif self.bidding_player_position == 'second':
                self.bidding_player_position = 'third'
        return self.bidding_player_position

    def get_bid_infoset(self):
        for pos in ['first', 'second', 'third']:
            self.bid_info_sets[pos].bid_over = self.bid_over

        self.bid_info_sets[self.bidding_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.bid_info_sets[self.bidding_player_position].three_landlord_cards = \
            self.three_landlord_cards

        self.bid_info_sets[self.bidding_player_position].bid_action_seq = \
            self.bid_action_seq

        self.bid_info_sets[
            self.bidding_player_position].all_handcards = \
            {pos: self.bid_info_sets[pos].player_hand_cards
             for pos in ['first', 'second', 'third']}
        for pos in ['first', 'second', 'third']:
            self.bid_info_sets[pos].bid_info = \
                self.bid_info

        return deepcopy(self.bid_info_sets[self.bidding_player_position])

    def card_play_init(self):
        self.info_sets[self.bid_info_sets["first"].play_card_position].player_hand_cards = \
            self.bid_info_sets["first"].player_hand_cards
        self.info_sets[self.bid_info_sets['second'].play_card_position].player_hand_cards = \
            self.bid_info_sets['second'].player_hand_cards
        self.info_sets[self.bid_info_sets['third'].play_card_position].player_hand_cards = \
            self.bid_info_sets['third'].player_hand_cards
        self.three_landlord_cards = self.bid_info_sets["first"].three_landlord_cards

        # print(self.info_sets[self.bid_info_sets["first"].play_card_position].player_hand_cards)
        # print(self.info_sets[self.bid_info_sets['second'].play_card_position].player_hand_cards)
        # print(self.info_sets[self.bid_info_sets['third'].play_card_position].player_hand_cards)

        self.bid_info = self.bid_info_sets["first"].bid_info
        for pos in ["landlord", "landlord_down", "landlord_up"]:
            self.info_sets[pos].bid_over = self.bid_over
            self.info_sets[pos].bid_count = self.bid_count
        self.get_acting_player_position()
        self.game_infoset = self.get_infoset()

    def game_done(self):
        if len(self.info_sets['landlord'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_up'].player_hand_cards) == 0 or \
                len(self.info_sets['landlord_down'].player_hand_cards) == 0:
            # if one of the three players discards his hand,
            # then game is over.
            self.compute_player_utility()
            self.update_num_wins_scores()
            print(self.winner)
            print(self.bid_winner)
            self.game_over = True

    def judge_spring(self):
        if self.spring_count['farmer'] > 0 and self.spring_count['landlord'] > 1:
            self.spring = False

    def compute_player_utility(self):

        if self.draw:
            self.player_utility_dict = {'landlord': 0,
                                        'farmer': 0}

        elif len(self.info_sets['landlord'].player_hand_cards) == 0:
            self.player_utility_dict = {'landlord': 2,
                                        'farmer': -1}
        else:
            self.player_utility_dict = {'landlord': -2,
                                        'farmer': 1}

    def update_num_wins_scores(self):
        for pos, utility in self.player_utility_dict.items():
            base_score = 2 * self.bid_count if pos == 'landlord' else self.bid_count
            if utility > 0:
                self.num_wins[pos] += 1
                self.winner = pos
                bomb_num = self.bomb_num + 1 if self.spring else self.bomb_num
                s = base_score + 4 * bomb_num if pos == 'landlord' else base_score + 2 * bomb_num
                if pos == "landlord":
                    if pos == self.bid_info_sets["first"].play_card_position:
                        self.bid_winner = "first"
                        self.num_wins["first"] += 1
                        self.num_scores["first"] += s
                        self.num_scores["second"] -= s / 2
                        self.num_scores["third"] -= s / 2
                    elif pos == self.bid_info_sets["second"].play_card_position:
                        self.bid_winner = "second"
                        self.num_wins["second"] += 1
                        self.num_scores["first"] -= s / 2
                        self.num_scores["second"] += s
                        self.num_scores["third"] -= s / 2
                    else:
                        self.bid_winner = "third"
                        self.num_wins["third"] += 1
                        self.num_scores["first"] -= s / 2
                        self.num_scores["second"] -= s / 2
                        self.num_scores["third"] += s
                else:
                    if self.bid_info_sets["first"].play_card_position == "landlord":
                        self.bid_winner = "second & third"
                        self.num_wins["second"] += 1
                        self.num_wins["third"] += 1
                        self.num_scores["first"] -= s * 2
                        self.num_scores["second"] += s
                        self.num_scores["third"] += s
                    elif self.bid_info_sets["second"].play_card_position == "landlord":
                        self.bid_winner = "first & third"
                        self.num_wins["first"] += 1
                        self.num_wins["third"] += 1
                        self.num_scores["first"] += s
                        self.num_scores["second"] -= s * 2
                        self.num_scores["third"] += s
                    else:
                        self.bid_winner = "first & second"
                        self.num_wins["second"] += 1
                        self.num_wins["first"] += 1
                        self.num_scores["first"] += s
                        self.num_scores["second"] += s
                        self.num_scores["third"] -= s * 2

                self.num_scores[pos] += s
            elif utility < 0:
                bomb_num = self.bomb_num + 1 if self.spring else self.bomb_num
                s = base_score + 4 * bomb_num if pos == 'landlord' else base_score + 2 * bomb_num
                self.num_scores[pos] -= s
            else:
                self.num_wins['draw'] += 1
                break

    def get_winner(self):
        return self.winner

    def get_winner_bid(self):
        if self.winner == 'landlord':
            if self.bid_info_sets["first"].play_card_position == 'landlord':
                return 'first'
            elif self.bid_info_sets["second"].play_card_position == 'landlord':
                return 'second'
            else:
                return 'third'
        else:
            if self.bid_info_sets["first"].play_card_position == 'landlord':
                return 'second & third'
            elif self.bid_info_sets["second"].play_card_position == 'landlord':
                return 'first & third'
            else:
                return 'first & second'

    def get_bomb_num(self):
        return self.bomb_num

    def step(self):
        if self.bid_over and not self.draw:
            if not isinstance(self.players[self.acting_player_position], dict):
                action = self.players[self.acting_player_position].act(self.info_sets[self.acting_player_position])
            else:
                action = self.players[self.acting_player_position][self.acting_player_position].act(
                    self.info_sets[self.acting_player_position])
            action_list = None
            self.step_count += 1
            if len(action) > 0:
                self.last_pid = self.acting_player_position

            if action in bombs:
                self.bomb_num += 1
                self.pos_bomb_num[self.acting_player_position] += 1

            self.last_move_dict[
                self.acting_player_position] = action.copy()

            self.card_play_action_seq.append((self.acting_player_position, action))
            self.update_acting_player_hand_cards(action)

            self.played_cards[self.acting_player_position] += action

            if self.acting_player_position == 'landlord' and \
                    len(action) > 0 and \
                    len(self.three_landlord_cards) > 0:
                for card in action:
                    if len(self.three_landlord_cards) > 0:
                        if card in self.three_landlord_cards:
                            self.three_landlord_cards.remove(card)
                    else:
                        break

            self.judge_spring()
            self.game_done()
            if not self.game_over:
                self.get_acting_player_position()
                self.get_infoset()
            # print(action)
            return action, action_list
        elif not self.bid_over:
            return self.bid_step()

    def get_last_move(self):
        last_move = []
        if len(self.card_play_action_seq) != 0:
            if len(self.card_play_action_seq[-1][1]) == 0:
                last_move = self.card_play_action_seq[-2][1]
            else:
                last_move = self.card_play_action_seq[-1][1]

        return last_move

    def get_last_two_moves(self):
        last_two_moves = [[], []]
        for card in self.card_play_action_seq[-2:]:
            last_two_moves.insert(0, card[1])
            last_two_moves = last_two_moves[:2]
        return last_two_moves

    def get_acting_player_position(self):
        if self.acting_player_position is None:
            self.acting_player_position = 'landlord'

        else:
            if self.acting_player_position == 'landlord':
                self.acting_player_position = 'landlord_down'

            elif self.acting_player_position == 'landlord_down':
                self.acting_player_position = 'landlord_up'

            else:
                self.acting_player_position = 'landlord'

        return self.acting_player_position

    def update_acting_player_hand_cards(self, action):
        if action != []:
            for card in action:
                self.info_sets[
                    self.acting_player_position].player_hand_cards.remove(card)
            # 统计地主出牌次数
            if self.acting_player_position == "landlord":
                self.spring_count["landlord"] += 1
            else:
                self.spring_count["farmer"] += 1
            self.info_sets[self.acting_player_position].player_hand_cards.sort()

    def get_legal_card_play_actions(self):
        if self.bid_over:
            mg = MovesGener(
                self.info_sets[self.acting_player_position].player_hand_cards)

            action_sequence = self.card_play_action_seq

            rival_move = []
            if len(action_sequence) != 0:
                if len(action_sequence[-1][1]) == 0:
                    rival_move = action_sequence[-2][1]
                else:
                    rival_move = action_sequence[-1][1]

            rival_type = md.get_move_type(rival_move)
            rival_move_type = rival_type['type']
            rival_move_len = rival_type.get('len', 1)
            moves = list()

            if rival_move_type == md.TYPE_0_PASS:
                moves = mg.gen_moves()

            elif rival_move_type == md.TYPE_1_SINGLE:
                all_moves = mg.gen_type_1_single()
                moves = ms.filter_type_1_single(all_moves, rival_move)

            elif rival_move_type == md.TYPE_2_PAIR:
                all_moves = mg.gen_type_2_pair()
                moves = ms.filter_type_2_pair(all_moves, rival_move)

            elif rival_move_type == md.TYPE_3_TRIPLE:
                all_moves = mg.gen_type_3_triple()
                moves = ms.filter_type_3_triple(all_moves, rival_move)

            elif rival_move_type == md.TYPE_4_BOMB:
                all_moves = mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()
                moves = ms.filter_type_4_bomb(all_moves, rival_move)

            elif rival_move_type == md.TYPE_5_KING_BOMB:
                moves = []

            elif rival_move_type == md.TYPE_6_3_1:
                all_moves = mg.gen_type_6_3_1()
                moves = ms.filter_type_6_3_1(all_moves, rival_move)

            elif rival_move_type == md.TYPE_7_3_2:
                all_moves = mg.gen_type_7_3_2()
                moves = ms.filter_type_7_3_2(all_moves, rival_move)

            elif rival_move_type == md.TYPE_8_SERIAL_SINGLE:
                all_moves = mg.gen_type_8_serial_single(repeat_num=rival_move_len)
                moves = ms.filter_type_8_serial_single(all_moves, rival_move)

            elif rival_move_type == md.TYPE_9_SERIAL_PAIR:
                all_moves = mg.gen_type_9_serial_pair(repeat_num=rival_move_len)
                moves = ms.filter_type_9_serial_pair(all_moves, rival_move)

            elif rival_move_type == md.TYPE_10_SERIAL_TRIPLE:
                all_moves = mg.gen_type_10_serial_triple(repeat_num=rival_move_len)
                moves = ms.filter_type_10_serial_triple(all_moves, rival_move)

            elif rival_move_type == md.TYPE_11_SERIAL_3_1:
                all_moves = mg.gen_type_11_serial_3_1(repeat_num=rival_move_len)
                moves = ms.filter_type_11_serial_3_1(all_moves, rival_move)

            elif rival_move_type == md.TYPE_12_SERIAL_3_2:
                all_moves = mg.gen_type_12_serial_3_2(repeat_num=rival_move_len)
                moves = ms.filter_type_12_serial_3_2(all_moves, rival_move)

            elif rival_move_type == md.TYPE_13_4_2:
                all_moves = mg.gen_type_13_4_2()
                moves = ms.filter_type_13_4_2(all_moves, rival_move)

            elif rival_move_type == md.TYPE_14_4_22:
                all_moves = mg.gen_type_14_4_22()
                moves = ms.filter_type_14_4_22(all_moves, rival_move)

            if rival_move_type not in [md.TYPE_0_PASS,
                                       md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB]:
                moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

            if len(rival_move) != 0:  # rival_move is not 'pass'
                moves = moves + [[]]

            for m in moves:
                m.sort()

            return moves

        else:
            if self.bid_count == 0:
                return [[0], [1], [2], [3]]
            elif self.bid_count == 1:
                return [[0], [2], [3]]
            elif self.bid_count == 2:
                return [[0], [3]]

    def reset(self):
        self.bid_over = False

        self.bidding_player_position = None

        self.bid_info_sets = {'first': InfoSet('first'),
                              'second': InfoSet('second'),
                              'third': InfoSet('third')}

        self.bid_action_seq = []

        self.bid_info = [-1, -1, -1]

        self.bid_count = 0

        self.position = ['landlord', 'landlord_down', 'landlord_up']

        self.bid_infoset = None

        self.draw = False

        self.bid_step_count = 0

        self.card_play_action_seq = []

        self.three_landlord_cards = None

        self.game_over = False

        self.acting_player_position = None

        self.player_utility_dict = None

        self.last_move_dict = {'landlord': [],
                               'landlord_up': [],
                               'landlord_down': []}

        self.played_cards = {'landlord': [],
                             'landlord_up': [],
                             'landlord_down': []}

        self.last_move = []

        self.last_two_moves = []

        self.info_sets = {'landlord': InfoSet('landlord'),
                          'landlord_up': InfoSet('landlord_up'),
                          'landlord_down': InfoSet('landlord_down')}

        self.bomb_num = 0

        self.pos_bomb_num = {
            "landlord": 0,
            "landlord_up": 0,
            "landlord_down": 0
        }
        self.last_pid = 'landlord'

        self.step_count = 0

        self.game_infoset = None

        self.spring = True

        self.spring_count = {'landlord': 0,
                             'farmer': 0}

        self.winner = None

        self.bid_winner = None

    def get_infoset(self):
        self.info_sets[
            self.acting_player_position].spring = self.spring

        self.info_sets[
            self.acting_player_position].bid_info = self.bid_info

        self.info_sets[
            self.acting_player_position].last_pid = self.last_pid

        self.info_sets[
            self.acting_player_position].legal_actions = \
            self.get_legal_card_play_actions()

        self.info_sets[
            self.acting_player_position].bomb_num = self.bomb_num

        self.info_sets[
            self.acting_player_position].last_move = self.get_last_move()

        self.info_sets[
            self.acting_player_position].last_two_moves = self.get_last_two_moves()

        self.info_sets[
            self.acting_player_position].last_move_dict = self.last_move_dict

        self.info_sets[self.acting_player_position].num_cards_left_dict = \
            {pos: len(self.info_sets[pos].player_hand_cards)
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        self.info_sets[self.acting_player_position].other_hand_cards = []
        for pos in ['landlord', 'landlord_up', 'landlord_down']:
            if pos != self.acting_player_position:
                self.info_sets[
                    self.acting_player_position].other_hand_cards += \
                    self.info_sets[pos].player_hand_cards

        self.info_sets[self.acting_player_position].played_cards = \
            self.played_cards
        self.info_sets[self.acting_player_position].three_landlord_cards = \
            self.three_landlord_cards
        self.info_sets[self.acting_player_position].card_play_action_seq = \
            self.card_play_action_seq

        self.info_sets[
            self.acting_player_position].all_handcards = \
            {pos: self.info_sets[pos].player_hand_cards
             for pos in ['landlord', 'landlord_up', 'landlord_down']}

        return deepcopy(self.info_sets[self.acting_player_position])

class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_position):
        # The player position, i.e., landlord, landlord_down, or landlord_up
        self.player_position = player_position
        # The hand cands of the current player. A list.
        self.player_hand_cards = None
        # The number of cards left for each player. It is a dict with str-->int
        self.num_cards_left_dict = None
        # The three landload cards. A list.
        self.three_landlord_cards = None

        self.play_card_position = None

        self.bid_action_seq = None

        self.bid_info = [-1, -1, -1]
        # The historical moves. It is a list of list
        self.card_play_action_seq = None
        # The union of the hand cards of the other two players for the current player
        self.other_hand_cards = None
        # The legal actions for the current move. It is a list of list
        self.legal_actions = None
        # The most recent valid move
        self.last_move = None
        # The most recent two moves
        self.last_two_moves = None
        # The last moves for all the postions
        self.last_move_dict = None
        # The played cands so far. It is a list.
        self.played_cards = None
        # The hand cards of all the players. It is a dict.
        self.all_handcards = None
        # Last player position that plays a valid move, i.e., not `pass`
        self.last_pid = None
        # The number of bombs played so far
        self.bomb_num = None

        self.bid_count = 0

        self.spring = None

        self.bid_over = None