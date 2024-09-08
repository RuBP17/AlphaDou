import builtins
import multiprocessing as mp
import os.path
import pickle
import copy
from game_eval import GameEnv


EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {3: '3', 4: '4', '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

output_to_file = False
output_list = []


def print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    if output_to_file:
        end = "\n"
        if kwargs.get("end") is not None:
            end = kwargs.get("end")
        output_list.append(" ".join(args) + end)


def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['first', 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
        if card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        elif card_play_model_path_dict[position] == 'Supervised':
            from .deep_agent import SupervisedModel
            players[position] = SupervisedModel()
        elif card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        else:
            from .deep_agent import DeepAgent
            if not isinstance(card_play_model_path_dict[position], list):
                players[position] = DeepAgent(position, card_play_model_path_dict[position])
            else:
                paths = card_play_model_path_dict[position]
                if "landlord" in paths[0]:
                    players[position] = {
                        "landlord": DeepAgent("landlord", paths[0]),
                        "landlord_down": DeepAgent("landlord_down", paths[1]),
                        "landlord_up": DeepAgent("landlord_up", paths[2]),
                    }
                elif "first" in paths[0]:
                    players[position] = {
                        "first": DeepAgent("first", paths[0]),
                        "second": DeepAgent("second", paths[1]),
                        "third": DeepAgent("third", paths[2])
                    }
    return players


def print_card(cards, end="\n"):
    print("".join(EnvCard2RealCard[card] for card in cards), end=end)


def format_action_list(action_list):
    if action_list is None:
        return ""
    return ",".join([action[1] + "(" + str(round(action[0], 3)) + ")" for action in action_list])


def get_modelname_by_path(model_path):
    sep = "/"
    if "|" in model_path:
        model_path = model_path.split("|")
    if isinstance(model_path, list):
        mlist = []
        for path in model_path:
            if "\\" in path:
                sep = "\\"
            mlist.append(path.split(sep)[-1].split(".")[0])
        return "|".join(mlist)
    else:
        if "\\" in model_path:
            sep = "\\"
        return model_path.split(sep)[-1].split(".")[0]


def mp_simulate(card_play_data_list, card_play_model_path_dict, q):
    for k in card_play_model_path_dict:
        if "|" in card_play_model_path_dict[k]:
            card_play_model_path_dict[k] = card_play_model_path_dict[k].split("|")
    players = load_card_play_models(card_play_model_path_dict)

    Env = GameEnv(players)
    enable_output = False
    bid_count = [0, 0, 0, 0]

    if enable_output:
        print("对局模型信息：")
        for position in ['first', 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            print("{}：{}".format(position, get_modelname_by_path(card_play_model_path_dict[position])))

    def start_game(env, idx, card_play_data):
        _card_play_data = copy.deepcopy(card_play_data)
        env.bid_init(_card_play_data)
        if enable_output:
            print_card(_card_play_data["first"])
            print_card(_card_play_data["second"])
            print_card(_card_play_data["third"])
        while not env.bid_over:
            action, action_list = env.step()
            if enable_output:
                if action[0] == 1:
                    print('叫1分', format_action_list(action_list))
                elif action[0] == 2:
                    print('叫2分', format_action_list(action_list))
                elif action[0] == 3:
                    print('叫3分', format_action_list(action_list))
                else:
                    print("不叫", format_action_list(action_list))
        step_index = 0
        if not env.draw:
            bid_count[env.bid_count-1] += 1
            if enable_output:
                print_card(env.info_sets["landlord"].player_hand_cards)
                print_card(env.info_sets["landlord_down"].player_hand_cards)
                print_card(env.info_sets["landlord_up"].player_hand_cards)
            while not env.game_over:
                step_index += 1
                action, action_list = env.step()
                if enable_output:
                    if action:
                        print("".join(EnvCard2RealCard[card] for card in action), end=" ")
                    else:
                        print("Pass", end=" ")
                    if step_index % 3 == 0:
                        print()
        else:
            print("Draw")
        env.reset()

    for idx, card_play_data in enumerate(card_play_data_list):
        print("\n----- 第%d局 -----" % (idx + 1))
        start_game(Env, idx, card_play_data)

    q.put((Env.num_wins['landlord'],

           Env.num_wins['farmer'],

           Env.num_wins['first'],

           Env.num_wins['second'],

           Env.num_wins['third'],

           Env.num_wins['draw'],

           Env.num_scores['landlord'],

           Env.num_scores['farmer'],

           Env.num_scores['first'],

           Env.num_scores['second'],

           Env.num_scores['third'],

           bid_count,

           Env.num_landlord['first'],

           Env.num_landlord['second'],

           Env.num_landlord['third'],
         ))


def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker


def evaluate(first, second, third, playcard_1, playcard_2, playcard_3, eval_data, num_workers):

    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': playcard_1,
        'landlord_down': playcard_2,
        'landlord_up': playcard_3,
        'first': first,
        'second': second,
        'third': third
    }
    model_name_dict = {
        'landlord': get_modelname_by_path(playcard_1),
        'landlord_down': get_modelname_by_path(playcard_2),
        'landlord_up': get_modelname_by_path(playcard_3),
        'first': get_modelname_by_path(first),
        'second': get_modelname_by_path(second),
        'third': get_modelname_by_path(third),
    }

    num_landlord_wins = 0
    num_farmer_wins = 0
    num_first_wins = 0
    num_second_wins = 0
    num_third_wins = 0
    num_draw = 0
    num_landlord_scores = 0
    num_farmer_scores = 0
    num_first_scores = 0
    num_second_scores = 0
    num_third_scores = 0
    num_first_landlord = 0
    num_second_landlord = 0
    num_third_landlord = 0


    ctx = mp.get_context('spawn')
    q = ctx.SimpleQueue()
    processes = []
    for card_paly_data in card_play_data_list_each_worker:
        p = ctx.Process(
                target=mp_simulate,
                args=(card_paly_data, card_play_model_path_dict, q))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    for i in range(num_workers):
        result = q.get()
        num_landlord_wins += result[0]
        num_farmer_wins += result[1]
        num_first_wins += result[2]
        num_second_wins += result[3]
        num_third_wins += result[4]
        num_draw += result[5]
        num_landlord_scores += result[6]
        num_farmer_scores += result[7]
        num_first_scores += result[8]
        num_second_scores += result[9]
        num_third_scores += result[10]
        num_first_landlord += result[12]
        num_second_landlord += result[13]
        num_third_landlord += result[14]

    num_total_wins = num_landlord_wins + num_farmer_wins
    print("\n对局模型信息：")
    print("landlord distribution:")
    print('First : Second : Third - {} : {} : {}'.format(num_first_landlord / num_total_wins,
                                                         num_second_landlord / num_total_wins,
                                                         num_third_landlord / num_total_wins))
    print('WP results:')
    print('First : Second : Third - {} : {} : {}'.format(num_first_wins / num_total_wins,
                                                         num_second_wins / num_total_wins,
                                                         num_third_wins / num_total_wins))
    print('landlord : Farmers - {} : {}'.format(num_landlord_wins / num_total_wins, num_farmer_wins / num_total_wins))
    print('ADP results:')
    print('First : Second : Third - {} : {} : {}'.format(num_first_scores / num_total_wins,
                                                         num_second_scores / num_total_wins,
                                                         num_third_scores / num_total_wins))
    print('landlord : Farmers - {} : {}'.format(num_landlord_scores / num_total_wins,
                                                num_farmer_scores / num_total_wins))
    print('number of draw: - {}'.format(num_draw))

    if output_to_file:
        eval_name = model_name_dict['first'] + '__' + model_name_dict['second'] + '__' + model_name_dict['landlord'] + '__' + model_name_dict['farmer']
        if not os.path.exists("./eval_results"):
            os.mkdir("./eval_results")
        with open("./eval_results/" + eval_name + ".txt", 'a') as f:
            f.write("".join(output_list))
