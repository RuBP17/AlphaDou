import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from douzero.evaluation.autosimu import evaluate
# 设置 random 模块的随机数种子
random.seed(78419)
# 设置 NumPy 的随机数种子
np.random.seed(78419)
# 设置 PyTorch 的随机数种子
torch.manual_seed(78419)
# 如果你的代码也将在CUDA设备上运行，还需要为所有GPU设置随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(78419)
torch.set_default_tensor_type('torch.FloatTensor')


def get_latest_model(dir_path):
    # 获取最新的模型文件
    files = os.listdir(dir_path)
    model_files = [f for f in files if f.startswith("landlord_") and not f.startswith("landlord_up_") and not f.startswith("landlord_down_")]
    
    if not model_files:  # 如果没有找到模型文件
        return None

    return max(model_files, key=lambda x: int(x.split('_')[1].split('.')[0]))


def main(checkpoint_dir, output_csv, output_all_csv, best_txt, dire, position):
    print("start_landlord")
    output_csv = dire + output_csv
    output_all_csv = dire + output_all_csv
    best_txt = dire + best_txt
    max_rows = 200
    
    if not os.path.exists(output_csv):
        pd.DataFrame(columns=["model_id", "win_rate", "score"]).to_csv(output_csv, index=False)
    
    if not os.path.exists(output_all_csv):
        pd.DataFrame(columns=["model_id", "win_rate", "score"]).to_csv(output_all_csv, index=False)
    
    # 读取已评估的模型编号
    evaluated_models = pd.read_csv(output_csv)["model_id"].tolist()

    bid = 6285000

    # player_1_bid = 'baseline/first_' + str(bid) + '.ckpt'
    # player_2_bid = 'baseline/second_' + str(bid) + '.ckpt'
    # player_3_bid = 'baseline/third_' + str(bid) + '.ckpt'

    player_1_bid = 'random'
    player_2_bid = 'random'
    player_3_bid = 'random'

    player_2_playcard = 'baseline/best/landlord_down.ckpt'
    player_3_playcard = 'baseline/best/landlord_up.ckpt'

    
    while True:
        latest_model = get_latest_model(checkpoint_dir)

        if not latest_model:  # 如果没有找到模型文件
            print("No model found. Waiting for the next check...")
            time.sleep(60)
            continue

        model_id = int(latest_model.split('_')[1].split('.')[0])

        latest_model_path = checkpoint_dir + '/' + latest_model
        
        # 如果模型已经评估过，跳过
        df = pd.read_csv(output_csv)
        if model_id in df["model_id"].values:
            time.sleep(60)
            print('no new model')
            continue
        
        num_workers = 2
        eval_data = 'eval_data.pkl'

        # 评估模型
        ti = time.time()

        win_rate, score = evaluate(
            player_1_bid,
            player_2_bid,
            player_3_bid,
            latest_model_path,
            player_2_playcard,
            player_3_playcard,
            eval_data,
            num_workers,
            position)

        print('eval landlord:', time.time() - ti)

        new_row = pd.DataFrame([[model_id, win_rate, score]], columns=["model_id", "win_rate", "score"])
        
        # 更新eval_out.csv
        df = pd.read_csv(output_csv)
        df = pd.concat([df, new_row], ignore_index=True)
        df = df.tail(max_rows)
        df.to_csv(output_csv, index=False)
        
        # 更新eval_out_all.csv
        df_all = pd.read_csv(output_all_csv)
        df = pd.concat([df, new_row], ignore_index=True)
        df_all.to_csv(output_all_csv, index=False)
        
        # 更新best.txt
        best_win_rate_model = df.loc[df["win_rate"].idxmax()]["model_id"]
        best_score_model = df.loc[df["score"].idxmax()]["model_id"]
        with open(best_txt, "w") as f:
            f.write(f"Best win rate model: {best_win_rate_model}\n")
            f.write(f"Best score model: {best_score_model}\n")
        
        # 绘制胜率图
        plt.figure(figsize=(10, 6))
        plt.plot(df["model_id"], df["win_rate"], marker='o')
        plt.xlabel("Model ID")
        plt.ylabel("Win Rate")
        plt.title("Win Rate over Models")
        plt.savefig(dire + "win_rate.jpg")
        plt.close()
        
        # 绘制得分图
        plt.figure(figsize=(10, 6))
        plt.plot(df["model_id"], df["score"], marker='o', color='r')
        plt.xlabel("Model ID")
        plt.ylabel("Score")
        plt.title("Score over Models")
        plt.savefig(dire + "score.jpg")
        plt.close()

        # 更新已评估的模型列表
        evaluated_models.append(model_id)
        
        # 等待下一个模型
        time.sleep(60)


if __name__ == "__main__":
    checkpoint_dir = "douzero_checkpoints/douzero/"
    output_csv = "landlord_eval_out.csv"
    output_all_csv = "landlord_eval_out_all.csv"
    best_txt = "landlord_best.txt"
    dire = "landlord/"
    position = 'landlord'
    main(checkpoint_dir, output_csv, output_all_csv, best_txt, dire, position)
