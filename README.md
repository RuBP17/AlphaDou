# AlphaDou: High-Performance End-to-End Doudizhu AI Integrating Bidding

AlphaDou is a reinforcement learning framework for [DouDizhu](https://en.wikipedia.org/wiki/Dou_dizhu) ([斗地主](https://baike.baidu.com/item/%E6%96%97%E5%9C%B0%E4%B8%BB/177997)), the most popular card game in China. 

Deep Monte Carlo framework modified from the open source project [DouZero ResNet](https://github.com/Vincentzyx/Douzero_Resnet).

Compared to the framework provided by the open source project [DouZero](https://github.com/kwai/DouZero), the buffers part has been removed. 

The framework introduces a bidding phase, which allows RL models to be trained in realistic landlord environments.

The trained model(Card Model) vs. the open source DouZero(ADP) model has a win rate of 61.7%, reaching to reach the state-of-the-art.

<img width="500" src="https://raw.githubusercontent.com/RuBP17/AlphaDou/main/pics/compare.png" alt="Logo" />



## Training
To use GPU for training, run
```
python3 train.py
```
This will train AlphaDou on one GPU. To train AlphaDou on multiple GPUs. Use the following arguments.
*   `--gpu_devices`: what gpu devices are visible
*   `--num_actor_devices`: how many of the GPU deveices will be used for simulation, i.e., self-play
*   `--num_actors`: how many actor processes will be used for each device
*   `--training_device`: which device will be used for training DouZero

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
To use CPU training or simulation, use the following arguments:
*   `--training_device cpu`: Use CPU to train the model
*   `--actor_device_cpu`: Use CPU as actors

For example, use the following command to run everything on CPU:
```
python3 train.py --actor_device_cpu --training_device cpu
```
The following command only runs actors on CPU:
```
python3 train.py --actor_device_cpu
```


## Evaluation

The evaluation can be performed with GPU or CPU (GPU will be much faster). The performance is evaluated through self-play. We have provided pre-trained models and some heuristics as baselines:
For the bidding phase, the following agendas are provided here for testing:
*   [random](douzero/evaluation/random_agent.py): agents that play randomly 
*   SLModel(`baselines/SLModel/`): Threshold bidding Model Trained by Supervised Learning. You can set up "Supervised" in evaluate.py to test the SLModel

For the cardplay phase, the following agendas are provided here for testing:
*   [random](douzero/evaluation/random_agent.py): agents that play randomly 
*   DouZero-ADP (`baselines/test/`): the pretrained DouZero agents with Average Difference Points (ADP) as objective
*   DouZero ResNet (`baselines/best/`): the pretrained DouZero ResNet agents with Average Difference Points (ADP) as objective

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
Some important hyperparameters are as follows.
*   `--output`: where the pickled data will be saved
*   `--num_games`: how many random games will be generated

### Step 2: Self-Play
```
python3 evaluate.py
```
Some important hyperparameters are as follows.
*   `--player_1_bid`: which agent will play as First bidding player, which can be random, Supervised, or the path of the pre-trained model
*   `--player_2_bid`: which agent will play as Second bidding player, which can be random, Supervised, or the path of the pre-trained model
*   `--player_3_bid`: which agent will play as Third bidding player, which can be random, Supervised, or the path of the pre-trained model
*   `--player_1_playcard`: which agent will play as Landlord, which can be random, or the path of the pre-trained model
*   `--player_2_playcard`: which agent will play as LandlordUp (the one plays before the Landlord), which can be random, or the path of the pre-trained model
*   `--player_3_playcard`: which agent will play as LandlordDown (the one plays after the Landlord), which can be random, or the path of the pre-trained model
*   `--eval_data`: the pickle file that contains evaluation data
*   `--num_workers`: how many subprocesses will be used
*   `--gpu_device`: which GPU to use. It will use CPU by default

### Evaluate while training
auto_test.py: can be used to automatically test new models while training.
```
python3 auto_test.py
```

