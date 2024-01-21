'''
@Author: Zuxin Liu
@Email: zuxinl@andrew.cmu.edu
@Date:   2020-03-01 21:53:06
@LastEditTime: 2020-03-25 22:42:13
@Description:
'''
import time

import yaml
import numpy as np
import argparse
from envs import grid_env
from envs.rendering import Window
import os
import torch
from torch import optim
import datetime
import tensorboardX
import utils
import sys

## new add

from utils.attention_sac import AttentionSAC
from torch.autograd import Variable
from torch import Tensor
from read_plot import R_P
from model1 import Actor
from utils.misc import soft_update, hard_update
import csv


actions = ['N','S','E','W','NW','WS','SE','EN','.']
idx_to_act = {0:"N",1:"S",2:"E",3:"W", 4:"NW",5:"WS",6:"SE",7:"EN",8:"."}
act_to_idx = dict(zip(idx_to_act.values(),idx_to_act.keys()))
avg_reward = []
succ_rate = []
coll_rate = []
avg_step = []
tot_step = []
ep_len = []
parser = argparse.ArgumentParser()

parser.add_argument("--exp", help="experiment number",type=int, default=3)#


parser.add_argument("--dataset-dir", help="static map dir", default="/home/andrea/Thesis/baselines/Dataset/")
parser.add_argument("--map-name", help="static map name", default="50_55_simple_warehouse")
parser.add_argument("--map-dir", help="static map dir", default="/home/andrea/Thesis/baselines/Dataset/")
parser.add_argument("--model-dir", help="model dir", default="weights")

parser.add_argument("--load", help="load from the given path", default=False)
parser.add_argument("--name", help="model name", default=None)
parser.add_argument("--lr", type=float, default=0.0003,
                    help="learning rate (default: 0.0003)")
parser.add_argument("--seed", type=int, default=825,
                    help="random seed (default: 1)")
parser.add_argument("--map", help="static map path", default="simple_warehouse")
parser.add_argument("--obs", type=int, default=0,
                    help="dynamic obstacle number (default: 4)")
parser.add_argument("--agents", type=int, default=4,
                    help="agents number (default: 4)")
parser.add_argument("--interval", type=int, default=100,
                    help="episode interval to update all the other agents' model to the best one's model parameter (default: 100)")
parser.add_argument("--entropy-weight", type=float, default=0.01,
                    help="entropy weight in the loss term (default: 0.01)")
parser.add_argument("--render", type=bool, default=False,
                    help="render the env to visualize (default: True)")
parser.add_argument("--goal-range", type=int, default=6,
                    help="goal sample range (default: 6)")
parser.add_argument("--agent-type", type=int, default=0,
                    help="0: full feature; 1: without global planner A_star guidance; 2: without dynamic obstacle trajectory; 3: without sub-goal guidance (default: 0)")
"""
new add
"""
parser.add_argument("--batch_size", type=int, default=10,
                    help= "Batch size of training")
parser.add_argument("--num_updates",default=4,type=int,
                    help="Number of updates per update cycle")
parser.add_argument("--pol_hidden_dim", default=128, type=int)
parser.add_argument("--critic_hidden_dim", default=175*8, type=int)
parser.add_argument("--attend_heads", default=4, type=int)
parser.add_argument("--pi_lr", default=0.001, type=float)
parser.add_argument("--q_lr", default=0.0001, type=float)
parser.add_argument("--tau", default=0.001, type=float)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--reward_scale", default=20., type=float)
parser.add_argument("--critic_model-dir", help="model dir", default="critic_weights")
parser.add_argument("--sub_num_agent", default=15, type=int)
parser.add_argument("--my_a_lr", default=0.000035, type=int)

parser.add_argument("--train", default=True,type=bool)
parser.add_argument("--test", default=True,type=bool)

args = parser.parse_args()

agent_type = args.agent_type
if agent_type==0:
    from agent import Agent, compute_returns
    print(" Import agent with full features...")
elif agent_type==1:
    from agent_no_A_star import Agent, compute_returns
    print(" Import agent without A_star planner...")
elif agent_type==2:
    from agent_no_trajectory import Agent, compute_returns
    print(" Import agent without dyanmic obstacle traj...")
elif agent_type==3:
    from agent_no_guidance import Agent, compute_returns
    print(" Import agent without sub goal guidance...")
else:
    sys.exit("without such type of agent!! check the --agent-type for more detail.")

def get_key1(dct, value):
   return list(filter(lambda k:dct[k] == value, dct))

RENDER=args.render
MAX_STEP_RATIO = 20 # 4
SUCCESS_RATE_THRES = 0.99
MODEL_SAVE_NAME = "AB-MAPPER"

model_dir, critic_model_dir = args.model_dir, args.critic_model_dir
if not os.path.exists(model_dir):
    os.mkdir(model_dir)



if args.load:
    load_model_path = os.path.join(model_dir, args.load)
    load_critic_model_path =  os.path.join(critic_model_dir, args.load)

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"exp{args.exp}_{args.map}_lr_{args.lr}_seed{args.seed}_{date}"

save_model_path = os.path.join(model_dir, default_model_name+".pth")

# Create output directory
output_dir = os.path.join(args.dataset_dir, args.map_name, "output", MODEL_SAVE_NAME)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.name:
    save_model_path = os.path.join(model_dir, args.name)

save_log_path = os.path.join(model_dir,"exp"+str(args.exp))

if not os.path.exists(save_log_path):
    os.mkdir(save_log_path)
    print("Exp log directory " , save_log_path ,  " Created ")
else:    
    print("Exp log directory ", save_log_path ,  " already exists")

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(save_log_path, default_model_name)

csv_file, csv_logger = utils.get_csv_logger(save_log_path, default_model_name)


tb_writer = tensorboardX.SummaryWriter(save_log_path)
# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

txt_logger.info(f"Device: {device}\n")

# environment initilization
map_dir = args.map_dir + args.map_name + "/input/map/"
with open( os.path.join(map_dir,args.map_name + ".yaml")) as map_file:
    map = yaml.load(map_file, Loader=yaml.FullLoader)

def count_actions(action_list):
    count = 0
    for action in action_list:
        if action != '.':
            count += 1
    return count

def insert_action(x):
    action_array = [0 for i in range(len(actions))]
    action_array[x]=1
    return action_array


file_name = ['./train_data/_reward/', './train_data/succ/','./train_data/_time/']

def train():
    num_agents_list = [64, 128, 256]
    for num_agents in num_agents_list:
        t1=time.time()
        time_list=[]
        # window = Window(default_model_name)
        utils.seed(args.seed)
        env = grid_env.GridEnv(args.dataset_dir, args.map_name, map, agent_num=num_agents, window=None, obs_num=args.obs, goal_range = args.goal_range)
        state = env.reset()

        txt_logger.info("Environments loaded\n")
        # if RENDER:
        #     env.render(show_traj=True)

        # create agent output directory
        agent_output_dir = os.path.join(output_dir, str(num_agents) + "_agent")
        if not os.path.exists(agent_output_dir):
            os.makedirs(agent_output_dir)

        agent_list = []
        for i in range(num_agents):
            agent = Agent(env.background_grid.copy(), ID=i)
            agent.set_max_step(state, ratio = MAX_STEP_RATIO)
            agent_list.append(agent)

            actor = Actor(number_of_agents=num_agents).to(device)
        if args.test == True:
            model = 'weights/exp3_60_65_hard_lr_0.0003_seed825_21-07-31-09-42-46.pth'
            actor.load_state_dict(torch.load(model,map_location=torch.device(device)))
            print('loading network')
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.my_a_lr)


        max_episode = 200 #1500
        step_count = 0
        save_model_count = 0
        save_model_frequency = args.interval
        sum_rewards_array = np.zeros(num_agents)
        entropy_weight = args.entropy_weight
        global_success_rate = []
        avg_step = []
        count_max_step_list = []

        model = AttentionSAC.init_from_env(env,
                                        agents=num_agents,
                                        tau=args.tau,
                                        pi_lr=args.pi_lr,
                                        q_lr=args.q_lr,
                                        gamma=args.gamma,
                                        pol_hidden_dim=args.pol_hidden_dim,
                                        critic_hidden_dim=args.critic_hidden_dim,
                                        attend_heads=args.attend_heads,
                                        reward_scale=args.reward_scale)


        for epi in range(max_episode):
                for agent in agent_list:
                    agent.reset_memory()

                save_model_count += 1
                max_step, max_step_list = utils.get_max_step(agent_list)
                # print("max_step",max_step)
                # print("max_step_list",max_step_list)
                success_rate = 0
                first_done = np.zeros(num_agents)
                # new add
                agent_values = [[] for i in range(len(agent_list))]
                episode_length = 0
                total_steps = 0
                steps_for_max_step = np.zeros(num_agents)
                
                # Get agents position coordinates
                solution = [[tuple(state["pose"][i]) + (0,)] for i in range(num_agents)]                    
                    

                for step_num in range(max_step):

                    input_img_list = []
                    input_val_list = []
                    attention_index = []
                    for i in range(num_agents):
                        agent = agent_list[i]

                        input_img, input_val, index = agent.preprocess(state,sub_num_agent=args.sub_num_agent, replan = True)
                        attention_index.append(index)
                        input_img_list.append(input_img)
                        input_val_list.append(input_val)

                    attention_actions_list, action_list, img_list, actions_prob, log_probs, entropy_s = actor.forward(state_img=input_img_list, state_val=input_val_list)

                    # count the number of actions different from '.' (stay)
                    total_steps += count_actions(action_list)

                    for i in range(num_agents):
                        if action_list[i] != '.':
                            steps_for_max_step[i] += 1

                    next_state, reward, done, _ = env.step(action_list)
                    
                    # Append coordinates to solution
                    for i in range(num_agents):
                        solution[i].append(tuple(next_state["pose"][i]) + (step_num+1,))

                    # print("Agents pose: ", next_state["pose"])

                    critic_in = list(zip(img_list, attention_actions_list))
                    # print("attention_index",attention_index)
                    values = model.critic(inps=critic_in, agent_index=attention_index, return_q = True, return_all_q = False)
                    # print('values', values,'len',len(values))

                    # for i in range(len(agent_list)):
                    #     agent_values[i].append((values[i]*actions_prob[i]).sum(dim=1, keepdim=True)) # V(s)
                    for i in range(num_agents):
                        agent = agent_list[i]
                        #print("agent %d done %d first_done %d "%(i, done[i], first_done[i]))
                        if done[i] and first_done[i]:
                            # agent.log_probs.pop()
                            # agent_values[i].pop()
                            first_done[i] = 1
                            continue
                        if done[i] and not first_done[i]:
                            first_done[i]=1
                        additional_reward = agent.compute_reward(next_state)
                        #print("agent %d reward: %3f, additional_reward: %3f"%(i, reward[i], additional_reward))
                        reward[i] += additional_reward
                        # agent.entropy += agent.current_ent
                        agent.collision = next_state["collision"]
                        agent.steps = next_state["steps"]
                        agent.rewards.append(torch.FloatTensor([reward[i]]).unsqueeze(1).to(device))
                        agent.masks.append(torch.FloatTensor([1 - done[i]]).unsqueeze(1).to(device))

                    state = next_state
                    episode_length += 1
                    if RENDER:
                        env.render(show_traj=True)
                    # time.sleep(1)
                    success_rate = np.sum(done)/num_agents
                    if success_rate>SUCCESS_RATE_THRES:
                        # if more than SUCCESS_RATE_THRES% agents reached the goal
                        break

                # print("Solution: ", solution)
                

                

                if success_rate>SUCCESS_RATE_THRES:
                    # if more than SUCCESS_RATE_THRES% agents reached the goal
                    global_success_rate.append(True)
                else:
                    global_success_rate.append(False)

                sum_reward_max = -9999
                total_reward = []
                extra_time = []
                next_input_img_list = []
                next_input_val_list = []

                attention_index_tar= []
                
                collision_rate = np.sum(state["collision"]) / (num_agents*(episode_length+1)) * 100

                for i in range(num_agents):
                    agent = agent_list[i]

                    #print("agent %d reward list "%(i), agent.rewards)
                    sum_reward = sum(agent.rewards).item()
                    sum_rewards_array[i] += sum_reward

                    total_reward.append(sum_reward)
                    collsions = np.array(state["collision"])
                    collsions[collsions>=1]=1
                    # steps = state["steps"]
                    extra_time.append( (state["steps"][i]-max_step_list[i])/max_step_list[i] )

                    input_img, input_val,index = agent.preprocess(next_state,sub_num_agent=args.sub_num_agent, replan = True)
                    attention_index_tar.append(index)

                    next_input_img_list.append(input_img)
                    next_input_val_list.append(input_val)

                attention_next_actions_list, next_action_list, next_img_list, next_actions_prob,next_log_probs, next_entropy_s = actor.forward(state_img=next_input_img_list, state_val=next_input_val_list)

                target_critic_in = list(zip(next_img_list, attention_next_actions_list))
                next_values = model.critic(inps=target_critic_in, agent_index=attention_index_tar) # Q(s,a)

                # loss_c = 0
                # loss_a = 0
                # loss_all = 0
                # for i in range(num_agents):
                #     # print(i)
                #     critic_delta = torch.FloatTensor([[reward[i]]]).cuda()+\
                #                 args.gamma*next_values[i]*\
                #                 torch.FloatTensor([[1 - done[i]]]).cuda()\
                #                 - values[i]
                #     loss_c += critic_delta**2
                #     loss_a += -(log_probs[0][i]*critic_delta)
                #     loss_all += loss_a+loss_c

                # actor_optimizer.zero_grad()
                # model.critic_optimizer.zero_grad()

                # loss_all.backward()

                # model.critic_optimizer.step()
                # actor_optimizer.step()

                state = env.reset()
                for agent in agent_list:
                    agent.set_max_step(state, ratio = MAX_STEP_RATIO)

                if RENDER:
                    env.render(show_traj=True)

                #print
                

                average_reward = np.mean(total_reward)
                # collision_rate = np.mean(collsions)
                extra_step = np.mean(extra_time)
                # total_steps = np.sum(steps)
                # average_step = np.mean(steps)
                average_step = total_steps/num_agents

                # print and save training info
                header = ["exp", "type", "update","success_rate", "avg_reward", "collision", "extra",  "max_reward"]
                data = [args.exp, agent_type , epi, success_rate, average_reward, collision_rate, extra_step, sum_reward_max]

                succ_rate.append(success_rate)
                avg_reward.append(average_reward)
                coll_rate.append(collision_rate)
                avg_step.append(average_step)
                count_max_step_list.append(np.max(steps_for_max_step))
                tot_step.append(total_steps)
                ep_len.append(episode_length)

                out = dict()
                out["finished"] = True if success_rate>SUCCESS_RATE_THRES else False
                if out["finished"]:
                    out["total_step"] = total_steps
                    out["avg_step"] = average_step
                    out["max_step"] = np.max(steps_for_max_step)
                    out["episode_length"] = episode_length
                out["collision_rate"] = collision_rate

                save_dict = {"metrics": out, "solution": solution}

                # save solution to file
                solution_file_name = "solution_" + MODEL_SAVE_NAME + "_" + args.map_name + "_" + str(num_agents) + "_agents_ID_" + str(epi).zfill(5) + ".npy"
                solution_file = os.path.join(agent_output_dir, solution_file_name)
                np.save(solution_file, save_dict)


                txt_logger.info(
                    "Exp {} | Agent {} | Epi {} | succ rate {:.2f}| avg reward {:.2f}| collision {:.2f}| extra {:.2f}| max reward {:.2f}|".format(*data))
                # if epi == 0:
                #     csv_logger.writerow(header)
                # csv_logger.writerow(data)
                # csv_file.flush()
                for field, value in zip(header, data):
                        tb_writer.add_scalar(field, value, epi)



                if epi % 100 == 0:
                    torch.cuda.empty_cache()

        # window.close()
        t2=time.time()
        time_list.append(t2-t1)
        # torch.save(model.critic.state_dict(),)
        glob_succ_rate = np.sum(global_success_rate)/len(global_success_rate) * 100
        glob_coll_rate = np.mean(coll_rate)
        glob_extra_time = np.mean(extra_time)
        glob_avg_step = np.mean(avg_step)
        glob_tot_step = np.mean(tot_step)
        glob_ep_len = np.mean(ep_len)
        glob_max_step = np.mean(count_max_step_list)
        header = ["n_agents", "success_rate", "collision_rate", "extra_time", "avg_step", 'total_step', 'max_step', 'episode_length', "total_step_std", "avg_step_std", "max_step_std", "episode_length_std", "total_step_min", "avg_step_min", "max_step_min", "episode_length_min", "total_step_max", "avg_step_max", "max_step_max", "episode_length_max"]
        data = [num_agents, glob_succ_rate, glob_coll_rate, glob_extra_time, glob_avg_step, glob_tot_step, glob_max_step, glob_ep_len, np.std(tot_step), np.std(avg_step), np.std(count_max_step_list), np.std(ep_len), np.min(tot_step), np.min(avg_step), np.min(count_max_step_list), np.min(ep_len), np.max(tot_step), np.max(avg_step), np.max(count_max_step_list), np.max(ep_len)]
        if num_agents == 4:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        data=[avg_reward, succ_rate, time_list, coll_rate, glob_succ_rate]

        i=0
        for name in file_name:
            values = data[i]
            with open(name+'exp'+str(args.exp)+'.txt', "w") as output:
                output.write(str(values))
                output.close()
            i+=1
        # torch.save(actor.state_dict(), save_model_path)
    


def read_show_data(path, smooth):
    rp = R_P(smooth=smooth)
    data = []
    for i in range(len(path)):
        data.append(rp.read(path[i]))
    rp.show(data)


def write_my_csv(header, data_list, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in data_list:
            writer.writerow(row)


if __name__ == "__main__":
    if args.train == True:
        train()










