'''
Test script for AB-Mapper algorithm on warehouse environments.
Adapted from train.py, with focus on testing rather than training.
'''
import time
import yaml
import numpy as np
import argparse
from envs import grid_env
import os
import torch
import datetime
import utils
import sys
from pathlib import Path
import json
from model1 import Actor
import csv

actions = ['N','S','E','W','.']
idx_to_act = {0:"N",1:"S",2:"E",3:"W", 5:"."}
act_to_idx = dict(zip(idx_to_act.values(),idx_to_act.keys()))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-dir", help="dataset directory", default="/home/andrea/CODE/master_thesis_MAPF_DRL/baselines/Dataset/")
parser.add_argument("--model-dir", help="model dir", default="weights")
parser.add_argument("--load", help="model to load", default="exp3_60_65_hard_lr_0.0003_seed825_21-07-31-09-42-46.pth")
parser.add_argument("--seed", type=int, default=825, help="random seed")
parser.add_argument("--obs", type=int, default=0, help="dynamic obstacle number")
parser.add_argument("--render", type=bool, default=False, help="render the env")
parser.add_argument("--goal-range", type=int, default=6, help="goal sample range")
parser.add_argument("--agent-type", type=int, default=0, help="agent type selection")
parser.add_argument("--sub_num_agent", default=15, type=int, help="number of sub-agents to consider")

args = parser.parse_args()

agent_type = args.agent_type
if agent_type==0:
    from agent import Agent, compute_returns
    print("Import agent with full features...")
elif agent_type==1:
    from agent_no_A_star import Agent, compute_returns
    print("Import agent without A_star planner...")
elif agent_type==2:
    from agent_no_trajectory import Agent, compute_returns
    print("Import agent without dyanmic obstacle traj...")
elif agent_type==3:
    from agent_no_guidance import Agent, compute_returns
    print("Import agent without sub goal guidance...")
else:
    sys.exit("without such type of agent!! check the --agent-type for more detail.")

RENDER = args.render
MAX_STEP_RATIO = 20
SUCCESS_RATE_THRES = 0.99
MODEL_SAVE_NAME = "AB-MAPPER"

def count_collisions(solution, obstacle_map):
    """Count agent-agent and obstacle collisions in the solution."""
    agent_agent_collisions = 0
    obstacle_collisions = 0
    num_agents = 0
    
    # Convert solution format to timestep-based format
    timestep_based_solution = []
    if len(solution) > 0:
        # Find max timestep
        max_timestep = 0
        for agent_path in solution:
            for pos in agent_path:
                max_timestep = max(max_timestep, pos[2])
                
        num_agents = len(solution)
        # Initialize timestep-based solution
        timestep_based_solution = [[] for _ in range(max_timestep + 1)]
        
        # Fill in the positions for each timestep
        for agent_idx, agent_path in enumerate(solution):
            positions = {}  # Dictionary to store position at each timestep
            for pos in agent_path:
                positions[pos[2]] = (pos[0], pos[1])
            
            # Ensure every timestep has a position
            for t in range(max_timestep + 1):
                if t in positions:
                    timestep_based_solution[t].append(positions[t])
                elif t > 0 and t-1 in positions:
                    # If missing, use previous position
                    timestep_based_solution[t].append(positions[t-1])
                else:
                    # Should not happen in proper solutions
                    timestep_based_solution[t].append((-1, -1))
    
    # Now count collisions
    for timestep in range(len(timestep_based_solution)):
        positions_at_timestep = timestep_based_solution[timestep]
        current_agent_positions = []
        
        for agent_idx in range(len(positions_at_timestep)):
            agent_pos = positions_at_timestep[agent_idx]
            
            # Check for obstacle collisions
            if agent_pos[0] >= 0 and agent_pos[1] >= 0:  # Valid position
                if agent_pos[0] < obstacle_map.shape[0] and agent_pos[1] < obstacle_map.shape[1]:
                    if obstacle_map[agent_pos[0], agent_pos[1]] == -1:
                        obstacle_collisions += 1
            
            # Prepare for agent-agent collision check
            current_agent_positions.append(agent_pos)
        
        # Agent-agent collision check
        for i in range(len(current_agent_positions)):
            for j in range(i+1, len(current_agent_positions)):
                if current_agent_positions[i] == current_agent_positions[j]:
                    # Count collision for both agents involved
                    agent_agent_collisions += 2
    
    return agent_agent_collisions, obstacle_collisions

def count_actions(action_list):
    """Count the number of actions that are not 'stay'."""
    count = 0
    for action in action_list:
        if action != '.':
            count += 1
    return count

def get_csv_logger(model_dir, default_model_name):
    model_dir_path = Path(model_dir)
    csv_path = model_dir_path / f"log-{default_model_name}.csv"
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def convert_numpy_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj

def run_simulations(dataset_dir, map_name, num_agents, size, iter_idx, actor, device):
    """Run simulation for a specific map and agent configuration."""
    dataset_dir_path = Path(dataset_dir)
    map_dir = dataset_dir_path / map_name / "input/map"
    
    # Load map configuration
    with open(os.path.join(map_dir, map_name + ".yaml")) as map_file:
        map_config = yaml.load(map_file, Loader=yaml.FullLoader)
    
    # Setup environment
    # Ensure dataset_dir has trailing slash for proper path concatenation in GridEnv
    dataset_dir_with_slash = str(dataset_dir_path) + os.path.sep
    env = grid_env.GridEnv(dataset_dir_with_slash, map_name, map_config, agent_num=num_agents, window=None,
                         obs_num=args.obs, goal_range=args.goal_range)
    state = env.reset()
    
    # Create agent list
    agent_list = []
    for i in range(num_agents):
        agent = Agent(env.background_grid.copy(), ID=i)
        agent.set_max_step(state, ratio=MAX_STEP_RATIO)
        agent_list.append(agent)
    
    # Get max step from agents
    # max_step, max_step_list = utils.get_max_step(agent_list)
    max_step = 256
    
    success_rate = 0
    first_done = np.zeros(num_agents)
    episode_length = 0
    total_steps = 0
    steps_for_max_step = np.zeros(num_agents)
    
    # Get agents initial position coordinates
    solution = [[tuple(state["pose"][i]) + (0,)] for i in range(num_agents)]
    
    start_time = time.time()
    results = dict()
    
    try:
        for step_num in range(max_step):
            input_img_list = []
            input_val_list = []
            attention_index = []
            
            for i in range(num_agents):
                agent = agent_list[i]
                input_img, input_val, index = agent.preprocess(state, sub_num_agent=args.sub_num_agent, replan=True)
                attention_index.append(index)
                input_img_list.append(input_img)
                input_val_list.append(input_val)
            
            attention_actions_list, action_list, img_list, actions_prob, log_probs, entropy_s = actor.forward(state_img=input_img_list, state_val=input_val_list)
            
            # Count the number of actions different from '.' (stay)
            total_steps += count_actions(action_list)
            
            for i in range(num_agents):
                if action_list[i] != '.':
                    steps_for_max_step[i] += 1
            
            next_state, reward, done, _ = env.step(action_list)
            
            # Append coordinates to solution
            for i in range(num_agents):
                solution[i].append(tuple(next_state["pose"][i]) + (step_num+1,))
            
            for i in range(num_agents):
                agent = agent_list[i]
                if done[i] and first_done[i]:
                    first_done[i] = 1
                    continue
                if done[i] and not first_done[i]:
                    first_done[i] = 1
                
                agent.collision = next_state["collision"]
                agent.steps = next_state["steps"]
            
            state = next_state
            episode_length += 1
            if RENDER:
                env.render(show_traj=True)
            
            success_rate = np.sum(done) / num_agents
            if success_rate > SUCCESS_RATE_THRES:
                # If more than SUCCESS_RATE_THRES% agents reached the goal
                break
        
        elapsed_time = time.time() - start_time
        
        # Prepare results
        results['finished'] = success_rate > SUCCESS_RATE_THRES
        if results['finished']:
            results['time'] = elapsed_time
            results['episode_length'] = episode_length
            results['total_steps'] = total_steps
            results['avg_steps'] = total_steps / num_agents if num_agents > 0 else 0
            results['max_steps'] = np.max(steps_for_max_step) if steps_for_max_step.size > 0 else 0
            results['min_steps'] = np.min(steps_for_max_step) if steps_for_max_step.size > 0 else 0
        
            # Calculate costs similar to PRIMAL's method
            agent_costs = np.zeros(num_agents)
            for i in range(num_agents):
                if first_done[i]:
                    # Agent reached the goal
                    agent_costs[i] = state["steps"][i] if "steps" in state else episode_length
                else:
                    # Agent didn't reach the goal
                    agent_costs[i] = episode_length
                    
            results['total_costs'] = np.sum(agent_costs)
            results['avg_costs'] = np.mean(agent_costs)
            results['max_costs'] = np.max(agent_costs)
            results['min_costs'] = np.min(agent_costs)
        
            # Calculate collisions
            agent_coll, obs_coll = count_collisions(solution, env.background_grid)
            results['crashed'] = (agent_coll + obs_coll) > 0
            results['agent_collisions'] = agent_coll
            results['obstacle_collisions'] = obs_coll
            results['collisions'] = agent_coll + obs_coll
            results['agent_coll_rate'] = agent_coll / (episode_length * num_agents)
            results['obstacle_coll_rate'] = obs_coll / (episode_length * num_agents)
            results['total_coll_rate'] = (agent_coll + obs_coll) / (episode_length * num_agents)
        else:
            results['time'] = None
            results['episode_length'] = None
            results['total_steps'] = None
            results['avg_steps'] = None
            results['max_steps'] = None
            results['min_steps'] = None
            results['crashed'] = None
            results['agent_collisions'] = None
            results['obstacle_collisions'] = None
            results['collisions'] = None
            results['agent_coll_rate'] = None
            results['obstacle_coll_rate'] = None
            results['total_coll_rate'] = None
    
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        results['finished'] = False
        results['time'] = time.time() - start_time
        results['agent_collisions'] = 0
        results['obstacle_collisions'] = 0
        results['crashed'] = False
        results['agent_coll_rate'] = 0
        results['obstacle_coll_rate'] = 0
        results['total_coll_rate'] = 0
        results['min_steps'] = 0
        results['min_costs'] = 0
    
    return results, solution

def generate_start_and_goals(dataset_dir, map_name, num_agents, id):
    """Load start and goal positions from dataset files."""
    dataset_dir_path = Path(dataset_dir)
    filepath = dataset_dir_path / map_name / "input" / "start_and_goal" / f"{num_agents}_agents"
    case_name = filepath / f"{map_name}_{num_agents}_agents_ID_{str(id).zfill(3)}.npy"
    
    if case_name.exists():
        return np.load(case_name, allow_pickle=True)
    else:
        print(f"Warning: Case file {case_name} not found")
        return None

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up paths
    ab_mapper_dir = Path(__file__).resolve().parent
    results_path = ab_mapper_dir / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    model_dir = ab_mapper_dir / args.model_dir
    
    # Load the model
    model_path = model_dir / args.load
    if not model_path.exists():
        model_path = str(model_dir / args.load)
    
    baseline_dir = ab_mapper_dir.parent
    dataset_dir = Path(args.dataset_dir)
    
    # Set random seed
    utils.seed(args.seed)
    
    # Map configurations for testing
    map_configurations = [
        # {
        #     "map_name": "15_15_simple_warehouse",
        #     "size": 15,
        #     "n_tests": 2,
        #     "list_num_agents": [4, 8, 12, 16, 20, 22]
        # },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4]
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4]
        }
    ]
    
    header = ["n_agents", 
              "success_rate", "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_step", "total_step_std", "total_step_min", "total_step_max",
              "avg_step", "avg_step_std", "avg_step_min", "avg_step_max",
              "max_step", "max_step_std", "max_step_min", "max_step_max",
              "min_step", "min_step_std", "min_step_min", "min_step_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]
    
    # Process each map configuration
    for config in map_configurations:
        map_name = config["map_name"]
        size = config["size"]
        n_tests = config["n_tests"]
        list_num_agents = config["list_num_agents"]
        
        print(f"\nProcessing map: {map_name}")
        
        # Create output directory for results
        output_dir = dataset_dir / map_name / "output" / MODEL_SAVE_NAME
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logger
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
        csv_filename_base = f'{MODEL_SAVE_NAME}_{sanitized_map_name}_{date}'
        csv_file, csv_logger = get_csv_logger(str(results_path), csv_filename_base)
        
        csv_logger.writerow(header)
        csv_file.flush()
    
        
        # Process each agent count
        for num_agents in list_num_agents:
            # Load actor model
            actor = Actor(number_of_agents=num_agents).to(device)
            actor.load_state_dict(torch.load(model_path, map_location=device))
            actor.eval()  # Set to evaluation mode

            output_agent_dir = output_dir / f"{num_agents}_agents"
            output_agent_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize result storage
            results = {
                'finished': [], 'time': [], 'episode_length': [],
                'total_steps': [], 'avg_steps': [], 'max_steps': [], 'min_steps': [],
                'total_costs': [], 'avg_costs': [], 'max_costs': [], 'min_costs': [],
                'crashed': [], 'agent_coll_rate': [], 'obstacle_coll_rate': [], 'total_coll_rate': []
            }
            
            print(f"Starting tests for {num_agents} agents on map {map_name}")
            
            # Run simulations for this agent count
            for iter_idx in range(n_tests):
                print(f"Running test {iter_idx+1}/{n_tests}")
                res, solution = run_simulations(
                    str(dataset_dir), map_name, num_agents, size, iter_idx, actor, device
                )
                
                # Save results
                results['finished'].append(res['finished'])
                if res['finished']:
                    results['time'].append(res['time'])
                    results['episode_length'].append(res['episode_length'])
                    results['total_steps'].append(res['total_steps'])
                    results['avg_steps'].append(res['avg_steps'])
                    results['max_steps'].append(res['max_steps'])
                    results['min_steps'].append(res['min_steps'])
                    results['total_costs'].append(res['total_costs'])
                    results['avg_costs'].append(res['avg_costs'])
                    results['max_costs'].append(res['max_costs'])
                    results['min_costs'].append(res['min_costs'])
                    results['agent_coll_rate'].append(res['agent_coll_rate'])
                    results['obstacle_coll_rate'].append(res['obstacle_coll_rate'])
                    results['total_coll_rate'].append(res['total_coll_rate'])
                    results['crashed'].append(res['crashed'])
                
                # Save solution to file
                solution_filepath = output_agent_dir / f"solution_{MODEL_SAVE_NAME}_{map_name}_{num_agents}_agents_ID_{str(iter_idx).zfill(3)}.txt"
                with open(solution_filepath, 'w') as f:
                    f.write("Metrics:\n")
                    # Convert numpy values to Python native types before serialization
                    serializable_res = convert_numpy_to_native(res)
                    json.dump(serializable_res, f, indent=4)
                    f.write("\n\nSolution:\n")
                    if solution:
                        for agent_path in solution:
                            f.write(f"{agent_path}\n")
                    else:
                        f.write("No solution found.\n")
            
            # Calculate aggregated metrics
            final_results = {}
            final_results['finished'] = np.sum(results['finished']) / len(results['finished']) if len(results['finished']) > 0 else 0
            
            # Calculate statistics for metrics when available
            metric_keys = ['time', 'episode_length', 'total_steps', 'avg_steps', 'max_steps', 'min_steps', 
                         'total_costs', 'avg_costs', 'max_costs', 'min_costs', 
                         'agent_coll_rate', 'obstacle_coll_rate', 'total_coll_rate']
            
            for key in metric_keys:
                if results[key]:
                    final_results[key] = np.mean(results[key])
                else:
                    final_results[key] = 0
            
            final_results['crashed'] = np.sum(results['crashed']) / len(results['crashed']) if len(results['crashed']) > 0 else 0
            
            print(final_results)
            
            # Write results to CSV
            data = [num_agents,
                    final_results['finished'] * 100,  # convert to percentage
                    final_results['time'],
                    np.std(results['time']) if results['time'] else 0,
                    np.min(results['time']) if results['time'] else 0,
                    np.max(results['time']) if results['time'] else 0,
                    final_results['episode_length'],
                    np.std(results['episode_length']) if results['episode_length'] else 0,
                    np.min(results['episode_length']) if results['episode_length'] else 0,
                    np.max(results['episode_length']) if results['episode_length'] else 0,
                    final_results['total_steps'],
                    np.std(results['total_steps']) if results['total_steps'] else 0,
                    np.min(results['total_steps']) if results['total_steps'] else 0,
                    np.max(results['total_steps']) if results['total_steps'] else 0,
                    final_results['avg_steps'],
                    np.std(results['avg_steps']) if results['avg_steps'] else 0,
                    np.min(results['avg_steps']) if results['avg_steps'] else 0,
                    np.max(results['avg_steps']) if results['avg_steps'] else 0,
                    final_results['max_steps'],
                    np.std(results['max_steps']) if results['max_steps'] else 0,
                    np.min(results['max_steps']) if results['max_steps'] else 0,
                    np.max(results['max_steps']) if results['max_steps'] else 0,
                    final_results['min_steps'],
                    np.std(results['min_steps']) if results['min_steps'] else 0,
                    np.min(results['min_steps']) if results['min_steps'] else 0,
                    np.max(results['min_steps']) if results['min_steps'] else 0,
                    final_results['total_costs'],
                    np.std(results['total_costs']) if results['total_costs'] else 0,
                    np.min(results['total_costs']) if results['total_costs'] else 0,
                    np.max(results['total_costs']) if results['total_costs'] else 0,
                    final_results['avg_costs'],
                    np.std(results['avg_costs']) if results['avg_costs'] else 0,
                    np.min(results['avg_costs']) if results['avg_costs'] else 0,
                    np.max(results['avg_costs']) if results['avg_costs'] else 0,
                    final_results['max_costs'],
                    np.std(results['max_costs']) if results['max_costs'] else 0,
                    np.min(results['max_costs']) if results['max_costs'] else 0,
                    np.max(results['max_costs']) if results['max_costs'] else 0,
                    final_results['min_costs'],
                    np.std(results['min_costs']) if results['min_costs'] else 0,
                    np.min(results['min_costs']) if results['min_costs'] else 0,
                    np.max(results['min_costs']) if results['min_costs'] else 0,
                    final_results['agent_coll_rate'] * 100,  # convert to percentage
                    np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    final_results['obstacle_coll_rate'] * 100,  # convert to percentage
                    np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    final_results['total_coll_rate'] * 100,  # convert to percentage
                    np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0
                   ]
            csv_logger.writerow(data)
            csv_file.flush()
        
        csv_file.close()

    print("Finished all tests!")
