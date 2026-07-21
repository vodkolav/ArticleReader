from Benchmarking.pipelines.RL import RLAgent, Strategy
from Benchmarking.telemetry_manager import TelemetryManager
from Benchmarking.Pipeline import Pipeline

import gymnasium as gym
import numpy as np
import psutil
import os
import time




class RLPileline(Pipeline):
    """
    Class to manage the RL pipeline, including training and evaluation.

        Args:
            env: The Gymnasium environment.
            algorithm: The RLAlgorithm instance.
    """
    def __init__(self, env: gym, agent: RLAgent, experiment_config: dict = None):
        self.env = env
        
        self.agent = agent

        if experiment_config is None:
            experiment_config = self.config_template()

        experiment_config["algorithm"] = agent.get_parameters()
        
        experiment_config["strategy"] = agent.strategy.get_parameters()
        
        self.output_dir = "data/results"
        # Create Telemetry Manager
        self.telemetry = TelemetryManager(experiment_config)

    def get_algo(self, algorithm_name):
        from algorithms.dynamic_programming import PolicyIteration
        from algorithms.monte_carlo import MonteCarlo
        from algorithms.temporal_difference import TemporalDifference
        from algorithms.q_learning import QLearning
        from algorithms.sarsa import SARSA         

        # Mapping of algorithm names to their classes ---
        ALGORITHM_CLASSES = {
            "DynamicProgramming": PolicyIteration, # DP is handled differently, might not fit here
            "SARSA": SARSA,
            "QLearning": QLearning,
            "MonteCarlo": MonteCarlo,
            "TemporalDifference": TemporalDifference,
        }
        # Get the algorithm class from the mapping
        algorithm_class = ALGORITHM_CLASSES.get(algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}. "
                            f"Available algorithms: {list(ALGORITHM_CLASSES.keys())}")
        return algorithm_class


    @staticmethod
    def config_template():
        """
        Returns a template for the experiment configuration.
        
        Args:
            config: A dictionary to fill with the experiment parameters.
        
        Returns:
            A dictionary with the experiment configuration template.
        """
        conf = {
                "metadata": {
                    "telemetry_episodes_limit": 256,
                },
                "env": {},
                "algorithm": {},
                "strategy": {}
            }
        return conf
    

    def new_episode(self, i_episode):
        """Resets metrics for a new episode."""
        self.i_episode = i_episode
        self._current_episode = { 
            "i": i_episode,             
            "reward": 0,
            "length": 0,
            "replay": [],
            "info": [],
            "internal_state":{}, 
            "start": time.time()
        } 

    def record_step(self, reward, info):
        """Records metrics for a single step."""
        if self.telemetry.timetorecord(self.i_episode):
            if self.render:
                frame = self.env.render()  
                self._current_episode["replay"].append(frame)      
                self._current_episode["info"] = info #TODO: do we need this? 
            self._current_episode["reward"] += reward
            self._current_episode["length"] += 1


    def record_episode(self, terminated, truncated):
        """Records metrics at the end of an episode."""
        if self.telemetry.timetorecord(self.i_episode):
            self._current_episode = {
            "is_training": self.is_training, #TODO: what is this? 
            "end": time.time(),
            "terminated": terminated,
            "truncated": truncated,
            "internal_state": dict(self.agent.get_intestines())            ,
            "epsilon": self.agent.strategy.epsilon,
            "rss_mb": self.get_memory_usage_mb(),
            "size_bytes": self.agent.size(),
            "avg_reward": self.get_average_reward()   
            }
            self.telemetry.record_episode(self._current_episode)


    def run_episode(self, i_episode):
        """
        Runs a single episode in the environment.

            Args:
                i_episode: episode number

            Returns:
                The total reward for the episode.
        """
        state, info = self.env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        while not terminated and not truncated:
            self.new_episode(i_episode)
            # Algorithm chooses an action
            # Note: For evaluation, choose_action should ideally be purely greedy.
            # The QLearning class handles this internally based on episode count for epsilon decay.
            # For evaluation phase, ensure epsilon is effectively 0.
            if self.is_training:
                action = self.agent.choose_action(state, i_episode) # choose_action now handles exploration strategy
            else: 
                action = self.agent.choose_greedy_action(state)

            if action is None:
                print("wtf")

            # Environment takes a step
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Algorithm updates its internal state if training
            if self.is_training:
                # Pass all relevant info to the update method
                self.agent.update(state, action, reward, next_state, terminated, truncated) # Update signature might vary

            total_reward += reward
            state = next_state # Move to the next state

            self.record_step(reward, info)


        self.record_episode( terminated, truncated ) # Record episode number

        return self.agent.terminate_prematurely

    def get_average_reward(self, window_size: int = 100) -> float:
        """Calculates the average reward over the last window_size episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([ep["reward"] for ep in self.episodes[-window_size:]])


    def train_algorithm(self, num_episodes: int):
        """
        Runs the training loop for an episode-based algorithm.

        Args:
            num_episodes: Total number of episodes for training.
        """
        self.is_training = True # Set to True for training

        for i_episode in range(num_episodes):

            self.run_episode(i_episode) # No rendering during training usually
            self.telemetry.progress()
            if self.agent.terminate_prematurely:
                self.telemetry.display("Algorithm decided to terminate prematurely.")
                break


    def evaluate_algorithm(self, num_episodes: int = 10, initial_episode: int = 0):
        """
        Evaluates the learned policy of an algorithm.

        Args:
            num_episodes: Number of evaluation episodes.
        """
        self.is_training = False # Set to False for evaluation
        
        for i_episode in range(initial_episode, initial_episode + num_episodes):

            self.run_episode(i_episode)
            self.telemetry.progress()


    # --- Special Handling for Dynamic Programming ---
    def run_dynamic_programming(algorithm):
        """
        Runs a Dynamic Programming algorithm (which doesn't use step-by-step interaction loops in main).
        """
        print(f"--- Running Dynamic Programming: {type(algorithm).__name__} ---")
        # DP algorithms typically have a 'solve' or 'run' method that computes the policy/value function once
        algorithm.solve() # Assuming DP class has a 'solve' method

        print("Dynamic Programming finished.")
        # You would then get the policy/value function using algorithm.get_policy() or algorithm.get_value_function()


    def run_case(self, train_episodes, eval_episodes, render = False):

        self.render = render

        self.telemetry.start(train_episodes + eval_episodes) 

        self.train_algorithm(train_episodes)

        self.telemetry.display("\nTraining done.", newline=True)

        # --- Optional: Evaluate the trained algorithm ---
        self.evaluate_algorithm(eval_episodes, train_episodes+1)

        self.telemetry.end()

        # --- Example for Dynamic Programming (Different Structure) ---
        # DP algorithms don't fit the step-by-step update in the same training loop
        # dp_agent = DynamicProgramming(env, gamma=0.99, theta=1e-5)
        # run_dynamic_programming(dp_agent)
        # After running DP, you would evaluate its *policy* directly using run_episode with is_training=False

        # Close the environment
        self.env.close()


        
    def run_experiment(self, exp_config: dict) -> dict:
        """
        Runs a single RL experiment based on the provided configuration.
        This function will be run by a separate process.

        Args:
            exp_config: A dictionary containing all parameters for this experiment.

        Returns:
            A dictionary containing key results and path to saved telemetry.
        """
        # exp_name = exp_config.get("name", "unnamed_experiment")
        # print(f"[{os.getpid()}] Starting experiment: {exp_name}")

        # --- Extract parameters ---


        meta = exp_config["metadata"]
        num_training_episodes = meta["num_training_episodes"]
        num_eval_episodes = meta["num_eval_episodes"]
        render_evaluation = meta.get("render_evaluation", False) # Don't render in parallel usually
        save_ansi_frames = meta.get("save_ansi_frames", False) # Or handle differently
        
        if meta.get("skip", False):
            print(f"[{os.getpid()}] Skipping experiment due to config: experiment.metadata.skip")
            return {"status": "skipped", "experiment_id": None, "timestamp": None, "telemetry_filepath": None}

        algorithm_name = exp_config["algorithm"]["name"]
        algo_params = exp_config["algorithm"]["params"]
        

        # --- Setup Environment ---
        # Environments are NOT picklable, so each process must create its own.
        # Set render_mode to None or 'ansi' as 'human' rendering often conflicts in parallel processes.
        env_params = exp_config["env"]["params"]      
        env_params["render_mode"] = 'ansi' if save_ansi_frames else None

        env = gym.make(**env_params)

        # --- Instantiate Strategy, Agent, Experiment ---
        strategy_params = exp_config["strategy"]["params"]
        strat = Strategy(env.action_space.n, **strategy_params)



        agent = algorithm_class(env, strat, **algo_params) # Adjust based on the actual algorithm class

        # --- Run Experiment ---
        # The run_experiment method from your Experiment class
        # You might want to return a summary directly or save it.
        # For parallel runs, it's best to save telemetry to a unique file.    
        
        # Assuming your Experiment.run_experiment takes a telemetry object and potentially an output path
        self.run_case(
            train_episodes=num_training_episodes,
            eval_episodes=num_eval_episodes,
            render=render_evaluation, # Likely False for parallel runs
        )

        # Save configuration and results specific to this run
        # config_filepath = os.path.join(output_dir, "config.json")
        # with open(config_filepath, 'w') as f:
        #     json.dump(exp_config, f, indent=4)


        run_timestamp = exper.telemetry.metadata["start_time"] 
        ex_id = exper.telemetry.metadata["id"]

        fname =  f"{ex_id}.json"
        os.makedirs(output_dir, exist_ok=True)
        telemetry_filepath = os.path.join(output_dir, fname)
        exper.telemetry.save_metrics(telemetry_filepath) # Save the telemetry

        env.close()

        result = {"status": "done", "experiment_id": str(ex_id), "timestamp":run_timestamp, "telemetry_filepath": telemetry_filepath}

        return result
    

    def load_experiment(data):
        meta = data["metadata"]

        # Flatten the algorithm parameters into the metadata
        # I'll deal with strategy parameters later
        algo = data["algorithm"]
        algo.update(algo["params"])
        algo.pop("params", None)
        meta.update(algo)

        strat = data["strategy"]
        meta.update({"decay": strat["params"]["decay"],
                    "initial_epsilon": strat["params"]["initial_epsilon"],})

        episodes = pd.DataFrame(data["episodes"])
        episodes["exp_id"] = meta["id"]

        return meta, episodes



    def span_grid(self, grid):
         #, processed_text, grid
        """
        grid = {"chunk_length": range(50, 500, 50),
                "batch_size": (1, 2, 3, 5, 10, 20, 30, 50, 70, 100, 200),
                "tts_model": ["tts-tacotron2-ljspeech"],
                "vocoder_model": ["tts-hifigan-ljspeech"],
                "device": ["CPU"], 
            }
        grid
        """        
        self.provider = "speechbrain"
        
        self.case_objects = {}
        self.case = {}

        for d in grid["device"]:
            self.init_device(d)

            for tts_model_name in grid["tts_model"]:                
                self.init_tts_model(tts_model_name)

                for voc_model_name in grid["vocoder_model"]:     
                    self.init_voc_model(voc_model_name)

                    for chunk_length in grid["chunk_length"]:
                        self.init_chunker(processed_text, chunk_length)

                        for batch_size in grid["batch_size"]:
                            self.init_batch(batch_size)

                            print("-"*30)

        print("experiment complete.")



    def make_case(C: Constants, algo_name ,alpha, gamma, lambda_, epsilon = ("linear", 1) , theta = 1e-5, ):
        
        decay, eps = epsilon

        # descr = {"case": i, "algo_name": algo_name , "alpha": alpha, "gamma": gamma, 
        #          "lambda_":lambda_, "epsilon": epsilon, "theta": theta}

        Case =  {
            "metadata": {
                "name": f"",
                "description": f"Experiment with {algo_name} algorithm, gamma={gamma}, lambda={lambda_}",
                "num_training_episodes": C.NUM_TRAINING_EPISODES,
                "num_eval_episodes": C.NUM_EVAL_EPISODES,
                "render_evaluation": C.RENDER_EVALUATION,
                "save_ansi_frames": False,
                "telemetry_episodes_limit": 256,
                "skip": C.SKIP
            },
            "env": {
                "name": C.ENV_ID,
            } ,
            "algorithm": {
                "name": algo_name ,
                "params": {
                    "alpha": alpha,  
                    "gamma": gamma,
                    "lambda_": lambda_,
                    "theta": theta,  # Only for Dynamic Programming
                }
            },
            "strategy": {
                "name": "EpsilonGreedy",
                "params":{
                    "decay": decay,
                    "initial_epsilon": eps,
                    "min_epsilon": 0.01,
                    "epsilon_decay_episodes": C.NUM_TRAINING_EPISODES
                }
            }
        }
        return Case