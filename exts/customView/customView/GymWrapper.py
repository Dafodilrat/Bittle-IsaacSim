

import gymnasium
from gymnasium import spaces
import numpy as np

class gym_env(gymnasium.Env):
    
    def __init__(self ,bittle ,env ,weights=[100,10,10,0.5,0.2,10]):

        super().__init__()

        print("initializing isaac sim env .....",flush=True)
        
        self.weights = weights

        self.bittle = bittle
        
        dof,low,high = self.bittle.get_robot_dof()
        
        self.prev_action = np.zeros(dof,dtype=np.float32)
        
        self.action_space = spaces.Box(low=low, high=high, shape=(dof,), dtype=np.float32)

        self.environment = env

        obs_low = np.concatenate([
            np.array([-np.inf, -np.inf, -np.inf]),            # Position
            -np.ones(3) * np.pi,                   # Orientation (radians)
            low,                 # Joint angles
            -np.ones(dof) * 10.0,                  # Joint velocities
        ])  


        obs_high = np.concatenate([
            np.array([np.inf, np.inf, np.inf]),            # Position
            np.ones(3) * np.pi,                   # Orientation (radians)
            high,                 # Joint angles
            np.ones(dof) * 10.0,                  # Joint velocities
        ])

        self.observation_space = spaces.Box(low=obs_low-0.01, high=obs_high+0.01, dtype=np.float64)

        self.goals = self.environment.get_valid_positions_on_terrain()

        self.current_goal = self.goals[np.random.choice(len(self.goals))]

        self.total_rewards = 0

        self.observations = [[0,0,0],[0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]

        self.upright_steps=1

        self.prev_distance = 0

        self.delta = 0

    def generate_info(self):
           
        pos, orientation, joint_angles, joint_velocities = self.observations
        
        info = {}
        
        info["goal"]= self.current_goal
        info["pose"]= pos
        info["orientation"] = orientation
        info["joint_angles"] = joint_angles
        info["joint_vel"] = joint_velocities
        info["new"] = False
        info["total_reward"] = self.total_rewards
        info["distance_to_goal"] = np.linalg.norm(np.array(pos[:2]) - np.array(list(self.current_goal)[:2]))
        info["delta movement"] = self.delta
        return info

    def step(self,action):

        self.bittle.set_robot_action(action)
        
        self.observations = self.bittle.get_robot_observation()
                
        reward = self.calculate_reward(action)
        self.total_rewards+=reward

        done = self.is_terminated()
        ended = self.is_truncated()
        
        info = self.generate_info()
        info["reward"] = reward
        info["terminated"] = done
        info["truncated"] = ended
        
        self.prev_action = action

        observations = np.concatenate([
            self.observations[0],
            self.observations[1],
            self.observations[2],
            self.observations[3], 
        ])

        for key, value in info.items():
            print(f"{key}: {value}")

        print("---------------------------------------")

        return observations,reward,done,ended,info

    def is_terminated(self):

        pos, orientation, joint_angles, joint_velocities = self.observations
        
        if np.linalg.norm(np.array(pos[:2]) - np.array(list(self.current_goal)[:2])) < 0.1:
            return True
        
        #roll, pitch, yaw = orientation   # orientation

        # if abs(roll)> 1 :
        #     return True

        # if abs(pitch)> 1 :
        #     return True
        

        return False
    
    def is_truncated(self):
        
        if not self.bittle.is_running():
            return False
        
        return False
    
    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        
        if len(self.goals) > 1 :

            self.goals.remove(self.current_goal)
        
        else:
            self.goals = self.environment.get_valid_positions_on_terrain()
        

        self.current_goal = self.goals[np.random.choice(len(self.goals))]
        
        self.bittle.reset_simulation()

        dof,low,high = self.bittle.get_robot_dof()
        
        self.prev_action =  np.zeros(dof,dtype=np.float32)

        observations = self.observations

        observations = np.concatenate([
            self.observations[0],
            self.observations[1],
            self.observations[2],
            self.observations[3], 
        ])

        self.total_rewards = 0

        info = self.generate_info()
        info["new"] = True

        print("------------------RESET---------------------")

        return (observations,info)       

    def calculate_reward(self, action):

        pos,orientation,joint_angles,joint_velocities = self.observations
        params=self.weights

        # --- Extract key values ---
        z = pos[2]                     # z-position of base
        roll, pitch, yaw = orientation   # orientation
        delta = np.abs(action - self.prev_action)

        # --- Posture penalties ---
        roll_penalty = max(0.0, abs(roll) - 1.2)
        pitch_penalty = max(0.0, abs(pitch) - 1.2)
        posture_penalty = roll_penalty**2 + pitch_penalty**2

        # --- Jerk penalty (action delta > 0.1 rad) ---
        jerk_penalty = np.sum(np.clip(delta - 0.1, 0, None))

        # --- Velocity penalty (joint speed > 8 rad/s) ---
        velocity_penalty = np.sum(np.clip(np.abs(joint_velocities) - 90.0, 0, None))

        # --- Height penalty if bot crouches or falls ---
        z_penalty = 0.4 - pos[2] if z < 0.4 else 0.0

        # --- Bonuses ---
        upright_bonus = 1.0 if abs(roll) < 1 and abs(pitch) < 1 else 0.0
        smooth_bonus = 1.0 if np.all(delta < 0.5) else 0.0

        # ---- location reward ---
        dist_to_goal = np.linalg.norm(list(self.current_goal)[:2] - pos[:2] )    
        self.delta = abs(self.prev_distance - dist_to_goal)
        # --- Final reward ---
        reward = 0.0
        reward += params[0] * upright_bonus
        reward += params[1] * smooth_bonus
        reward -= params[2] * posture_penalty
        reward -= params[3] * jerk_penalty
        reward -= params[4] * velocity_penalty
        reward -= z_penalty
        reward -= params[5]* dist_to_goal
        # self.steps_before_reset+=1

        if self.delta < 0.05 and self.prev_distance!=0 :
            reward -= 200 

        self.prev_action = action.copy()
        self.prev_distance = dist_to_goal
        
        return reward

