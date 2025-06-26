from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

from .PPO import stb3_PPO
from .world import Environment

if __name__ == "__main__":

    agents = []
    w = Environment()
    n = 1
    w.add_bittles(n = n)
    #b = w.bittlles[0]
    print("done",flush=True)
    # while simulation_app.is_running():
    #     simulation_app.update()
    
    for bittle in w.bittlles:
        t=stb3_PPO(params = [100,10,10,0.5,0.2,10],bittle = bittle,env = w) 
        agents.append(t)

    for agent in agents:
        agent.start_training()