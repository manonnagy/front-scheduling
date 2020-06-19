import gym
import gym_sch
import sys
from agentsch import *
from time import time

def timeit(func):
	def new_func(*args, **kwargs):
		init_time = time()
		res = func(*args, **kwargs)
		print("{0:10s} : {1:5f}s.".format(func.__name__, time()-init_time))
		return res
	return new_func

def gen_new_demand(env, k): 
    env.products["Farine"]["stock"] = 20
    env.products["Plastique"]["stock"] = 20
    env.products["PainDeMieEmballe"]["demand"] = k
    env.products["PainDeMieSansCrouteEmballe"]["demand"] = 20-k
    env.machines["Four"]["is_on"] = True
    env.machines["Decrouteur"]["is_on"] = True
    env.machines["Emballeur"]["is_on"] = True

@timeit
def train(episodes, agent, env):
    scores = list()
    t = time()
    for i in range(episodes+1):
        env.reset(stoch = True)
        gen_new_demand(env, random.randrange(21))
        reward = None
        compt = 0
        while not env.done or compt<10:
            state = np.array([env.observation_space])
            action = agent.get_best_action(state)
            state, action, reward, done, next_state = env.step([action])
            agent.update_sequential_replay(state, action[0], reward, done, next_state)
            compt += 1
        if i%5==0:
            grad = agent.train(batch_size = 32, epochs = 1, use_loss_scale = True, use_grad_clip = False)
        if i%500 == 0:
            score = 0
            for k in range(21):
                env.reset(stoch = False)
                gen_new_demand(env, k)
                compt = 0
                while not env.done or compt<10:
                    state = np.array([env.observation_space])
                    action = agent.get_best_action(state, rand = False)
                    env.step([action])
                    compt += 1
                score += env.score
            scores.append(score/21)
            print("episode: {0:5d}/{1:5d}, score: {2:7.2f}, time: {3:5.2f}s, eps: {4:4f}"
                  .format(i, episodes, scores[-1], time()-t, agent.epsilon))
            t = time()
            
    fig, ax = plt.subplots()
    ax.plot(scores)
    ax.plot([0]*len(scores), color = 'r')
    print(max(scores))

def validate(agent, env):
	for k in range(21):
		env.reset(stoch = False)
		env.products["PainDeMieEmballe"]["demand"] = k
		env.products["PainDeMieSansCrouteEmballe"]["demand"] = 20-k
		compt = 0
		while not env.done or compt<10:
			state = np.array([env.observation_space])
			action = agent.get_best_action(state, rand = False)
			env.step([action])
			compt += 1
		print("Demand : {:2d}:{:2d} Score : {}.".format(k,20-k,env.score))

if __name__ == "__main__":
	env = gym.make("sch-v0", stoch = True)
	env.from_json("PraindeMine.json")
	agent = Agent(env.observation_space.__len__(), env.action_space.__len__())
	train(1000000, agent, env)
	validate(agent, env)
	try:
		agent.model_saving(sys.argv[1])
	except IndexError:
		agent.model_saving("agent")
