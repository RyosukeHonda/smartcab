import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        
        #Set traffic_light and movement
        
        traffic_light=["red","green"]
        motion = [None, 'forward', 'left', 'right']
        waypoint,oncoming,left=motion,motion,motion

        
        #Initialize q_table
        #Set all features to 0
        self.q_table = {}
        for light in traffic_light:
            for point in waypoint:
                for on in oncoming:
                    for lf in left:
                        self.q_table[(light, point, on, lf)] = {None: 0, 'forward': 0, 'left': 0, 'right': 0}

        print self.q_table

    def reset(self,destination=None,total=0):

        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward=total



        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        
        # TODO: Update state
        self.state = (inputs['light'],
                      self.next_waypoint,
                      inputs['oncoming'],
                      inputs['left'])

        print "The current state is: {}".format(self.state)

        print "t:{}".format(t)
        
        # TODO: Select action according to your policy
        epsilon=0.1
        rand=random.random()
        if rand<epsilon:
            action=random.choice(Environment.valid_actions[0:])
            print "RANDOM ACTION"
        else:
            if max(self.q_table[self.state].values())==0:
                action=random.choice(Environment.valid_actions[0:])
                print self.q_table[self.state]
                print action
            else:
                action = max(self.q_table[self.state],
                     key=self.q_table[self.state].get)

            print action

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        print "REWARD IS:",reward
        self.total_reward+=reward
     
        print "Total Reward",self.total_reward
        
        # TODO: Learn policy based on state, action, reward
        # Set the tuning parameters
        alpha = 1.0/(1.0+t) # learning rate
        gamma = 1.0/(1.0+deadline) # discount factor
      #  alpha=0.5
       # gamma=0.2
        # Get the new state after the above action

        inputs_new = self.env.sense(self)
        state_new = (inputs_new['light'],
                     self.planner.next_waypoint(),
                     inputs_new['oncoming'],
                     inputs_new['left'])
        print "The new state is: {}".format(state_new)
        print "t:{}".format(t)
        print alpha,gamma
        



        
        # Calculate the Q_value
        q_value = (1 - alpha) * self.q_table[self.state][action] + \
                  alpha * (reward + gamma * max(self.q_table[state_new].values()))
        # Update the Q_table
        self.q_table[self.state][action] = q_value
        # Set current state and action as previous state and action
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print "\n"
        print "\n"
    
def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    # Now simulate it
    sim = Simulator(e, update_delay=1,display=True)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit
    print 'TRIALTRIALTRIALTRIAL'trial


    

if __name__ == '__main__':
    run()
    
