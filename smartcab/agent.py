import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import numpy as np

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
        waypoint,oncoming,left,right=motion,motion,motion,motion


        #Initialize q_table(States are traffic_light,waypoint,oncoming,left and right)
        #Set all features to 0
        self.q_table = {}
        for light in traffic_light:
            for point in waypoint:
                for on in oncoming:
                    for lf in left:
                        for ri in right:
                            self.q_table[(light, point, on, lf,ri)] = {None: 0, 'forward': 0, 'left': 0, 'right': 0}
       # print self.q_table


        self.episode=0
        self.preserve=[]
        self.failure=0

    def reset(self,destination=None,total=0):

        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.total_reward=total


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        old_time=t
        print 'Old_TIME',old_time

        # TODO: Update state
        self.state = (inputs['light'],
                      self.next_waypoint,
                      inputs['oncoming'],
                      inputs['left'],
                      inputs['right'])

        print "The current state is: {}".format(self.state)

        print "t:{}".format(t)

        # TODO: Select action according to your policy
        epsilon=0.1
        #epsilon=1.0
        rand=random.random()
        if rand<epsilon:
            action=random.choice(Environment.valid_actions[0:])
            print "RANDOM ACTION"
        else:
            if max(self.q_table[self.state].values())==0:
                action=random.choice(Environment.valid_actions[0:])
                print self.q_table[self.state]

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


        #alpha = 1.0/(1.0+t) # learning rate
        #gamma = 1.0/(1.0+deadline) # discount factor
        alpha=0.5 #learning rate(constant version)
        gamma=0.9 #discount factor(constant version)
        # Get the new state after the above action

        inputs_new = self.env.sense(self)
        state_new = (inputs_new['light'],
                     self.planner.next_waypoint(),
                     inputs_new['oncoming'],
                     inputs_new['left'],
                     inputs['right'])
        print "The new state is: {}".format(state_new)
        print "t:{}".format(t)
        print alpha,gamma





        # Calculate the Q_value
        q_value = (1 - alpha) * self.q_table[self.state][action] + \
                  alpha * (reward + gamma * max(self.q_table[state_new].values()))
        # Update the Q_table
        self.q_table[self.state][action] = q_value

        #Print the LearningAgent information
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        total_preserve=self.total_reward
        print "\n"
        print "\n"


        #update the episode number
        if old_time==0:
            self.episode+=1


        self.preserve.append([new_time,self.total_reward,deadline,self.episode])

        #calculate the failure
        if deadline==0:
            self.failure+=1
            print self.failure



        #Preserve all episodes information
        if self.episode==100:
            df=pd.DataFrame(self.preserve,columns=['Time','Reward','Deadline','Episode'])
          #  print self.preserve
          #  print df
            #df.to_csv('better.csv')
           # df.to_csv('random.csv')
            df.to_csv('constant.csv')
           # df.to_csv('gamma_con.csv')
          #  return df1

def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    # Now simulate it
    sim = Simulator(e, update_delay=1,display=True)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
