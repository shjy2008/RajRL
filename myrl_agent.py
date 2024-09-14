__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np
import pickle
import os
import time

agentName = "Junyi_rl_agent"

# Example of a training specification - in this case it's two sessions,
# one 100 games against two opponents, value_agent and valueplus_agent,
# the other 50 games against random_agent and value_agent. 
training = [ ("value_agent.py", "valueplus_agent.py", 50000),
               ("random_agent.py", "random_agent.py", 50000),
            #("random_agent.py", "value_agent.py", 2500),
            #("random_agent.py", "valueplus_agent.py", 2500),
            #("my_agent.py", "value_agent.py", 2500),
            #("my_agent.py", "valueplus_agent.py", 2500),
           ]
# training = [ ("valueplus_agent.py", 100000)
#            ]

# Name of the file to save the agent to.  If you want to retrain your agent
# delete that file.
save_filename="saved_myrl_agent.pkl"

class RajAgent():
   """
             A class that encapsulates the code dictating the
             behaviour of the agent playing the game of Raj.

             ...

             Attributes
             ----------
             item_values : list of ints
                 values of items to bid on
             card_values: list of ints
                 cards agent bids with

             Methods
             -------
             AgentFunction(percepts)
                 Returns the card value from hand to bid with
             """

   def __init__(self, item_values, card_values):
      """
      :param item_values: list of ints, values of items to bid on
      :card_values: list of ints, cards agent bids with
      """

      self.card_values = card_values
      self.item_values = item_values

      self.T = 0.6 # T > 0. (How explorative) The bigger T is, the more explorative. The smaller T is, the more exploitative
      self.gamma = 0.8 # 0 < gamma <= 1. (Future discount) The percentage of reward of the next state passes to the current state
      self.alpha = 0.5 # alpha > 0. (Learning speed) The percentage of Q-value of the next state passes to the current state

      self.Q = dict() # Q-table

      # Previous percepts(state) and action(action_index), in the next round, these will be used to set the Q-value of the previous Q(s, a)
      self.previous_percepts = None
      self.previous_state = None
      self.previous_action_index = None
      self.banks = None
      self.initial_cards = None

      self.is_training = False

      self.training_start_time = None

      self.action_has_Q_value_count = 0
      self.action_count = 0

   """ Load and save function for the agent.
      
       Currently, these load the object properties from a file 
       (all things that are stored in 'self').  You may modify
       them if you need to store more things.

       The load function is called by the engine if the save_file
       is found, the save function is called by the engine after
       the training (which is carried out only if the save file
       hasn't been found)
   """
   def load(self, filename):
        print(f'Loading trained {agentName} agent from {filename}...')
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.update(tmp_dict) 

   def save(self, filename):
        print(f'Saving trained {agentName} agent to {filename}...')
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()

   def train_start(self):
        """ Invoked once by the engine at the start of the training.

            You may use it to initialise training variables
            You may remove this method if you don't need it.
        """
        self.is_training = True
        self.training_start_time = time.time()
   
   def train_end(self):
       """ Invoked once by the engine at the start of the training.

           You may use it to finalise training
           You may remove this method if you don't need it.
       """
       self.is_training = False
       training_time = time.time() - self.training_start_time
       print(f"Training finished. Training time: {training_time}s")

   def train_session_start(self):
       """ Invoked by the engine at the start of the training session
            with new opponents (once per tuple in your training variable)

           You may use it to initialise training session against new opponents.
           You may remove this method if you don't need it.
       """
       pass

   def train_session_end(self):
       """ Invoked by the engine at the end of the training session
            with new opponents (once per tuple in your training variable)

           You may use it to finalise training session against 
           You may remove this method if you don't need it.
       """

       pass
   
   def train_game_start(self):
       """ Invoked by the engine at the start of each game in training

           You may use it to initialise game-specific training variables 
           You may remove this method if you don't need it.
       """
       self.previous_percepts = None
       self.previous_state = None
       self.previous_action_index = None
       self.banks = None
       
      
   def train_game_end(self, banks):
        """ Invoked by the engine at the end of each game training,
            passing in the banks of all players

            Args: banks - a list of integers, the banks of all players at the end of the game
            
            You may remove this method if you don't need it.
        """

        state = self.previous_state
        for i in range(len(self.Q[state])):
            self.Q[state][i] = self.get_state_reward(state, banks) #banks[0] * 2 - banks[1] - banks[2] #self.get_state_reward(state)

        self.previous_percepts = None
        self.previous_state = None
        self.previous_action_index = None
        self.banks = None

        self.action_has_Q_value_count = 0
        self.action_count = 0

   def AgentFunction(self, percepts):
      """Returns the bid value of the next bid

            :param percepts: a tuple of four items: bidding_on, items_left, my_cards, opponents_cards

                     , where

                     bidding_on - is an integer value of the item to bid on;

                     items_left - the items still to bid on after this bid (the length of the list is the number of
                                  bids left in the game)

                     my_cards - the list of cards in the agent's hand

                     bank - total value of items won by this agent in this game
                     
                     opponents_cards - a list of lists of cards in the opponents' hands, so in two player game, this is
                                      a list of one list of cards, in three player game, this is a list of two lists, etc.


            :return: value - card value to bid with, must be a number from my_cards
      """

      # Extract different parts of percepts.
      bidding_on = percepts[0]
      items_left = percepts[1]
      my_cards = percepts[2]
      bank = percepts[3]
      opponents_cards = percepts[4:]

      self.update_banks(my_cards, opponents_cards)

      state = (bidding_on, items_left, my_cards)

      best_action_index = 0
      
      if self.is_training:
        # If no records in Q-table for this state, create an empty record
        # Q-table key: state
        # Q-table value: actions(index of cards in hand to choose)
        if state not in self.Q:
            self.Q[state] = np.zeros(len(my_cards))

        if self.previous_state == None:
            self.initial_cards = my_cards
        
        # Set the previous Q(s, a)
        if self.previous_state != None:
            previous_state_reward = self.get_state_reward(self.previous_state, self.banks)
            Q_star = previous_state_reward + self.gamma * max(self.Q[state])
            previous_Q = self.Q[self.previous_state][self.previous_action_index]
            self.Q[self.previous_state][self.previous_action_index] = previous_Q + self.alpha * (Q_star - previous_Q)

        is_terminal_state = (len(items_left) == 0)
        if not is_terminal_state:
            # Compute pi(softmax function), and choose an action to take based on probability
            p = self.Q[state]
            p = np.exp(p / self.T)
            p = p / np.sum(p)

            p_addup = np.copy(p)
            for i in range(1, len(p)):
                for j in range(0, i):
                    p_addup[i] += p[j]
            
            rand_0_to_1 = np.random.random()
            best_action_index = 0
            for i in range(len(p_addup)):
                if p_addup[i] > rand_0_to_1:
                    best_action_index = i
                    break
      else: # Not training
          if (state in self.Q):
              best_action_index = np.argmax(self.Q[state])
              self.action_has_Q_value_count += 1
          else:
              best_action_index = np.random.randint(0, len(my_cards))
          self.action_count += 1

      self.previous_percepts = percepts
      self.previous_state = state
      self.previous_action_index = best_action_index
      
    
    #   if self.action_count > 0 and self.action_count % 1000 == 0:
    #     print(f"Action has Q value rate: {self.action_has_Q_value_count} / {self.action_count} = {self.action_has_Q_value_count / self.action_count}")

      action = my_cards[best_action_index]

      # Return the bid
      return action
   
   def get_state_reward(self, state, banks):
      items_left = state[1]
      is_terminal = (len(items_left) == 0)

      bank_score = banks[0] * (len(banks) - 1)
      for opponent_bank in banks[1:]:
          bank_score -= opponent_bank

      if not is_terminal:
        my_cards = state[2]
        avg_init_hand_card_value = sum(self.initial_cards) / len(self.initial_cards)
        avg_hand_card_value = sum(my_cards) / len(my_cards)
        card_score = (avg_hand_card_value - avg_init_hand_card_value) * len(my_cards)
        score = (card_score + bank_score) / 10
      else:
        score = bank_score

      return score

   # Get the previous bid by comparing the cards in hand
   def get_previous_bid(self, previous_cards, curr_cards):
       return next(iter(set(previous_cards) - set(curr_cards)))
   
   # Calculate self.banks, the first one is my bank, and the others are opponents' banks
   def update_banks(self, my_cards, opponents_cards):
      if self.previous_percepts != None and len(my_cards) >= len(self.previous_percepts[2]): # It's new round
          self.banks = None
      if self.banks == None:
         self.banks = [0] * (len(opponents_cards) + 1) #np.zeros((len(opponents_cards)) + 1)
      else:
         my_previous_bid = self.get_previous_bid(self.previous_percepts[2], my_cards)

         bids = np.array([my_previous_bid])
         for i in range(len(opponents_cards)):
             opponent_previous_bid = self.get_previous_bid(self.previous_percepts[4:][i], opponents_cards[i])
             bids = np.append(bids, opponent_previous_bid)
         item_value = self.previous_percepts[0]

         # Only unique bids count
         unique_bids ,counts = np.unique(bids,return_counts=True)
         I = np.where(counts==1)[0]

         if len(I) == 0:
             pass
         else:
            Is = unique_bids[I]
            if item_value >= 0:
               winning_bid = Is[-1]
            else:
               winning_bid = Is[0]

            p = np.where(bids==winning_bid)[0][0]

            self.banks[p] += item_value
