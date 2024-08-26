__author__ = "<your name>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<your e-mail>"

import numpy as np
import pickle
import os

agentName = "Junyi_rl_agent"

# Example of a training specification - in this case it's two sessions,
# one 100 games against two opponents, value_agent and valueplus_agent,
# the other 50 games against random_agent and value_agent. 
training = [ ("value_agent.py", "valueplus_agent.py", 100),
             ("random_agent.py", "value_agent.py", 50),
           ]

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
        pass
   
   def train_end(self):
       """ Invoked once by the engine at the start of the training.

           You may use it to finalise training
           You may remove this method if you don't need it.
       """

       pass

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
       pass
      
   def train_game_end(self, banks):
        """ Invoked by the engine at the end of each game training,
            passing in the banks of all players

            Args: banks - a list of integers, the banks of all players at the end of the game
            
            You may remove this method if you don't need it.
        """
        pass

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

      # Extract different parts of percepts.
      bidding_on = percepts[0]
      items_left = percepts[1]
      my_cards = percepts[2]
      bank = percepts[3]
      opponents_cards = percepts[4:]

      # Currently this agent just bids the first card in its hand - you need to make it smarter!
      action = my_cards[0]

      # Return the bid
      return action
