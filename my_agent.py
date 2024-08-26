__author__ = "Junyi Shen"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "sheju347@student.otago.ac.nz"

import numpy as np

agentName = "Limited-depth-minimax"

# Node in the search tree
class Node:
    def __init__(self, state):
        self.state = state
        self.value = None
        self.best_child = None

# State of a node
class State:
    def __init__(self, bidding_on, items_left, my_cards, my_bank, opponents_bank, opponents_cards):
        self.bidding_on = bidding_on    # The value of the item to bid on. e.g. 4
        self.items_left = items_left    # The tuple of the items left. e.g. (-1, 1, 2, 3)
        self.my_cards = my_cards        # The tuple of my cards. e.g. (1, 2, 3, 4)
        self.my_bank = my_bank          # The total score of my bank. e.g. 2
        self.opponents_bank = opponents_bank    # The total score of the opponent's bank. e.g. 1 (Only consider there's only one opponent)
        self.opponents_cards = opponents_cards  # The tuple of the opponent's cards. e.g. (3, 4, 5, 6)

        # The value of my action in this round. Should be None on the start of a round
        # Assigned after my action, and assigned None again after the opponent's move and the new round starts
        self.my_action = None           

    def __str__(self):
        return f"State: \nitems_left:{self.items_left}, bidding_on:{self.bidding_on}, \n\
            my_cards:{self.my_cards}, my_action:{self.my_action}, opponents_cards:{self.opponents_cards}, \n\
            my_bank:{self.my_bank}, opponents_bank:{self.opponents_bank}, \n\
            is_terminal:{self.is_terminal_state()}, evaluation:{self.get_evaluation_value()}\n"
   
    # Evaluation function: value is calculated according to the bank and cards of both sides
    def get_evaluation_value(self):
        # Compare cards one by one by positions after sorting
        # For each position, if my card is greater than the opponent's, score += 1, if less than the opponent's, score -= 1
        my_cards_sorted = list(self.my_cards[:])
        if self.my_action != None:
            my_cards_sorted.append(self.my_action)
        my_cards_sorted.sort()
        opponents_cards_sorted = sorted(self.opponents_cards)
        card_value_compare_score = 0
        for i in range(len(my_cards_sorted)):
            if my_cards_sorted[i] > opponents_cards_sorted[i]:
                card_value_compare_score += 1
            elif my_cards_sorted[i] < opponents_cards_sorted[i]:
                card_value_compare_score -= 1
        
        # Multiply compare score with the average value of the items left
        items_left_abs = [abs(card) for card in self.items_left]
        items_left_abs.append(self.bidding_on)
        items_left_abs_average = sum(items_left_abs) / len(items_left_abs)
        potential_value = items_left_abs_average * card_value_compare_score

        # Calculate bank difference score
        bank_diff = self.my_bank - self.opponents_bank

        # Score is calculated by 
        total_score = potential_value + bank_diff * 1.2
        return total_score
    
    # Check if it is the terminal state
    def is_terminal_state(self):
        return len(self.items_left) <= 0
    
    # Get all possible next states on my round
    def get_my_possible_next_states(self):
        next_states = []
        for i in range(len(self.my_cards)):
            # Get a new tuple of my cards after using a card
            my_action = self.my_cards[i]
            next_my_cards = tuple([card for card in self.my_cards if card != my_action])

            # Generate a new state. In this state, I finished my bid and wait for the opponent to bid
            # (Assume the opponent bid after me and he know how much I bid, since it's the worse case)
            next_state = State(self.bidding_on, self.items_left[:], next_my_cards, self.my_bank, self.opponents_bank, self.opponents_cards)
            next_state.my_action = my_action
            next_states.append(next_state)

        return next_states
    
    # Get all possible next states on the opponent's round(Assume the opponent's action is after mine and he know how much I bid. It's the worst case)
    def get_opponents_possible_next_states(self):
        next_states = []
        for i in range(len(self.opponents_cards)):
            opponent_action = self.opponents_cards[i]
            next_my_bank = self.my_bank
            next_opponents_bank = self.opponents_bank
            next_bidding_on = self.items_left[0] # Note: In real cases, next item should be random. Here I simply assume bidding the first item in the list
            next_items_left = self.items_left[1:]
            next_opponents_cards = tuple([card for card in self.opponents_cards if card != opponent_action]) # Get a new tuple of opponent's cards

            # Modify my_bank or opponents_bank by checking who win this bidding. If tie, add to the next bid.
            if self.my_action > opponent_action: # I win
                if self.bidding_on >= 0:
                    next_my_bank += self.bidding_on
                else:
                    next_opponents_bank += self.bidding_on
            elif self.my_action < opponent_action: # I lose
                if self.bidding_on >= 0:
                    next_opponents_bank += self.bidding_on
                else:
                    next_my_bank += self.bidding_on
            else: # Draw
                next_bidding_on += self.bidding_on
            
            # Generate a new state. In this state, both sides finish bidding and a new round starts
            next_state = State(next_bidding_on, next_items_left, self.my_cards[:], next_my_bank, next_opponents_bank, next_opponents_cards)
            next_states.append(next_state)

        return next_states


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

    # Depth limit. 2 depth limits = 1 round. Don't enter odd number.
    DEPTH_LIMIT = 4

    def __init__(self, item_values, card_values):
        """
        :param item_values: list of ints, values of items to bid on
        :card_values: list of ints, cards agent bids with
        """

        self.card_values = card_values
        self.item_values = item_values

    # Recursively get the best child of a node
    def get_best_child(self, node, depth, alpha, beta):
        state = node.state
        next_states = []
        is_max = depth % 2 == 0

        # If is max, then it's my turn. Otherwise, it's the opponent's turn.
        if is_max:
            next_states = state.get_my_possible_next_states()
        else:
            next_states = state.get_opponents_possible_next_states()

        next_depth = depth + 1
        for next_state in next_states:
            child_node = Node(next_state)
            # If already reaches the depth limit, or this child is already a terminal state, calculate the value of the child with evaluation function
            if next_depth == self.DEPTH_LIMIT or next_state.is_terminal_state():
                child_node.value = next_state.get_evaluation_value()
            else: # If not reaches the depth limit or terminal state, keep going deep
                self.get_best_child(child_node, next_depth, alpha, beta)

            # Back up the value to parent if this child is better than the previous ones
            if node.value == None or (is_max and child_node.value > node.value) or ((not is_max) and child_node.value < node.value):
                node.value = child_node.value
                node.best_child = child_node

                # Assign the new value of the parent to the alpha/beta value
                if is_max:
                    alpha = max(alpha, node.value)
                else:
                    beta = min(beta, node.value)

                # Alpha-beta pruning
                # alpha >= beta means:
                # In MAX's level, the best value MAX can guarentee is greater than or equal to the best value for MIN in this branch, so MAX will not choose this branch
                # In MIN's level, the best value MIN can guarentee is smaller than or equal to the best value for MAX in this branch, so MIN will not choose this branch
                # Thus, the subsequent siblings don't need to be calculated because MAX or MIN has a better choice
                if alpha >= beta:
                    break

        # Finally, return the best child of the root node
        return node.best_child


    def AgentFunction(self, percepts):
        """Returns the bid value of the next bid

            :param percepts: a tuple of four items: bidding_on, items_left, my_cards, opponents_cards

                        , where

                        bidding_on - is an integer value of the item to bid on;

                        items_left - the items still to bid on after this bid (the length of the list is the number of
                                    bids left in the game)

                        my_cards - the list of cards in the agent's hand

                        bank - total value of items in this game
                        
                        opponents_cards - a list of lists of cards in the opponents' hands, so in two player game, this is
                                        a list of one list of cards, in three player game, this is a list of two lists, etc.


            :return: value - card value to bid with, must be a number from my_cards
        """

        # Extract different parts of percepts.
        bidding_on = percepts[0] # e.g. 4
        items_left = percepts[1] # e.g. (-1, 1, 2, 3)
        my_cards = percepts[2] # e.g. (1, 2, 3, 4)
        bank = percepts[3] # e.g. 2
        opponents_cards = percepts[4:] # e.g. ((3, 4, 5, 6),)

        # Note 1: Only consider that there's only one opponent
        # Note 2: Percepts don't provide the opponent's bank, but I can simply assume it's 0.
        # I calculate "my_bank - opponents_bank" in the evaluation function, so it doesn't matter what opponent_bank originally is.
        opponent_bank = 0
        root_state = State(bidding_on, items_left, my_cards, bank, opponent_bank, opponents_cards[0])
        root_node = Node(root_state)

        best_child = self.get_best_child(root_node, 0, -float("inf"), float("inf"))
        best_next_state = best_child.state
        action = best_next_state.my_action

        # Return the bid
        return action
