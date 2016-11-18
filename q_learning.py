__author__ = "Abhimanyu Banerjee"

from random import random, choice
import matplotlib.pyplot as plt
import numpy as np
import timeit
from sys import exit
from copy import deepcopy
import pdb

def empty_state():
	'''establish the empty wherein each cell is filled by zero'''
	return np.array([ [0] * 3 ] * 3)

def print_board(state):

	board_format = "-------------\n| {0} | {1} | {2} |\n|-------------|\n| {3} | {4} | {5} |\n|-------------|\n| {6} | {7} | {8} |\n-------------"
	cell_values = []
	symbols = ["O", " ", "X"]

	for i in xrange(len(state)):
		for j in xrange(len(state[i])):
			cell_values.append(symbols[int(state[i][j] + 1)])

	print board_format.format(*cell_values)

def open_spots(state):
	open_cells = []
	for i in xrange(len(state)):
		for j in xrange(len(state[i])):
			if state[i][j] == 0:
				open_cells.append(i*len(state) + j)
	return open_cells

def is_game_over(state):
	''' check if any of the columns or rows or diagonals when summed are 
	divisible by the width or height of the board - indication that the 
	game has been won. The sign of the sum tells us who has won'''

	for i in xrange(len(state)):

		state_trans = np.array(state).transpose() # transposed board state

		#check for winner row-wise
		if np.array(state[i]).sum() != 0 and np.array(state[i]).sum() % 3 == 0:
			return np.array(state[i]).sum() / 3

		#check for winner column-wise
		elif state_trans[i].sum() !=0 and state_trans[i].sum() % 3 == 0:
			return state_trans.sum() / 3

	#extract major diagonal from the state
	major_diag = np.multiply(np.array(state), np.identity(len(state))) 
	if major_diag.sum() != 0 and major_diag.sum() % 3 == 0:
		return major_diag.sum() / 3
	
	#extract minor diagonal from the state
	minor_diag = np.multiply(np.array(state), np.fliplr(major_diag))
	if minor_diag.sum() != 0 and minor_diag.sum() % 3 == 0:
		return minor_diag.sum() / 3

	return 0 #no clear winner

def get_state_key(state):

	''' convert the state into a unique key by concatenating 
	all the flattened values in the state'''

	flat_state = [cell for row in state for cell in row]
	key = "".join(map(str, flat_state))
	#print key
	return key

def generate_state_value_table(state, turn, player):

	winner = int(is_game_over(state)) #check if for the current turn and state, game has finished and if so who won
	print "\nWinner is ", winner
	print "\nBoard at turn: ", turn
	print_board(state)	
	player.add_state(state, winner/2 + 0.5) #add the current state with the appropriate value to the state table

	board_size = len(state)*len(state)

	open_cells = open_spots(state) 
	if len(open_cells) > 0:
		for cell in open_cells:
			#pdb.set_trace()
			row, col = cell / len(state), cell % len(state)
			num_cells_occupied = (board_size - len(open_cells))
			if state[row][col] == 0:
				new_state = deepcopy(state)
				if num_cells_occupied % 2 == 0:
					new_state[row][col] = 1
				else:
					new_state[row][col] = -1		
				try:
					if not player.check_duplicates(new_state):
						generate_state_value_table(new_state, turn+1, player)
					else:
						return
				except:
					print "Recursive depth exceeded"
					exit()

		else:
			return
		'''#either someone has won the game or it is a draw
		if winner != 0 or or col > len(state) or turn > 8:	
			return 

		#the game is still playable, so fill in a new cell
		i, j = turn / 3, turn % 3
		
		if turn % 2 == 0:
			state[row][col] = 1
		else:
			state[row][col] = -1

		print "\nBoard after adding symbol: ", symbol, " at turn: ", turn
		print_board(state)
		#pdb.set_trace()
		generate_state_value_table(state, turn+1, player) 
	'''
	
class Agent(object):
	
	def __init__(self, player):
		self.state_values = {}
		self.symbol = player
        #self.is_learning = is_learning
		self.behaviour_threshold = 0.1
		self.learning_rate = 0.9
		self.prev_state = None
		self.prev_score = 0
		self.num_states = 0

		print "\nInitializing state table for player ", self.symbol, ". Please wait ..."
		start_time = timeit.default_timer()
		generate_state_value_table(empty_state(), 0, self)
		print "Time taken to initialize state table: ", (timeit.default_timer() - start_time) 

	def check_duplicates(self, state):
		''' return true if state passed is already registered in the state table'''
		
		if get_state_key(state) in self.state_values:
			return True
		return False

	def add_state(self, state, value):
		print "\nAdded state", len(self.state_values) + 1

		if len(self.state_values) < self.num_states:
			exit("Duplicate state!")

		self.state_values[get_state_key(state)] = value
		self.num_states += 1

	def count_states(self):
		return len(self.state_values)

	def action(self, state):

		toss = random()

		# explore if toss is more than threshold, else choose next move greedy-ly
		if toss > behaviour_threshold:
			i, j = self.explore(state)
		else:
			i, j = self.greedy(state)

		#update the current state and the previous state
		self.state[i][j] = self.symbol


	def explore(self, state):
    	
		open_cells = open_spots(state)
		random_choice = choice(open_cells)
		return random_choice[0], random_choice[1]

	def greedy(self, state):
		pass


player1 = Agent(player=1)
print "\nNumber of possible states: ", player1.count_states()