from .board import *

class tttAgent2D(object):
	
	def __init__(self, symbol, is_learning=True, behaviour_threshold=0.1):
		self.state_values = {}
		self.symbol = symbol
		self.is_learning = is_learning
		self.learning_rate = 0.9
		self.prev_state = None
		self.prev_score = 0
		self.num_actions = 9
		self.num_states = 0

		if self.is_learning:
			
			self.behaviour_threshold = behaviour_threshold
			print("\nInitializing state table for player ", self.symbol, ". Please wait ...")
			#start_time = timeit.default_timer()
			self.generate_state_value_table(empty_state(), 0)

		else:
			self.behavior = 1 #set behaviour threshold such that agent is always in exploit mode
			
			#just to be safe that the pickle exists before we load it 
			try:
				if self.symbol == 1:
					self.state_values = pickle.load(open("state_table_X.p", "rb"))
				else:
					self.state_values = pickle.load(open("state_table_O.p", "rb"))
			except (OSError, IOError):
				exit("The AI is not ready. Please train it first!")

	def generate_state_value_table(self, state, turn):

		winner = int(is_game_over(state)) #check if, for the current turn and state, game has finished and if so who won
		#print("\nWinner is ", winner)
		#print("\nBoard at turn: ", turn)
		#print_board(state)	
		self.add_state(state, winner/2 + 0.5) #add the current state with the appropriate value to the state table
		
		#check if the winner returned is one of the players and go back to the previous state if so
		if winner != 0:
			return

		open_cells = open_spots(state) #find the index (from 0 to total no. of cells) of all the empty cells in the board 

		#check if there are any empty cells in the board
		if len(open_cells) > 0:
			
			for cell in open_cells:
			
				#pdb.set_trace()
				row, col = int(cell / len(state)), cell % len(state)
				new_state = deepcopy(state) #make a copy of the current state 
				
				#check which player's turn it is 
				if turn % 2 == 0:
					new_state[row][col] = 1
				else:
					new_state[row][col] = -1		
				
				#using a try block because recursive depth may be exceeded
				try:
					#check if the new state has not been generated somewhere else in the search tree
					if not self.check_duplicates(new_state):
						self.generate_state_value_table(new_state, turn+1)
					
				except:
					#print("Recursive depth exceeded")
					exit("Recursive depth exceeded")

		else:
			return

	def check_duplicates(self, state):
		''' return true if state passed is already registered in the state table'''
		
		if get_state_key(state) in self.state_values:
			return True
		return False

	def add_state(self, state, value):
		#print("\nAdded state", len(self.state_values) + 1)

		if len(self.state_values) < self.num_states:
			exit("Duplicate state!")

		self.state_values[get_state_key(state)] = value
		self.num_states += 1

	def count_states(self):
		return len(self.state_values)

	def update_state_table(self, next_val):
		''' Back up the value to the previous state using ....'''
		if self.prev_state != None and self.is_learning:
			prev_state_key = get_state_key(self.prev_state)
			prev_score = self.state_values[prev_state_key]
			self.state_values[prev_state_key] += self.learning_rate * (next_val - prev_score)

	def action(self, state):

		toss = random()

		# explore if toss is more than threshold, else choose next move greedily
		if toss > self.behaviour_threshold:
			i, j = self.explore(state)
		else:
			i, j = self.greedy(state)

		#set the current state and update the previous state
		state[i][j] = self.symbol
		self.prev_state = state

		return i, j # return the chosen move

	def explore(self, state):
    	
		open_cells = open_spots(state)
		random_choice = choice(open_cells)
		return random_choice / len(state), random_choice % len(state)

	def greedy(self, state):
		max_value = -1*np.inf
		best_move = None
		potential_state = None
		open_cells = open_spots(state) #find the index (from 0 to total no. of cells) of all the empty cells in the board 

			
		for cell in open_cells:
			i, j = int(cell / len(state)), cell % len(state)
			potential_state = deepcopy(state)
			potential_state[i][j] = self.symbol
			
			#putting this in a try block because we may encounter states not generated before
			try:
				val = self.state_values[get_state_key(potential_state)]
				if val > max_value:
					max_value = val
					best_move = (i, j)
			except KeyError:
				#exit("State not seen before")
				print(get_state_key(potential_state))
				pdb.set_trace()

		self.update_state_table(max_value)
		return best_move[0], best_move[1]

	def self_play(self, opponent):
		''' play a game of ttt against a predefined opponent'''

		state = empty_state()
		num_cells = len(state)*len(state[0])
		for turn in range(num_cells):
			if turn % 2 == 0:
				i, j = self.action(deepcopy(state))
				symbol = self.symbol
			else:
				i, j = opponent.action(deepcopy(state))
				symbol = opponent.symbol
			state[i][j] = symbol
			winner = int(is_game_over(state))
			if winner != 0: #one of the players won
				return winner 
		
		return winner #nobody won


	def play_board(self, state, action):
		''' register the current move and then make a move against an unknown opponent '''
		
		row, col = int(action / len(state)), action % len(state) #get the row and column to mark from the chosen action
		state[row][col] = -1 * self.symbol # invert the agent's symbol to get the opponent's
		reward = -1*self.get_reward(deepcopy(state)) #invert the reward from the agent's perspective to get that of the opponent

		i, j = self.action(deepcopy(state)) #get the agent's move
		symbol = self.symbol
		state[i][j] = symbol #mark the board with the agent's move
		
		return state #, reward