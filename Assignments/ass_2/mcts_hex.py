# Reincorment Learning 2020 - Assignment 2: Adaptive Sampling
# Auke Bruinsma (s1594443), Meng Yao (s2308266), Ella (s2384949).
# This file contains part 1 of the assignment: MCTS Hex - 3 points

# Imports.
import numpy as np
import copy
from hex_skeleton import HexBoard
from node import Node
import random as rd
import sys 

# Global variables
BOARD_SIZE = 2
SEARCH_DEPTH = 2
AI = HexBoard.BLUE
PLAYER = HexBoard.RED
EMPTY = HexBoard.EMPTY
inf = float('inf')
C_p = 1

def MCTS(rootstate,itermax):
	# Initialise rootnote.
	rootnode = Node(state = rootstate)
	node = rootnode

	# Loop until the max number of iterations is reached.
	for i in range(itermax):
		board = copy.deepcopy(rootstate)

		action = node.check_visits(node)

		while action != True:
			action = node.check_visits(node)
			if action == False:
				node.collapse(node,board,BOARD_SIZE)
			elif action == 2:
				node,move = node.UCTSelectChild(C_p,inf)
				board.virtual_place(move,AI)
				action = node.check_visits(node)
				if action == False:
					node.collapse(node,board,BOARD_SIZE)

		# Select
		while node.childNodes != []: # If there are children ...
			node,move = node.UCTSelectChild(C_p,inf)
			board.virtual_place(move,AI)

		print('val:', node.V)

		# Check if the node has been visited.
		if node.V == 0:
			result = board.move_check_win(board,BOARD_SIZE) # Playout.

		# Backpropagate:
		while node != None:
			node.update(result)
			if node.parent == None:
				break;
			node = node.parent

		node.tree_info(rootnode,C_p,inf)

	return node.UCTSelectChild(C_p,inf)[1]

if __name__ == '__main__':
	# Initialise a board.
	board = HexBoard(BOARD_SIZE)

	# Number of iterations.
	itermax = 6

	best_move = MCTS(board,itermax)
	print(best_move)