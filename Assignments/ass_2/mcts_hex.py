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
import time 

# Global variables
BOARD_SIZE = 3
#SEARCH_DEPTH = 2
AI = HexBoard.BLUE
PLAYER = HexBoard.RED
EMPTY = HexBoard.EMPTY
inf = float('inf')
C_p = 1

# Digit to letter conversion.
def d2l_conversion(x_coor):
	letter_arr = np.array(['a','b','c','d','e','f','g','h','i','j']) # Max a playfield of 10 by 10.
	return letter_arr[x_coor]

# Letter to digit conversion.
def l2d_conversion(letter):
	letter_arr = np.array(['a','b','c','d','e','f','g','h','i','j'])
	for i in range(len(letter_arr)):
		if letter == letter_arr[i]:
			return i

def MCTS(rootstate,itermax,timemax=600):
	# Initialise rootnote.
	rootnode = Node(state = rootstate)
	node = rootnode
	start_time = time.time()
	# Loop until the max number of iterations is reached.
	for i in range(itermax):
		board = copy.deepcopy(rootstate)

		action = node.check_visits(node)

		counter = 0 # Solves a bug.

		while action != True:
			counter += 1
			action = node.check_visits(node)
			if action == False:
				node.collapse(node,board,BOARD_SIZE)
			elif action == 2:
				node,move = node.UCTSelectChild(C_p,inf)
				board.virtual_place(move,AI)
				action = node.check_visits(node)
				if action == False:
					node.collapse(node,board,BOARD_SIZE)
			if counter == 10:
				action = True

		# Select
		while node.childNodes != []: # If there are children ...
			node,move = node.UCTSelectChild(C_p,inf)
			board.virtual_place(move,AI)

		# Check if the node has been visited.
		if node.V == 0:
			result = board.move_check_win(board,BOARD_SIZE) # Playout.
			# Or we can use possible moves to accomplish playout
			"""
			moves = board.searchmoves()
			empty_num = len(moves)
			for _ in range(empty_num):
				board.place(moves.pop(rd.randint(0,len(moves)-1)),color)
				color = board.get_opposite_color(color)
			"""

		# Backpropagate:
		while node != None:
			node.update(result)
			if node.parent == None:
				break;
			node = node.parent

		#node.tree_info(rootnode,C_p,inf)
		if time.time() - start_time > timemax:
			break
	return node.UCTSelectChild(C_p,inf)[1]

# Player makes a move.
def player_make_move(board):
	print('Next move.')
	x = l2d_conversion(input(' x: '))
	y = int(input(' y: '))
	if (x >= 0 and x < BOARD_SIZE and y >= 0 and y < BOARD_SIZE) == 0:
		print("Invalid move: Outside the board. Please play again.")
		player_make_move(board)
	else:
		if(board.is_empty((x,y))):
			# Only place the move when the position is inside the board and empty.
			board.place((x,y),PLAYER)
		else:
			print("Invalid move: Position has been occupied. Please play again.")
			player_make_move(board)
	
if __name__ == '__main__':
	# Initialise a board.
	board = HexBoard(BOARD_SIZE)
	board.print()

	itermax = 100

	while not board.game_over:
		board.place((MCTS(board,itermax)),AI)
		board.print()
		if board.check_win(AI):
			print('AI has won.')
			break
		player_make_move(board)
		board.print()
		if board.check_win(PLAYER):
			print('PLAYER has won.')


	