# Reincorment Learning 2020 - Assignment 1: Heuristic Planning
# Auke Bruinsma (s1594443), Meng Yao
# This file contains part  of the assignment: Eval.

# Imports.
import numpy as np
from hex_skeleton import HexBoard
import random as rd
import search
import sys 

# Global variables
BOARD_SIZE = 6
SEARCH_DEPTH = 2
AI = HexBoard.BLUE
PLAYER = HexBoard.RED
EMPTY = HexBoard.EMPTY
INF = 11

'''
To test Dijkstra's shortest path algorithm, we are not going to play
the game yet, but first populate the board in a certain way and then
run Dijkstra's algorithm on that certain board state. If we have succeeded
in that, we can implement in playing the game.
'''

def populate_board(board):
	# Place player pieces.
	#board.place((3,1),PLAYER)
	#board.place((4,3),PLAYER)
	#board.place((5,3),PLAYER)
	#board.place((5,4),PLAYER)

	# Play AI pieces.
	board.place((3,2),AI)
	#board.place((4,2),AI)
	#board.place((5,2),AI)
	#board.place((3,3),AI)

	board.print()

def AI_shortest_path(board):
	# Create list of possible starting positions.
	starting_point = []
	for i in range(BOARD_SIZE):
		starting_point.append((0,i))

	# Make a distance graph.
	distance_graph = np.zeros((BOARD_SIZE,BOARD_SIZE))
	distance_graph.fill(INF)
	
	for i in range(len(starting_point)):
		current_point = starting_point[i]
		
		visited = []
		distance_graph[current_point[1],current_point[0]] = 0

		update_distances(board,current_point,distance_graph,visited)

	print(distance_graph)



def PLAYER_shortest_path(board):
	# Create list of possible starting positions.
	starting_point = []
	for i in range(BOARD_SIZE):
		starting_point.append((0,i))
	print(starting_point)


def update_distances(board,current_point,distance_graph,visited):
	cur_distance = distance_graph[current_point[1],current_point[0]]
	shortest = INF
	next_point = []

	neighbors = board.get_neighbors(current_point)

	#print(neighbors)
	#print(visited)

	for i in range(len(neighbors)):
		if neighbors[i] not in visited and board.is_color(neighbors[i],PLAYER) == False:
			next_distance = cur_distance + 1

			# If a point has the same color, distance won't change.
			if board.is_color(neighbors[i],AI) == True:
				distance_graph[neighbors[i][1],neighbors[i][0]] = cur_distance
				print('hoi')

			# Change the value of the neighbours to the current value + 1, if it is lower than the current value.
			if next_distance < distance_graph[neighbors[i][1],neighbors[i][0]]:
				distance_graph[neighbors[i][1],neighbors[i][0]] = next_distance
			if next_distance < shortest:
				next_point = neighbors[i]

	visited.append(current_point)

	#print(distance_graph)

	if len(next_point) != 0:
		update_distances(board,next_point,distance_graph,visited)



	


if __name__ == '__main__':
	# Initialise the board.
	board = HexBoard(BOARD_SIZE)

	# Populate board and print it.
	populate_board(board)

	AI_shortest_path(board)




