# -*- coding: utf-8 -*-

class TaxiSEM(object):
	"""
	Causal model for the Taxi RL problem using a
	structural equation.
	+---------+
	|R: | : :G|
	| : : : : |
	| : : : : |
	| | : | : |
	|Y| : |B: |
	+---------+

	R = 0
	G = 1
	Y = 2
	B = 3
	IN_THE_CAB = 4
	"""
	def __init__(self, taxi_row, taxi_col, passenger_idx, destination_idx):
		initial_pos = self.decode_tuple((taxi_row, taxi_col))
		self.initial_self = [0 for i in range(25)]
		self.initial_self[initial_pos] = 1
		self.pos = [0 for i in range(25)]
		self.pick_up = 0
		self.drop_off = 0
		self.destination = [0 for i in range[5]]

		self.passenger = self.pick_up * (sum(self.destination[:-1])) + self.destination[-1]
		self.goal = self.drop_off * (sum(self.destination[:-1])) + self.destination[-1]
		self.pos[0] = self.pos[1] + self.pos[5] + self.initial_self.pos[0]
		self.pos[1] = self.pos[2] + self.pos[6] + self.pos[0] + self.initial_self.pos[1]
		self.pos[2] = self.pos[3] + self.pos[7] + self.pos[1] + self.initial_self.pos[2]
		self.pos[3] = self.pos[4] + self.pos[8] + self.pos[2] + self.initial_self.pos[3]
		self.pos[4] = self.pos[9] + self.pos[3] + self.initial_self.pos[4]
		self.pos[5] = self.pos[0] + self.pos[6] + self.pos[10] + self.initial_self.pos[5]
		self.pos[6] = self.pos[1] + self.pos[7] + self.pos[11] + self.pos[5] + self.initial_self.pos[6]
		self.pos[7] = self.pos[2] + self.pos[8] + self.pos[12] + self.pos[6] + self.initial_self.pos[7]
		self.pos[8] = self.pos[3] + self.pos[9] + self.pos[13] + self.pos[7] + self.initial_self.pos[8]
		self.pos[9] = self.pos[4] + self.pos[14] + self.pos[8] + self.initial_self.pos[9]
		self.pos[10] = self.pos[5] + self.pos[11] + self.pos[15] + self.initial_self.pos[10]
		self.pos[11] = self.pos[6] + self.pos[12] + self.pos[16] + self.pos[10] + self.initial_self.pos[11]
		self.pos[12] = self.pos[7] + self.pos[13] + self.pos[17] + self.pos[11] + self.initial_self.pos[12]
		self.pos[13] = self.pos[8] + self.pos[14] + self.pos[18] + self.pos[12] + self.initial_self.pos[13]
		self.pos[14] = self.pos[9] + self.pos[19] + self.pos[13] + self.initial_self.pos[14]
		self.pos[15] = self.pos[10] + self.pos[16] + self.pos[20] + self.initial_self.pos[15]
		self.pos[16] = self.pos[11] + self.pos[17] + self.pos[21] + self.pos[15] + self.initial_self.pos[16]
		self.pos[17] = self.pos[12] + self.pos[18] + self.pos[22] + self.pos[16] + self.initial_self.pos[17]
		self.pos[18] = self.pos[13] + self.pos[19] + self.pos[23] + self.pos[17] + self.initial_self.pos[18]
		self.pos[19] = self.pos[14] + self.pos[24] + self.pos[18] + self.initial_self.pos[19]
		self.pos[20] = self.pos[15] + self.pos[21] + self.initial_self.pos[20]
		self.pos[21] = self.pos[16] + self.pos[22] + self.pos[20] + self.initial_self.pos[21]
		self.pos[22] = self.pos[17] + self.pos[23] + self.pos[21] + self.initial_self.pos[22]
		self.pos[23] = self.pos[18] + self.pos[24] + self.pos[22] + self.initial_self.pos[23]
		self.pos[24] = self.pos[19] + self.pos[23] + self.initial_self.pos[24]
	def decode_tuple(coordinates_x_y):
		dict_pos = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (2, 0): 10, (2, 1): 11, (2, 2): 12, (2, 3): 13, (2, 4): 14, (3, 0): 15, (3, 1): 16, (3, 2): 17, (3, 3): 18, (3, 4): 19, (4, 0): 20, (4, 1): 21, (4, 2): 22, (4, 3): 23, (4, 4): 24}
		return dict_pos[coordinates_x_y]


