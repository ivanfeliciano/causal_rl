POSITIONS = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]

DICT_POS = {}
for i in range(5):
	for j in range(5):
		vecinos = []
		if (i - 1 >= 0):
			vecinos.append(POSITIONS[i-1][j])
		if (j + 1 < 5):
			vecinos.append(POSITIONS[i][j+1])
		if (i + 1 < 5):
			vecinos.append(POSITIONS[i+1][j])
		if (j - 1 >= 0):
			vecinos.append(POSITIONS[i][j-1])
		print("pos[{0}] = {1} + initial_pos[{0}]".format(POSITIONS[i][j], " + ".join(["pos[" + str(s) + "]" for s in vecinos])))
		DICT_POS[(i, j)] = "pos_" + str(POSITIONS[i][j])
