"""
This file contains useful constants, tables etc for use in the modules.
"""

# cumulative table
cumulative_table = [
    0.00000,    0.00399,    0.00798,	0.01197,	0.01595,	0.01994,	0.02392,	0.02790,	0.03188,	0.03586,
	0.03983,	0.04380,	0.04776,	0.05172,	0.05567,	0.05962,	0.06356,	0.06749,	0.07142,	0.07535,
	0.07926,	0.08317,	0.08706,	0.09095,	0.09483,	0.09871,	0.10257,	0.10642,	0.11026,	0.11409,
	0.11791,	0.12172,	0.12552,	0.12930,	0.13307,	0.13683,	0.14058,	0.14431,	0.14803,	0.15173,
	0.15542,	0.15910,	0.16276,	0.16640,	0.17003,	0.17364,	0.17724,	0.18082,	0.18439,	0.18793,
	0.19146,	0.19497,	0.19847,	0.20194,	0.20540,	0.20884,	0.21226,	0.21566,	0.21904,	0.22240,
	0.22575,	0.22907,	0.23237,	0.23565,	0.23891,	0.24215,	0.24537,	0.24857,	0.25175,	0.25490,
	0.25804,	0.26115,	0.26424,	0.26730,	0.27035,	0.27337,	0.27637,	0.27935,	0.28230,	0.28524,
	0.28814,	0.29103,	0.29389,	0.29673,	0.29955,	0.30234,	0.30511,	0.30785,	0.31057,	0.31327,
	0.31594,	0.31859,	0.32121,	0.32381,	0.32639,	0.32894,	0.33147,	0.33398,	0.33646,	0.33891,
	0.34134,	0.34375,	0.34614,	0.34849,	0.35083,	0.35314,	0.35543,	0.35769,	0.35993,	0.36214,
	0.36433,	0.36650,	0.36864,	0.37076,	0.37286,	0.37493,	0.37698,	0.37900,	0.38100,	0.38298,
	0.38493,	0.38686,	0.38877,	0.39065,	0.39251,	0.39435,    0.39617,	0.39796,	0.39973,	0.40147,
	0.40320,	0.40490,	0.40658,	0.40824,	0.40988,	0.41149,	0.41308,	0.41466,	0.41621,	0.41774,
	0.41924,	0.42073,	0.42220,	0.42364,	0.42507,	0.42647,	0.42785,	0.42922,	0.43056,	0.43189,
	0.43319,	0.43448,	0.43574,	0.43699,	0.43822,	0.43943,	0.44062,	0.44179,	0.44295,	0.44408,
	0.44520,	0.44630,	0.44738,	0.44845,	0.44950,	0.45053,	0.45154,0.45254,	0.45352,	0.45449,
	0.45543,	0.45637,	0.45728,	0.45818,	0.45907,	0.45994,	0.46080,	0.46164,	0.46246,	0.46327,
	0.46407,	0.46485,	0.46562,	0.46638,	0.46712,	0.46784,	0.46856,	0.46926,	0.46995,	0.47062,
	0.47128,	0.47193,	0.47257,	0.47320,	0.47381,	0.47441,	0.47500,	0.47558,	0.47615,	0.47670,
	0.47725,	0.47778,	0.47831,	0.47882,	0.47932,	0.47982,	0.48030,	0.48077,	0.48124,	0.48169,
	0.48214,	0.48257,	0.48300,	0.48341,	0.48382,	0.48422,	0.48461,	0.48500,	0.48537,	0.48574,
	0.48610,	0.48645,	0.48679,	0.48713,	0.48745,	0.48778,	0.48809,	0.48840,	0.48870,	0.48899,
	0.48928,	0.48956,	0.48983,	0.49010,	0.49036,	0.49061,	0.49086,	0.49111,	0.49134,	0.49158,
	0.49180,	0.49202,	0.49224,	0.49245,	0.49266,	0.49286,	0.49305,	0.49324,	0.49343,	0.49361,
	0.49379,	0.49396,	0.49413,	0.49430,	0.49446,	0.49461,	0.49477,	0.49492,	0.49506,	0.49520,
	0.49534,	0.49547,	0.49560,	0.49573,	0.49585,	0.49598,	0.49609,	0.49621,	0.49632,	0.49643,
	0.49653,	0.49664,	0.49674,	0.49683,	0.49693,	0.49702,	0.49711,	0.49720,	0.49728,	0.49736,
	0.49744,	0.49752,	0.49760,	0.49767,	0.49774,	0.49781,	0.49788,	0.49795,	0.49801,	0.49807,
	0.49813,	0.49819,	0.49825,	0.49831,	0.49836,	0.49841,	0.49846,	0.49851,	0.49856,	0.49861,
	0.49865,	0.49869,	0.49874,	0.49878,	0.49882,	0.49886,	0.49889,	0.49893,	0.49896,	0.49900,
	0.49903,	0.49906,	0.49910,	0.49913,	0.49916,	0.49918,	0.49921,	0.49924,	0.49926,	0.49929,
	0.49931,	0.49934,	0.49936,	0.49938,	0.49940,	0.49942,	0.49944,	0.49946,	0.49948,	0.49950,
	0.49952,	0.49953,	0.49955,	0.49957,	0.49958,	0.49960,	0.49961,	0.49962,	0.49964,	0.49965,
	0.49966,	0.49968,	0.49969,	0.49970,	0.49971,	0.49972,	0.49973,	0.49974,	0.49975,	0.49976,
	0.49977,	0.49978,	0.49978,	0.49979,	0.49980,	0.49981,	0.49981,	0.49982,	0.49983,	0.49983,
	0.49984,	0.49985,	0.49985,	0.49986,	0.49986,	0.49987,	0.49987,	0.49988,	0.49988,	0.49989,
	0.49989,	0.49990,	0.49990,	0.49990,	0.49991,	0.49991,	0.49992,	0.49992,	0.49992,	0.49992,
	0.49993,	0.49993,	0.49993,	0.49994,	0.49994,	0.49994,	0.49994,	0.49995,	0.49995,	0.49995,
	0.49995,	0.49995,	0.49996,	0.49996,	0.49996,	0.49996,	0.49996,	0.49996,	0.49997,    0.49997,
	0.49997,	0.49997,	0.49997,	0.49997,	0.49997,	0.49997,	0.49998,	0.49998,	0.49998,	0.49998
]

# Z-table
z_table = [0.5, 0.50399, 0.50798, 0.51197, 0.51595, 0.51994, 0.52392, 0.5279, 0.53188, 0.53586, 0.53983, 0.5438, 0.54776, 0.55172, 0.55567, 0.55966, 0.5636, 0.56749, 0.57142, 0.57535, 0.57926, 0.58317, 0.58706, 0.59095, 0.59483, 0.59871, 0.60257, 0.60642, 0.61026, 0.61409, 0.61791, 0.62172, 0.62552, 0.6293, 0.63307, 0.63683, 0.64058, 0.64431, 0.64803, 0.65173, 0.65542, 0.6591, 0.66276, 0.6664, 0.67003, 0.67364, 0.67724, 0.68082, 0.68439, 0.68793, 0.69146, 0.69497, 0.69847, 0.70194, 0.7054, 0.70884, 0.71226, 0.71566, 0.71904, 0.7224, 0.72575, 0.72907, 0.73237, 0.73565, 0.73891, 0.74215, 0.74537, 0.74857, 0.75175, 0.7549, 0.75804, 0.76115, 0.76424, 0.7673, 0.77035, 0.77337, 0.77637, 0.77935, 0.7823, 0.78524, 0.78814,
0.79103, 0.79389, 0.79673, 0.79955, 0.80234, 0.80511, 0.80785, 0.81057, 0.81327, 0.81594, 0.81859, 0.82121, 0.82381, 0.82639, 0.82894, 0.83147, 0.83398, 0.83646, 0.83891, 0.84134, 0.84375, 0.84614, 0.84849, 0.85083, 0.85314, 0.85543, 0.85769, 0.85993, 0.86214, 0.86433, 0.8665, 0.86864, 0.87076, 0.87286, 0.87493, 0.87698, 0.879, 0.881, 0.88298, 0.88493, 0.88686, 0.88877, 0.89065, 0.89251, 0.89435, 0.89617, 0.89796, 0.89973, 0.90147, 0.9032, 0.9049, 0.90658, 0.90824, 0.90988, 0.91149, 0.91308, 0.91466, 0.91621, 0.91774, 0.91924, 0.92073, 0.9222, 0.92364, 0.92507, 0.92647, 0.92785, 0.92922, 0.93056, 0.93189, 0.93319, 0.93448, 0.93574, 0.93699, 0.93822, 0.93943, 0.94062, 0.94179, 0.94295, 0.94408, 0.9452, 0.9463, 0.94738, 0.94845, 0.9495, 0.95053, 0.95154, 0.95254, 0.95352, 0.95449, 0.95543, 0.95637, 0.95728, 0.95818, 0.95907, 0.95994, 0.9608, 0.96164, 0.96246, 0.96327, 0.96407, 0.96485, 0.96562, 0.96638, 0.96712, 0.96784, 0.96856, 0.96926, 0.96995, 0.97062, 0.97128, 0.97193, 0.97257, 0.9732, 0.97381, 0.97441, 0.975, 0.97558, 0.97615, 0.9767, 0.97725, 0.97778, 0.97831, 0.97882, 0.97932, 0.97982, 0.9803, 0.98077, 0.98124, 0.98169,
0.98214, 0.98257, 0.983, 0.98341, 0.98382, 0.98422, 0.98461, 0.985, 0.98537, 0.98574, 0.9861, 0.98645, 0.98679, 0.98713, 0.98745, 0.98778, 0.98809, 0.9884, 0.9887, 0.98899, 0.98928, 0.98956, 0.98983, 0.9901, 0.99036, 0.99061, 0.99086, 0.99111, 0.99134, 0.99158, 0.9918, 0.99202, 0.99224, 0.99245, 0.99266, 0.99286, 0.99305, 0.99324, 0.99343, 0.99361, 0.99379, 0.99396, 0.99413, 0.9943, 0.99446, 0.99461, 0.99477, 0.99492, 0.99506, 0.9952, 0.99534, 0.99547, 0.9956, 0.99573, 0.99585, 0.99598, 0.99609, 0.99621, 0.99632, 0.99643, 0.99653, 0.99664, 0.99674, 0.99683, 0.99693, 0.99702, 0.99711, 0.9972, 0.99728, 0.99736, 0.99744, 0.99752, 0.9976, 0.99767, 0.99774, 0.99781, 0.99788, 0.99795, 0.99801, 0.99807, 0.99813,
0.99819, 0.99825, 0.99831, 0.99836, 0.99841, 0.99846, 0.99851, 0.99856, 0.99861, 0.99865, 0.99869, 0.99874, 0.99878, 0.99882, 0.99886, 0.99889, 0.99893, 0.99896, 0.999, 0.99903, 0.99906, 0.9991, 0.99913, 0.99916, 0.99918, 0.99921, 0.99924, 0.99926, 0.99929, 0.99931, 0.99934, 0.99936, 0.99938, 0.9994, 0.99942, 0.99944, 0.99946, 0.99948, 0.9995, 0.99952, 0.99953, 0.99955, 0.99957, 0.99958, 0.9996, 0.99961, 0.99962, 0.99964, 0.99965, 0.99966, 0.99968, 0.99969, 0.9997, 0.99971, 0.99972, 0.99973, 0.99974, 0.99975, 0.99976, 0.99977, 0.99978, 0.99978, 0.99979, 0.9998, 0.99981, 0.99981, 0.99982, 0.99983, 0.99983, 0.99984, 0.99985, 0.99985, 0.99986, 0.99986, 0.99987, 0.99987, 0.99988, 0.99988, 0.99989, 0.99989, 0.9999, 0.9999, 0.9999, 0.99991, 0.99991, 0.99992, 0.99992, 0.99992, 0.99992, 0.99993, 0.99993, 0.99993, 0.99994, 0.99994, 0.99994, 0.99994, 0.99995, 0.99995, 0.99995, 0.99995, 0.99995, 0.99996, 0.99996, 0.99996, 0.99996, 0.99996, 0.99996, 0.99997, 0.99997, 0.99997, 0.99997, 0.99997, 0.99997, 0.99997, 0.99997, 0.99998, 0.99998, 0.99998, 0.99998]

# print(len(z_table))
# print(z_table[250])

# t-table
from utilities import Table

names = ["dof", .25, .20, .15, .10, .05, .025, .02, .01, .005, .0025, .001, .0005]

rows = [
	[1, 1.000, 1.376, 1.963, 3.078, 6.314, 12.71, 15.89,31.82, 63.66, 127.3, 318.3, 636.6],
	[2, 0.816, 1.061, 1.386, 1.886, 2.920, 4.303, 4.849, 6.965, 9.925, 14.09, 22.33, 31.60],
	[3, 0.765, 0.978, 1.250, 1.638, 2.353, 3.182, 3.482, 4.541, 5.841, 7.453, 10.21, 12.92],
	[4, 0.741, 0.941, 1.190, 1.533, 2.132, 2.776, 2.999, 3.747, 4.604, 5.598, 7.173, 8.610],
	[5, 0.727, 0.920, 1.156, 1.476, 2.015, 2.571, 2.757, 3.365, 4.032, 4.773, 5.893, 6.869],
	[6, 0.718, 0.906, 1.134, 1.440, 1.943, 2.447, 2.612, 3.143, 3.707, 4.317, 5.208, 5.959],
	[7, 0.711, 0.896, 1.119, 1.415, 1.895, 2.365, 2.517, 2.998, 3.499, 4.029, 4.785, 5.408],
	[8, 0.706, 0.889, 1.108, 1.397, 1.860, 2.306, 2.449, 2.896, 3.355, 3.833, 4.501, 5.041],
	[9, 0.703, 0.883, 1.100, 1.383, 1.833, 2.262, 2.398, 2.821, 3.250, 3.690, 4.297, 4.781],
	[10, 0.700, 0.879, 1.093, 1.372, 1.812, 2.228, 2.359, 2.764, 3.169, 3.581, 4.144, 4.587],
	[11, 0.697, 0.876, 1.088, 1.363, 1.796, 2.201, 2.328, 2.718, 3.106, 3.497, 4.025, 4.437],
	[12, 0.695, 0.873, 1.083, 1.356, 1.782, 2.179, 2.303, 2.681, 3.055, 3.428, 3.930, 4.318],
	[13, 0.694, 0.870, 1.079, 1.350, 1.771, 2.160, 2.282, 2.650, 3.012, 3.372, 3.852, 4.221],
	[14, 0.692, 0.868, 1.076, 1.345, 1.761, 2.145, 2.264, 2.624, 2.977, 3.326, 3.787, 4.140],
	[15, 0.691, 0.866, 1.074, 1.341, 1.753, 2.131, 2.249, 2.602, 2.947, 3.286, 3.733, 4.073],
	[16, 0.690, 0.865, 1.071, 1.337, 1.746, 2.120, 2.235, 2.583, 2.921, 3.252, 3.686, 4.015],
	[17, 0.689, 0.863, 1.069, 1.333, 1.740, 2.110, 2.224, 2.567, 2.898, 3.222, 3.646, 3.965],
	[18, 0.688, 0.862, 1.067, 1.330, 1.734, 2.101, 2.214, 2.552, 2.878, 3.197, 3.611, 3.922],
	[19, 0.688, 0.861, 1.066, 1.328, 1.729, 2.093, 2.205, 2.539, 2.861, 3.174, 3.579, 3.883],
	[20, 0.687, 0.860, 1.064, 1.325, 1.725, 2.086, 2.197, 2.528, 2.845, 3.153, 3.552, 3.850],
	[21, 0.686, 0.859, 1.063, 1.323, 1.721, 2.080, 2.189, 2.518, 2.831, 3.135, 3.527, 3.819],
	[22, 0.686, 0.858, 1.061, 1.321, 1.717, 2.074, 2.183, 2.508, 2.819, 3.119, 3.505, 3.792],
	[23, 0.685, 0.858, 1.060, 1.319, 1.714, 2.069, 2.177, 2.500, 2.807, 3.104, 3.485, 3.768],
	[24, 0.685, 0.857, 1.059, 1.318, 1.711, 2.064, 2.172, 2.492, 2.797, 3.091, 3.467, 3.745],
	[25, 0.684, 0.856, 1.058, 1.316, 1.708, 2.060, 2.167, 2.485, 2.787, 3.078, 3.450, 3.725],
	[26, 0.684, 0.856, 1.058, 1.315, 1.706, 2.056, 2.162, 2.479, 2.779, 3.067, 3.435, 3.707],
	[27, 0.684, 0.855, 1.057, 1.314, 1.703, 2.052, 2.158, 2.473, 2.771, 3.057, 3.421, 3.690],
	[28, 0.683, 0.855, 1.056, 1.313, 1.701, 2.048, 2.154, 2.467, 2.763, 3.047, 3.408, 3.674],
	[29, 0.683, 0.854, 1.055, 1.311, 1.699, 2.045, 2.150, 2.462, 2.756, 3.038, 3.396, 3.659],
	[30, 0.683, 0.854, 1.055, 1.310, 1.697, 2.042, 2.147, 2.457, 2.750, 3.030, 3.385, 3.646],
	[40, 0.681, 0.851, 1.050, 1.303, 1.684, 2.021, 2.123, 2.423, 2.704, 2.971, 3.307, 3.551],
	[50, 0.679, 0.849, 1.047, 1.299, 1.676, 2.009, 2.109, 2.403, 2.678, 2.937, 3.261, 3.496],
	[60, 0.679, 0.848, 1.045, 1.296, 1.671, 2.000, 2.099, 2.390, 2.660, 2.915, 3.232, 3.460],
	[80, 0.678, 0.846, 1.043, 1.292, 1.664, 1.990, 2.088, 2.374, 2.639, 2.887, 3.195, 3.416],
	[100, 0.677, 0.845, 1.042, 1.290, 1.660, 1.984, 2.081, 2.364, 2.626, 2.871, 3.174, 3.390],
	[1000, 0.675, 0.842, 1.037, 1.282, 1.646, 1.962, 2.056, 2.330, 2.581, 2.813, 3.098, 3.300],
	["inf", 0.674, 0.841, 1.036, 1.282, 1.645, 1.960, 2.054, 2.326, 2.576, 2.807, 3.091, 3.291]
]

obj = Table(names, rows)
t_table = obj.make_table()
dof = obj.select(t_table, "dof", 12)
# obj.show_table(dof)  # uncomment to see the table
# t = obj.project()



