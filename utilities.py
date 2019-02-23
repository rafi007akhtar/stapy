class Table:
	def __init__(self, names, rows):
		self.names = names
		self.rows = rows
	
	def make_table(self):
		table = []
		table.append(self.names)
		for row in self.rows:
			table.append(row)
		return table
	
	def select(self, table, attr, value):
		# return all the tuples with attr=value true
		
		# first, find the index of the given attr in the header
		names = table[0]
		ind = 0
		for name in names:
			if name is attr: break
			ind += 1
		
		results = [] # tuples containing the results
		results.append(table[0]) # first row as header
		for row in table[1:]:
			if row[ind] == value:
				results.append(row)

		# now return the tuples
		return results		

	def project(self, table, attr):
		# return all the elements of a column named attr

		# first, find the index of attr in table
		ind = 0
		for attrs in table[0]:
			if attrs is attr: break
			ind += 1
		
		# now, append all elements in index ind in the given tuples
		temp = []
		for row in table[1:]:
			temp.append(row[ind])
		
		# remove duplicates, as the relation is a set
		temp = list(set(temp))
		results = []
		results.append([table[0][ind]])
		results.append(temp)

		# finally, return the result column
		return results
	
	def show_table(self, table):
		head = table[0]
		dashes = 0
		for h in head:
			print(h, end="\t")
			dashes += len(str(h))
		print()
		print("-" * 2*dashes)

		for row in table[1:]:
			for elem in row:
				print(elem, end="\t")
			print()


# test
# uncomment the following lines to test
# names = ["roll", "name"]
# rows = [[37, "Rafi"], [100, "NA"], [200, "NADA"], [40, "Rafi"], [20, "ifar"], [30, "Rafi"]]
# students = Table(names, rows)
# table = students.make_table()
# # for student in table:
# # 	print(student)
# # students.show_table(table)
# rafis = students.select(table, "name", "Rafi")
# # students.show_table(rafis)
# rafis = students.project(rafis, "name")
# students.show_table(rafis)
# # print(table)
# # print(rafis)
