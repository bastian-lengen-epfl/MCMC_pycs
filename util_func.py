import sys

def proquest(askquestions):
	"""
	Asks the user if he wants to proceed. If not, exits python.
	askquestions is a switch, True or False, that allows to skip the
	questions.
	"""
	if askquestions:
		answer = raw_input("Tell me, do you want to go on ? (yes/no) ")
		if answer[:3] != "yes":
			sys.exit("Ok, bye.")
		print ""	# to skip one line after the question.