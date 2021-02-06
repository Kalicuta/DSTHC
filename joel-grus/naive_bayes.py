import re

def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+", message)
	return set(all_words)

def count_words(training_set):
	"""training set consists of pairs (message, is_spam)"""
	counts = defaultdict(lambda: [0,0])
	for message, is_spam in training_set:
		for word in tokenize(message):
			counts[word][0 if is_spam else 1] += 1
	return counts




if __name__ == "__main__":
	msg = "Hello, This is Kalicuta. Hello World"
	print msg
	print tokenize(msg)
 
