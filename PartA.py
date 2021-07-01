import sys
class textProcess:

	# run time complexity is o(n) where n is the number of characters in the textFile
	# opening file is constant time
	# the for loop runs for each char in the text and this results in the o(n) runtime
	# work done in the loop is constant time checks
	def tokenize(self, s):
		tokens = []
		prev = ""

		valid = set()
		for i in range(26):
			valid.add(chr(i + ord('a')))
			valid.add(chr(i + ord('A')))
		for i in ('1','2','3','4','5','6','7','8','9','0'):
			valid.add(i)

		for char in s:
			if char in valid:
				if prev not in valid:
					tokens.append("")
				tokens[-1] += char.lower()
			prev = char
		return tokens


	# run time complexity is o(n) where n is the number of tokens. so linear time algorithm scaling linearly with respect to the number of items in the list of tokens
	# addition to the dictionary is constant time and so totally the runtime is o(n)
	def computeWordFrequencies(self, tokens):
		tokenFreqMap = {}
		for token in tokens:
			if token in tokenFreqMap:
				tokenFreqMap[token] += 1
			else:
				tokenFreqMap[token] = 1
		return tokenFreqMap


	# run time complexity is o(nlogn) where n is the number of key,value pairs in the tokenFreqMap because the map is sorted in descending order
	def printFreq(self, tokenFreqMap):
		freqTokenMap = []									# freq is mapped with token so that we can sort based on the frequencies in descending order
		for k,v in tokenFreqMap.items():
			freqTokenMap.append((v,k))

		freqTokenMap = sorted(freqTokenMap, key = lambda x: x[0] , reverse = True)
		for freqToken in freqTokenMap:
			s = freqToken[1] + " -> " + str(freqToken[0])
			print(s)

					


		
if __name__ == "__main__":
	t = textProcess()
	tokens = t.tokenize(sys.argv[1])						# compute list of tokens by passing it the text file path

	tfmap = t.computeWordFrequencies(tokens)				# compute token frequencies 

	t.printFreq(tfmap)										# print the token and corresponding frequency








