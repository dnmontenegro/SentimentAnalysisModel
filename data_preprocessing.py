import re
from collections import Counter
import json
import pandas

def clean(data):
    data['Clean_Content'] = data['Content'].map(lambda x: re.sub(r'[^\w\s]',' ',x)) # Convert non-word or non-whitespace characters to whitespace
    data['Clean_Content'] = data['Clean_Content'].apply(lambda x: x.lower()) # Convert to lower case
    return data

def length(data):
    data['seq_words'] = data['Clean_Content'].apply(lambda x: x.split()) 
    data['seq_len'] = data['seq_words'].apply(lambda x: len(x))
    print(data['seq_len'].describe())
    min_seq_len = 100
    max_seq_len = 600
    data = data[min_seq_len <= data['seq_len']] # Remove short sentences
    data = data[data['seq_len'] <= max_seq_len] # Remove long sentences
    return data

def tokenize(data, top_K = 500):
    words = data['seq_words'].tolist()
    tokens_list = []
    for i in words:
        tokens_list.extend(i)
    tokens_count = Counter(tokens_list) # Count frequency of words
    tokens_sorted = tokens_count.most_common(len(tokens_list)) # Sort frequency counts
    tokens_top = tokens_sorted[:top_K] # Choose the top K tokens
    tokens_dict = {w:i+2 for i, (w,c) in enumerate(tokens_top)}
    tokens_dict['<pad>'] = 0
    tokens_dict['<unk>'] = 1
    with open('tokens_dict.json', 'w') as outfile:
        json.dump(tokens_dict, outfile, indent=4)
    return tokens_dict

def encode(x, tokens_dict):
    tokens = [tokens_dict.get(w, 1) for w in x]
    return tokens

def pad_truncate(x, seq_len):
    if len(x) >= seq_len: # Truncating
        return x[:seq_len]
    else: # Add padding
    	return x+[0]*(seq_len-len(x))

def main():
	# Data preprocessing

	# Reading raw data from files
	training_data = pandas.read_csv('training_raw_data.csv', index_col=None, encoding='utf8')
	testing_data = pandas.read_csv('testing_raw_data.csv', index_col=None, encoding='utf8')

	#Cleaning data
	training_data = clean(training_data)
	testing_data = clean(testing_data) 

	# Removing long and short sentences
	training_data = length(training_data) 
	testing_data = length(testing_data)
		
	# Converting string labels to boolean
	training_data['Label'] = training_data['Label'].apply(lambda x: 0 if x=='neg' else 1)
	testing_data['Label'] = testing_data['Label'].apply(lambda x: 0 if x=='neg' else 1)
		
	# Tokenizing data
	top_K = 10000
	tokens_dict = tokenize(training_data, top_K)
	print("Number of tokens", len(tokens_dict))
		
	# Encoding data to index
	training_data['input_x'] = training_data['seq_words'].apply(lambda x: encode(x, tokens_dict))
	testing_data['input_x'] = testing_data['seq_words'].apply(lambda x: encode(x, tokens_dict))
	print(testing_data['input_x'][:10])

	# Padding and truncating data
	batch_seq_len = 150
	training_data['input_x'] = training_data['input_x'].apply(lambda x: pad_truncate(x, batch_seq_len))
	training_data.to_csv('training_data.csv', index=False)
	testing_data['input_x'] = testing_data['input_x'].apply(lambda x: pad_truncate(x, batch_seq_len))
	testing_data.to_csv('testing_data.csv', index=False)
	
if __name__ == '__main__':
	main()
    


