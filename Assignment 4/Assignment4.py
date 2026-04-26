# !!!
# EXERCISE 1
# !!!

# Read training data (text file)
book_dir = ''
book_fname = book_dir + 'Assignment 4/goblet_book.txt'
fid = open(book_fname, "r")
book_data = fid.read()
fid.close()

unique_chars = list(set(book_data))

K = len(unique_chars)

# help taken from: https://www.geeksforgeeks.org/python/adding-items-to-a-dictionary-in-a-loop-in-python/

# Initalize empty dictionaries
char_to_ind = {}
ind_to_char = {}

# from 0 to K-1
for char, i in enumerate(unique_chars): 
    char_to_ind = {char: i}
    ind_to_char = {i: char}

print(f"Number of unique characters, K={K}")

