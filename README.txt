INDEX:
You can build an index by running the index.py script directly
- The program will generate an index in the index.txt file
- The index is a text file where each line starts with a term and followed by a list of postings
- Each posting is composed of a docID and a tfidf score and is stored in the form of (docID, tfidf) separete by a single space
Addtionally, the program will also produce a hash table in the hashtable.p file and a term index in the termindex.p file
Both of them will be automatically place in the same directory as index.py
- The hash table is a dictionary where the keys are integers and the values are the corresponding urls
- The term index is an index for the index. It is a dictionary where the keys are the terms and the values 
  are the positions in bytes of where the lines of the terms start in index.txt
Note: The files need to the placed in the /DEV directory and it needs to be in the same directory of index.py

SEARCH:
You can perform search by running the search.py file directly
- When the program is started, it will prompt you for a query in the terminal
- After entering your query, the program will display the top K urls
- You can modify the value of K directly in search.py, the default value for K is 10
- To exit search, simply enter "exit search" when prompted for query
Note: Make sure both hashtable.p and termindex.p are in the same directory as search.py

Libaries: Please make sure the numpy and BeautifulSoup libraries are installed