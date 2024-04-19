"""Goal is to implement Levenshtein's distance algorithm, then make a GUI"""

def lev_naive(s1,s2):# O(max(n,m)^2*3^(n+m)) given n=len(s1),m=len(s2); auxillary complexity of O(n+m)
    if not len(s1): return len(s2)
    if not len(s2): return len(s1)
    if s1[0]==s2[0]: return lev_naive(s1[1:],s2[1:])
    
    return 1 + min(lev_naive(s1[1:],s2),        #next move is deletion
                   lev_naive(s1[1:],s2[1:]),    #next move is substitution
                   lev_naive(s1,s2[1:])         #next move is insertion
                   )
    
    
    

def lev_naive_better(s1,s2,l1=None,l2=None): #O(3^(n+m)) given n=len(s1), m=len(s2); auxillary complexity of O(n+m)
    if l1 is None or l2 is None:
        l1 = len(s1)
        l2 = len(s2)
        
    if l1<=0: return l2
    if l2<=0: return l1
    
    if s1[l1-1]==s2[l2-1]: return lev_naive_better(s1,s2,l1-1,l2-1)

    return 1 + min(lev_naive_better(s1,s2,l1-1,l2),   #next move is deletion
                   lev_naive_better(s1,s2,l1-1,l2-1), # next move is substitution
                   lev_naive_better(s1,s2,l1,l2-1)    # next move is insertion
                    )





def lev_matrix(s1,s2): # O(n*m) time complexity and auxillary complexity
    """Optimized levenshtein distance alg, should run in O(n*m) time"""
    l1, l2 = len(s1),len(s2)
    
    mat = [[0 for _ in range(l2+1)] for _ in range(l1+1)]
    
    for i in range(l1+1):
        mat[i][0]=i# initialize first row
    for j in range(l2+1):
        mat[0][j]=j # initialize first col
        
    
    #now, use dynamic programming to find mat[l1][l2]
    
    for i in range(1,l1+1):
        for j in range(1,l2+1):
            if s1[i-1]==s2[j-1]:#if curr chars are same, equal to top right ting
                mat[i][j] = mat[i-1][j-1]
            else:
                mat[i][j] = 1 + min(mat[i][j-1], #insertion move
                                    mat[i-1][j], #deletion move
                                    mat[i-1][j-1] #sub move
                                    )
                
    return mat[l1][l2], mat        
    
    
    
def lev_matrix_better(s1,s2): # O(n*m) time complexity; O(min(n,m)) auxillary complexity
    """Like the one before but stores only two rows of the matrix at a time; swapped lists for numpy arrays, which would decrease runtime but not in an asymptotically significant manner"""
    if len(s1)< len(s2):
        s1, s2 = s2,s1 # swap so s2 is the smaller string
    
    l1,l2 = len(s1),len(s2)
    
    if l1==0: return l2
    if l2==0: return l1
    
    prev = [j for j in range(l2+1)]
    curr = [0]* (l2+1)
    
    for i in range(1,l1+1): # outer loop moves curr to prev, updates curr
        curr[0]=i
        
        for j in range(1,l2+1):
            if s1[i-1]==s2[j-1]:#char match
                curr[j]=prev[j-1]#Same as before, use top left
            else: #choose min cost operation
                curr[j]=1 + min(curr[j-1],   # insertion
                                     prev[j], #removal
                                     prev[j-1] # replacement
                                )
                
        prev = curr.copy()
        
    return curr[l2]
        
    
    
    
    

    
    
    
    
    
    
    
def matRepr(mat):
    """Helper function for representing the matrix"""
    ret = "" 
    for row in mat:
        r = []
        for entry in row:
            if round(entry,6)==int(entry): 
                r.append(int(entry))
            else:
                r.append(round(entry,3))
        ret += f'|{str(r)[1:-1]}|\n'
    return ret





import tkinter as tk
import matplotlib.pyplot as plt

from time import time

def timeit(f, *args): #returns average time taken
    times = 25
    total = 0
    for _ in range(times):
        start = time()
        f(*args)
        end = time()
        total += (end-start)
    return round(total/times,8)




allWords = []

with open(r"Words.txt",'r') as f:
    for line in f:
        allWords.append(line.strip())


from random import choice, choices


maxim = 20
iterations = maxim - 3
words = [[] for _ in range(iterations+1)]#words of length 3 inclusive to maxim exclusive


for word in allWords:
    l = len(word)
    if l >maxim: continue
    words[l-3].append(word)


# we have words of sizes l = 3,6,...,27
#We now want to choose 20 pairs at random, same for all trials, then find total time 
#NOTE: exclude len=27 since there are only three words here, not good enough data set

# Plot lev_naive, lev_naive_better, lev_matrix, lev_matrix_better

chosenPairs = [[] for _ in range(iterations)]
for i in range(iterations): # 0,1, ... 7 --> 3,6, ..., 24
    for _ in range(60):#20 pairs
        chosenPairs[i].append(choices(words[i],k=2))



# for pairs in chosenPairs: # This is just to test that we did it right :)
#     print(pairs)
#     print('\n\n')
#     print(len(pairs[1][0]))

#Now we have chosenPairs which is a list[i=0..8] of words, each item in chosenPairs is a list of 20 pairs of 2 words of length (3*i)+1



# we are gonna do v1, v2, v3, v4 for each implementation
v1 = [[] for _ in range(iterations)] 
v2 = [[] for _ in range(iterations)] 
v3 = [[] for _ in range(iterations)] 
v4 = [[] for _ in range(iterations)] 

for version in [[v1,lev_naive],[v2,lev_naive_better],[v3,lev_matrix],[v4,lev_matrix_better]]:
    for i in range(iterations): # 0,1, ... 7 --> 3,6, ..., 24
        avgTime = 0
        for j in range(20):#20 pairs
            avgTime += timeit(version[1],*chosenPairs[i][j])
        version[0][i] = (avgTime/20)


nums = [i+3 for i in range(iterations)]
plt.plot(nums, v1,'b-', nums,v2,'g-',nums, v3,'r-', nums,v4,'y-')
plt.xlabel('Size of input')
plt.ylabel('Time in seconds')
#plt.yscale('log')
plt.title(' Difference in Levenshtein Implementations')
plt.show()
 



if __name__=='__main__balls':
    tup = ('catherine','chatters')
    print(lev_naive(*tup))
    print(lev_naive_better(*tup))
    print(lev_matrix(*tup)[0])
   # print(matRepr(lev_matrix(*tup)[1]))
    print(lev_matrix_better(*tup))
