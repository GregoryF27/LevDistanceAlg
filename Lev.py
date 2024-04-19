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
    if not l1 or not l2:
        l1 = len(s1)
        l2 = len(s2)
        
    if l1==0: return l2
    if l2==0: return l1
    
    if s1[l1-1]==s2[l2-1]: return lev_naive_better(s1,s2,l1-1,l2-1)
    
    return 1 + min(lev_naive_better(s1,s2,l1-1,l2),   #next move is deletion (are we sure?)
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
    
print(lev_matrix("hello","hemblopp")[1])
    
def lev_matrix_better(s1,s2): # O(n*m) time complexity; O(max(n,m)) auxillary complexity
    """Like the one before but stores only two rows of the matrix at a time"""
    l1,l2 = len(s1),len(s2)
    
    prev = [j for j in range(l2+1)]
    curr = [0]* (l2+1)
    
    for i in range(1,l1+1): # outer loop moves curr to prev, updates curr
        curr[0]=i
        
        for j in range(1,l1+1):
            if s1[i-1]==s2[j-1]:#char match
                curr[j]=prev[j-1]#Same as before, use top left
            else: #choose min cost operation
                curr[j]=1 + min(curr[j-1],   # insertion
                                     prev[j], #removal
                                     prev[j-1] # replacement
                                )
        print(curr)
        prev = curr.copy()
        
    return curr[l1]


print(lev_matrix_better("hello","hemblopp"))
        
import numpy as np
def lev_matrix_better_numpy(s1, s2): # O(n*m) time complexity; O(max(n,m)) auxillary complexity
    """swapped lists for numpy arrays, which would decrease runtime but not in an asymptotically significant manner"""
    l1,l2 = len(s1),len(s2)
    
    prev = np.array([j for j in range(l2+1)])
    curr = np.array([0] * (l2+1))
    print()
    for i in range(1,l1+1): # outer loop moves curr to prev, updates curr
        curr[0]=i
        
        for j in range(1,l1+1):
            if s1[i-1]==s2[j-1]:#char match
                curr[j]=prev[j-1]#Same as before, use top left
            else: #choose min cost operation
                curr[j]=1 + min(curr[j-1],   # insertion
                                     prev[j], #removal
                                     prev[j-1] # replacement
                                )
                
        print(curr)
        prev = curr.copy()
        
    return curr[l1]

# problem here
print(lev_matrix_better_numpy("hello","hemblopp"))

    
    
    
    
    
    
    
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

def timeit(f, *args):
    times = 10
    minim = float('inf')
    for _ in range(times):
        start = time()
        f(*args)
        if (end := time() - start) < minim: mim = end
    return round(minim,8)




allWords = []

with open(r"Words.txt",'r') as f:
    for line in f:
        allWords.append(line.strip())


from random import choice, choices
    
words = [[] for _ in range(1,9)]
# 0 is len=3, 1 is len=6, ... 9 is len=27


lens = {3*i for i in range(1,9)}


allThem = set()
for word in allWords:
    if l:=len(word) in lens: 
        allThem.add(int(l/3-1))
        words[int(l/3)-1].append(word)


#print(allThem)
# we have words of sizes l = 3,6,...,27
#We now want to choose 40 pairs at random, same for all trials, then find total time
# Plot lev_naive, lev_naive_better, lev_matrix, lev_matrix_better (implement numpy first?)

chosenPairs = [[] for _ in range(1,9)]
for i in range(1,9):
    for _ in range(5):
        chosenPairs[i-1].append(choices(words))


#print(chosenPairs)
# print(words[1])
# print(choices(words[1]))

# plt.plot(nums, bruteData,'r^', nums,binaryData,'g^')
# plt.xlabel('Size of input')
# plt.ylabel('Time in seconds')
# plt.xscale('log')
# plt.title(' Brute Force vs Binary Search Effeciency')
# plt.show()
 



if __name__=='__main__':
    tup = ('catherine','chatters')
    # print(lev_naive(*tup))

    # print(lev_matrix(*tup)[0])
    # print(matRepr(lev_matrix(*tup)[1]))
    # print(lev_matrix_better(*tup))
