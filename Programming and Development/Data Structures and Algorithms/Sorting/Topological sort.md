# Topological sort

Topological sort is a sorting algorithm for [[graphs#DAG|Graph]]
. It returns a list of nodes such that all
nodes later in the list depend on nodes earlier in the list.

## The Algorithm
- Initialize a stack and a visited array of size n (number of vertices)
- Loop over all unvisited vertices and do the following:
	- call [[Depth first search|DPS]] with the vertex as the parameter
	- In [[Depth first search|DPS]], mark the vertex as visited and recursively call DFS for all unvisited neighbors
	- Once all neighbors are visited, push that vertex onto the stack
- Pop each vertex from the stack and append to output list
- The list is now topologically sorted.
# References
- https://www.geeksforgeeks.org/topological-sorting