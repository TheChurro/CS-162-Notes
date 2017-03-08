#!/usr/bin/env python3

from bisect import insort_left

from UnionFind import UpTree
from QueueAndStack import PriorityQueue, FIFOQueue

class Graph(object):

  def __init__(self):

    # Essentially an adjacency list, just using a table
    self.adjacencyTable = {}

  def addNode(self, newNode):

    # Create a new entry in the table for newNode
    self.adjacencyTable[newNode] = []

  def connect(self, nodeA, nodeB, weight):

    # Connect the two nodes A and B with the given weight
    if nodeA in self.adjacencyTable and nodeB in self.adjacencyTable:
      self.adjacencyTable[nodeA].append((weight, nodeB))

  def connections(self, nodeA):

    # Get the list of connections for a given node
    if nodeA in self.adjacencyTable:
      return self.adjacencyTable[nodeA]
    return []

  def dfs(self, start, end, join=lambda x, y: [x] + y, base=lambda x:[x],
                noneVal=None):

    return self._dfs(start, end, [], join, base, noneVal)

  def _dfs(self, start, end, passedNodes, join=lambda x, y: [x] + y, base=lambda x:[x],
                noneVal=None):

    """
    Search from start to find end. This function will call base on end when it is
    found and return that value. The function will return join of start and the
    return value of the recursive depth first search calls. DFS uses noneVal
    to determine when a depth first search failed.

    The default values for join, base, and noneVal will cause this function to
    find the shortest path kept in a list.

    This function will consider every node exactly 1 time. Hence, the number of
    times _dfs is called is proportional to |V|, the number of nodes in the graph.
    This is because if a node is already visited, it will not be visited again.
    Every edge must be connected to two nodes, and in the worst case every edge
    will be seen one time. Hence, _dfs has O(|E| + |V|), for going through
    |E| calls in the for loop and |V| calls of _dfs.

    This specific implementation will have O(|E| * |V| + |V|) because we know
    if a node is marked by its existence in an array. A faster implementation
    could remove this barrier but for the sake of clarity I have included it.
    """

    # We have traveled to this node so mark it as not available
    passedNodes.append(start)

    # If we have encountered the end node
    if start == end:

      # Return base(end) to signify that end has been found
      return base(end)

    # If we are not at the termination condition, search through the set
    # of nodes connected to this node (start)
    for weight, node in self.connections(start):

      # If this node has been encountered when searching, skip it
      if node in passedNodes:

        continue

      # Search down this path.
      searchVal = self._dfs(node, end, passedNodes, join, base, noneVal)

      # If the search was a success
      if searchVal != noneVal:

        # Join the node we are at (start) with the result of the search
        return join(start, searchVal)

    # If we failed to find end down any of the connections on this node
    # return noneVal to signify the search failed.

    return noneVal

  def bfs(self, start, end, join=lambda x, y: [x] + y, base=lambda x:[x],
                noneVal=None):

    """
    Breadth First Search will find the path to end which uses the shortest number
    of edges.
    """

    # If we start at the end then we know we're done
    if start == end:
      return base(end)

    # Queue which holds the next item to search through as well as
    # the previous path (node, path)
    queue = FIFOQueue()

    # Enqueue the start node to start searching from there
    queue.enqueue((start, []))

    # While the queue is not empty
    while queue:

      # Pop off the next node to search through
      nodeToSearch, path = queue.dequeue()

      # Append the current node onto the path for the nodes connected to it
      newPath = path + [nodeToSearch]

      # For each edge
      for weight, nextNode in self.connections(nodeToSearch):

        # If we have found the end
        if nextNode == end:

          # We start to build the return value. The base is base(end)
          retVal = base(end)
          # Then we will go backwards through the path
          newPath.reverse()

          for prevNode in newPath:

            # And join each node in the path with the return value
            retVal = join(prevNode, retVal)

          # And finally return the result of the path visited
          return retVal

        # Otherwise we have not found the end so we will keep searching

        # If we have already searched next node while traversing this path,
        # we ignore it as not to have infinite loops
        if nextNode in newPath:
          continue

        # Otherwise we enqueue nextNode with the path to nextNode
        # so that we can search through those options.
        queue.enqueue((nextNode, newPath))

    # So because the queue is empty and we have not returned,
    # end was not reachable from start. Therefore, we return noneVal

    return noneVal

  def kruskal(self):

    """
    Create a min-spanning tree from this graph.
    """

    # Up tree initialized using one tree for every node in the graph
    upTree = UpTree(self.adjacencyTable.keys())

    # Create a PriorityQueue which uses edge weight to determine priority
    # Edges are stored in the queue as (start, end, weight)
    priorityQueue = PriorityQueue(lambda x:x[2])
    for node, edges in self.adjacencyTable.items():
      priorityQueue.enqueueAll([(node, nextNode, weight) for weight, nextNode in edges])

    # For keeping track of the termination condition
    numVertices = len(self.adjacencyTable.keys())

    # Edges in the new graph
    edges = []

    # While the queue is not empty and while not every node has been connected
    while priorityQueue and len(edges) != numVertices - 1:

      # Remove the next least weighted edge from the queue.
      node0, node1, weight = priorityQueue.deleteMin()

      # If these nodes have not yet been connected, connect them
      if upTree.find(node0) != upTree.find(node1):

        # Union the uptree and add an edge to the end list of edges.
        upTree.union(node0, node1)
        edges.append((node0, node1, weight))

    # Return the edges connected.
    return edges

  def djikstra(self, start, join=lambda x, y: None, base=lambda x:None,
                     noneVal = None, returnVisited=False, reduce=lambda n, x:n):

    # List of visited nodes stored as (nodeName, pathToNode, closestNodeWeight, distanceTraveled)
    visited = []

    def hasVisited(node):

      for n, ret, weight, dist in visited:

        if n == node:
          return True

      return False

    # List of nodes to visit stored in 3 tuples, (nodeName, path, prevWeight, distanceTraveled)
    toVisit = PriorityQueue(lambda x: x[3])

    # Enqeue the start node
    toVisit.enqueue((start, [], 0, 0))

    # While there are more nodes to visit
    while toVisit:

      # Remove the node with the next least distance to visit
      node, path, prevWeight, distance = toVisit.deleteMin()

      # If this node has already been visited, skip it
      if hasVisited(node):
        continue

      # Mark this node as visited
      visited.append((node, path, prevWeight, distance))

      # Go through each connection to this node
      for weight, nextNode in self.connections(node):

        # and enqueue it in toVisit with increased distance
        toVisit.enqueue((nextNode, path + [node], weight, distance + weight))

    # In case one wishes the whole tree of options they may have it.
    if returnVisited:

      return visited

    # Otherwise we shall build up a graph from the paths traversed

    djiGraph = Graph()

    # Function which folds join over the given list after applying base to the
    # last element. This is a right fold
    def foldList(ls):

      if not ls:
        return noneVal

      accumulator = base(ls[-1])
      for i in range(1, len(ls)):

        accumulator = join(ls[-1 - i], accumulator)

      return accumulator

    # Dictionary which keeps track of which node in the original
    # graph corresponds to a new node in the djikstra graph
    reducedNodes = {}

    # For each node we visited in order of visitation
    for node, path, weight, distance in visited:

      reducedNodes[node] = reduce(node, foldList(path))
      # Add the node to the graph
      djiGraph.addNode(reducedNodes[node])

      # If the path to this node is non-empty
      if path:
        # Then make a connection from the last node in the path
        # to this node
        djiGraph.connect(reducedNodes[path[-1]], reducedNodes[node], weight)

    return djiGraph

  def __str__(self):

    string = ""
    for node, connections in self.adjacencyTable.items():

      string += str(node) + ": " + str([(n, w) for w, n in connections]) + "\n"

    return string[:-1]

  def __eq__(self, other):

    # Ensure that the same nodes are in each graph
    if self.adjacencyTable.keys() != other.adjacencyTable.keys():
      return False

    # For each node n in this graph
    for node, connections in self.adjacencyTable.items():

      # And ensure that the edges are the same for that node in each graph
      if set(connections) != set(other.connections(node)):

        return False

    return True

class UnweightedUndirectedGraph(Graph):

  def connect(self, nodeA, nodeB):

    super().connect(nodeA, nodeB, 1)
    super().connect(nodeB, nodeA, 1)

  def __str__(self):

    string = ""

    string += "Nodes: " + str(self.adjacencyTable.keys()) + "\n"

    parsedNodes = []
    edges = []

    for node, connections in self.adjacencyTable.items():

      for weight, otherNode in connections:

        if otherNode not in parsedNodes:

          edges.append("({0} <-> {1})".format(node, otherNode))

      parsedNodes.append(node)

    string += "Edges: " + str(edges)

    return string

class UndirectedGraph(Graph):

  def connect(self, nodeA, nodeB, weight):

    super().connect(nodeA, nodeB, weight)
    super().connect(nodeB, nodeA, weight)

  def __str__(self):

    string = ""

    string += "Nodes: " + str(self.adjacencyTable.keys()) + "\n"

    parsedNodes = []
    edges = []

    for node, connections in self.adjacencyTable.items():

      for weight, otherNode in connections:

        if otherNode not in parsedNodes:

          edges.append("({0} <-> {1} with weight {2})".format(node, otherNode, weight))

      parsedNodes.append(node)

    string += "Edges: " + str(edges)

    return string

class UnweightedGraph(Graph):

  def connect(self, nodeA, nodeB):

    super().connect(nodeA, nodeB, 1)

  def __str__(self):

    string = ""
    for node, connections in self.adjacencyTable.items():

      string += str(node) + ": " + str([n for w, n in connections]) + "\n"

    return string[:-1]

if __name__ == "__main__":

  print("----------------------------")
  print(" Testing Depth First Search")
  print("----------------------------")

  udwGraph = UnweightedUndirectedGraph()
  udGraph  = UndirectedGraph()
  uwGraph  = UnweightedGraph()
  graph    = Graph()

  nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

  for node in nodes:

    udwGraph.addNode(node)
    udGraph.addNode(node)
    uwGraph.addNode(node)
    graph.addNode(node)

  edges = [('a', 'b', 5), ('a', 'c', 1), ('h', 'b', 0), ('b', 'd', 9),
           ('d', 'c', 10), ('c', 'f', 3), ('c', 'e', 1), ('e', 'g', 20),
           ('g', 'd', 7), ('g', 'h', 2)]

  for start, end, weight in edges:

    udwGraph.connect(start, end)
    udGraph.connect(start, end, weight)
    uwGraph.connect(start, end)
    graph.connect(start, end, weight)

  path = udwGraph.dfs('a', 'g')

  print("UnweightedUndirectedGraph:")
  print(udwGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'h', 'g'] else "Failure")

  print("----------------------------")

  path = udGraph.dfs('a', 'g')

  print("UndirectedGraph:")
  print(udGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'h', 'g'] else "Failure")

  print("----------------------------")

  path = uwGraph.dfs('a', 'g')

  print("UnweightedGraph:")
  print(uwGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'd', 'c', 'e', 'g'] else "Failure")

  print("----------------------------")

  path = graph.dfs('a', 'g')

  print("Graph:")
  print(graph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'd', 'c', 'e', 'g'] else "Failure")

  print("----------------------------")
  print("Testing Breadth First Search")
  print("----------------------------")

  path = udwGraph.bfs('a', 'g')

  print("UnweightedUndirectedGraph:")
  print(udwGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'h', 'g'] else "Failure")

  print("----------------------------")

  path = udGraph.bfs('a', 'g')

  print("UndirectedGraph:")
  print(udGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'b', 'h', 'g'] else "Failure")

  print("----------------------------")

  path = uwGraph.bfs('a', 'g')

  print("UnweightedGraph:")
  print(uwGraph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'c', 'e', 'g'] else "Failure")

  print("----------------------------")

  path = graph.bfs('a', 'g')

  print("Graph:")
  print(graph)
  print("Path: {0}".format(path))
  print("Successful" if path == ['a', 'c', 'e', 'g'] else "Failure")

  print("----------------------------")
  print("Testing Djikstra's Algorithm")
  print("----------------------------")

  path = udwGraph.djikstra('a')

  expectedGraph = Graph()

  for node in nodes:
    expectedGraph.addNode(node)

  expectedEdges = [('a', 'b', 1), ('a', 'c', 1), ('b', 'd', 1), ('c', 'f', 1),
                   ('c', 'e', 1), ('b', 'h', 1), ('e', 'g', 1)]
  for start, end, weight in expectedEdges:
    expectedGraph.connect(start, end, weight)

  print("UnweightedUndirectedGraph:")
  print(udwGraph)
  print("Djikstra:\n{0}".format(path))
  print("Successful" if path == expectedGraph else "Failure")

  print("----------------------------")

  path = udGraph.djikstra('a')

  expectedGraph = Graph()

  for node in nodes:
    expectedGraph.addNode(node)

  expectedEdges = [('a', 'b', 5), ('a', 'c', 1), ('c', 'd', 10), ('c', 'f', 3),
                   ('c', 'e', 1), ('b', 'h', 0), ('h', 'g', 2)]
  for start, end, weight in expectedEdges:
    expectedGraph.connect(start, end, weight)

  print("UndirectedGraph:")
  print(udGraph)
  print("Djikstra:\n{0}".format(path))
  print("Successful" if path == expectedGraph else "Failure")

  print("----------------------------")

  path = uwGraph.djikstra('a')

  expectedGraph = Graph()

  for node in nodes:
    expectedGraph.addNode(node)

  expectedEdges = [('a', 'b', 1), ('a', 'c', 1), ('b', 'd', 1), ('c', 'f', 1),
                   ('c', 'e', 1), ('e', 'g', 1), ('g', 'h', 1)]
  for start, end, weight in expectedEdges:
    expectedGraph.connect(start, end, weight)

  print("UnweightedGraph:")
  print(uwGraph)
  print("Djikstra:\n{0}".format(path))
  print("Successful" if path == expectedGraph else "Failure")

  print("----------------------------")

  path = graph.djikstra('a')

  expectedGraph = Graph()

  for node in nodes:
    expectedGraph.addNode(node)

  expectedEdges = [('a', 'b', 5), ('a', 'c', 1), ('b', 'd', 9), ('c', 'f', 3),
                   ('c', 'e', 1), ('e', 'g', 20), ('g', 'h', 2)]
  for start, end, weight in expectedEdges:
    expectedGraph.connect(start, end, weight)

  print("Graph:")
  print(graph)
  print("Djikstra:\n{0}".format(path))
  print("Successful" if path == expectedGraph else "Failure")

  print("----------------------------")
  print("Testing Kruskal's Algorithm ")
  print("----------------------------")

  # Note, due to the method that I store the edges (in a hashtable)
  # edge order cannot be consistent through runs and so I will not be
  # attempting kruskal's for unweighted graphs or undirected graphs
  # (it will never be consistent.)

  path = graph.kruskal()

  kruskalEdges = set([('h', 'b', 0), ('a', 'c', 1), ('c', 'e', 1),
                      ('g', 'h', 2), ('c', 'f', 3), ('a', 'b', 5),
                      ('g', 'd', 7)])

  print("Graph:")
  print(graph)
  print("Edges:\n{0}".format(path))
  print("Successful" if set(path) == kruskalEdges else "Failure")
