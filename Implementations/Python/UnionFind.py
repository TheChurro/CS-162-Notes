#!/usr/bin/env python3

class UnionSize(object):

  """
  Class used to keep track of the size of subtrees of UpTrees.
  """

  def __init__(self, value):

    self.value = value

  def __lt__(self, other):

    return self.value < other.value

  def __le__(self, other):

    return self.value <= other.value

  def __iadd__(self, other):

    self.value += other.value
    return self

  def __repr__(self):

    return "US" + repr(self.value)

  def __str__(self):

    return "US" + str(self.value)

def find(upTree, name):

  """
  Finds the root node of the tree which this name belongs to
  """

  # If the node "name" in the uptree is a root node
  if isinstance(upTree[name], UnionSize):

    # Then return this node as the root
    return name

  # Otherwise find the root node
  root = find(upTree, upTree[name])
  # Link this node directly to that root
  upTree[name] = root

  # And return the root
  return root

def union(upTree, name0, name1):

  """
  Unions two trees together within the uptree. This function
  will find the root nodes of the two names passed in.

  Returns true if the two passed in names are unioned.
  """

  # Get the root nodes of the two trees.
  tree0 = find(upTree, name0)
  tree1 = find(upTree, name1)

  # If the two names are connected to the same root, return
  if tree0 == tree1: return False

  if upTree[tree0] >= upTree[tree1]:

    # If tree0 is larger, connect tree1 to tree0 as its root
    # This will preserve the log(n) search behaviour
    # Also, increase the size of tree0 by the number of nodes added
    upTree[tree0] += upTree[tree1]
    upTree[tree1] = tree0

  else:

    # If tree1 is larger, connect tree0 to tree1 as its root
    # This will preserve the log(n) search behaviour
    # Also, increase the size of tree1 by the number of nodes added
    upTree[tree1] += upTree[tree0]
    upTree[tree0] = tree1

  return True

def makeUpTree(data):

  """
  Create an up tree from the data where each element in the data
  set is its own tree.
  """
  return {name:UnionSize(1) for name in data}

class UnionFind(object):

  def __init__(self, data):

    self.data = data
    self.upTree = makeUpTree(data)

    self.find  = lambda x: find(self.upTree, x)
    self.union = lambda x, y: union(self.upTree, x, y)

if __name__ == "__main__":

  from random import randrange

  # Test case: Take the first 99 numbers and partition them such
  # that things with the same remainder will be in the same tree

  # Create the set of data used
  data = [i for i in range(100)]

  # Generate an empty uptree from the data
  upTree = makeUpTree(data)

  # For each point in the data
  for point in data:

    # Union the tree attached to that point
    # with the tree attached to its mod value
    union(upTree, point % 10, point)

  # Determine if any of the unions failed
  failed = False

  # For each point in the data
  for point in data:

    # Ensure that the unions were correctly performed
    if find(upTree, point % 10) != find(upTree, point):

      failed = True
      print("Found mismatch from expected: {0}:{1} -- {2}:{3}".format(
            point % 10, find(upTree, point % 10), point, find(upTree, point)))

  # As long as the unions were performed correctly
  if not failed:

    # Print out that this case was successful
    print("Test 1 Successful.")

  else:

    print("Test 1 Failed.")


  """
  Test 2. We will now check using randomness that
  unions are performed correctly.
  """

  failed = False

  upTree = makeUpTree(data)

  fakeTree = []

  def findWithin(item):

    """
    Find the list which item is in within the list of equivalence
    classes, fakeTree.
    """

    for i in range(len(fakeTree)):

      if item in fakeTree[i]:

        return i

    return -1

  for point in data:

    # Take a random index from the list
    index = randrange(len(data))

    # If the index found is that of the point we are at
    # Then skip forwards one
    if data.index(point) == index:

      index += 1
      index %= len(data)

    # Store which point in the data we will be
    # joining with the point we are at
    toUnion = data[index]

    # Union the trees of the two points
    if union(upTree, point, toUnion):

      # If the union was successful then we need to update our
      # fake tree
      pointIndex   = findWithin(point)
      toUnionIndex = findWithin(toUnion)

      # If the point was not yet included in any equivalence class
      if pointIndex == -1:

        # If the toUnion element was not yet included in any equivalence class
        if toUnionIndex == -1:

          # Then add a new equivalence class to the fakeTree made up of the
          # two new members
          fakeTree.append([point, toUnion])

        else:

          # Otherwise add point to the equivalence class containing toUnion
          fakeTree[toUnionIndex].append(point)

      # If toUnion is not yet in any equivalence class but point is
      elif toUnionIndex == -1:

        # Then add toUnion to point's equivalence class
        fakeTree[pointIndex].append(toUnion)

      else: # Both toUnion and point are held within the fake tree

        # So combine toUnion's equivalence class with points
        fakeTree[pointIndex] += fakeTree[toUnionIndex]
        # Remove toUnion's equivalence class from the tree because it
        # is now contained in point's.
        fakeTree.pop(toUnionIndex)

  # For each equivalenceClass
  for point in data:

    # Gather the equivalenceClass for point
    equivalenceClass = findWithin(point)
    if equivalenceClass == -1:
      equivalenceClass = [point]
    else:
      equivalenceClass = fakeTree[equivalenceClass]

    # Get the root of point within the upTree
    upTreeRoot = find(upTree, point)

    # For every point in data
    for someOtherPoint in data:

      # Check if someOtherPoint is in the equivalenceClass of point
      # aka if their root in the uptree should be the same
      if someOtherPoint in equivalenceClass:

        # If their roots are different, log this discrepency
        # And signal that the unions failed
        if upTreeRoot != find(upTree, someOtherPoint):

          failed = True
          print("Found mismatch from expected: {0}:{1} != {2}:{3}".format(
                point, upTreeRoot, someOtherPoint, find(upTree, someOtherPoint)
          ))

      else: # Then these two points should not have the same root

        # If their roots are the same, log this discrepency
        # And signal that the unions failed
        if upTreeRoot == find(upTree, someOtherPoint):

          failed = True
          print("Found mismatch from expected: {0}:{1} == {2}:{3}".format(
                point, upTreeRoot, someOtherPoint, find(upTree, someOtherPoint)
          ))

  if not failed:

    print("Test 2 Successful.")

  else:

    print("Test 2 Failed.")
