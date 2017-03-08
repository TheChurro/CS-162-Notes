#!/usr/bin/env python3

class PriorityQueue(object):

  def __init__(self, priorityFunction):

    self.size = 0
    self.heap = []
    self.priority = priorityFunction

  def min(self):

    return self.heap[0]

  def __len__(self):

    return self.size

  def __nonzero__(self):

    return self.size > 0

  def nonEmpty(self):

    return self.size > 0

  def __getitem__(self, index):

    return self.heap[index - 1]

  def __setitem__(self, index, value):

    self.heap[index - 1] = value

  def enqueueAll(self, items):

    for item in items:

      self.enqueue(item)

  def enqueue(self, item):

    """
    Add an item to the priority queue.
    """

    # Add the new element to the end of the queue
    self.heap.append(item)

    # Increase the size of the queue
    self.size += 1
    # Move the last item in the queue upwards based on priority
    self._percolateUp(self.size)

  def _percolateUp(self, index):

    # If we are at the start index, return
    if index == 1: return

    # Otherwise, if the priority of the parent of "index" is greater
    # than that at index, then swap the two and continue percolating
    if self.priority(self[index // 2]) >= self.priority(self[index]):

      # Swap
      self[index // 2], self[index] = self[index], self[index // 2]

      self._percolateUp(index // 2)

  def deleteMin(self):

    # Swap the first item with the last
    self[1], self[self.size] = self[self.size], self[1]

    # Take off the last item (the one with the minimum priority)
    item = self.heap.pop()
    # Decrease the size
    self.size -= 1

    if self.size > 0: # If the queue is not empty

      # Percolate the first item down
      self._percolateDown(1)

    # Return the lowest item
    return item

  def _percolateDown(self, index):

    toSwap  = -1
    minPriority = self.priority(self[index])

    # Determine if the left child has lower priority
    # than the node at index
    if index * 2 <= self.size:

      if self.priority(self[index * 2]) < minPriority:

        # If so, swap with that child
        toSwap = index * 2
        minPriority = self.priority(self[index * 2])

    # Determine if the right child has the lowest priority
    # of the node at index and the left child
    if index * 2 + 1 <= self.size:

      if self.priority(self[index * 2 + 1]) < minPriority:

        # If so, swap with that child
        toSwap = index * 2 + 1
        minPriority = self.priority(self[index * 2])

    if toSwap != -1:

      # Swap the element at index with the element at toSwap
      self[index], self[toSwap] = self[toSwap], self[index]

      # Percolate down from there
      self._percolateDown(toSwap)

class FIFOLList(object):

  def __init__(self, item, rest):

    self.item = item
    self.rest = rest

class FIFOQueue(object):

  def __init__(self):

    self.head = None
    self.end  = None

  def __nonzero__(self):

    return self.head != None

  def nonEmpty(self):

    return self.head != None

  def enqueue(self, item):

    # If the queue is not empty
    if self.end is not None:
      # Then add another item onto the end of the LList
      self.end.rest = FIFOLList(item, None)
      self.end = self.end.rest

    else:

      # otherwise create the LList representing the queue
      self.end = FIFOLList(item, None)
      self.head = self.end

  def dequeue(self):

    # As long as there is something to dequeue
    if self.head == None:
      return None

    # Grab it from the front of the list
    item = self.head.item
    # Remove that element from the head of the list
    self.head = self.head.rest

    # If we have exhausted the list
    if self.head is None:

      # then clear out the end
      self.end = None

    # And then return the dequeued item
    return item

  def __str__(self):

    ls = []

    it = self.head
    while it:

      ls.append(it.item)
      it = it.rest

    return str(ls)

  def __repr__(self):

    return str(self)

class Stack(object):

  def __init__(self):

    # Holds the items (linked list-esque implementation)
    self.items = None

  def push(self, item):

    # Takes in an item and appends it to the front of the list
    # (Note that the head of the list is the first item in the tuple)
    self.items = (item, self.items)

  def pop(self):

    # Removes the first item from the stack
    item = None

    # If the list is not empty
    if self.items:

      # Take off the first item
      item, self.items = self.items

    # return the item taken of if any
    return item

if __name__ == "__main__":

  from random import randrange, choice
  import sys

  # Function which takes a list and returns a new list
  # with the items in random order
  def randomize(ls):

    newls = []

    for i in range(len(ls)):

      item = choice(ls)
      ls.remove(item)
      newls.append(item)

    return newls

  # Items to be added into the queue
  items = [i for i in range(1000)]

  for test in range(1000):

    # Randomized order of items
    randomizedItems = randomize(items)

    # Gives integers the priority of what they are
    def intPriority(num): return num

    # Create a new queue and add all the items in items in random order
    itemsQueue = PriorityQueue(intPriority)
    itemsQueue.enqueueAll(randomizedItems)

    # Keep track of if anything was removed in the wrong order
    failed = False

    for item in items:

      removed = itemsQueue.deleteMin()

      if item != removed:

        print("Failed removal: expected {0} | got {1}".format(item, removed))
        failed = True

    if failed:

      print("Failed test with:")
      print(randomizedItems)
      sys.exit()

  print("All PriorityQueue tests successful.")

  for i in range(1000):

    # Randomized order of items
    randomizedItems = randomize(items)

    # Create a new stack and add all the items in items in random order
    itemsStack = Stack()
    for item in randomizedItems:
      itemsStack.push(item)

    # Keep track of if anything was removed in the wrong order
    failed = False

    randomizedItems.reverse()

    for item in randomizedItems:

      removed = itemsStack.pop()

      if item != removed:

        print("Failed removal: expected {0} | got {1}".format(item, removed))
        failed = True

    if failed:

      print("Failed test with:")
      print(randomizedItems)
      sys.exit()

  print("All Stack tests successful.")
