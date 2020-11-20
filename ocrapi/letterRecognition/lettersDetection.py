import matplotlib.pyplot as plt
import cv2
import numpy as np
import itertools
import networkx as nx
import matplotlib.colors as col
import matplotlib.pyplot as plt
%matplotlib inline
from google.colab.patches import cv2_imshow
from imutils import contours

class Node(object):
  """
    Private tree node class.
    Node:
      value: the node's value
      parent: reference to the node's parent
      rank: the rank of the tree (only valid if node is a root)
  """
  # Constructor
  def __init__(self, value):
    self.value = value
    self.parent = self # Node is its own parent, therefore it's a root node
    self.rank = 0 # Tree of single node has rank 0

  # Print format for debugging
  def __str__(self):
    st = "[value: " + str(self.value) + ", parent: " + str(self.parent.value) 
    st += ", rank: " + str(self.rank) +  "]"
    return st

class UnionFind:

  # Constructor
  def __init__(self):
    self.__nodes_addressed_by_value = {} # To keep track of nodes


  # Required Union-Find functions ############################################################
  def MakeSet(self, value):
    """
      MakeSet(value):
        Makes a new set containing one node (with value 'value').
    """
    
    # If node already exists, return it
    if self.GetNode(value):
      return self.GetNode(value)

    # Otherwise create node
    node = Node(value)

    # Keep track of node
    self.__nodes_addressed_by_value[value] = node
    
    return node


  def Find(self, x):
    """
      Find(Node x):
        Returns the representative node of the set containing node x, by recursively 
        getting the node's parent.
      Optimisation using path compression: 
        Once you've found the root of the tree, set each visited node's parent to the 
        root, therefore flattening the tree along that path, speeding up future 
        operations.
        This is only a constant time complexity increase, but means future Find 
        operations along the same path are O(1).
    """

    # Node is not its own parent, therefore it's not the root node
    if x.parent  != x:  
      x.parent = self.Find(x.parent) # Flatten tree as you go (Path Compression)
    
    # If node is its own parent, then it is the root node -> return it
    return x.parent


  def Union(self, x, y):
    """
      Union(Node x, Node y):
        Performs a union on the two sets containing nodes x and y.
        Gets the representative nodes of x's and y's respective containing sets, and 
        makes one of them the other's parent (depending on their rank).
      Optimisation using union-by-rank:
        Always add the lower ranked ('smaller') tree to the larger one, ensuring no 
        increase in tree depth. If the two trees have the same rank (worst case), the 
        depth will increase by one. Without union-by-rank, each union operation is much 
        more likely to cause an increase in tree depth.
    """

    # If x and y are the same node, do nothing
    if x == y:
      return

    # Get the roots of both nodes' trees (= the representative elements of each of their 
    # containing sets)
    x_root = self.Find(x)
    y_root = self.Find(y)

    # If x and y are already members of the same set, do nothing
    if x_root == y_root:
      return

    # Perform set union
    # Union-by-rank optimisation: always add 'smaller' tree to 'larger' tree
    if x_root.rank > y_root.rank: 
      # Tree x has higher rank (therefore 'bigger' tree), so add y to x
      y_root.parent = x_root

    elif x_root.rank < y_root.rank: 
      # Tree y has higher rank, so add x to y
      x_root.parent = y_root

    else: 
      # Trees x and y have the same rank (same 'size')
      # Therefore add one tree to other arbitrarily and increase the resulting tree's rank 
      # by one
      x_root.parent = y_root
      y_root.rank = y_root.rank + 1


  # Other functions ##########################################################################
  def GetNode(self, value): # Get node with value 'value' (O(1))
    if value in self.__nodes_addressed_by_value:
      return self.__nodes_addressed_by_value[value]
    else:
      return False


  # Debugging functions ######################################################################
  def display_all_nodes(self):
    print("All nodes:")
    for item in self.__nodes_addressed_by_value.values():
      print(item)


  def display_all_sets(self):
    sets = {} # Initialise so nodes can't be added twice

    # Add all nodes to set dictionary 
    #   keys: representative element of each set
    #  values: the elements of the set with that representative
    for item in self.__nodes_addressed_by_value.values():
      if self.Find(item).value not in sets.keys():
        sets[self.Find(item).value] = [] # initialise list for this key
      sets[self.Find(item).value].append(item)

    # Display each representative key's set of items 
    st = ""
    for representative in sets.keys():
      st = st +  "("
      for item in sets[representative]:
        st = st + str(item.value) + ","
      st = st[:-1] # remove final ','
      st = st + ") "
    print(st)



CONNECTIVITY_4 = 4
CONNECTIVITY_8 = 8

def addspace(gray, tresh, w):
  # Sum white pixels in each row
  # Create blank space array and and final image 
  pixels = np.sum(thresh, axis=1).tolist()
  space = np.ones((1, w), dtype=np.uint8) * 255
  result = np.zeros((0, w), dtype=np.uint8)

  # Iterate through each row and add space if entire row is empty
  # otherwise add original section of image to final image
  for index, value in enumerate(pixels):
      if value < 255*len(tresh[0])/10:
          result = np.concatenate((result, space), axis=0)
      row = gray[index:index+1, 0:w]
      result = np.concatenate((result, row), axis=0)
  return result

def addspace_v(gray, thresh, h):
  # Sum white pixels in each row
  # Create blank space array and and final image 
  pixels = np.sum(thresh, axis=0).tolist()
  space = np.ones((h, 1), dtype=np.uint8) * 255
  result = np.zeros((h, 0), dtype=np.uint8)

  # Iterate through each row and add space if entire row is empty
  # otherwise add original section of image to final image
  for index, value in enumerate(pixels):
      if value < 255*len(thresh[0])/10:
          result = np.concatenate((result, space), axis=1)
      row = gray[0:h, index:index+1]
      result = np.concatenate((result, row), axis=1)
  return result

def connected_component_labelling(bool_input_image, connectivity_type=CONNECTIVITY_8):
  """
    2 pass algorithm using disjoint-set data structure with Union-Find algorithms to maintain 
    record of label equivalences.
    Input: binary image as 2D boolean array.
    Output: 2D integer array of labelled pixels.
    1st pass: label image and record label equivalence classes.
    2nd pass: replace labels with their root labels.
    (optional 3rd pass: Flatten labels so they are consecutive integers starting from 1.)
  """
  if connectivity_type !=4 and connectivity_type != 8:
    raise ValueError("Invalid connectivity type (choose 4 or 8)")


  image_width = len(bool_input_image[0])
  image_height = len(bool_input_image)

  # initialise efficient 2D int array with numpy
  # N.B. numpy matrix addressing syntax: array[y,x]
  labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
  uf = UnionFind() # initialise union find data structure
  current_label = 1 # initialise label counter

  # 1st Pass: label image and record label equivalences
  for y, row in enumerate(bool_input_image):
    for x, pixel in enumerate(row):
      
      if pixel == False:
        # Background pixel - leave output pixel value as 0
        pass
      else: 
        # Foreground pixel - work out what its label should be

        # Get set of neighbour's labels
        labels = neighbouring_labels(labelled_image, connectivity_type, x, y)

        if not labels:
          # If no neighbouring foreground pixels, new label -> use current_label 
          labelled_image[y,x] = current_label
          uf.MakeSet(current_label) # record label in disjoint set
          current_label = current_label + 1 # increment for next time        
        
        else:
          # Pixel is definitely part of a connected component: get smallest label of 
          # neighbours
          smallest_label = min(labels)
          labelled_image[y,x] = smallest_label

          if len(labels) > 1: # More than one type of label in component -> add 
                    # equivalence class
            for label in labels:
              uf.Union(uf.GetNode(smallest_label), uf.GetNode(label))


  # 2nd Pass: replace labels with their root labels
  final_labels = {}
  new_label_number = 1

  for y, row in enumerate(labelled_image):
    for x, pixel_value in enumerate(row):
      
      if pixel_value > 0: # Foreground pixel
        # Get element's set's representative value and use as the pixel's new label
        new_label = uf.Find(uf.GetNode(pixel_value)).value 
        labelled_image[y,x] = new_label

        # Add label to list of labels used, for 3rd pass (flattening label list)
        if new_label not in final_labels:
          final_labels[new_label] = new_label_number
          new_label_number = new_label_number + 1


  # 3rd Pass: flatten label list so labels are consecutive integers starting from 1 (in order 
  # of top to bottom, left to right)
  # Different implementation of disjoint-set may remove the need for 3rd pass?
  for y, row in enumerate(labelled_image):
    for x, pixel_value in enumerate(row):
      
      if pixel_value > 0: # Foreground pixel
        labelled_image[y,x] = final_labels[pixel_value]

  return labelled_image

# Private functions ############################################################################
def neighbouring_labels(image, connectivity_type, x, y):
  """
    Gets the set of neighbouring labels of pixel(x,y), depending on the connectivity type.
    Labelling kernel (only includes neighbouring pixels that have already been labelled - 
    row above and column to the left):
      Connectivity 4:
            n
         w  x  
       
      Connectivity 8:
        nw  n  ne
         w  x   
  """

  labels = set()

  if (connectivity_type == CONNECTIVITY_4) or (connectivity_type == CONNECTIVITY_8):
    # West neighbour
    if x > 0: # Pixel is not on left edge of image
      west_neighbour = image[y,x-1]
      if west_neighbour > 0: # It's a labelled pixel
        labels.add(west_neighbour)

    # North neighbour
    if y > 0: # Pixel is not on top edge of image
      north_neighbour = image[y-1,x]
      if north_neighbour > 0: # It's a labelled pixel
        labels.add(north_neighbour)


    if connectivity_type == CONNECTIVITY_8:
      # North-West neighbour
      if x > 0 and y > 0: # pixel is not on left or top edges of image
        northwest_neighbour = image[y-1,x-1]
        if northwest_neighbour > 0: # it's a labelled pixel
          labels.add(northwest_neighbour)

      # North-East neighbour
      if y > 0 and x < len(image[y]) - 1: # Pixel is not on top or right edges of image
        northeast_neighbour = image[y-1,x+1]
        if northeast_neighbour > 0: # It's a labelled pixel
          labels.add(northeast_neighbour)
  else:
    print("Connectivity type not found.")

  return labels


def print_image(image):
  """ 
    Prints a 2D array nicely. For debugging.
  """
  for y, row in enumerate(image):
    print(row)


def image_to_2d_bool_array(image):
  arr = np.asarray(image)
  arr = arr == 255

  return arr


def border(coordinates):
  i_min, j_min = 1000, 1000
  i_max, j_max = 0, 0
  for (i, j) in coordinates:
    if i < i_min:
      i_min = i
    if j < j_min:
      j_min = j
    if i > i_max:
      i_max = i
    if j > j_max:
      j_max = j
  return [(i_min, j_min), (i_max, j_max)]


CONNECTIVITY_4 = 4
CONNECTIVITY_8 = 8

def lettersDetection(img):
  #img = cv2.imread("/content/drive/MyDrive/lettres.PNG")
  #img = np.array(img, dtype=np.uint8)
  h, w = img.shape[:2]
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  
  
  img = addspace(gray, thresh, w)
  
  kernel = np.ones((3,3), np.uint8)
  kernel[0, 0] = 0
  kernel[0, 2] = 0
  kernel[2, 0] = 0
  kernel[2, 2] = 0
  
  erosion = cv2.erode(img, kernel, iterations = 10)
  cv2_imshow(erosion)
  
  erosion = cv2.GaussianBlur(erosion,(13,13),0)
  tresh = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
  
  label_img = connected_component_labelling(image_to_2d_bool_array(tresh), 4)
  
  label_hue = np.uint8(179*label_img/np.max(label_img))
  blank_ch = 255*np.ones_like(label_hue)
  labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
  
  # cvt to BGR for display
  labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
  
  # set bg label to black
  labeled_img[label_hue==0] = 0
  
  sub_imgs = []
  for i in range(1, np.max(label_img) + 1):
    indices = np.where(label_img == i)
    coordinates = zip(indices[0], indices[1])
    borders = border(coordinates)
    if borders[1][0] - borders[0][0] + borders[1][1] - borders[1][1] < 50:
    continue
    sub_img = img[borders[0][0] : borders[1][0], borders[0][1] : borders[1][1]]
    j=0
    while j < len(sub_img):
    if sum(sub_img[j]) == 255*len(sub_img[j]):
      sub_img =  np.delete(sub_img, (j), axis=0)
    else:
      j+=1 
    sub_imgs.append(sub_img)
  
  phrases = []
  for sub_img in sub_imgs:
    h, w = sub_img.shape[:2]
    thresh = cv2.threshold(sub_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    sub_img = addspace(sub_img, thresh, w)
    #erosion = cv2.erode(sub_img, kernel, iterations = 1)
    #blur = cv2.GaussianBlur(erosion,(13,13),0)
    #img2 = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
    print("__")
    
    ind_phrases = []
    i_min = 0
    for i in range(len(sub_img)):
    s = sum(sub_img[i])
    if s == 255*len(sub_img[i]):
      i_max = i
      if i_max - i_min < 4:
      i_min = i+1
      continue
      ind_phrases.append((max(i_min-5, 0), min(i_max+5, len(sub_img)-1)))
      i_min = i+1
    for i in ind_phrases:
    sub_im = sub_img[i[0] : i[1], :]
    j=0
    while j < len(sub_im):
      if sum(sub_im[j]) == 255*len(sub_im[j]):
      sub_im =  np.delete(sub_im, (j), axis=0)
      else:
      j+=1 
    phrases.append(sub_im)
  
  sub_words = []
  for phrase in phrases:
    h, w = phrase.shape[:2]
    thresh = cv2.threshold(phrase, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    phrase=addspace_v(phrase, thresh, h)
  
    erosion = cv2.erode(phrase, kernel, iterations = 8)
    erosion = cv2.GaussianBlur(erosion,(13,13),0)
    tresh = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY_INV)[1]  # ensure binary
  
    phrase_label = connected_component_labelling(image_to_2d_bool_array(tresh), 4)
    label_hue = np.uint8(179*phrase_label/np.max(phrase_label))
    blank_ch = 255*np.ones_like(label_hue)
    phrased_label = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    phrased_label = cv2.cvtColor(phrased_label, cv2.COLOR_HSV2BGR)
  
    # set bg label to black
    phrased_label[label_hue==0] = 0
    for i in range(1, np.max(phrase_label) + 1):
    indices = np.where(phrase_label == i)
    coordinates = zip(indices[0], indices[1])
    borders = border(coordinates)
    if borders[1][0] - borders[0][0] + borders[1][1] - borders[1][1] < 5:
      continue
    sub_img = phrase[borders[0][0] : borders[1][0], borders[0][1] : borders[1][1]]
    h, w = sub_img.shape[:2]
    j=0
    sum_col = np.sum(sub_img, axis=0)
    while j < len(sum_col):
      if sum_col[j] == 255*h:
      sub_img =  np.delete(sub_img, (j), axis=1)
      sum_col =  np.delete(sum_col, (j), axis=0)
      else:
      j+=1
    sub_words.append(sub_img)
  return sub_words
    

  
