"""
Class implementing KMeans for our Game Server
This is done to show the step-by-step process of KMeans
"""
import os, itertools, random, time, json, argparse
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1) randomnly assign 3 centroids
# 2) add data and determine distance from each data point to closest centroid
# 3) shift new centroids by averaging data points of each cluster 
# 4) convergence stops when centroids approach fixed values (i.e. centroid-data distance minimizes) 

# player 1: Speed Runners(time=0, level=all levels, coins=0, murders=0); player 2: Achievers (coins=~, levels, time=~, murders=0); player 3: Killers (murders=~, levels, time=~, coins=0)

# k= # of clusters (3 : one for each player profile)
# c= # of initial clusters 

# Standard Color list
COLORS_LIST = ['b', 'g', 'c', 'm', 'y', 'k']
# Standard Marker list
MARKERS_LIST = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

class KMeans(object):
  """
  Defines the Kmeans class for our game server
  Each point is: [time, coins, murders]
  'time' == time take to beat stage
  'coins' == percentage of coins collected in stage
  'murders' == number of cubes killed on the stage
  """


  def __init__(self, class_name, k=1, max_iterations=300, canvas=None, sleep_time=0, interactive=True, log=False):
    
    # Store k-means variables and constants
    self.__class_name = class_name
    self.__k = k
    self.__cm = list()
    self.__max_iterations = max_iterations
    self.__log_enabled = log
    self.__colors = self.new_color_list()
    self.__markers = self.new_markers_list()
    self.__canvas = canvas
    self.__sleep_time = sleep_time
    self.__interactive = interactive

    # predefine random centroids
    start = time.clock()
    self.centroids = self.__random_centroid(k)
    self.__log ( 'Initial random centroids generation Time: {0}'.format( (time.clock() - start) ) )

    # predefine empty clusters
    self.clusters = [ [] for i in range(k) ]


  def put(self, items):
    """
    Put new entries in the dataSet,
    Recalculate KMeans

    Parameters
      ----------
      items : array_ofarrays
        [item, item, item, ....]
        item == [time, coins, murders]
    """

    # Add item to the last cluster before we begin
    start = time.clock()
    self.clusters[self.__k-1].extend(items)
    self.__log ( 'Adding new data Time: {0}'.format( (time.clock() - start) ) )

    # Create list of old/new centroids and clusters before re-calculation
    start = time.clock()
    old_centroids = [ [] for i in range(self.__k) ]
    new_centroids = self.__random_centroid(self.__k)
    new_clusters = [ [] for i in range(self.__k) ]
    self.__log ( 'New temp centroids and clusters generation Time: {0}'.format( (time.clock() - start) ) )
    
    # Keep track of iterations of moving centroids
    iterations = 0
    calc_times = list()

    ### Before we begin, we draw raw data...
    x = list()
    y = list()
    z = list()
    for item in data:
      x.append(item[0])
      y.append(item[1])
      z.append(item[2])

    # find the 'x, y, z' value per centroid
    c_x = list()
    c_y = list()
    c_z = list()
    for item in new_centroids:
      c_x.append(item[0])
      c_y.append(item[1])
      c_z.append(item[2])
      
    # Plot the raw data as black +
    self.__canvas.plot(x, y, z, '+k')
    self.__canvas.plot(c_x, c_y, c_z, 'or', markersize=8)
    plt.pause(0.05)
    raw_input("Press enter to begin... ")
    #################

    # Continue converging data points until __max_iterations is reached
    self.__log ("*********************************")
    while not (self.__should_stop(old_centroids, new_centroids, iterations)):
      iterations += 1 # increase iterations count
      self.__log("Iteration: {0}".format(iterations))
      start = time.clock()
      new_clusters = [ [] for i in range(self.__k) ] # Empty the 'new_clusters' for this iteration
      old_centroids = new_centroids[:]  # Copies new_centroids into old_centroids
      self.__log ( 'Copy new centroids to old centroids Time: {0}'.format( (time.clock() - start) ) )

      # chain together all clusters (old ones) instead of copying, 
      # this way we don't waste soo much memory
      start = time.clock()
      data_set = itertools.chain( *(self.clusters[i] for i in range(self.__k)) )
      self.__log ( 'Chaining data set Time: {0}'.format( (time.clock() - start) ) )
      
      # For each item in the data_set, we: 
      # 1) find the closest centroid to the data item
      # 2) add data item to the cluster it belongs to
      # 3) recalculate the centroid for that cluster
      start = time.clock()
      for item in data_set:
        centroid_index = self.__closest_centroid(item, new_centroids) 
        new_clusters[centroid_index].append(item)
      
      # recalculate all centroids
      for centroid_index in range(self.__k):
        new_centroids[centroid_index] = self.__recalculate(new_clusters[centroid_index])

      calc_time = (time.clock() - start)
      calc_times.append(calc_time)
      self.__log ( 'New centroids calculation Time: {0}'.format( calc_time ) )
      self.__log ( '~~~~ centroids ~~~~')
      self.__log ('Old: {0}'.format(old_centroids))
      self.__log ('New: {0}'.format(new_centroids))
      self.__log ( '~~~~~~~~~~~~~~~~~~~')
      self.__plot(centroids=new_centroids, clusters=new_clusters)
      self.__log ("+++++++++++++++++++++++++++++++++")

      # If interactive, use "enter" to continue, else use timers
      if self.__interactive:
        raw_input("Press enter when done with iteration {0}... ".format(iterations))
      else:
        time.sleep(self.__sleep_time)

    # Once we have converged (or reached max_iterations)
    # Re-assign centroids and clusters to the new values
    self.centroids = new_centroids
    self.clusters = new_clusters
    self.__log ( 'Total calculation Time: {0}'.format( sum(calc_times) ) )
    self.__log ("*********************************")

  def __random_centroid(self, k):
    """
    Generate a list of 'k' random centroids
    """
    return [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)] for i in range(k)]


  def __recalculate(self, cluster):
    """
    Recalculate centroid as the new mean of the cluster
    """
    return map(lambda x:sum(x)/float(len(x)), zip(*cluster))


  def __closest_centroid(self, item, centroids):
    """
    Return the index of the centroid the 'item' is closer to
    """
    return min([(i[0], sqrt(
      ((item[0] - centroids[i[0]][0])**2) + 
      ((item[1] - centroids[i[0]][1])**2) + 
      ((item[2] - centroids[i[0]][2])**2)
    )) for i in enumerate(centroids)], key=lambda t:t[1])[0]

  def __should_stop(self, old_centroids, centroids, iterations):
    """
    Verify if we should stop recalculating centroids
    """
    if iterations > self.__max_iterations:
      return True
    return old_centroids == centroids

  def dot_product(self, item):
    return [ sum([item[i] * self.centroids[j][i] for i in range(len(item))]) for j in range(len(self.centroids))]

  def k(self):
    """
    Return the value of 'k'
    """
    return self.__k

  def class_name(self):
    """
    Return the value of class_name
    """
    return self.__class_name

  # Logging Helper function
  def __log(self, txt):
    if self.__log_enabled:
      print(txt)

  def new_color_list(self):
    """
    Get a new random list of colors.
    """
    new_list = COLORS_LIST[:]
    random.shuffle(new_list)
    return new_list

  def new_markers_list(self):
    """
    Get a new random list of markers
    """
    new_list = MARKERS_LIST[:]
    random.shuffle(new_list)
    return new_list

  def __get_marker_color(self, index):
    if len(self.__cm) >= index:
      # If color/markers are gone, we get a new random list
      # We do this to recycle the list
      if len(self.__colors) <= 0:
        self.__colors = self.new_color_list()
      if len(self.__markers) <= 0:
        self.__markers = self.new_markers_list()

      # Get a color/marker combination
      c = self.__colors.pop()
      m = self.__markers.pop()
      self.__cm.append( '{0}{1}'.format(c, m) )
    return self.__cm[index]

  def __plot(self, centroids, clusters):
    """
    Plot this iteration.
    """
    
    # Clear old canvas
    if self.__canvas:
      self.__canvas.clear()
      # Set-up axes
      self.__canvas.set_xlabel('Time')
      self.__canvas.set_ylabel('Coins')
      self.__canvas.set_zlabel('Kills')

    # find the 'x, y, z' value per cluster
    x = [ list() for i in range(self.__k) ]
    y = [ list() for i in range(self.__k) ]
    z = [ list() for i in range(self.__k) ]
    for i in range(self.__k):
      for item in clusters[i]:
        x[i].append(item[0])
        y[i].append(item[1])
        z[i].append(item[2])

    # find the 'x, y, z' value per centroid
    c_x = list()
    c_y = list()
    c_z = list()
    for item in centroids:
      c_x.append(item[0])
      c_y.append(item[1])
      c_z.append(item[2])

    # Plot each cluster
    for i in range(self.__k):
      # Get the marker/color for this centroid
      mc = self.__get_marker_color(i)
      # Plot this cluster
      self.__canvas.plot(x[i], y[i], z[i], mc)

    # Plot the centroids
    self.__canvas.plot(c_x, c_y, c_z, 'or', markersize=8)

    # Show window with new plot...
    plt.pause(0.05)



# Run our plot
if __name__ == '__main__':

  # Instantiate argument parser
  parser = argparse.ArgumentParser(description='Calculate KMeans in a graphical way')
  parser.add_argument('-s', '--sleep', help='Time in seconds to wait before new re-draw', type=int, default=5)
  parser.add_argument('-i', '--interactive', help='If the graph should be interactive or driven by time', action='store_true', default=False)
  parser.add_argument('-m', '--max_iterations', help='Max convergance iterations', type=int, default=5)
  parser.add_argument('file', help='path to the file containing json data from the game server', default='')

  # Get/parse arguments
  args = parser.parse_args()
  file_name = args.file

  # Open the file, assuming that we have a json object per line
  data = list()
  with open(file_name, 'rb') as f:
    for line in f:
      stat = json.loads(line)
      item = [
        float(stat['time_used']) / float(stat['time_total']), 
        float(stat['coins_collected']) / float(stat['coins_total']), 
        float(stat['enemies_killed']) / float(stat['enemies_total'])
      ]
      data.append(item)

  # Create plot figure
  fig = plt.figure()
  # Create new Canvas
  ax = fig.add_subplot(111, projection='3d')
  # Set-up axes
  ax.set_xlabel('Time')
  ax.set_ylabel('Coins')
  ax.set_zlabel('Kills')

  # enable interactive plotting
  plt.ion()

  k = KMeans(
    class_name="no idea", 
    k=3, 
    max_iterations=args.max_iterations, 
    canvas=ax, 
    sleep_time=args.sleep, 
    interactive=args.interactive, 
    log=False
  )

  # Put the data and start plotting
  k.put(data)

  # We force a "press enter to exit..."
  raw_input("Done, press enter to exit... ")
