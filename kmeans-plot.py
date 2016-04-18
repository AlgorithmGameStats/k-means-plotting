"""
Calculate KMeans from the data input
Input:
------
date := Game stats data input. file with 1 JSON object per line.
k := number of centroids
class_name := name of KMeans cluster
save := If we should save the resulting cluster data to a file (./clusters/class_name.json), in JSON format (defaults to True)
transmit := If we should transmit the results to the server (defaults to False)

Example:

Calculate KMeans cluster with k=2, class='killer', save and transmit for all levels (1-4)
python k-means-calculator.py -st -k 2 -c "killer" ./data/data.20160414.txt
"""
import sys, json, argparse, random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Standard Color list
COLORS_LIST = ['b', 'g', 'c', 'm', 'y', 'k']
# Standard Marker list
MARKERS_LIST = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']

def new_color_list():
  """
  Get a new random list of colors.
  """
  new_list = COLORS_LIST[:]
  random.shuffle(new_list)
  return new_list

def new_markers_list():
  """
  Get a new random list of markers
  """
  new_list = MARKERS_LIST[:]
  random.shuffle(new_list)
  return new_list

def plot_kmeans(file_name=None):
  """
  Calculate KMeans....
  """
  if (not file_name):
    # If no 'file_name' or no 'class_name', we error
    sys.exit(1)
  
  final = ""
  with open(file_name, 'rb') as f:
    final += f.read()

  # Load the json cluster data
  kmeans = json.loads(final)

  # find the 'x, y, z' value per cluster
  x = [ list() for i in range(kmeans['k']) ]
  y = [ list() for i in range(kmeans['k']) ]
  z = [ list() for i in range(kmeans['k']) ]
  for i in range(kmeans['k']):
    for item in kmeans['clusters'][i]:
      x[i].append(item[0])
      y[i].append(item[1])
      z[i].append(item[2])


  # find the 'x, y, z' value per centroid
  c_x = list()
  c_y = list()
  c_z = list()
  for item in kmeans['centroids']:
    c_x.append(item[0])
    c_y.append(item[1])
    c_z.append(item[2])

  # Put the cluster data into a color plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # Get a random list for colors and markers
  colors = new_color_list()
  markers = new_markers_list()
  cm = list()

  # Plot all clusters
  for i in range(kmeans['k']):
    # If color/markers are gone, we get a new random list
    # We do this to recycle the list
    if len(colors) <= 0:
      colors = new_color_list()
    if len(markers) <= 0:
      markers = new_markers_list()

    # Get a color/marker combination
    c = colors.pop()
    m = markers.pop()
    cm.append( '{0}{1}'.format(c, m) )

    # Plot this cluster
    ax.plot(
      x[i], y[i], z[i], cm[i]
    )

  # Plotting Centroids
  ax.plot(
    c_x, c_y, c_z, 'or',
    markersize=8
  )

  # Set-up axes
  ax.set_xlabel('Time')
  ax.set_ylabel('Coins')
  ax.set_zlabel('Kills')

  # Show plot
  plt.show()


if __name__ == '__main__':
  
  # Instantiate argument parser
  parser = argparse.ArgumentParser(description='Plot KMeans from input cluster data')
  parser.add_argument('file', help='path to the file containing cluster data', default='')

  # Get/parse arguments
  args = parser.parse_args()

  plot_kmeans(file_name=args.file)
