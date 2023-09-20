import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import TimedAnimation
from matplotlib.animation import PillowWriter

array = np.load('mpc_decentralized_arrays.npz')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5,5)

boat_radius = 0.15
boat_side = np.sqrt((boat_radius**2)/2)
 
def animate(i):
    ax.clear()
    x1 = array['x1'][i]
    y1 = array['y1'][i]
    #circle1 = plt.Circle((y1,x1),0.15, fc='b',ec="k")
    #ax.add_patch(circle1)
    rect1 = plt.Rectangle((y1-boat_side,x1-boat_side),2*boat_side,2*boat_side, fc='b',ec="k")
    ax.add_patch(rect1)
    x2 = array['x2'][i]
    y2 = array['y2'][i]
    #circle2 = plt.Circle((y2,x2),0.15, fc='g',ec="k")
    #ax.add_patch(circle2)
    rect2 = plt.Rectangle((y2-boat_side,x2-boat_side),2*boat_side,2*boat_side, fc='g',ec="k")
    ax.add_patch(rect2)
    x3 = array['x3'][i]
    y3 = array['y3'][i]
    #circle3 = plt.Circle((y3,x3),0.15, fc='r',ec="k")
    #ax.add_patch(circle3)
    rect3 = plt.Rectangle((y3-boat_side,x3-boat_side),2*boat_side,2*boat_side, fc='r',ec="k")
    ax.add_patch(rect3)
    x4 = array['x4'][i]
    y4 = array['y4'][i]
    #circle4 = plt.Circle((y4,x4),0.15, fc='c',ec="k")
    #ax.add_patch(circle4)
    rect4 = plt.Rectangle((y4-boat_side,x4-boat_side),2*boat_side,2*boat_side, fc='c',ec="k")
    ax.add_patch(rect4)
    x5 = array['x5'][i]
    y5 = array['y5'][i]
    #circle5 = plt.Circle((y5,x5),0.15, fc='m',ec="k")
    #ax.add_patch(circle5)
    rect5 = plt.Rectangle((y5-boat_side,x5-boat_side),2*boat_side,2*boat_side, fc='m',ec="k")
    ax.add_patch(rect5)
    x6 = array['x6'][i]
    y6 = array['y6'][i]
    #circle6 = plt.Circle((y6,x6),0.15, fc='y',ec="k")
    #ax.add_patch(circle6)
    rect6 = plt.Rectangle((y6-boat_side,x6-boat_side),2*boat_side,2*boat_side, fc='y',ec="k")
    ax.add_patch(rect6)
    x7 = array['x7'][i]
    y7 = array['y7'][i]
    #circle7 = plt.Circle((y7,x7),0.15, fc='k',ec="k")
    #ax.add_patch(circle7)
    rect7 = plt.Rectangle((y7-boat_side,x7-boat_side),2*boat_side,2*boat_side, fc='k',ec="k")
    ax.add_patch(rect7)
    x8 = array['x8'][i]
    y8 = array['y8'][i]
    #circle8 = plt.Circle((y8,x8),0.15, fc='w',ec="k")
    #ax.add_patch(circle8)
    rect8 = plt.Rectangle((y8-boat_side,x8-boat_side),2*boat_side,2*boat_side, fc='w',ec="k")
    ax.add_patch(rect8)
    ax.set_xlim([0,3])
    ax.set_ylim([0,3])

ani = FuncAnimation(fig, animate, frames=len(array['x1']),interval=100,repeat=False)
plt.close()

from matplotlib.animation import PillowWriter
# Save the animation as an animated GIF
ani.save("formation.gif", dpi=300, writer=PillowWriter(fps=10))