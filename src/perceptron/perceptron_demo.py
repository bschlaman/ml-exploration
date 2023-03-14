import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import random

import numpy as np
from colorama import Fore
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from utils.helpers import data_print
from utils.math.vectors import Vector


class FeaturePlotEventHandler:
    def __init__(self, event_modifier_key):
        self.event_modifier_key = event_modifier_key
        # there may be better alternatives to this class persisting the features
        self.features = []

    def __call__(self, event):
        if event.key != self.event_modifier_key:
            return
        if event.xdata < AX_VIEWPORT[0]:
            return
        if event.ydata < AX_VIEWPORT[0]:
            return
        if event.xdata > AX_VIEWPORT[1]:
            return
        if event.ydata > AX_VIEWPORT[1]:
            return
        self.features.append((event.xdata, event.ydata))
        render()


# globals
GLOBAL_VECTOR_OFFSET = (0.4, 0.4)
AX_VIEWPORT = (-0.5, 1)
gvo_vec = Vector(*GLOBAL_VECTOR_OFFSET)
# figure, axes, and event handlers
fig, ax = plt.subplots()
ind = np.linspace(*AX_VIEWPORT)  # reusable linear space
plot1 = ax.scatter([], [], color="b", marker="o")
plot2 = ax.scatter([], [], color="r", marker="o")
fpeh1 = FeaturePlotEventHandler(None)
fpeh2 = FeaturePlotEventHandler("shift")
cid1 = fig.canvas.mpl_connect("button_press_event", fpeh1)
cid2 = fig.canvas.mpl_connect("button_press_event", fpeh2)
# w vector
# TODO: make sure to get rid of gvo_vec mention
# quiver = ax.quiver(0, 0, gvo_vec.x, gvo_vec.y, angles="xy", scale_units="xy", scale=1)
adj_quiver = ax.quiver(
    0, 0, gvo_vec.x, gvo_vec.y, color="black", angles="xy", scale_units="xy", scale=1
)
# hyperplane
# (hyperplane_plot,) = ax.plot(ind, ind, ":g")
(adj_hyperplane_plot,) = ax.plot(ind, ind, ":")
# section filling
pos_fill = ax.fill_between(ind, 0, 0, facecolor="b", alpha=0.2)
neg_fill = ax.fill_between(ind, 0, 0, facecolor="r", alpha=0.2)
# test point indicator
circle = Circle((0, 0), 0.03, fill=False, color="g", label="test point")
# text
text = ax.text(AX_VIEWPORT[0] + 0.02, AX_VIEWPORT[0] + 0.02, "")
text2 = ax.text(AX_VIEWPORT[0] + 0.02, AX_VIEWPORT[0] + 0.10 * 1, "")
text3 = ax.text(AX_VIEWPORT[0] + 0.02, AX_VIEWPORT[0] + 0.10 * 2, "")
# text4 = ax.text(AX_VIEWPORT[0] + 0.02, AX_VIEWPORT[0] + 0.10*3, "")
# w vector
w = Vector(0, 0)
w3 = Vector(1, 1, 0)
w3.normalize()
w3.scale(0.3)


def init_ax():
    # initialize the figure, plots, and feature plotters
    ax.set_xlim(*AX_VIEWPORT)
    ax.set_ylim(*AX_VIEWPORT)
    ax.set_aspect("equal")
    ax.autoscale(False)
    ax.set_title(
        "Perceptron Feature Plotter\n(click to add points, hold shift for red)"
    )
    ax.axhline(y=0, linewidth=0.1)
    ax.axvline(x=0, linewidth=0.1)
    ax.add_patch(circle)
    # create w vector

    # global w
    # w = Vector(11.8, 1.9)
    # w.normalize()
    # w.scale(0.5)

    global w3
    w3 = Vector(1.8, 1.9, -1.9)
    w3.normalize()
    w3.scale(0.5)


def calculate_normal_line(w: Vector, ind: np.ndarray) -> np.ndarray:
    slope = -w.x / w.y
    offset = -w.z / w.y
    return slope, offset, slope * ind + offset


def render():
    # update w
    # quiver.set_UVC(w.x, w.y)
    adj_quiver.set_UVC(w3.x, w3.y)

    # update features
    if fpeh1.features:
        plot1.set_offsets(fpeh1.features)
    if fpeh2.features:
        plot2.set_offsets(fpeh2.features)

    # update hyperplane
    # w_vector_slope_rot = -w.x / w.y  # swap w.x, w.y; w.x *= -1
    # # dep = w_vector_slope_rot * (ind - gvo_vec.x) + gvo_vec.y
    # dep = w_vector_slope_rot * ind
    # hyperplane_plot.set_data(ind, dep)

    slope, offset, adj_dep = calculate_normal_line(w3, ind)
    adj_hyperplane_plot.set_data(ind, adj_dep)

    # update section fills
    global pos_fill
    global neg_fill
    pos_fill.remove()
    neg_fill.remove()
    pos_fill = ax.fill_between(
        ind, adj_dep.tolist(), AX_VIEWPORT[int(w3.y > 0)], facecolor="b", alpha=0.2
    )
    neg_fill = ax.fill_between(
        ind, adj_dep.tolist(), AX_VIEWPORT[int(w3.y < 0)], facecolor="r", alpha=0.2
    )

    # update text
    text.set_text(f"magnitude of w\u20D73: {round(w3.magnitude(), 5)}")
    text2.set_text(f"slope: {slope}")
    text3.set_text(f"offset: {offset}")
    # text4.set_text(f"h/m: {'miss' if miss else 'hit'}")

    # redraw canvas
    ax.figure.canvas.draw()


init_ax()
plt.show()
render()
input("press enter")
print("START")
for x in range(1000):
    print(x)
    perceptron(False)

# claim 1: since hyperplane offset (in y direction w.l.o.g.) depends on w_y,
# w_z can be literally anything because I can find a w_y to compensate


import random
import time

from colorama import Fore

from utils.helpers import data_print

LABEL_SPACE = (-1, 1)

# data will be of the form ((x, y), laassociationbel)
data = []
data.extend([(Vector(x, y), LABEL_SPACE[1]) for x, y in plot1.get_offsets()])
data.extend([(Vector(x, y), LABEL_SPACE[0]) for x, y in plot2.get_offsets()])
random.shuffle(data)

data3d = []
data3d.extend([(Vector(x, y, 1), LABEL_SPACE[1]) for x, y in plot1.get_offsets()])
data3d.extend([(Vector(x, y, 1), LABEL_SPACE[0]) for x, y in plot2.get_offsets()])
random.shuffle(data3d)

# w3 = Vector(0,0,0)


def perceptron(human: bool):
    global w3
    misses = 0
    for vec, label in data3d:
        miss = False
        circle.center = vec.x, vec.y  # + gvo_vec.x, vec.y + gvo_vec.y
        ax.figure.canvas.draw()

        if human:
            input(f"about to test feature: {vec}")
        else:
            time.sleep(0.1)

        if label * w3.dot_product(vec) <= 0:
            misses += 1
            miss = True
        labeled_data = {
            "current w\u20D7": w3,
            "miss count": misses,
            "testing feature": vec,
            "want": label,
            "got": w3.dot_product(vec),
            "result": "miss! adjusting w\u20D7" if miss else "hit",
        }
        for line in data_print(labeled_data, Fore.BLUE):
            print(line)

        if human:
            input(f"done testing: {'miss' if miss else 'hit'}")
        else:
            time.sleep(0.1)

        if miss:
            tmp_vec = Vector(vec.x, vec.y, vec.z)
            tmp_vec.scale(label)
            print(tmp_vec.x, tmp_vec.y, label)
            w3.add(tmp_vec)
            render()
