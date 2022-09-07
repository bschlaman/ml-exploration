from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import random
from colorama import Fore
from utils.math import Vector2D
from utils.helpers import data_print


def update_plot(vector):
    global pos_fill
    global neg_fill
    # update w vector
    w.add(vector)
    quiver.set_UVC(w.x, w.y)
    # update hyperplane
    w_vector_slope_rot = -w.x / w.y  # swap w.x, w.y; w.x *= -1
    y = w_vector_slope_rot * (x - gvo_vec.x) + gvo_vec.y
    hyperplane_plot.set_data(x, y)
    # update section fills
    pos_fill.remove()
    neg_fill.remove()
    pos_fill = ax.fill_between(
        x, y.tolist(), PLOT_VIEWPORT[int(w.y > 0)], facecolor="b", alpha=0.2
    )
    neg_fill = ax.fill_between(
        x, y.tolist(), PLOT_VIEWPORT[int(w.y < 0)], facecolor="r", alpha=0.2
    )
    # update text
    text.set_text(f"magnitude of w: {round(w.magnitude(), 5)}")
    ax.figure.canvas.draw()


class FeaturePlotter:
    def __init__(self, plot, event_key):
        self.plot = plot
        self.event_key = event_key
        self.points = []
        self.cid = plot.figure.canvas.mpl_connect("button_press_event", self)

    def __call__(self, event):
        if event.key != self.event_key:
            return
        self.points.append((event.xdata, event.ydata))
        self.plot.set_offsets(self.points)
        offset_event_data = event.xdata - gvo_vec.x, event.ydata - gvo_vec.y
        update_plot(Vector2D(*offset_event_data))


GLOBAL_VECTOR_OFFSET = (0.4, 0.4)
gvo_vec = Vector2D(*GLOBAL_VECTOR_OFFSET)
PLOT_VIEWPORT = (-0.5, 1)

# initialize the figure, plots, and feature plotters
fig, ax = plt.subplots()
ax.set_title("Perceptron Feature Plotter\n(click to add points, hold shift for red)")
ax.set_xlim(*PLOT_VIEWPORT)
ax.set_ylim(*PLOT_VIEWPORT)
ax.set_aspect("equal")
ax.autoscale(False)
plot1 = ax.scatter([], [], color="b", marker="o")
plot2 = ax.scatter([], [], color="r", marker="o")
fp1 = FeaturePlotter(plot1, None)
fp2 = FeaturePlotter(plot2, "shift")
# reusable linear space
x = np.linspace(*PLOT_VIEWPORT)
circle = Circle((0, 0), 0.03, fill=False, color="g", label="test point")
ax.add_patch(circle)


# w vector
w = Vector2D(11.8, 1.9)
w.normalize()
w.scale(0.5)
quiver = ax.quiver(
    gvo_vec.x, gvo_vec.y, w.x, w.y, angles="xy", scale_units="xy", scale=1
)

# hyperplane
w_vector_slope_rot = -w.x / w.y  # swap w.x, w.y; w.x *= -1
y = w_vector_slope_rot * (x - gvo_vec.x) + gvo_vec.y
(hyperplane_plot,) = ax.plot(x, y, ":g")

# section filling
pos_fill = ax.fill_between(
    x, y.tolist(), PLOT_VIEWPORT[int(w.y > 0)], facecolor="b", alpha=0.2
)
neg_fill = ax.fill_between(
    x, y.tolist(), PLOT_VIEWPORT[int(w.y < 0)], facecolor="r", alpha=0.2
)

# text
text = ax.text(0.02, 0.02, f"magnitude of w\u20D7: {round(w.magnitude(), 5)}")

# data will be of the form ((x, y), label)
data = []
data.extend([(Vector2D(x, y).sub(gvo_vec), 1) for x, y in plot1.get_offsets()])
data.extend([(Vector2D(x, y).sub(gvo_vec), -1) for x, y in plot2.get_offsets()])
random.shuffle(data)


def perceptron():
    misses = 0
    for vec, label in data:
        miss = False
        circle.center = vec.x + gvo_vec.x, vec.y + gvo_vec.y
        ax.figure.canvas.draw()
        if label * w.dot_product(vec) <= 0:
            misses += 1
            vec.scale(label)
            # w.add(vec)
            miss = True
        labeled_data = {
            "current w\u20D7": w,
            "miss count": misses,
            "testing feature": vec,
            "want": label,
            "got": w.dot_product(vec),
            "result": "miss! adjusting \u20D7" if miss else "hit",
        }
        for line in data_print(labeled_data, Fore.BLUE):
            print(line)
        input("press a key to continue...")
        if miss:
            update_plot(vec)
