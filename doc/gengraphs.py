#!/usr/bin/python
from __future__ import division
from pylab import *

lightCol = array((1,1,0.8))
blueCol = array((0.75, 0.75, 1))
textSize = 'x-large'

def toneMap(Y, exposure, power):
    Y1 = exposure*Y**power
    return Y1 / (1.0 + Y1)

def formatAxes(ax):
    ax.patch.set_alpha(0)
    ax.patch.set_color((0,0,0))
    ax.patch.set_edgecolor(lightCol)
    def setCol(objs):
        for obj in objs:
            obj.set_color(lightCol)
    setCol(ax.get_xticklines() + ax.get_yticklines())
    setCol(ax.get_xticklabels() + ax.get_yticklabels())
    setCol(ax.spines.values())
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_size(textSize)

Y = linspace(0,4,200)
clf()
subplot(1,2,1)
plot(Y, toneMap(Y, 0.5, 1), color=blueCol)
plot(Y, toneMap(Y, 1, 1), color=lightCol)
plot(Y, toneMap(Y, 2, 1), color=blueCol)
xlabel('$Y$', color=lightCol, size=textSize)
ylabel("$Y'$", color=lightCol, size=textSize)
formatAxes(gca())

subplot(1,2,2)
plot(Y, toneMap(Y, 1, 0.5), color=blueCol)
plot(Y, toneMap(Y, 1, 1), color=lightCol)
plot(Y, toneMap(Y, 1, 2), color=blueCol)
xlabel('$Y$', color=lightCol, size=textSize)
ylabel("$Y'$", color=lightCol, size=textSize)
formatAxes(gca())

#figure(1, figsize=(8,4))
gcf().savefig('tone_map.pdf', transparent=True)


show()
