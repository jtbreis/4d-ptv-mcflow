# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:04:24 2020

@author: ferran6am
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.colors as mcolors

#%% FIGURE COLORS
# Create a colormap with an existing colormap
def truncate_colormap(cmap, a=0.0, b=1.0, minval=0.0, maxval=1.0, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def discrete_cmap(n, n1=0, n2=1, base_cmap=None):
    """Create an n-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(n1, n2, n))
    return color_list

def concatenate_cmap(N, cmaps):
    colors = []
    for n,cmap in zip(N,cmaps):
        dis_cmap = discrete_cmap(n, 0.3, 0.8, cmap)
        colors.append(dis_cmap)
    final_colors = np.vstack(colors)
    
    return final_colors

# =============================================================================
# Get a color proportional to a value from a cmap.
#       v = value to get the color
#       vmin, vmax = min and max values for normalization
#       base_cmap = cmap
# usage: z = structure_f(u, n,maxlag)
# =============================================================================
def get_color_from_cmap(val, vmin, vmax, base_cmap=None, a=0.0, b=1.0):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    base = plt.cm.get_cmap(base_cmap)
    trunc_base = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=base.name, a=a, b=b),
        base(np.linspace(a, b, 100)))
    color = trunc_base(norm(val))
    return color

def discrete_proportional_cmap(phi_v, pmin, pmax, base_cmap, a=0.0, b=1.0):
    color_list = []
    for p in phi_v:
        if base_cmap in CBLDcm.keys() :
            color = get_color_from_cmap(float(p), pmin, pmax, CBLDcm[base_cmap], a=a, b=b)
        elif base_cmap in CB2cm.keys() :
            color = get_color_from_cmap(float(p), pmin, pmax, CB2cm[base_cmap], a=a, b=b)
        else:
            color = base_cmap
        color_list.append(color)
    return color_list


# This very cool piece of Python code was taken from github.com/nesanders/colorblind-colormap
# and slightly modified. It provides colour blind friendly colour maps according to
# Wong, B. (2011) Nature Methods 8:441

CBcdict={
    #'Bl':(0,0,0),
    #'Or':(.9,.6,0),
    #'SB':(.35,.7,.9),
    #'bG':(0,.6,.5),
    #'Ye':(.95,.9,.25),
    #'Bu':(0,.45,.7),
    #'Ve':(.8,.4,0),
    #'rP':(.8,.6,.7),
    # updated acc. to http://mkweb.bcgsc.ca/colorblind/img/colorblindness.palettes.trivial.png
    'Bl':(0.00000, 0.00000, 0.000000), # Black            '#000000'
    'Or':(0.90196, 0.62353, 0.000000), # Orange           '#E69F00'
    'SB':(0.33725, 0.70588, 0.913730), # Sky Blue         '#56B4E9'
    'bG':(0.00000, 0.61961, 0.450980), # bluish Green     '#009E73'
    'Ye':(0.94118, 0.89412, 0.258820), # Yellow           '#F0E442'
    'Bu':(0.00000, 0.44706, 0.698040), # Blue             '#0072B2'
    'Ve':(0.83529, 0.36863, 0.000000), # Vermillion       '#D55E00' 
    'rP':(0.80000, 0.47451, 0.654900), # reddish Purple   '#CC79A7'
    'Pu':(0.40000, 0.00000, 0.400000),
    'Re':(0.65000, 0.18000, 0.050000),
    'Gr':(0.50000, 0.70000, 0.200000)
}

# Single color gradient maps
def lighter(colors):
    li = lambda x: x+.5*(1-x)
    return (li(colors[0]),li(colors[1]),li(colors[2]))

def darker(colors):
    return (.5*colors[0],.5*colors[1],.5*colors[2])

CBLDcm={}
for key in CBcdict:
    CBLDcm[key]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key,[lighter(CBcdict[key]),darker(CBcdict[key])])

# Two color gradient maps
CB2cm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CB2cm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],CBcdict[key2]])

# Two color gradient maps with white in the middle
CBWcm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CBWcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(1,1,1),CBcdict[key2]])

# Two color gradient maps with Black in the middle
CBBcm={}
for key in CBcdict:
    for key2 in CBcdict:
        if key!=key2: CBBcm[key+key2]=matplotlib.colors.LinearSegmentedColormap.from_list('CMcm'+key+key2,[CBcdict[key],(0,0,0),CBcdict[key2]])


#%% DEFINE GRADIANT CMAP FROM A COLOR
colordict={
    'Bl' : (0,0,0),
    'Or' : (.9,.6,0),
    'SB' : (.35,.7,.9),
    'bG' : (0,.6,.5),
    'Ye' : (.95,.9,.25),
    'Bu' : (0,.45,.7),
    'Ve' : (.8,.4,0),
    'rP' : (.8,.6,.7)}

def gradient_cmap_from_color(color):
    def lighter(colors):
        li = lambda x: x+.2*(1-x)
        return (li(colors[0]), li(colors[1]), li(colors[2]))
    
    def darker(colors):
        return (.7*colors[0], .7*colors[1], .7*colors[2])

   
    CMgrad={}
    for key in colordict:
        CMgrad[key]=mcolors.LinearSegmentedColormap.from_list('CMgrad' + key,
                            [lighter(colordict[key]),darker(colordict[key])])

    try:
        cmap = CMgrad[color]
    except KeyError:
        print('Color not available')
        cmap = None
    return cmap

#%% FIGURE SIZES
# =============================================================================
# Function to choose the size of matplotlib figures
# 
# =============================================================================
def set_size(width, fraction=1, subplots=(1, 1), extra_space=0.0):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 455.24411
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'powerpoint':
        width_pt = 750  # 750 correspond to 8 inches
    else:
        width_pt = width
        
    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    # add extra space in the bottom
    fig_height_in = fig_height_in + fig_height_in*extra_space
    
    return fig_width_in, fig_height_in



def move_axes(ax, fig, subplot_spec=111):
      """Move an Axes object from a figure to a new pyplot managed Figure in
      the specified subplot."""

      # get a reference to the old figure context so we can release it
      old_fig = ax.figure

      # remove the Axes from it's original Figure context
      ax.remove()

      # set the pointer from the Axes to the new figure
      ax.figure = fig

      # add the Axes to the registry of axes for the figure
      fig.axes.append(ax)
      # twice, I don't know why...
      fig.add_axes(ax, subplot_spec)

      # then to actually show the Axes in the new figure we have to make
      # a subplot with the positions etc for the Axes to go, so make a
      # subplot which will have a dummy Axes
      dummy_ax = fig.add_subplot(subplot_spec)

      # # then copy the relevant data from the dummy to the ax
      ax.set_position(dummy_ax.get_position())

      # # then remove the dummy
      dummy_ax.remove()

      # close the figure the original axis was bound to
      plt.close(old_fig)


### TESTS COLORMAP


# ## Plot gradient bars for each palette
# ## Using recipe from http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps
# plt.rc('text', usetex=False)
# a=np.outer(np.arange(0,1,0.01),np.ones(10))

# names=['CB2cm','CBWcm','CBBcm','CBLDcm']
# CBcm=[CB2cm,CBWcm,CBBcm,CBLDcm]
# for i in range(len(names)):
#     plt.figure(figsize=(10,5))
#     plt.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
#     maps=sorted(CBcm[i].keys())
#     l=len(maps)+1
#     for j in range(len(maps)):
#         m=maps[j]
#         plt.subplot(1,l,j+1)
#         plt.axis("off")
#         plt.imshow(a,aspect='auto',cmap=CBcm[i][m],origin="lower")
#         plt.text(0,110,m,rotation=90,fontsize=10)
#     fname='test_palette_'+names[i]+'.png'
#     plt.savefig(fname)
#     plt.close()