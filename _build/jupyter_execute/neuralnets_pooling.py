#!/usr/bin/env python
# coding: utf-8

# ## Pooling 
# 
# Pooling is a standard operation in convolutional neural networks (CNNs) used to **downsample** feature maps. It reduces the spatial dimensions (height and width) while keeping the number of channels unchanged.
# 
# Pooling is not a learnable operation — it applies a fixed function (like max or average) over small regions of the input.
# 
# A pooling layer slides a small window (like a kernel) across the input and applies a function.
# Like convolution, pooling has:
# 
# - **Kernel size**: window size ($k\times k$)
# - **Stride**: step size (often equals the kernel size for downsampling)
# - **Padding**: rarely used in pooling, but available
# ### Max Pooling
# The example below shows a Max-Pooling with a $2\times 2$ window. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation
from IPython.display import HTML
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cm_0 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff","#a0c3ff"])
cm_1 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff", "#ffa1cf"])
cm_2 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff", "#f37726"])
#####################
# Array preparation
#####################

#input array
a = np.random.randint(1,10, size=(5,5))
# kernel
kernel = np.array([[ 0,-1, 0], [-1, 5,-1], [ 0,-1, 0]])

# visualization array (2 bigger in each direction)
va = np.zeros((a.shape[0], a.shape[1]), dtype=int)
va = a

#output array
res = np.zeros((a.shape[0]-kernel.shape[0],a.shape[1]-kernel.shape[1]))

#colorarray
va_color = np.zeros((a.shape[0], a.shape[1])) 

#####################
# Create inital plot
#####################
fig = plt.figure(figsize=(10,3))

def add_axes_inches(fig, rect):
    w,h = fig.get_size_inches()
    return fig.add_axes([rect[0]/(w+2), rect[1]/(h+2), rect[2]/(w+2), rect[3]/(h+2)])

axwidth = 3.
cellsize = axwidth/va.shape[1]
axheight = cellsize*va.shape[0]

ax_va  = add_axes_inches(fig, [cellsize, cellsize, axwidth, axheight])

ax_res = add_axes_inches(fig, [cellsize*3+axwidth,
                               2*cellsize, 
                               res.shape[1]*cellsize,  
                               res.shape[0]*cellsize])

im_va = ax_va.imshow(va_color, vmin=0., vmax=1.3, cmap=cm_0)
for i in range(va.shape[0]):
    for j in range(va.shape[1]):
        ax_va.text(j,i, va[i,j], va="center", ha="center")


im_res = ax_res.imshow(res, vmin=0, vmax=1.3, cmap=cm_1)
res_texts = []
for i in range(res.shape[0]):
    row = []
    for j in range(res.shape[1]):
        row.append(ax_res.text(j,i, "", va="center", ha="center"))
    res_texts.append(row)    


for ax  in [ax_va, ax_res]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.yaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.xaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.grid(color="k")

###############
# Animation
###############
def init():
    for row in res_texts:
        for text in row:
            text.set_text("")

def animate(ij):
    i,j=ij
    o = kernel.shape[1]//2
    # calculate result
    res_ij = (np.max(va[2*i-o+1:1+2*i+o+1, 1+2*j-o:1+2*j+o+1]))
    res_texts[i][j].set_text(res_ij)
    # make colors
    c = va_color.copy()
    c[1+2*i-o:1+2*i+o+1, 1+2*j-o:1+2*j+o+1] = 1.
    im_va.set_array(c)

    r = res.copy()
    r[i,j] = 1
    im_res.set_array(r)

i,j = np.indices(res.shape)
anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, 
                                         frames=zip(i.flat, j.flat), interval=400)
plt.close()
HTML(anim.to_jshtml())


# In Python, we can apply a max pooling layer as follows. 

# In[2]:


import torch
import torch.nn as nn

x = torch.tensor([[[[1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                    [9.,10.,11.,12.],
                    [13.,14.,15.,16.]]]])

pool = nn.MaxPool2d(kernel_size=2, stride=2)
y = pool(x)

print(y)  # shape: (1, 1, 2, 2)


# The effect of max-pooling is mainly the amplification of features. It compresses the input to a summary that contains the most prevalent features of the previous feature map.    
# ### Average Pooling
# Average pooling computes the average of the window. The example below shows the input on the right and the average pooling output on the right.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation
from IPython.display import HTML
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cm_0 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff","#a0c3ff"])
cm_1 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff", "#ffa1cf"])
cm_2 = LinearSegmentedColormap.from_list("mycmap", ["#ffffff", "#f37726"])
#####################
# Array preparation
#####################

#input array
a = np.random.randint(1,10, size=(5,5))
# kernel
kernel = np.array([[ 0,-1, 0], [-1, 5,-1], [ 0,-1, 0]])

# visualization array (2 bigger in each direction)
va = np.zeros((a.shape[0], a.shape[1]), dtype=int)
va = a

#output array
res = np.zeros((a.shape[0]-kernel.shape[0],a.shape[1]-kernel.shape[1]))

#colorarray
va_color = np.zeros((a.shape[0], a.shape[1])) 

#####################
# Create inital plot
#####################
fig = plt.figure(figsize=(10,3))

def add_axes_inches(fig, rect):
    w,h = fig.get_size_inches()
    return fig.add_axes([rect[0]/(w+2), rect[1]/(h+2), rect[2]/(w+2), rect[3]/(h+2)])

axwidth = 3.
cellsize = axwidth/va.shape[1]
axheight = cellsize*va.shape[0]

ax_va  = add_axes_inches(fig, [cellsize, cellsize, axwidth, axheight])

ax_res = add_axes_inches(fig, [cellsize*3+axwidth,
                               2*cellsize, 
                               res.shape[1]*cellsize,  
                               res.shape[0]*cellsize])

im_va = ax_va.imshow(va_color, vmin=0., vmax=1.3, cmap=cm_0)
for i in range(va.shape[0]):
    for j in range(va.shape[1]):
        ax_va.text(j,i, va[i,j], va="center", ha="center")


im_res = ax_res.imshow(res, vmin=0, vmax=1.3, cmap=cm_1)
res_texts = []
for i in range(res.shape[0]):
    row = []
    for j in range(res.shape[1]):
        row.append(ax_res.text(j,i, "", va="center", ha="center"))
    res_texts.append(row)    


for ax  in [ax_va, ax_res]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.yaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.xaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.grid(color="k")

###############
# Animation
###############
def init():
    for row in res_texts:
        for text in row:
            text.set_text("")

def animate(ij):
    i,j=ij
    o = kernel.shape[1]//2
    # calculate result
    res_ij = (np.mean(va[2*i-o+1:1+2*i+o+1, 1+2*j-o:1+2*j+o+1]))
    res_texts[i][j].set_text(f"{res_ij:.1f}")
    # make colors
    c = va_color.copy()
    c[1+2*i-o:1+2*i+o+1, 1+2*j-o:1+2*j+o+1] = 1.
    im_va.set_array(c)

    r = res.copy()
    r[i,j] = 1
    im_res.set_array(r)

i,j = np.indices(res.shape)
anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, 
                                         frames=zip(i.flat, j.flat), interval=400)
plt.close()
HTML(anim.to_jshtml())


# Average pooling smoothes the features in the input feature map.
