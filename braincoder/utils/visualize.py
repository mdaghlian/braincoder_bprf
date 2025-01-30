import numpy as np


def show_animation(images, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML

    fig = plt.figure(figsize=(6, 4))
    ims = []

    if vmin is None:
        vmin = np.percentile(images, 5)

    if vmax is None:
        vmax = np.percentile(images, 95)

    for t in range(len(images)):
        im = plt.imshow(images[t], animated=True, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig, ims, interval=150, blit=True, repeat_delay=1500)

    return HTML(ani.to_html5_video())



import matplotlib.pyplot as plt
import pandas as pd
from .stats import get_rsq
def quick_plot(model, parameters, data=None, color=None):
    if isinstance(parameters, pd.Series):
        parameters = parameters.to_frame().T
    elif isinstance(parameters, dict):
        parameters = pd.DataFrame(parameters)
    elif isinstance(parameters, pd.DataFrame):
        pass
    # print(parameters)
    rf = model.get_rf(
        parameters=parameters,
    ).reshape(model.n_x, model.n_y).T

    pred = model.predict(
        paradigm=model.paradigm,
        parameters=parameters, 
    )
    
    # Matplotlib figure
    fig, ax = plt.subplots(
        1, 2, figsize=(10, 5),
        width_ratios=[1, 4]
        )
    ax[0].imshow(rf, cmap='gray', origin='lower')
    ax[0].axis('off')
    ax[0].set_title('Receptive Field')
    if data is not None:
        ax[1].plot(data.T, '--k', label='Data', alpha=0.5, )
    ax[1].plot(pred, '-', color=color, label='Prediction', alpha=0.5,)
    ax[1].legend()
    return 




def edit_pair_plot(axes, **kwargs):
    """
    Adds vertical and horizontal lines to each subplot in a Seaborn pairplot.

    Annoyingly the x and y axis aren't consistently named in pairplot...

    Parameters:
    - pairplot (sns.PairGrid): The Seaborn pairplot to modify.
    - lines_dict (dict): Dictionary where keys are column names in the dataset, 
                         and values are the x or y values where lines should be drawn.
    """
    lines_dict = kwargs.pop('lines_dict', None)    
    lim_dict = kwargs.pop('lim_dict', None)
    n_r, n_c = axes.shape
    x_labels = [['']*n_c for _ in range(n_r)]
    y_labels = [['']*n_c for _ in range(n_r)]
    # Sometimes each row has a label...    
    for iR,axR in enumerate(axes):
        for iC,ax in enumerate(axR):
            if ax is None:
                continue
            x_labels[iR][iC] = axes[iR,iC].get_xlabel()
            y_labels[iR][iC] = axes[iR,iC].get_ylabel()

    # Now lets consolidate
    x_by_col = []
    for iC in range(n_c):
        v = ''
        i = 0 
        while v == '':
            v = x_labels[i][iC]
            i += 1
        x_by_col.append(v)

    y_by_row = []
    for iR in range(n_r):
        v = ''
        i = 0 
        while v == '':
            v = y_labels[iR][i]
            i += 1
        y_by_row.append(v)
    if lines_dict is not None:
        label_list = list(lines_dict.keys())
        for iR,vR in enumerate(y_by_row):
            for iC,vC in enumerate(x_by_col):
                if axes[iR,iC] is None:
                    continue

                x_label = axes[iR,iC].get_xlabel()
                y_label = axes[iR,iC].get_ylabel()
                if (x_label in label_list) & (y_label in label_list):
                    # Both in there? then it must be a scatter plot 
                    # Plot them both
                    axes[iR,iC].axvline(x=lines_dict[x_label], **kwargs)
                    axes[iR,iC].axhline(y=lines_dict[y_label], **kwargs)
                else:
                    # Missing labels... must be a histogram
                    if vR in label_list:
                        axes[iR,iC].axvline(x=lines_dict[vR], **kwargs)
    if lim_dict is not None:
        label_list = list(lim_dict.keys())
        for iR,vR in enumerate(y_by_row):
            for iC,vC in enumerate(x_by_col):
                if axes[iR,iC] is None:
                    continue

                x_label = axes[iR,iC].get_xlabel()
                y_label = axes[iR,iC].get_ylabel()
                if (x_label in label_list) & (y_label in label_list):
                    # Both in there? then it must be a scatter plot 
                    # Plot them both
                    axes[iR,iC].set_xlim(lim_dict[x_label])
                    axes[iR,iC].set_ylim(lim_dict[y_label])
                else:
                    # Missing labels... must be a histogram
                    if vR in label_list:
                        axes[iR,iC].set_xlim(lim_dict[vR])

    return 

import matplotlib as mpl
def dag_update_fig_fontsize(fig, new_font_size, **kwargs):
    '''dag_update_fig_fontsize
    Description:
        Update the font size of a figure
    Input:
        fig             matplotlib figure
        new_font_size   int/float             
    Return:
        None        
    '''
    fig_kids = fig.get_children() # Get the children of the figure, i.e., the axes
    for i_kid in fig_kids: # Loop through the children
        if isinstance(i_kid, mpl.axes.Axes): # If the child is an axes, update the font size of the axes
            dag_update_ax_fontsize(i_kid, new_font_size, **kwargs)
        elif isinstance(i_kid, mpl.text.Text): # If the child is a text, update the font size of the text
            i_kid.set_fontsize(new_font_size)            

def dag_update_ax_fontsize(ax, new_font_size, include=None, do_extra_search=True):
    '''dag_update_ax_fontsize
    Description:
        Update the font size of am axes
    Input:
        ax              matplotlib axes
        new_font_size   int/float
        *Optional*
        include         list of strings     What to update the font size of. 
                                            Options are: 'title', 'xlabel', 'ylabel', 'xticks','yticks'
        do_extra_search bool                Whether to search through the children of the axes, and update the font size of any text
    Return:
        None        
    '''
    if include is None: # If no include is specified, update all the text       
        include = ['title', 'xlabel', 'ylabel', 'xticks','yticks']
    if not isinstance(include, list): # If include is not a list, make it a list
        include = [include]
    incl_list = []
    for i in include: # Loop through the include list, and add the relevant text to the list
        if i=='title': 
            incl_list += [ax.title]
        elif i=='xlabel':
            incl_list += [ax.xaxis.label]
        elif i=='ylabel':
            incl_list += [ax.yaxis.label]
        elif i=='xticks':
            incl_list += ax.get_xticklabels()
        elif i=='yticks':
            incl_list += ax.get_yticklabels()
        elif i=='legend':
            incl_list += ax.get_legend().get_texts()

    for item in (incl_list): # Loop through the text, and update the font size
        item.set_fontsize(new_font_size)        
    if do_extra_search:
        for item in ax.get_children():
            if isinstance(item, mpl.legend.Legend):
                texts = item.get_texts()
                if not isinstance(texts, list):
                    texts = [texts]
                for i_txt in texts:
                    i_txt.set_fontsize(new_font_size)
            elif isinstance(item, mpl.text.Text):
                item.set_fontsize(new_font_size)                
