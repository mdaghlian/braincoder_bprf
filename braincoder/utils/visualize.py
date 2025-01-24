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
def quick_plot(model, parameters, data=None):
    if isinstance(parameters, pd.Series):
        parameters = parameters.to_frame().T
    elif isinstance(parameters, dict):
        parameters = pd.DataFrame(parameters)
    elif isinstance(parameters, pd.DataFrame):
        pass

    rf = model.get_rf(
        parameters=parameters,
    ).reshape(model.n_x, model.n_y).T

    pred = model.predict(
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
    ax[1].plot(pred, '-g', label='Prediction', alpha=0.5,)
    ax[1].legend()
    return 

    