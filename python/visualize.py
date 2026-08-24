import matplotlib.pyplot as plt
import numpy as np

def show_pipeline_stages(stages):
    n = len(stages)
    fig,axes = plt.subplots(1,n,figsize=(4*n,5))
    if n==1:
        axes = [axes]
    for ax,(title,img)  in zip(axes,stages):
        ax.imshow(img,cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.show()