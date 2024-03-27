from os import listdir
from os.path import join, isfile
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def getSpectralFiles( mypath):
    return [(join(mypath, f),f) for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]

if __name__ == "__main__":
    mylist = getSpectralFiles(mypath="./img1")
    df= {}
    for i, (myp, f) in enumerate(mylist):
        if f.startswith("aitoff"):
            a=f.replace(".png","").split("_")[3:]
            df[i]=[float(a[x]) for x in [0,2,3,4]]
    dff = pd.DataFrame.from_dict(df, orient="index")
    dff.to_pickle("./PG_data/globeOptimization")
    dff.to_csv("./PG_data/globeOptimization.csv")
    print(dff.shape)
    dff.columns=["error","ksi","theta","phi"]
    dff["r"]=np.sqrt(dff.ksi**2+dff.theta**2+dff.phi**2)
    dff["xx"]=dff.ksi/dff.r
    dff["yy"] = dff.theta/ dff.r
    dff["zz"] = dff.phi / dff.r
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]

    dff.error = -dff.error
    colormax = dff.error.max()
    colormin = dff.error.min()
    dff["color"] = (dff.error + colormin) / (colormax - colormin)

    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    xx, yy, zz = dff.xx, dff.yy, dff.zz


    # Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color='c', alpha=0.1, linewidth=0)

    color =dff.color
    ax.scatter(xx  , yy , zz , alpha=0.2, c=color, cmap=cm.inferno, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([1, -1])
    ax.set_zlim([-1, 1])
    plt.tight_layout()
    plt.show()

