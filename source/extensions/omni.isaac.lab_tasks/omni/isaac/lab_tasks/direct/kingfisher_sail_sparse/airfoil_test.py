from xfoil import XFoil
from xfoil.model import Airfoil
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

# File path to the Naca profile downloaded from xfoil
filepath = "/home/GTL/fsangare/Usd/naca0018.csv"
df = pd.read_csv(filepath, skiprows=8, nrows=201)
#print(df.head)

x_coords_naca = 1e-2*df[["X(mm)"]].values
y_coords_naca = 1e-2*df[["Y(mm)"]].values
xfoil = XFoil()
xfoil.print = False
airfoil_set = False
xfoil.airfoil = Airfoil(x_coords_naca, y_coords_naca)  # set later

xfoil.n_crit = 9
max_cl_cd_ratio = 0
max_cl = 0

xfoil.Re = 200000
xfoil.max_iter = 100
a, cl, cd, cm, co = xfoil.aseq(-20, 20, 1)

dir = "/home/GTL/fsangare/Isaaclab_kingfisher_sail/"

plt.plot(a, cl, label="Cl")
plt.plot(a, cd, label="Cd")
plt.savefig(dir+"cl_cd_naca0018.png")