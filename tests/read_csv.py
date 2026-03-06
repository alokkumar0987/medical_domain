import os
import pydicom
import numpy as np

# Use test_data (LIDC-IDRI-0001) or original manifest path
folder = r"D:\project\medical_3Dmodel\test_data\LIDC-IDRI-0001"

files = []

for f in os.listdir(folder):
    if f.endswith(".dcm"):
        path = os.path.join(folder, f)
        files.append(pydicom.dcmread(path))

# sort slices correctly
files.sort(key=lambda x: float(x.ImagePositionPatient[2]))

# create 3D volume
volume = np.stack([f.pixel_array for f in files])

print("3D Volume Shape:", volume.shape)

import pandas as pd

pd.set_option('display.max_columns', None)
# Use test_data manifest or full digest
path = r"D:\project\medical_3Dmodel\test_data\LIDC-IDRI-0001_digest.xlsx"

df = pd.read_excel(path)
# Keep only CT series (133 .dcm); drop DX row (2 .dcm) if present
df = df[df["Image Count"] > 10]

print("Columns:", df.columns.tolist())
print(df.head())

ct_series = df  # already filtered to CT
print(ct_series[["Patient ID", "Image Count"]].head())


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

slice_index = volume.shape[0] // 2

img = ax.imshow(volume[slice_index], cmap="bone")
ax.set_title(f"Slice {slice_index}")
ax.axis("off")

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, volume.shape[0]-1,
                valinit=slice_index, valstep=1)

def update(val):
    idx = int(slider.val)
    img.set_data(volume[idx])
    ax.set_title(f"Slice {idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


