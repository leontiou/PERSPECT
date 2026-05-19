# Projection matrix generation for SPECT with GATE 10  
## Example: 3D Shepp–Logan phantom

This folder contains a Jupyter notebook demonstrating **projection matrix generation for SPECT using GATE 10** with a **3D voxelized Shepp–Logan phantom**.

The example is intended as a controlled synthetic test case complementary to the anthropomorphic-phantom workflows in the main `GATE10` folder.

---

## What this notebook does

The notebook demonstrates the following workflow:

1. **Create a 3D voxelized Shepp–Logan phantom** and a corresponding attenuation map.
2. **Write the phantom and attenuation map** in `.mhd/.raw` MetaImage format for use in GATE.
3. **Create or load a geometric projection matrix** based on the 2D Radon transform.
4. **Run GATE10 simulations** in a dual-head SPECT configuration for a set of projection angles.
5. **Extract sinograms and voxel-wise projection data** from the GATE ROOT outputs.
6. **Construct a GATE-corrected 2D projection matrix** for a selected slice.
7. **Apply MLEM reconstruction**, both with and without GATE-based correction, and compare the results.

---

## Main files

- **Notebook file**  
  Main workflow for phantom generation, simulation, matrix construction, and reconstruction.

- **`GATE10.py`**  
  Python script that runs the OpenGATE / GATE10 simulation for each projection angle.

Generated files may include:

- **`phantom.mhd/.raw`**  
  3D voxelized source phantom written in MetaImage format.

- **`attenuation_map.mhd/.raw`**  
  3D attenuation map used by GATE.

- **`pmatrix_geo.pk`**  
  Geometric projection matrix generated from the Radon transform.

- **`sinograms_ideal_NC.pk`**  
  Sinograms assembled from the GATE simulation output.

- **`projections_ideal_NC.pk`**  
  Voxel-wise projection information extracted from GATE output.

- **`phantom_data/output_<angle>/`** and **`projection_data/output_<angle>/`**  
  Per-angle GATE output folders containing ROOT files, projection files, and logs.

---

## Requirements

The notebook expects the project files to be available in:

```python
PROJECT_PATH = "./"
