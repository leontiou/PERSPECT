# Hamiltonian Monte Carlo reconstruction for SPECT

This folder contains an illustrative implementation of the **Hamiltonian Monte Carlo (HMC)** reconstruction framework developed within the **PERSPECT** project.

The notebook demonstrates:

- reconstruction from a measured SPECT sinogram;
- comparison with a conventional MLEM reconstruction;
- HMC sampling using a geometric projection matrix;
- HMC reconstruction using a GATE-corrected forward model;
- generation of ensembles of reconstructed images;
- analysis of χ² evolution, ensemble correlations, and uncertainty-related image diagnostics.

The main notebook is:

- `HMC.ipynb`

## Required data files

The following large intermediate files are not distributed through GitHub because of their size:

- `pmatrix_geo.pk`
- `projections_ideal_WC.pk`

`pmatrix_geo.pk` contains the geometric projection matrix and can be generated directly by the notebook.

`projections_ideal_WC.pk` contains voxel-dependent projection information obtained from the corresponding GATE simulation workflow and is used to construct the GATE-corrected projection matrix.

The example also expects the measured SPECT DICOM file used by the notebook.

## HMC output

The HMC reconstruction generates an ensemble of statistically acceptable reconstructed images rather than a single deterministic solution.

The returned results include, among other quantities:

- `all_images` — reconstructed image ensemble;
- `chi2list` — χ² values associated with the sampled states.

The ensemble can be used to study the mean reconstructed image, image fluctuations, correlations, and data-visible uncertainty.

## PERSPECT

This material is shared as part of the dissemination activities of the **PERSPECT — Personalized SPECT** project.

Project website:  
https://perspect.frederick.ac.cy

## Funding

This work is supported by the **Cyprus Research and Innovation Foundation** through the project **“Personalized SPECT” (PERSPECT)**  
(**Project No. EXCELLENCE/0524/0410**).

The project is implemented within the framework of the **Cohesion Policy Programme “THALIA 2021–2027”** and is **co-funded by the European Union**.

