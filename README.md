# **Per**sonalized **SPECT** 
The PERSPECT project improves SPECT imaging by using personalized Monte Carlo-based system matrices and advanced GPU-accelerated reconstruction methods to enhance detection of small lesions and brain diseases, leading to better diagnosis, treatment, and patient outcomes.

This material is shared as part of the dissemination activities of the **PERSPECT** project and is intended to support transparent and reproducible research in advanced SPECT simulation and reconstruction.


## GATE10 notebook workflow for SPECT projection-matrix generation

The folder [`GATE10`](./GATE10) contains a Google Colab / Jupyter workflow for **projection matrix generation for SPECT using GATE 10**. It includes:

Folder GATE10:
- a complete demonstration notebook,
- the `GATE10.py` simulation script,
- examples of hardware phantom DICOM inputs,
- a workflow for generating GATE-compatible attenuation/source maps, running dual-head SPECT simulations, and assembling a GATE-based projection matrix,
- Subfolders GATE10/Jaszczak3D and GATE10/SheppLogan3D contain two additional examples using voxelized 3D software phantoms.

## Hamiltonian Monte Carlo reconstruction

The [`HMC`](./HMC) folder contains an illustrative implementation of the **Hamiltonian Monte Carlo (HMC)** reconstruction methodology developed within PERSPECT.

The notebook demonstrates:

- reconstruction from measured SPECT sinograms;
- comparison with conventional MLEM reconstruction;
- HMC sampling using a geometric projection matrix;
- HMC reconstruction using a **GATE-corrected system model** incorporating attenuation and scatter information;
- generation of ensembles of statistically acceptable reconstructed images;
- analysis of ensemble fluctuations, correlations, χ² distributions, and data-visible uncertainty.

The purpose of the example is to illustrate how stochastic reconstruction can provide information beyond a single reconstructed image by allowing the statistical properties of the reconstruction ensemble to be investigated.

---


## PERSPECT Database for GATE v9.0

A public database of 2D analytic phantoms, sinograms, and GATE v9.0 macro files is available here:

**[Open the database webpage](https://leontiou.github.io/PERSPECT/)**

The page provides visual previews and downloadable files for the available examples.


# Funding
This work is supported by the **Cyprus Research and Innovation Foundation** through the project **“Personalized SPECT” (PERSPECT)**  
(**Project No. EXCELLENCE/0524/0410**).  
The PERSPECT project is implemented within the framework of the **Cohesion Policy Programme “THALIA 2021–2027”** and is **co-funded by the European Union**.
