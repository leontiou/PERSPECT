from pathlib import Path
import numpy as np
import opengate as gate
import opengate.contrib.spect.ge_discovery_nm670 as discovery

# ===========================================================
#  Radionuclide discrete gamma emission spectra for SPECT
# ===========================================================
RADIONUCLIDE_SPECTRA = {
    "Lu177": {
        "energies": [71.6418, 112.9498, 136.7245, 208.3662, 249.6742, 321.3159],
        "weights": [0.001726, 0.0620, 0.000470, 0.1038, 0.002012, 0.002160],
    },
    "Tc99m": {
        "energies": [140.511],
        "weights": [0.889],
    },
    "I123": {
        "energies": [159.0, 127.0, 529.0],
        "weights": [0.837, 0.016, 0.014],
    },
    "I131": {
        "energies": [364.49, 284.3, 637.0, 723.0],
        "weights": [0.816, 0.067, 0.071, 0.018],
    },
    "In111": {
        "energies": [171.28, 245.39],
        "weights": [0.901, 0.941],
    },
    "Tl201": {
        "energies": [68.9, 70.8, 135.3, 167.4],
        "weights": [0.26, 0.34, 0.10, 0.14],
    },
    "Xe133": {
        "energies": [81.0, 31.0],
        "weights": [0.373, 0.589],
    },
    "Co57": {
        "energies": [122.06, 136.47],
        "weights": [0.856, 0.107],
    },
}


def apply_radionuclide_spectrum(src, isotope_name, keV):
    if isotope_name not in RADIONUCLIDE_SPECTRA:
        raise ValueError(f"Unknown isotope '{isotope_name}'. Available: {list(RADIONUCLIDE_SPECTRA.keys())}")
    spec = RADIONUCLIDE_SPECTRA[isotope_name]
    src.energy.type = "spectrum_discrete"
    src.energy.spectrum_energies = [e * keV for e in spec["energies"]]
    src.energy.spectrum_weights = spec["weights"]


def rot_x(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=float)


def rot_y(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=float)


def rot_z(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=float)


def get_spect_energy_windows(radionuclide, tag, keV):
    """
    Return a list of channel dicts for the given radionuclide.
    'tag' keeps names unique per head, e.g. 'h1_peak'.
    """
    rn = radionuclide.lower().replace("-", "").replace("_", "")

    if rn in ("lu177", "lu177q", "lu177lu"):
        return [
            {"name": f"{tag}_spectrum", "min": 3 * keV,      "max": 515 * keV},
            {"name": f"{tag}_scatter1", "min": 96 * keV,     "max": 104 * keV},
            {"name": f"{tag}_peak113",  "min": 104.52 * keV, "max": 121.48 * keV},
            {"name": f"{tag}_scatter2", "min": 122.48 * keV, "max": 133.12 * keV},
            {"name": f"{tag}_scatter3", "min": 176.46 * keV, "max": 191.36 * keV},
            {"name": f"{tag}_peak208",  "min": 192.4 * keV,  "max": 223.6 * keV},
            {"name": f"{tag}_scatter4", "min": 224.64 * keV, "max": 243.3 * keV},
        ]

    if rn in ("tc99m", "tc99", "tc"):
        return [
            {"name": f"{tag}_spectrum", "min": 40 * keV,  "max": 200 * keV},
            {"name": f"{tag}_scatter",  "min": 80 * keV,  "max": 126 * keV},
            {"name": f"{tag}_peak140",  "min": 126 * keV, "max": 154 * keV},
            {"name": f"{tag}_high",     "min": 154 * keV, "max": 190 * keV},
        ]

    if rn in ("i123", "i123q", "i123i"):
        return [
            {"name": f"{tag}_spectrum", "min": 60 * keV,   "max": 260 * keV},
            {"name": f"{tag}_scatter",  "min": 110 * keV,  "max": 145 * keV},
            {"name": f"{tag}_peak159",  "min": 145 * keV,  "max": 175 * keV},
            {"name": f"{tag}_high",     "min": 175 * keV,  "max": 230 * keV},
        ]

    if rn in ("i131", "i131q", "i131i"):
        return [
            {"name": f"{tag}_spectrum", "min": 80 * keV,   "max": 600 * keV},
            {"name": f"{tag}_scatter",  "min": 250 * keV,  "max": 340 * keV},
            {"name": f"{tag}_peak364",  "min": 340 * keV,  "max": 388 * keV},
            {"name": f"{tag}_high",     "min": 388 * keV,  "max": 500 * keV},
        ]

    # Fallback
    return [
        {"name": f"{tag}_spectrum", "min": 3 * keV, "max": 600 * keV},
    ]


def add_spect_digitizer_chain(
    sim,
    crystal,
    tag,
    radionuclide,
    keV,
    pixel_size_mm,
    ndim,
    out_dir,
    root_prefix="spect",
):
    """
    Build the full digitizer chain (Hits -> Singles -> Energy windows -> Projection)
    for one crystal.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    root_file = out_dir / f"{root_prefix}_{tag}.root"

    channels = get_spect_energy_windows(radionuclide, tag, keV)

    # Hits
    hc = sim.add_actor("DigitizerHitsCollectionActor", f"Hits_{tag}")
    hc.attached_to = crystal.name
    hc.output_filename = str(root_file)
    hc.attributes = [
        "PostPosition",
        "TrackVertexPosition",
        "TotalEnergyDeposit",
        "PreStepUniqueVolumeID",
        "GlobalTime",
        "LocalTime",
        "StepLength",
        "TrackLength",
        "ParentID",
    ]

    # Singles
    sc = sim.add_actor("DigitizerAdderActor", f"Singles_{tag}")
    sc.attached_to = crystal.name
    sc.input_digi_collection = hc.name
    sc.policy = "EnergyWinnerPosition"
    sc.output_filename = str(root_file)

    # Energy windows
    ew = sim.add_actor("DigitizerEnergyWindowsActor", f"EnergyWindows_{tag}")
    ew.attached_to = crystal.name
    ew.input_digi_collection = sc.name
    ew.channels = channels
    ew.output_filename = str(root_file)

    # Projection
    proj = sim.add_actor("DigitizerProjectionActor", f"Projection_{tag}")
    proj.attached_to = crystal.name
    proj.input_digi_collections = [c["name"] for c in channels]
    proj.spacing = [pixel_size_mm, pixel_size_mm]
    proj.size = [ndim, ndim]

    peak_candidates = [c for c in channels if "peak" in c["name"]]
    if peak_candidates:
        peak_channel = max(peak_candidates, key=lambda c: c["max"])
    else:
        peak_channel = channels[0]

    peak_ch_name = peak_channel["name"]
    peak_label = peak_ch_name.split("_", 1)[1]

    proj.output_filename = str(out_dir / f"projection_{tag}_{peak_label}.mhd")


def run(
    angle,
    isotope="I123",
    usecollimator=True,
    visualize=False,
    PIXEL_SIZE=4.416968,
    ndim=128,
    dtheta=0.0,
    nthreads=1,
    activity=1,
    mu_map_path="/content/mu_map.mhd",
    source_map_path="/content/source_map.mhd",
    output_base="/content",
):
    """
    Run one simulation case from a notebook.
    """

    mu_map_path = str(mu_map_path)
    output_base = Path(output_base)

    if not Path(mu_map_path).exists():
        raise FileNotFoundError(
            f"mu_map file not found: {mu_map_path}\n"
            f"Upload mu_map.mhd and mu_map.raw first, or change mu_map_path."
        )

    outname = output_base / f"output_{angle}"
    outname.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 0. Simulation setup
    # --------------------------------------------------
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = visualize
    sim.visu_type = "qt"
    sim.number_of_threads = int(nthreads)
    sim.random_seed = "auto"

    # units
    u = gate.g4_units
    m = u.m
    sec = u.second
    days = 3600 * 24 * sec
    cm = u.cm
    mm = u.mm
    keV = u.keV
    Bq = u.Bq
    MBq = 1000 * 1000 * Bq

    # --------------------------------------------------
    # 1. World
    # --------------------------------------------------
    sim.world.size = [2 * m, 2 * m, 2 * m]
    sim.world.material = "G4_AIR"

    # --------------------------------------------------
    # 2. Voxelized phantom
    # --------------------------------------------------
    phantom = sim.add_volume("Image", "phantom")
    phantom.mother = "world"
    phantom.image = mu_map_path
    phantom.material = "G4_AIR"
    phantom.translation = [0.0, 0.0, 0.0]
    phantom.color = [1, 0, 0, 0.5]
    phantom.voxel_materials = [
        [0, 1, "G4_AIR"],
        [1, 2, "G4_TISSUE_SOFT_ICRP"],
        [2, 3, "G4_B-100_BONE"],
        [3, 4, "G4_BONE_CORTICAL_ICRP"],
    ]

    # --------------------------------------------------
    # 3. Dual-head GE Discovery NM670
    # --------------------------------------------------
    head_distance = 50 * cm

    if usecollimator:
        discovery.add_spect_head(sim, "head1", collimator_type="lehr")
        discovery.add_spect_head(sim, "head2", collimator_type="lehr")
    else:
        discovery.add_spect_head(sim, "head1", collimator_type=False)
        discovery.add_spect_head(sim, "head2", collimator_type=False)

    head1 = sim.volume_manager.get_volume("head1")
    head2 = sim.volume_manager.get_volume("head2")

    theta = np.deg2rad(angle)
    rmatrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0,              0,             1]])

    r1 = rmatrix @ np.array([0, head_distance, 0])
    r2 = rmatrix @ np.array([0, -head_distance, 0])
    head1.translation = r1
    head2.translation = r2

    head1.rotation = rot_z(angle) @ rot_y(180) @ rot_x(90)
    head2.rotation = rot_z(angle) @ rot_x(-90)

    crystal1 = sim.volume_manager.get_volume("head1_crystal")
    crystal2 = sim.volume_manager.get_volume("head2_crystal")

    # --------------------------------------------------
    # 4. Digitizer chains
    # --------------------------------------------------
    pixel_size_mm = PIXEL_SIZE * mm

    add_spect_digitizer_chain(
        sim, crystal1, tag="h1", radionuclide=isotope,
        keV=keV, pixel_size_mm=pixel_size_mm,
        ndim=ndim, out_dir=outname, root_prefix=isotope
    )
    add_spect_digitizer_chain(
        sim, crystal2, tag="h2", radionuclide=isotope,
        keV=keV, pixel_size_mm=pixel_size_mm,
        ndim=ndim, out_dir=outname, root_prefix=isotope
    )

    # --------------------------------------------------
    # 5. Sources
    # --------------------------------------------------
    # offset = float(dtheta) * u.deg
    # ANG = angle * u.deg

    # if dtheta == -1:
    #     TH_min = 0 * u.deg
    #     TH_max = 180 * u.deg
    #     PHI1_min = 0 * u.deg
    #     PHI1_max = 360 * u.deg
    #     PHI2_min = 0 * u.deg
    #     PHI2_max = 360 * u.deg
    # else:
    #     TH_min = 90 * u.deg - offset
    #     TH_max = 90 * u.deg + offset
    #     PHI1_min = ANG + 90 * u.deg - offset
    #     PHI1_max = ANG + 90 * u.deg + offset
    #     PHI2_min = ANG - 90 * u.deg - offset
    #     PHI2_max = ANG - 90 * u.deg + offset

    def wrap_deg(x):
        return x % 360.0

    offset_deg = float(dtheta)
    ANG_deg = float(angle)

    if dtheta == -1:
        TH_min = 0 * u.deg
        TH_max = 180 * u.deg
        PHI1_min = 0 * u.deg
        PHI1_max = 360 * u.deg
        PHI2_min = 0 * u.deg
        PHI2_max = 360 * u.deg
    else:
        TH_min = (90.0 - offset_deg) * u.deg
        TH_max = (90.0 + offset_deg) * u.deg

        phi1_min_deg = wrap_deg(ANG_deg + 90.0 - offset_deg)
        phi1_max_deg = wrap_deg(ANG_deg + 90.0 + offset_deg)
        phi2_min_deg = wrap_deg(ANG_deg - 90.0 - offset_deg)
        phi2_max_deg = wrap_deg(ANG_deg - 90.0 + offset_deg)

        PHI1_min = phi1_min_deg * u.deg
        PHI1_max = phi1_max_deg * u.deg
        PHI2_min = phi2_min_deg * u.deg
        PHI2_max = phi2_max_deg * u.deg


    src1 = sim.add_source("VoxelSource", "voxel_source1")
    src1.particle = "gamma"
    src1.attached_to = "world"
    src1.image = source_map_path
    src1.position.translation = [0.0, 0.0, 0.0]
    src1.direction.type = "iso"
    src1.direction.theta = [TH_min, TH_max]
    src1.direction.phi = [PHI1_min, PHI1_max]

    src2 = sim.add_source("VoxelSource", "voxel_source2")
    src2.particle = "gamma"
    src2.attached_to = "world"
    src2.image = source_map_path
    src2.position.translation = [0.0, 0.0, 0.0]
    src2.direction.type = "iso"
    src2.direction.theta = [TH_min, TH_max]
    src2.direction.phi = [PHI2_min, PHI2_max]

    apply_radionuclide_spectrum(src1, isotope, keV)
    apply_radionuclide_spectrum(src2, isotope, keV)

    if sim.visu:
        sim.number_of_threads = 1
        src1.activity = 100 * Bq
        src2.activity = 100 * Bq
    else:
        src1.activity = activity * MBq
        src2.activity = activity * MBq

    # --------------------------------------------------
    # 6. Statistics
    # --------------------------------------------------
    stats_actor = sim.add_actor("SimulationStatisticsActor", "stats")
    stats_actor.track_types_flag = True
    stats_actor.output_filename = str(outname / "stats_dual.txt")

    # --------------------------------------------------
    # 7. Physics + Run
    # --------------------------------------------------
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)

    sim.run_timing_intervals = [[0, 1 * sec]]

    print("=" * 70)
    print(f"Running angle={angle}, isotope={isotope}, ndim={ndim}, PIXEL_SIZE={PIXEL_SIZE}, dtheta={dtheta}")
    print(f"mu_map_path = {mu_map_path}")
    print(f"output dir  = {outname}")
    print("=" * 70)

    sim.run(start_new_process=True)

    print(stats_actor)
    return str(outname)


