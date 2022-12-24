from pathlib import Path

multiSiteMri_int_to_site = {0: 'ISBI', 1: "ISBI_1.5", 2: 'I2CVB', 3: "UCL", 4: "BIDMC", 5: "HK"}
multiSiteMri_site_to_int = {v: k for k, v in multiSiteMri_int_to_site.items()}
cc359_data_path = '/home/dsi/shaya/cc359_data/CC359/'
cc359_splits_dir = Path('/home/dsi/shaya/unsup_splits/')
cc359_results = Path('/home/dsi/shaya/tomer/CC359_results/')
msm_data_path = '/home/dsi/shaya/multiSiteMRI'
msm_splits_dir = Path('/home/dsi/shaya/unsup_splits_msm/')
msm_results = Path('/home/dsi/shaya/tomer/msm_results/')

