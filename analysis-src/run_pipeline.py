'''
The main script to analyse images from a deep screening experiment.

This script will read in a json file, which describes how to
process the the experimental data, and where to find it.
On processing the fastq and image data, this script will write
per cluster intensities to a .pkl file for each tile.

Cluster intensities are referenced by a cluster id, and need to
be further processed to pair the sequence.

This script has support for processing tile images in parallel
via MPI, and can be near linearly scaled out to 1 core per tile.

'''
import os, sys, time, argparse
import json, logging, shutil, gc
from mpi4py import MPI
from detect_extract_pipeline import ProcessTile

parser = argparse.ArgumentParser(description='run_pipeline.py: Processes images from a deep screening experiment.')
parser.add_argument('-config', help='Json config file that describes the experiment. REQUIRED.', required=True)

args = parser.parse_args()


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)

## Setup for MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


## Read and parse the config json file.
experiment_config = json.load(open(args.config, 'r'))

## Prepare the output folder
output_folder = os.path.join(experiment_config['pipeline_output_location'],
    experiment_config['experiment_id'])
logs_folder = os.path.join(output_folder, 'logs')
intensities_folder = os.path.join(output_folder, 'intensities')
temp_folder = os.path.join(output_folder, 'tmp')


if rank == 0:
    # Create the experiment output folder.
    if not os.path.exists(output_folder):
        logging.info("Creating folder: %s" % (output_folder))
        os.mkdir(output_folder)
    else:
        logging.info("Folder already exists: %s" % (output_folder))

    # Create logs folder
    if not os.path.exists(logs_folder):
        logging.info("Creating folder: %s" % (logs_folder))
        os.mkdir(logs_folder)
    else:
        logging.info("Folder already exists: %s" % (logs_folder))

    # Create intensities folder
    if not os.path.exists(intensities_folder):
        logging.info("Creating folder: %s" % (intensities_folder))
        os.mkdir(intensities_folder)
    else:
        logging.info("Folder already exists: %s" % (intensities_folder))

    # Create temp folder
    if not os.path.exists(temp_folder):
        logging.info("Creating folder: %s" % (temp_folder))
        os.mkdir(temp_folder)
    else:
        logging.info("Folder already exists: %s" % (temp_folder))

    ## Copy the json run file into the output folder.
    shutil.copy(args.config, os.path.join(output_folder, os.path.basename(args.config)))


## Setup other vars
if experiment_config['log_images'] == 'True':
    log_dir = logs_folder
else:
    log_dir = None

if experiment_config['normalise_ccd_strips'] == 'True':
    norm_ccd_strips = True
else:
    norm_ccd_strips = False



comm.barrier()



## Prepare the flowcell descriptor and number of tiles that need to be processed.
## Currently only supports a rapid v2 flowcell.
if experiment_config['flowcell_type'] == 'rapid_v2':
    n_lanes = 2
    n_surfaces = 2
    n_swaths = 2
    n_tiles = 16
elif experiment_config['flowcell_type'] == 'half':
    n_lanes = 2
    n_surfaces = 2
    n_swaths = 2
    n_tiles = 4
else: ## Default is the rapid v2 descriptor.
    n_lanes = 2
    n_surfaces = 2
    n_swaths = 2
    n_tiles = 16

if experiment_config['process_tiles'] == 'all':
    tileList = []
    for lane in range(n_lanes):
        for surface in range(n_surfaces):
            for swath in range(n_swaths):
                for tile in range(n_tiles):
                    tile_id = "%d_%d%d%02d" % (lane+1, surface+1, swath+1, tile+1)
                    tileList.append(tile_id)
else:
    tileList = [x.lstrip().rstrip() for x in experiment_config['process_tiles'].split(',')]
    if len(tileList) < size:
        logging.error("More processors in use than tiles designated for processing.")
        # exit()


## If using the neural network to detect clusters,
## we need to designate rank 0 to function as a model server.
rank_corr = rank
size_corr = size

subdiv = int(len(tileList)/size_corr)
start = rank_corr*subdiv
end = rank_corr*subdiv+subdiv
tileListSplit = tileList[start:end]

logging.info("Rank %d is processing tiles: %s" % (rank, tileListSplit))

comm.barrier() ## Comm barrier to stop procs from executing before everything has been setup.



for tile_id in tileListSplit:
    fastq_path = os.path.join(experiment_config['tile_fastq_file_location'], tile_id, 'Undetermined_S0_R1_001.fastq.gz')
    exp_images = experiment_config['experimental_image_location']


    ProcessTile(tile_id, fastq_path=fastq_path, exp_images=exp_images, log_dir=log_dir,
        data_out_dir=output_folder, temp_dir=temp_folder,
        chromatic_correction_image_location=experiment_config['chromatic_correction_image_location'],
        detect_method=experiment_config['spot_detection_method'],
        extract_method=experiment_config['spot_extraction_method'],
        local_max_threshold=float(experiment_config['local_max_threshold']),
        max_cluster_threshold=float(experiment_config['max_cluster_intensity_threshold']),
        sub_image_size=int(experiment_config['sub_img_size']),
        alignment_subdiv_x=int(experiment_config['alignment_subdiv_x']),
        alignment_subdiv_y=int(experiment_config['alignment_subdiv_y']),
        assignment_max_dist=float(experiment_config['assignment_max_dist']),
        assignment_method=experiment_config['assignment_method'],
        normalise_ccd_strips=norm_ccd_strips,
        )




