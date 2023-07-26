import os, logging
import sys, glob, json
from PIL import Image
import gzip
import numpy as np
import scipy as sp
import scipy.misc
from skimage.feature import register_translation
from skimage.filters import threshold_otsu, threshold_local, threshold_li, gaussian, sobel, laplace
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage import exposure
from skimage import morphology
from skimage import transform, util, io
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage.filters import median_filter
import scipy.ndimage as ndimage
import random
from skimage.morphology import disk, erosion
from skimage.morphology import opening, white_tophat

import time, gc
import cv2

from collections import defaultdict
import pickle, itertools

from mpi4py import MPI
from mpiTaskQueue import TaskQueue

## Setup for MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Image.MAX_IMAGE_PIXELS = None

## Tile image dimensions (pixels)
tile_width = 2048
tile_width_nodet = 2046
tile_height = 10048

right_strip_width = 12


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)


def GetTileImageFromBrunoScan(tile_id, scandir, channel):
    '''Fetch a specific tile and channel from a Bruno scan.'''

    ## Parse the tile id.
    tile_id_split = tile_id.split('_')
    lane = int(tile_id_split[0])
    surface = int(tile_id_split[1][0])
    swath = int(tile_id_split[1][1])
    tile = int(tile_id_split[1][2:4])

    if surface == 1:
        surface_name = "top"
    elif surface == 2:
        surface_name = "bot"

    images_dir = "Images"
    image_to_load = "L00%d/C*/*_c00*_l%d_t001_%s_s%d_%s.tif" % (lane, lane, surface_name, swath, channel)

    full_image_path_approx = os.path.join(scandir, images_dir, image_to_load)

    full_image_path = glob.glob(full_image_path_approx)[0]

    ## Load the swath image.
    img = Image.open(full_image_path)

    logging.info("%s - Loaded image: %s" % (tile_id, full_image_path))

    ## Extract the tile from the loaded swath image.
    y_tile_top = (tile-1)*tile_height
    y_tile_bottom = tile*tile_height
    tile_coordinates = (0, y_tile_top, tile_width, y_tile_bottom)

    logging.info("%s - Extracting tile %s from swath image with coordinates %s" % (tile_id, tile_id, tile_coordinates))

    extractedTile = np.array(img.crop(tile_coordinates))

    extractedTile[:,tile_width-right_strip_width:tile_width] = 0 ## Remove the 12px image stripe that HiSeq adds to the image.

    img.close()

    return extractedTile


def readFastq(fastqFile, tile_id=None):
    '''A very basic fastq reader.'''
    if fastqFile.split('.')[-1] == "gz":
        fh = gzip.open(fastqFile, 'rt')
    else:
        fh = open(fastqFile, 'r')

    for line in fh:
        if line[0] == '@':
            header = line.rstrip()
            seq = ''
            line = fh.readline()
            while line[0] != '+':
                seq += line.rstrip()
                line = fh.readline()
            if tile_id == None:
                yield [header, seq]
            else:
                header_split = header.split(':')
                lane = int(header_split[3])
                tile_id_fasta = "%d_%s" % (lane, header_split[4])
                if tile_id == tile_id_fasta:
                    yield [header, seq]
            fh.readline()
    fh.close()


def GenerateClusterImageFromFastq(fastq_path):
    '''Generates a tif file containing cluster positions defined in the fastq file.'''

    pseudo_cluster_map = np.zeros((tile_height, tile_width), dtype=np.uint8)

    for sequence in readFastq(fastq_path):

        ## Parse the record header
        header = sequence[0].split(':')

        cluster_x = int(header[5])
        cluster_y = int(header[6].split(' ')[0])

        try:
            pseudo_cluster_map[cluster_y, cluster_x] = 255
        except:
            continue

    return pseudo_cluster_map


def AlignImages(im0_path, im1_path, subdiv_x, subdiv_y, tile_id, global_only=False):

    deviance_max = 15.0

    # The template - usually the pseudo cluster map.
    try:
        im0 = Image.open(im0_path)
    except:
        im0 = im0_path
    # The image to be transformed - usually the raw image or the peak map from the raw image.
    try:
        im1 = Image.open(im1_path)
    except:
        im1 = im1_path

    ## Image subdivision
    n_blocks_x = int(tile_width/subdiv_x)
    n_blocks_y = int(tile_height/subdiv_y)

    offsets = []

    # Global alignment
    result = register_translation(im0, im1, upsample_factor=10)
    vector = result[0]

    yOffsetGlobal = vector[0]
    xOffsetGlobal = vector[1]

    logging.info("%s - Global alignment results: %s" % (tile_id, vector))

    im1_ = np.fft.fft2(im1)
    im1_tfm = ndimage.fourier_shift(im1_, shift=vector)
    im1_tfm = np.fft.ifft2(im1_tfm).real

    offset_map = np.zeros((n_blocks_y, n_blocks_x, 2), dtype=np.float)

    ## Split im1 into n_blocks_x x n_blocks_y images for local alignment
    left = 0
    right = subdiv_x
    top = 0
    bottom = subdiv_y

    ## Scan left to right, one row at a time.
    for i in range(n_blocks_y): ## Top to bottom:: y axis

        for j in range(n_blocks_x): ## Left to right:: x axis

            if not global_only:
                im1_split_coords = (top, bottom, left, right)
                im1_split = im1_tfm[top:bottom, left:right]

                result = register_translation(im0[top:bottom, left:right], im1_split, upsample_factor=10)
                vector = result[0]


                yOffset = vector[0]
                xOffset = vector[1]

                if yOffset > deviance_max or yOffset < -deviance_max:
                    logging.warn("%s - Alignment y axis deviation > +-10 px. Check subdivision dimensions." % (tile_id))
                    yOffset = 0 ## Set offset to 0, and just use global.
                    xOffset = 0

                if xOffset > deviance_max or xOffset < -deviance_max:
                    logging.warn("%s - Alignment x axis deviation > +-10 px. Check subdivision dimensions." % (tile_id))
                    yOffset = 0 ## Set offset to 0, and just use global.
                    xOffset = 0

                ## Save offsets into the grid map
                offset_map[i,j,0] = yOffset
                offset_map[i,j,1] = xOffset


            left += subdiv_x
            right += subdiv_x

        ## Reset left and right counters
        left = 0
        right = subdiv_x

        top += subdiv_y
        bottom += subdiv_y


    offset_map_global = offset_map.copy()
    offset_map_global[:,:,0] += yOffsetGlobal
    offset_map_global[:,:,1] += xOffsetGlobal

    return offset_map_global, offset_map, (yOffsetGlobal, xOffsetGlobal)


def discMask(array, radius):
    """Returns a disc masked array of radius."""

    size = array.shape[0]
    x0 = y0 = size // 2

    y, x = np.ogrid[-y0:size-y0, -x0:size-x0]
    mask = x*x + y*y <= radius*radius

    array_mask = np.zeros((size, size))
    array_mask[mask] = 1

    return array*array_mask


def GaussianTwoDimensions(size, sigma = 1):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    c_x = c_y = size // 2

    return 1*np.exp(-((((c_x-x)**2)/(2*sigma**2))+(((c_y-y)**2)/(2*sigma**2))))


def correctImages(tile_img, diskRadius):

    kernel = np.array(disk(diskRadius))
    bg = cv2.morphologyEx(tile_img, cv2.MORPH_OPEN, kernel)

    correctedTileImg = tile_img - bg

    return correctedTileImg


def DetectClustersLocalMax(tile_id, tile_image, local_max_threshold, max_cluster_threshold=8000.0):

    start_time = time.time()

    local_max = peak_local_max(tile_image, min_distance=2, indices=True, footprint=np.ones((3, 3)), threshold_abs=local_max_threshold,
                                  exclude_border=False)

    clusterMap = np.zeros((tile_height, tile_width), dtype=np.uint8)
    clusterArray = []

    counter = 0
    for cluster in local_max:
        y_coord = int(cluster[0])
        x_coord = int(cluster[1])

        if tile_image[y_coord, x_coord] >= max_cluster_threshold: ## Filter very bright spots.
            continue

        try:
            clusterMap[int(y_coord), int(x_coord)] = 255
            clusterArray.append([y_coord, x_coord])
        except:
            ## Pixel is out of bounds for some reason
            logging.warn("%s - Cluster coordinates (%s, %s) are out of bounds." % (tile_id, y_coord, x_coord))
            continue

        counter += 1

    logging.info("%s - Detected %d clusters via local max in %f seconds." % (tile_id, counter, time.time()-start_time))

    return clusterMap, clusterArray


def ExtractClustersPsfDot(tile_id, tile_image, clusterArray, log_dir=None, sub_image_size=7):
    '''
    Extracts fluorescent intensity values of detected clusters.

    Extraction works by creating a 2d gaussian psf and multiplying it by
    a NxN pixel sub-image centered on each cluster.
    Intensity is then the mean or median intensity of the dot product.
    Clusters can be filtered by a signal to noise ratio.
    '''

    start_time = time.time()

    intensityMap = np.zeros((tile_height, tile_width), dtype=np.uint16)
    intensityArray = []

    sigma = 1.0
    psf = GaussianTwoDimensions(sub_image_size, sigma)
    sub_img_half = int((sub_image_size-1)/2)

    extract_count = 0

    for cluster in clusterArray:
        cy = int(cluster[0])
        cx = int(cluster[1])

        ## Fetch the sub image centred on this cluster.
        sub_img = np.array(tile_image[int(cy)-sub_img_half:int(cy)+sub_img_half+1, int(cx)-sub_img_half:int(cx)+sub_img_half+1])

        if sub_img.shape != (sub_image_size, sub_image_size):
            continue

        ## Compute median intensity.
        gaussian = psf*sub_img
        intensity = np.sum(gaussian)

        ## Append intensity information to map and array.
        intensityMap[cy, cx] = intensity
        intensityArray.append([cy, cx, intensity])
        extract_count += 1


    logging.info("%s - Extracted %d/%d clusters via psf_dot in %f seconds." % (tile_id, extract_count, len(clusterArray), time.time()-start_time))

    return intensityMap, intensityArray


def DetectExtractClusters(tile_id, tile_image, log_dir=None, detect_method='local_max',
    extract_method='psf_dot', local_max_threshold=600, max_cluster_threshold=8000.0,
    sub_image_size=7):
    '''
    Detects clusters and extracts their intensities via several available methods.

    Detection methods:
        'local_max' uses peak_local_max from sci-kit image to identify cluster centroids.

    Extraction methods:
        'psf_dot' creates a 2d gaussian psf and multiplies it against a 7x7 pixel sub-image centered on each detected cluster.

    This method aims to detect clusters, remove high intensity artefacts, extract intensities, then return centroid and intensity information.
    '''

    tile_image = tile_image[:,:tile_width_nodet]

    ## Detect clusters.
    if detect_method == 'local_max':
        clusterMap, clusterArray = DetectClustersLocalMax(tile_id, tile_image, local_max_threshold, max_cluster_threshold)

    ## Extract cluster intensities.
    if extract_method == 'psf_dot':
        intensityMap, intensityArray = ExtractClustersPsfDot(tile_id, tile_image, clusterArray,
            log_dir=log_dir, sub_image_size=sub_image_size)

    elif extract_method == 'none':
        clusterMapFiltered, intensityMap, intensityArray = None, None, None
        return clusterMap, clusterMap, intensityMap, intensityArray

    ## This produces a clusterMap like array that has essentially been filtered by the snr.
    ## Using clusterMapFiltered for alignments may be more accurate.
    clusterMapFiltered = intensityMap > 0
    clusterMapFiltered = clusterMapFiltered*255

    return clusterMap, clusterMapFiltered, intensityMap, intensityArray


def ChromaticAberationCorrection(tile_id, chromatic_correction_image_location, seq_cluster_map, subdiv_x, subdiv_y, local_max_threshold, max_cluster_threshold):
    '''This function generates correction maps and offset coords between channels C and T.'''

    logging.info("%s - Measuring chromatic aberation for channels C and T." % (tile_id))

    tile_img_c = GetTileImageFromBrunoScan(tile_id, chromatic_correction_image_location, 'c')
    det_cluster_map, det_cluster_array = DetectClustersLocalMax(tile_id, tile_img_c, local_max_threshold, max_cluster_threshold)
    offset_map_c, chrom_c, global_c = AlignImages(seq_cluster_map, det_cluster_map, subdiv_x, subdiv_y, tile_id+"_c")


    tile_img_t = GetTileImageFromBrunoScan(tile_id, chromatic_correction_image_location, 't')
    det_cluster_map, det_cluster_array = DetectClustersLocalMax(tile_id, tile_img_t, local_max_threshold, max_cluster_threshold)
    offset_map_t, chrom_t, global_t = AlignImages(seq_cluster_map, det_cluster_map, subdiv_x, subdiv_y, tile_id+"_t")

    t_c_offset_y = global_t[0]-global_c[0]
    t_c_offset_x = global_t[1]-global_c[1]
    logging.info("%s - Channel offset between T and C = %f, %f" % (tile_id, t_c_offset_y, t_c_offset_x))

    return chrom_c, chrom_t, t_c_offset_y, t_c_offset_x


def ProcessTile(tile_id, fastq_path=None, exp_images=None, chromatic_correction_image_location=None,
    log_dir=None, data_out_dir=None, temp_dir=None,
    detect_method='local_max', extract_method='psf_dot',
    local_max_threshold=600, max_cluster_threshold=8000.0,
    sub_image_size=7, alignment_subdiv_x=256, alignment_subdiv_y=5000,
    assignment_max_dist=2.0, assignment_method='nearest', normalise_ccd_strips=False):

    ## Generate the expected pseudo cluster map from the fastq file.
    logging.info("%s - Generating the pseudo cluster map from: %s" % (tile_id, fastq_path))
    sequencing_cluster_map = GenerateClusterImageFromFastq(fastq_path)

    use_chC_w_known_offset = True ## Remove later and force dual alignment if we want to return intensities in both channels.

    ## Measure chromatic aberation for this tile.
    if chromatic_correction_image_location != None and use_chC_w_known_offset == True:
        chrom_c, chrom_t, t_c_offset_y, t_c_offset_x = ChromaticAberationCorrection(
            tile_id, chromatic_correction_image_location, sequencing_cluster_map, alignment_subdiv_x, alignment_subdiv_y,
            local_max_threshold, max_cluster_threshold
        )


    ## Populate the cluster array.
    ## This is a list, ordered by cluster index in the tile fastq file
    ## and contains a dictionary for each cluster, coordinates and observations.
    ## Sequences need to be paired later.

    cluster_array = []

    logging.info("%s - Populating the cluster array for this tile." % (tile_id))
    for i, cluster in enumerate(readFastq(fastq_path)):
        ## Parse the record header
        header = cluster[0].split(':')
        seq = cluster[1]
        lane = int(header[3])
        surface = int(header[4][0])
        swath = int(header[4][1])
        tile = int(header[4][2:4])
        tile_id = "%d_%s" % (lane, header[4])

        cx = int(header[5])
        cy = int(header[6].split(' ')[0])

        ## We don't store the sequence in this dict, as it becomes too memory heavy.
        cluster_dict = {
            'cluster_id': i,
            'x': cx,
            'y': cy,
            'measurements_t': [],
            'measurements_c': [],
        }
        cluster_array.append(cluster_dict)

    logging.info("%s - Populated the cluster array with %d clusters." % (tile_id, len(cluster_array)))


    ## Process the images from each independent variable (timepoint, concentration).
    ## Note, the json file MUST put the experimental images in a list and not a dict.
    ## If a dict is used, order will be lost.
    for datapoint in exp_images:
        idep_var = datapoint[0]
        idep_var_path = datapoint[1]
        tile_id_idep = "%s_%s" % (tile_id, idep_var)

        logging.info("%s - Processing images from independent variable: %s" % (tile_id_idep, idep_var))

        ## Fetch the tile image
        tile_image_t = GetTileImageFromBrunoScan(tile_id_idep, idep_var_path, 't')
        tile_image_c = GetTileImageFromBrunoScan(tile_id_idep, idep_var_path, 'c')

        logging.info("%s - Detecting and extracting clusters in channel C." % (tile_id_idep))
        clusterMap_c, clusterMapFiltered_c, intensityMap_c, intensityArray_c = DetectExtractClusters(
            tile_id_idep, tile_image_c,
            log_dir=log_dir, detect_method='local_max', extract_method='none',
            local_max_threshold=800, max_cluster_threshold=100000,
            sub_image_size=7)

        logging.info("%s - Aligning the pseudo cluster map with detected clusters in channel C." % (tile_id_idep))
        offset_map_c, chrom_c, global_c = AlignImages(sequencing_cluster_map, clusterMap_c, alignment_subdiv_x, alignment_subdiv_y, tile_id_idep, global_only=False)


        if normalise_ccd_strips:
            logging.info("%s - Correcting non-uniform illumination in tile image." % (tile_id_idep))
            tile_image_t = correctImages(tile_image_t, 25)


        logging.info("%s - Detecting and extracting clusters in channel T." % (tile_id_idep))
        clusterMap_t, clusterMapFiltered_t, intensityMap_t, intensityArray_t = DetectExtractClusters(
            tile_id_idep, tile_image_t,
            log_dir=log_dir, detect_method=detect_method, extract_method=extract_method,
            local_max_threshold=local_max_threshold, max_cluster_threshold=max_cluster_threshold,
            sub_image_size=sub_image_size)

        logging.info("%s - Aligning the pseudo cluster map with detected/filtered clusters." % (tile_id_idep))

        if not use_chC_w_known_offset:
            offset_map_t, chrom_t, global_t = AlignImages(sequencing_cluster_map, clusterMapFiltered_t, alignment_subdiv_x, alignment_subdiv_y, tile_id_idep)
        else:
            offset_map_t = None

        if offset_map_t is None or use_chC_w_known_offset == True:
            logging.warn("%s - Attempting to use channel C with known offsets and chromatic aberation for T." % (tile_id_idep))

            if offset_map_c is None:
                logging.warn("%s - Aborting idep in tile due to problems in alignment. Setting observations to -2." % (tile_id_idep))
                ## Do what we said we would do and append -2 to the measurements.
                ## This is probably because this tile is out of focus.
                ## Could implement some sort of focus metric to save us the hassle of trying to detect and extract.

                for cluster in cluster_array:
                    cluster['measurements_t'].append(-2)
                    cluster['measurements_c'].append(-2)
                ## Once all set to -2, let's skip the rest of the processing for this variable.
                continue
            else:
                offset_map_t = np.zeros_like(offset_map_c)
                offset_map_t[:,:,0] += t_c_offset_y+global_c[0]
                offset_map_t[:,:,1] += t_c_offset_x+global_c[1]
                offset_map_t += chrom_t


        logging.info("%s - Corrected offset map:\n%s" % (tile_id_idep, offset_map_t))


        ## With aligned images, assign intensities to the sequencing clusters.

        ### Direct cluster assignment.
        if assignment_method == 'direct':
            logging.info("%s - Assigning intensities directly." % (tile_id_idep))

            sub_img_half = sub_image_size // 2
            sigma = 1.0
            psf = GaussianTwoDimensions(sub_image_size, sigma)
            iinz_c = 0


            for cluster in cluster_array:

                cluster_x = cluster['x']
                cluster_y = cluster['y']

               ## Determine the correct offset to use for this cluster.
                x_index = int((cluster_x/tile_width)*(tile_width/alignment_subdiv_x))
                y_index = int((cluster_y/tile_height)*(tile_height/alignment_subdiv_y))

                try:
                    offset_y_t, offset_x_t = offset_map_t[y_index, x_index]
                    offset_y_c, offset_x_c = offset_map_c[y_index, x_index]
                except:
                    # logging.info("Cluster %d is out of bounds. Ignoring." % (cluster['cluster_id']))
                    cluster['measurements_t'].append(-1)
                    cluster['measurements_c'].append(-1)
                    continue

                cluster_y_t_offset = cluster_y-offset_y_t
                cluster_x_t_offset = cluster_x-offset_x_t

                cluster_y_c_offset = cluster_y-offset_y_c
                cluster_x_c_offset = cluster_x-offset_x_c

                ## Discard clusters that fall outside the bounds of the binding image.
                if not 0 < cluster_y_t_offset < tile_height or not 0 < cluster_x_t_offset < tile_width:
                    cluster['measurements_t'].append(-1)
                    cluster['measurements_c'].append(-1)
                    continue

                if not 0 < cluster_y_c_offset < tile_height or not 0 < cluster_x_c_offset < tile_width:
                    cluster['measurements_c'].append(-1)
                    cluster['measurements_t'].append(-1)
                    continue


                ## Extract cluster intensity.
                # Image t - binding image
                sub_img_t = np.array(tile_image_t[int(cluster_y_t_offset)-sub_img_half:int(cluster_y_t_offset)+sub_img_half+1,
                    int(cluster_x_t_offset)-sub_img_half:int(cluster_x_t_offset)+sub_img_half+1])
                # Image c - RNA probe image
                sub_img_c = np.array(tile_image_c[int(cluster_y_c_offset)-sub_img_half:int(cluster_y_c_offset)+sub_img_half+1,
                    int(cluster_x_c_offset)-sub_img_half:int(cluster_x_c_offset)+sub_img_half+1])

                if sub_img_t.shape != (sub_image_size, sub_image_size) or sub_img_c.shape != (sub_image_size, sub_image_size):
                    cluster['measurements_t'].append(-1)
                    cluster['measurements_c'].append(-1)
                    continue

                gaussian_t = psf*sub_img_t
                gaussian_c = psf*sub_img_c
                integrated_intensity_t = np.sum(gaussian_t)
                integrated_intensity_c = np.sum(gaussian_c)

                cluster['measurements_t'].append(float(integrated_intensity_t))
                cluster['measurements_c'].append(float(integrated_intensity_c))
                iinz_c += 1
                del sub_img_t, sub_img_c

            logging.info("%s - Assigned %d/%d non-zero intensities to %d clusters." % (tile_id_idep, iinz_c, len(cluster_array), len(cluster_array)))


    ## Once we have finished processing each datapoint in the tile, save the cluster_array as an intensity file.
    ## Format is: cluster_ud,x,y,*measurements
    out_file = os.path.join(data_out_dir, 'intensities' , '%s.int' % (tile_id))
    logging.info("%s - Saving measurements to disk: %s" % (tile_id_idep, out_file))
    out_fp = open(out_file, "w")

    for cluster in cluster_array:
        out_line = "%d,%d,%d" % (cluster['cluster_id'], cluster['x'], cluster['y'])
        for m in cluster['measurements_t']:
            out_line += ',%.2f' % (m)
        for m in cluster['measurements_c']:
            out_line += ',%.2f' % (m)

        out_line += '\n'
        out_fp.write(out_line)

    out_fp.close()
    logging.info("%s - Measurements saved to disk." % (tile_id_idep))

    # Clean up arrays to free memory.
    for cdict in cluster_array:
        del cdict
    del cluster_array
    del tile_image_t
    del tile_image_c
    gc.collect()



