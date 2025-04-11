import shutil
import logging
import sys

from tqdm import tqdm

import vtk
from slicer.util import *
import numpy as np
from numba import jit

from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.optimize import linear_sum_assignment
from skimage import segmentation, measure

import json
import os
import time
import slicer

from utils.config import module_path, debug, dynunet_pipeline_path


def run_segmentation(inputVolume_list,
                     do_affinereg,
                     input_is_bet,
                     task_dir,
                     tmp_path_in,
                     tmp_path_out,
                     python_executable,
                     ):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume_list: list of volumes passed as input to the model
    :param do_affinereg: whether to register input volumes to template image
    :param input_is_bet: whether the input image is already brain extracted
    :param task_dir: path to the task directory
    :param python_executable: path to python executable (for example a virtual environment)
    """

    if len(inputVolume_list) == 0:
        print("No input volume specified for current timepoint.")
        return
    for i in range(len(inputVolume_list)):
        if not inputVolume_list[i]:
            raise ValueError("Channel " + str(i + 1) + " input volume is invalid")

    startTime = time.time()
    logging.info('Processing started')

    # get the path to the external inference script
    ext_inference_script_path = os.path.join(dynunet_pipeline_path, "src", "inference.py")

    if os.path.isdir(tmp_path_in):
        shutil.rmtree(tmp_path_in)
    if os.path.isdir(tmp_path_out):
        shutil.rmtree(tmp_path_out)
    os.makedirs(tmp_path_in, exist_ok=True)
    os.makedirs(tmp_path_out, exist_ok=True)

    # create json file with the input and output paths
    inference_files_list = [
        {
            "images": {str(ch): os.path.join(tmp_path_in, "img_tmp_" + "{:04d}".format(ch) + ".nii.gz") for ch in
                       range(len(inputVolume_list))
                       },
            "save_path": os.path.join(tmp_path_out, "output.nii.gz")
        },
    ]

    inference_files_path = os.path.join(tmp_path_in, "inference_files.json")
    with open(inference_files_path, "w") as f:
        json.dump(inference_files_list, f, indent=4)

    # save image to nifti file that can be read by the docker
    for channel, input_volume in enumerate(inputVolume_list):
        img_tmp_path = os.path.join(tmp_path_in, "img_tmp_" + "{:04d}".format(channel) + ".nii.gz")
        # saveNode overwrites the storage node with the original filename, which is needed when the user wants to
        # rerun a section to get the report directory, hence, we overwrite the filename again
        if input_volume.GetStorageNode():
            orig_fname = input_volume.GetStorageNode().GetFileName()
        else:
            orig_fname = ""

        saveNode(input_volume, img_tmp_path)

        if orig_fname:
            input_volume.GetStorageNode().SetFileName(orig_fname)

    # provide paths (they must be absolute paths)
    out_folder = os.path.realpath(tmp_path_out)

    # check if all paths exist
    assert (os.path.isfile(inference_files_path)), f"Is not a file: {inference_files_path} ..."
    assert (os.path.isdir(out_folder)), f"Is not a folder: {out_folder} ..."

    if python_executable == "":
        python_executable = "python"

    # check if the system can find the external python
    if shutil.which(python_executable) is None:
        # warning
        logging.warning(f"Python executable {python_executable} not found ... \n"
                        f"Falling back to PythonSlicer executable ...")
        python_executable = sys.executable


    # check if the external python is a virtual environment and if so, add the activate command to the beginning
    activate_path = python_executable.split(os.sep)[:-1] + ["activate"]
    if os.path.isfile(os.sep.join(activate_path)):
        python_executable = os.sep.join(activate_path) + "; " + python_executable

    cmd = (f"{python_executable} {ext_inference_script_path} "
           f"-task_dir {task_dir} "
           f"-args_file config/infer_args.json "
           f"-inference_files_path {inference_files_path} "
           f"-out_dir {out_folder}")

    if do_affinereg:
        cmd += " -reg "
    else:
        cmd += " -no-reg "

    if input_is_bet:
        cmd += " -input_is_bet "
    else:
        cmd += " -no-input_is_bet "

    if debug: print(f"Running command in run_command module: ")
    if debug: print(cmd)

    # run scripted CLI
    parameters = {"command": cmd}
    cliNode = slicer.cli.run(slicer.modules.run_command, None, parameters=parameters)

    stopTime = time.time()
    logging.info(f'Python script completed in {stopTime - startTime:.2f} seconds')

    return cliNode



@jit(nopython=True)
def keep_contained_lines(boundary_coords, mask, sample_distance=0.2):
    """
    Given a set of boundary coordinates, this function returns the lines that are fully contained within the mask.
    :param boundary_coords: numpy array of shape (nb_boundary_pixels, 2)
    :param mask: binary numpy array of shape (width, height)
    :param sample_distance: the distance between samples on the line to check if the line is contained
    :return: a list of lines, where each line is a list of two points
    """
    nb_coords = len(boundary_coords)
    contained_lines = []
    # loop over all pairs of boundary pixels
    for i in range(nb_coords):
        for j in range(i + 1, nb_coords):

            # get the samples on the line between the two boundary pixels
            distance = np.sqrt((boundary_coords[i, 0] - boundary_coords[j, 0]) ** 2 +
                               (boundary_coords[i, 1] - boundary_coords[j, 1]) ** 2)
            num_samples = int(distance / sample_distance)
            x_vals = np.linspace(boundary_coords[i, 0], boundary_coords[j, 0], num=num_samples)
            y_vals = np.linspace(boundary_coords[i, 1], boundary_coords[j, 1], num=num_samples)

            # check if all samples are within the mask
            for x, y in zip(x_vals, y_vals):
                x_mask = int(np.round(x))
                y_mask = int(np.round(y))
                if mask[x_mask, y_mask] == 0:
                    break
            else:  # if the loop completes without breaking
                # the line is fully contained within the mask
                contained_lines.append([boundary_coords[i], boundary_coords[j]])

    return contained_lines


def get_all_contained_lines(plane):
    """
    Given a plane, this function returns all the lines from one boundary pixel to another that are fully contained
    within the foreground.

    :param plane: a 2D numpy array where background is 0 and foreground is > 0
    :return: a list of lines, where each line is a list of two points
    """
    mask = plane > 0

    # empty planes have no lines
    if np.sum(mask) == 0:
        return []

    # get the boundary of the mask
    mask_boundary = segmentation.find_boundaries(mask, connectivity=1, mode='inner', background=0)

    # pixels at the image boundary should also be considered as boundary pixels if they are part of the mask foreground
    mask_boundary[0, :] = mask[0, :]
    mask_boundary[-1, :] = mask[-1, :]
    mask_boundary[:, 0] = mask[:, 0]
    mask_boundary[:, -1] = mask[:, -1]

    # get the coordinates of the boundary
    boundary_coords = np.argwhere(mask_boundary)  # shape: (nb_boundary_pixels, 2)

    # get the lines that are fully contained within the mask
    contained_lines = keep_contained_lines(boundary_coords, mask, sample_distance=0.2 if not debug else 3)

    return contained_lines



@jit(nopython=True)
def get_max_orthogonal_line_product_coords_plane(all_line_coords_np, degree_tol=0):
    """
    Given a set of lines, this function returns the two lines that are orthogonal to each other and have the largest
    product of their lengths.
    :param all_line_coords_np: numpy array of shape (nb_lines, 2, 2) (lines x 2 points x 2 coordinates)
    :param degree_tol: the tolerance in degrees for the orthogonality of the lines
    :return: a list of two lines, where each line is a list of two points and the product of their lengths is maximal
    """

    if debug:
        # pick only 1000 lines
        all_line_coords_np = all_line_coords_np[:10]

    radians_tol = np.deg2rad(degree_tol)

    ortho_line_products_max = 0  # current maximum product of line lengths

    # precalculate the line lengths and angles with respect to the x-axis
    vecs = all_line_coords_np[:, 1, :] - all_line_coords_np[:, 0, :]
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    line_lengths = np.sqrt(np.sum(vecs ** 2, axis=1))

    # sort all_line_coords_np, vecs, angles, line_lengths by angle
    # this is necessary to make the search for orthogonal lines more efficient
    line_lengths_sorted_idx = np.argsort(angles)
    all_line_coords_np = all_line_coords_np[line_lengths_sorted_idx]
    angles = angles[line_lengths_sorted_idx]
    line_lengths = line_lengths[line_lengths_sorted_idx]

    # get the product of all pairs of lines
    num_lines = all_line_coords_np.shape[0]

    # indices of the two lines with the maximum product
    max_l1 = -1
    max_l2 = -1

    start_idx = 0  # start index for the search of orthogonal lines (initially 0) to avoid checking all angles
    for l1 in range(num_lines):
        angle = angles[l1]

        # condition 1: the angle must be orthogonal (within a certain tolerance radians_tol)
        # exact orthogonal angle
        o = angle + np.pi / 2

        # range of acceptable directions
        r_min = o - radians_tol - 1e-7  # subtract a small value to avoid numerical issues
        r_max = o + radians_tol + 1e-7

        # to avoid checking all angles, we can update the start index to the first angle that is within the orthogonal
        # range
        # update the start idx to the first angle that is >= r_min
        while angles[start_idx] >= r_min and start_idx > 0:
            start_idx -= 1
        while start_idx < num_lines and angles[start_idx] < r_min:
            start_idx += 1

        if start_idx == num_lines:
            break

        # loop over all angles from the start index (break when the angle becomes too large for the orthogonal range)
        for l2 in range(start_idx, num_lines):
            if angles[l2] > r_max:  # this angle and following angles (larger, because of sorting) are too large
                break

            # condition 2: vector length product must be larger than the current maximum
            line_length_product = line_lengths[l1] * line_lengths[l2]

            if not line_length_product > ortho_line_products_max:
                continue

            # condition 3: make sure that the two lines are intersecting
            l1x_min = min(all_line_coords_np[l1, 0, 0], all_line_coords_np[l1, 1, 0])
            l2x_max = max(all_line_coords_np[l2, 0, 0], all_line_coords_np[l2, 1, 0])
            if l1x_min > l2x_max:
                continue

            l2x_min = min(all_line_coords_np[l2, 0, 0], all_line_coords_np[l2, 1, 0])
            l1x_max = max(all_line_coords_np[l1, 0, 0], all_line_coords_np[l1, 1, 0])
            if l2x_min > l1x_max:
                continue

            l1y_min = min(all_line_coords_np[l1, 0, 1], all_line_coords_np[l1, 1, 1])
            l2y_max = max(all_line_coords_np[l2, 0, 1], all_line_coords_np[l2, 1, 1])
            if l1y_min > l2y_max:
                continue

            l2y_min = min(all_line_coords_np[l2, 0, 1], all_line_coords_np[l2, 1, 1])
            l1y_max = max(all_line_coords_np[l1, 0, 1], all_line_coords_np[l1, 1, 1])
            if l2y_min > l1y_max:
                continue

            # all conditions are met, so we can update the maximum product
            ortho_line_products_max = line_length_product
            max_l1 = l1
            max_l2 = l2

    # get the coordinates of the two lines with the maximum product
    ortho_line_products_max_idx_coords = [all_line_coords_np[max_l1, :, :], all_line_coords_np[max_l2, :, :]]

    return ortho_line_products_max_idx_coords



def get_instance_segmentation_by_connected_component_analysis(bin_seg):
    """
    Convert a binary segmentation into a segmentation where each connected component gets a different label > 0.
    :param bin_seg: Binary segmentation with shape W, H, D and labels 0 and 1
    :return: instance_seg: Segmentation with shape W, H, D and labels 0, ..., num_ccs (number of connected components)
    """
    # get connected components (each CC will get a different label)
    instance_seg, num_ccs = measure.label(bin_seg, background=0, return_num=True)
    return instance_seg



def match_instance_segmentations_by_IoU(instance_segs):
    def get_matched_labels(seg1, seg2):
        # calculate the overlap between each pair of connected components
        ccs_t1 = np.array([c for c in np.unique(seg1) if c != 0])
        ccs_t2 = np.array([c for c in np.unique(seg2) if c != 0])

        iou = np.zeros((len(ccs_t1), len(ccs_t2)))
        for i, cc1 in enumerate(ccs_t1):
            for j, cc2 in enumerate(ccs_t2):
                overlap = np.sum(np.logical_and(seg1 == cc1, seg2 == cc2))
                iou[i, j] = overlap / np.sum(np.logical_or(seg1 == cc1, seg2 == cc2))

        # find optimal matching based on minimizing average iou
        row_ind, col_ind = linear_sum_assignment(-iou)

        labels_t1 = ccs_t1[row_ind]
        labels_t2 = ccs_t2[col_ind]

        return labels_t1, labels_t2

    def relabel_seg(ref_labs, matched_labs, seg):
        # if some labels in seg are not in matched_labs, we need to assign them to new (arbitrary) labels
        orig_labels = np.unique(seg)
        unmatched_labels = np.setdiff1d(orig_labels, matched_labs)
        # assign new labels are the next available label (not in ref_labs)
        unmatched_labels_new = np.array(list(set(range(len(orig_labels))) - set(ref_labs))[0:len(unmatched_labels)])

        seg_out = np.zeros_like(seg)
        for src, dest in zip(matched_labs.tolist() + unmatched_labels.tolist(),
                             ref_labs.tolist() + unmatched_labels_new.tolist()):
            seg_out[seg == src] = dest

        return seg_out

    num_segs = len(instance_segs)

    # change labels of each segmentation to match to the previous timepoint
    for i in range(1, num_segs):
        seg1 = instance_segs[i - 1]
        seg2 = instance_segs[i]

        labels_t1, labels_t2 = get_matched_labels(seg1, seg2)
        # update the labels in seg2
        instance_segs[i] = relabel_seg(labels_t1, labels_t2, seg2)

    return instance_segs


def get_max_orthogonal_line_product_coords_multipairs_and_volumes(seg, opening_radius=None):
    """

    :param seg: a 3D numpy array where background is 0 and foreground is > 0
    :param opening_radius: the radius of the circle opening operation to apply to the slice segmentation before finding
    the orthogonal lines
    :return: a list of line pairs, where each line pair is a list of two lines and is associated with one of the
    connected components. Shape: (num connected components, 2 lines, 2 points, 3 coordinates)
    """

    # get connected components (each CC will get a different label)
    seg_cc, num_ccs = measure.label(seg, background=0, return_num=True)

    # get the largest connected component
    counts = np.bincount(seg_cc.flat)
    cc_label_sorted = np.argsort(counts)[::-1]

    # get a line pair that maximizes the product of the orthogonal lines for each connected component
    coords_ijk = np.zeros((num_ccs, 2, 2, 3))  # number of connected components, 2 lines, 2 points, 3 coordinates
    volumes = np.zeros(num_ccs)
    cc_idx = 0  # cc_idx is the index of the connected component sorted by size
    for cc_lab in cc_label_sorted:  # , cc_label is the arbitrary label of the connected component
        if cc_lab == 0:  # skip background
            continue
        # for this component, get coordinates of orthogonal lines
        seg_one_cc = np.array(seg_cc == cc_lab).astype(np.int16)
        coords_ijk[cc_idx] = get_max_orthogonal_line_product_coords(seg_one_cc, opening_radius)
        volumes[cc_idx] = np.sum(seg_one_cc)
        cc_idx += 1

    # discard entries with nan values
    has_nan = np.isnan(coords_ijk).any(axis=(1, 2, 3))
    print("Removing entries of line coordinates with nan values for connected components. has_nan = ", has_nan)
    coords_ijk = coords_ijk[~has_nan]
    volumes = volumes[~has_nan]

    return coords_ijk, volumes


def get_max_orthogonal_line_product_coords(seg, valid_axes=(0, 1, 2), center=None, center_tol=5,
                                           opening_radius=None):
    """
    Given a segmentation, this function returns the two lines that are orthogonal to each other and have the largest
    product of their lengths. It does so by looping over views (axial, coronal, sagittal) and planes and finding the
    two lines with the largest product of their lengths that are orthogonal to each other.
    :param valid_axes: the views to consider for finding the orthogonal lines
    :param center: coordinates of a point that the plane must be close to
    :param center_tol: tolerance for the distance between the plane and the center point
    :param seg: a 3D numpy array where background is 0 and foreground is > 0
    :param opening_radius: the radius of the circle opening operation to apply to the slice segmentation before finding
    the orthogonal lines
    :return: a list of two lines, where each line is a list of two points and the product of their lengths is maximal
    """

    # loop over all planes
    max_ortho_line_coords = []
    max_line_length_product = 0

    for view in valid_axes:  # loop over all views
        slices = range(seg.shape[view])
        for i in tqdm(slices):  # loop over all planes
            # get distance of the slice to the center point
            if center is not None:
                dist = np.abs(center[view] - i)
                if dist > center_tol:
                    continue  # skip the slice if it is too far from the center point

            # get the plane
            plane = seg.take(i, axis=view)

            # check if the plane has any object
            if np.sum(plane) == 0:
                continue

            # apply circle opening operation
            if opening_radius is not None:
                plane = circle_opening(plane, [1], opening_radius)

            # check if the plane has any object after circle opening
            if np.sum(plane) == 0:
                continue

            # get all contained lines
            all_line_coords = get_all_contained_lines(plane)
            if len(all_line_coords) < 2:
                continue

            all_line_coords_np = np.array(all_line_coords)  # lines x 2 points x 2 coordinates
            ortho_line_coords = get_max_orthogonal_line_product_coords_plane(all_line_coords_np)

            # calculate line length product
            line_length1 = np.sqrt(np.sum((ortho_line_coords[0][1] - ortho_line_coords[0][0]) ** 2))
            line_length2 = np.sqrt(np.sum((ortho_line_coords[1][1] - ortho_line_coords[1][0]) ** 2))
            line_length_product = line_length1 * line_length2

            if line_length_product > max_line_length_product:
                max_line_length_product = line_length_product

                # insert the slice index i in the orthogonal line coordinates in the view index dimension to go from
                # 2D to 3D coordinates
                ortho_line_coords[0] = np.insert(ortho_line_coords[0], view, i, axis=1)
                ortho_line_coords[1] = np.insert(ortho_line_coords[1], view, i, axis=1)

                max_ortho_line_coords = ortho_line_coords

                if debug:
                    break

    max_ortho_line_coords = np.array(max_ortho_line_coords)

    # print("Max product:", max_line_length_product)
    # print("Max product coords:", max_ortho_line_coords)

    return max_ortho_line_coords



def circle_opening(seg, labels, radius):
    """
    Given a segmentation, this function performs a circle opening operation on the specified labels.
    This means it returns a new binary segmentation where the specified labels are removed if they are within a certain
    distance from the boundary of the combined labels. The threshold corresponds to the radius of a circle that can
    be placed in the combined structure without intersecting the boundary.
    :param seg: input segmentation with multiple labels
    :param labels: list of labels to combine for the circle opening operation
    :param radius: the distance from the boundary of the combined labels (corresponds to the radius of the circle)
    :return: the circle-opened segmentation
    """
    # create a mask
    mask = np.zeros_like(seg)
    for label in labels:
        mask = np.logical_or(mask, seg == label)

    # get the distance map
    dist = distance_transform_edt(mask)

    # threshold the distance map
    dist_thresh_mask = dist >= radius

    # make circular structuring element
    x, y = np.ogrid[:radius * 2 - 1, :radius * 2 - 1]
    mask = (x - (radius - 1)) ** 2 + (y - (radius - 1)) ** 2 < radius ** 2
    structure = np.zeros((radius * 2 - 1, radius * 2 - 1))
    structure[mask] = 1

    # dilate the mask by the treshold
    seg_circle_open = binary_dilation(dist_thresh_mask, structure)

    return seg_circle_open

    # find the plane in which the line coordinates are placed



def sphere_opening(seg, labels, radius):
    # create a mask
    mask = np.zeros_like(seg)
    for label in labels:
        mask = np.logical_or(mask, seg == label)

    dist = distance_transform_edt(mask)

    dist_thresh_mask = dist >= radius

    # make spherical structuring element
    x, y, z = np.ogrid[:radius * 2 - 1, :radius * 2 - 1, :radius * 2 - 1]
    mask = (x - (radius - 1)) ** 2 + (y - (radius - 1)) ** 2 + (z - (radius - 1)) ** 2 < radius ** 2
    structure = np.zeros((radius * 2 - 1, radius * 2 - 1, radius * 2 - 1))
    structure[mask] = 1

    # dilate the mask by the threshold
    seg_open = binary_dilation(dist_thresh_mask, structure)

    return seg_open



def find_plane_of_coords(coords):
    """
    Given two lines, this function returns the axis of the plane in which the lines are placed.
    :param coords: a list of two lines, where each line is a list of two points with shape (2 lines x 2 points x 3 coordinates)
    :return: the axis that is perpendicular to the plane in which the lines are placed and the constant plane coordinate
    """
    l1p1 = coords[0][0]
    l1p2 = coords[0][1]
    l2p1 = coords[1][0]
    l2p2 = coords[1][1]

    const_val = None
    for axis in range(3):
        if l1p1[axis] == l1p2[axis] and l2p1[axis] == l2p2[axis]:
            axis_of_const_coordinate_values = axis
            const_val = l1p1[axis]
            break
    else:
        raise ValueError(f"The lines are not parallel to any of the planes. The coordinates are: \n{coords}")

    assert (const_val.is_integer()), f"The constant coordinate value is not an integer: {const_val}"

    return axis_of_const_coordinate_values, const_val.astype(int)



def find_closest_plane(coords):
    l1p1 = coords[0][0]
    l1p2 = coords[0][1]
    l2p1 = coords[1][0]
    l2p2 = coords[1][1]

    v1 = l1p2 - l1p1
    v2 = l2p2 - l2p1

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # find the axis with the largest component of the normal
    axis = np.argmax(np.abs(normal))
    return axis



def point_closest_to_two_lines(coords):
    p11 = coords[0][0]
    p12 = coords[0][1]
    p21 = coords[1][0]
    p22 = coords[1][1]

    e1 = p12 - p11
    e2 = p22 - p21
    normal = np.cross(e1, e2)
    if normal[0] == 0 and normal[1] == 0 and normal[2] == 0:
        # return the center of all four points
        return (p11 + p12 + p21 + p22) / 4
    t1 = np.dot(np.cross(e2, normal), (p21 - p11)) / np.dot(normal, normal)
    t2 = np.dot(np.cross(e1, normal), (p21 - p11)) / np.dot(normal, normal)
    i1 = p11 + t1 * e1
    i2 = p21 + t2 * e2
    out = (i1 + i2) / 2

    # if coordinates are larger or smaller than the original coordinates, return the original coordinates
    if np.any(out > np.max(coords)) or np.any(out < np.min(coords)):
        return (p11 + p12 + p21 + p22) / 4
    return out


def circle_opening_on_slices_perpendicular_to_axis(segmentationArray, axes, labels, radius, slice_idx=None):
    """
    Given a 3D segmentation, this function performs a circle opening operation on the specified labels in the slices
    perpendicular to each of the specified axes, then returns the union of segmentations from all axes.
    :param segmentationArray: a 3D numpy array with the segmentation
    :param axes: the axis perpendicular to the slices where the circle opening operation is performed
    :param labels: list of labels to combine for the circle opening operation
    :param radius: radius of the circle for the circle opening operation
    :param slice_idx: if the slice index is specified, the circle opening operation is only performed on that slice. If
    None, the operation is performed on all slices along the axis. If multiple axes are specified, slice_idx must be
    a list of the same length as axes.
    :return: the circle-opened segmentation along the specified axis
    """
    out = np.zeros_like(segmentationArray)

    for axis in axes:
        out_ax = np.zeros_like(segmentationArray)
        if slice_idx is not None:
            if isinstance(slice_idx, list):
                slice_range = slice_idx[axis]
            else:
                slice_range = [slice_idx]
        else:
            slice_range = range(segmentationArray.shape[axis])

        for slc_idx in slice_range:
            slc = np.take(segmentationArray, slc_idx, axis=axis)
            out_slc = circle_opening(slc, labels, radius=radius).astype(np.int16)

            # replace the slice in the output with the circle-opened slice  not using np.insert
            if axis == 0:
                out[slc_idx, :, :] = out_slc
            elif axis == 1:
                out[:, slc_idx, :] = out_slc
            elif axis == 2:
                out[:, :, slc_idx] = out_slc

        # combine output with the previous outputs
        out = np.logical_or(out, out_ax).astype(segmentationArray.dtype)

    return out



def get_ijk_to_world_matrix(node):
    """
    Given a node, this function returns the IJK to world matrix of the binary labelmap
    :param node: the segmentation node or volume node
    :return: the IJK to world matrix of the binary labelmap
    """
    binaryLabelmapRepresentation = slicer.vtkOrientedImageData()

    ijkToWorld = vtk.vtkMatrix4x4()
    if node.IsA("vtkMRMLSegmentationNode"):
        segmentId = node.GetSegmentation().GetSegmentIDs()[0]
        node.GetBinaryLabelmapRepresentation(segmentId, binaryLabelmapRepresentation)
        binaryLabelmapRepresentation.GetImageToWorldMatrix(ijkToWorld)
    elif node.IsA("vtkMRMLScalarVolumeNode"):
        node.GetIJKToRASMatrix(ijkToWorld)
    else:
        raise ValueError("Node must be a segmentation node or a volume node")
    return ijkToWorld



def transform_ijk_to_world_coord(ijk, ijkToWorld_matrix):
    """
    Given an IJK coordinate and the IJK to world matrix, this function returns the world coordinate
    :param ijk: the IJK coordinates (x, y, z)
    :param ijkToWorld_matrix: the IJK to world matrix
    :return: the world coordinate (x, y, z)
    """
    x_idx, y_idx, z_idx = 2, 1, 0  # swap z and x from numpy to slicer
    return ijkToWorld_matrix.MultiplyPoint([ijk[x_idx], ijk[y_idx], ijk[z_idx], 1])[
           :3]  # [:3] to remove the homogeneous coordinate (1)



def transform_world_to_ijk_coord(world, worldToIJK_matrix):
    """
    Given a world coordinate and the world to IJK matrix, this function returns the IJK coordinate
    :param world: the world coordinates (x, y, z)
    :param worldToIJK_matrix: the world to IJK matrix
    :return: the IJK coordinate (x, y, z)
    """

    kji = worldToIJK_matrix.MultiplyPoint([world[0], world[1], world[2], 1])[
          :3]  # [:3] to remove the homogeneous coordinate (1)
    return [int(round(kji[2])), int(round(kji[1])), int(round(kji[0]))]  # swap z and x from slicer
