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
    Run the segmentation model using the provided input volumes and parameters.

    Args:
        inputVolume_list: list of volumes passed as input to the model
        do_affinereg: whether to register input volumes to template image
        input_is_bet: whether to register input volumes to template image
        task_dir: path to the task directory that contains the segmentation model
        tmp_path_in: path to the temporary input directory
        tmp_path_out: path to the temporary output directory
        python_executable: path to python executable (for example a virtual environment)
    """

    if len(inputVolume_list) == 0:
        print("No input volume specified for current timepoint.")
        return
    for i in range(len(inputVolume_list)):
        if not inputVolume_list[i]:
            raise ValueError("Channel " + str(i + 1) + " input volume is invalid")

    startTime = time.time()  # to time the segmentation
    logging.info('Processing started')

    # get the path to the external inference script
    ext_inference_script_path = os.path.join(dynunet_pipeline_path, "src", "inference.py")

    # empty the temporary input and output directories
    if os.path.isdir(tmp_path_in):
        shutil.rmtree(tmp_path_in)
    if os.path.isdir(tmp_path_out):
        shutil.rmtree(tmp_path_out)
    os.makedirs(tmp_path_in, exist_ok=True)
    os.makedirs(tmp_path_out, exist_ok=True)

    # create json file with the input and output paths as input for the inference script
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

    # save input nodes as nifti files in the temporary input directory
    for channel, input_volume in enumerate(inputVolume_list):
        img_tmp_path = os.path.join(tmp_path_in, "img_tmp_" + "{:04d}".format(channel) + ".nii.gz")
        # saveNode overwrites the storage node with the original filename, which is needed when the user wants to
        # rerun a section to get the report directory, hence, we overwrite the filename again
        if input_volume.GetStorageNode():
            orig_fname = input_volume.GetStorageNode().GetFileName()
        else:
            orig_fname = ""

        saveNode(input_volume, img_tmp_path)

        input_volume.GetStorageNode().SetFileName(orig_fname)

    # provide paths (they must be absolute paths)
    out_folder = os.path.realpath(tmp_path_out)

    # check if all paths exist
    assert (os.path.isfile(inference_files_path)), f"Is not a file: {inference_files_path} ..."
    assert (os.path.isdir(out_folder)), f"Is not a folder: {out_folder} ..."

    # By default, use the PythonSlicer executable
    if python_executable == "":
        python_executable = sys.executable

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

    cmd = (
        f'"{python_executable}" "{ext_inference_script_path}" '
        f'-task_dir "{task_dir}" '
        f'-args_file "config/infer_args.json" '
        f'-inference_files_path "{inference_files_path}" '
        f'-out_dir "{out_folder}"'
    )

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
    Given a set of boundary coordinates, this method returns the lines that are fully contained within the mask.

    Args:
        boundary_coords: numpy array of shape (nb_boundary_pixels, 2)
        mask: binary numpy array of shape (width, height)
        sample_distance: the distance between samples on the line to check if the line is contained

    Returns:
        contained_lines: a list of lines, where each line is a list of two points.
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

    Args:
        plane: a 2D numpy array where background is 0 and foreground is > 0

    Returns:
        contained_lines: a list of lines, where each line is a list of two points
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
def get_max_orthogonal_line_product_coords_plane(line_coords, degree_tol=0):
    """

    Given a set of lines, this function returns the two lines that are orthogonal to each other and have the largest
    product of their lengths. The third condition is that the lines must intersect. The function
    loops over all lines and checks if the angle between the lines is orthogonal (within the given tolerance).
    It is assumed that all lines lie in a plane (one of the acquisition planes). This plane can have any orientation in
    world space.

    Args:
        line_coords: numpy array of shape (nb_lines, 2, 3) (lines x 2 points x 3 coordinates)
        degree_tol: the tolerance in degrees for the orthogonality of the lines

    Returns:
        ortho_line_max_prod_coords: a list of two lines, where each line is a list of two points and the product
        of their lengths is maximal
    """

    num_lines = len(line_coords)
    if not num_lines >= 2:
        return None

    radians_tol = np.deg2rad(degree_tol)

    # precalculate the line lengths and angles with respect to one of the lines in the plane
    vecs_world = line_coords[:, 1, :] - line_coords[:, 0, :]

    ref_line = vecs_world[0, :]  # reference line

    for vec_idx in range(1, len(vecs_world)):
        ref_line2 = vecs_world[vec_idx, :]  # reference line 2 to calculate the normal
        crossprod = np.cross(ref_line, ref_line2)
        normcrossprod = np.linalg.norm(crossprod)

        if normcrossprod > 1e-6:  # check if the two lines are not parallel. set threshold to avoid numerical issues
            break
    else:
        #print("No orthogonal vectors found.")
        return None

    # get the plane normal
    normal = crossprod / np.linalg.norm(crossprod)

    # get the signed angle between the reference line and the other lines by projecting on the normal
    # sorting the angles is done to assess only roughly orthogonal lines and avoid O(n^2) complexity
    angles = np.arctan2(np.dot(normal, np.cross(ref_line, vecs_world).T), np.dot(ref_line, vecs_world.T))

    # calculate the line lengths
    line_lengths = np.sqrt(np.sum(vecs_world ** 2, axis=1))

    # sort angles, line_coords, line_lengths by angle
    # this is necessary to make the search for orthogonal lines more efficient
    angles_sorted_idx = np.argsort(angles)
    angles = angles[angles_sorted_idx]
    line_coords = line_coords[angles_sorted_idx]
    line_lengths = line_lengths[angles_sorted_idx]

    # keep track of current maximum product of line lengths and corresponding line indices
    ortho_line_products_max = 0
    max_l1 = -1
    max_l2 = -1

    # to check if the lines are in the same plane, we later project the lines onto a 2D plane
    # we need to define 2 basis vectors and a coordinate center to describe coordinates in the plane in 2D
    # pick an arbitrary vector that is orthogonal to the normal
    ax1 = np.argsort(np.abs(normal))[-1]  # largest component of the normal
    ax2 = np.argsort(np.abs(normal))[-2]  # second largest component of the normal

    basis1 = np.zeros(3)
    basis1[ax1] = normal[ax2]
    basis1[ax2] = -normal[ax1]

    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    # the center can be the first point of the first line
    center2D = line_coords[0, 0, :]

    start_idx = 0  # start index for the search of orthogonal lines (initially 0) to avoid checking all angles
    for l1 in range(num_lines):
        angle = angles[l1]

        # condition 1: the angle must be orthogonal (within tolerance radians_tol)
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

            # condition 3: make sure that the two lines are intersecting in the plane
            # get the 2x2 points of the two lines in terms of the basis vectors defined above
            l1p1 = line_coords[l1, 0, :]
            l1p2 = line_coords[l1, 1, :]
            l2p1 = line_coords[l2, 0, :]
            l2p2 = line_coords[l2, 1, :]

            # project the points onto the 2D plane
            l1p1 = [np.dot(l1p1 - center2D, basis1), np.dot(l1p1 - center2D, basis2)]
            l1p2 = [np.dot(l1p2 - center2D, basis1), np.dot(l1p2 - center2D, basis2)]
            l2p1 = [np.dot(l2p1 - center2D, basis1), np.dot(l2p1 - center2D, basis2)]
            l2p2 = [np.dot(l2p2 - center2D, basis1), np.dot(l2p2 - center2D, basis2)]

            # this solution of checking if the lines intersect is from https://stackoverflow.com/a/9997374
            # it doesn't work for parallel lines, but we already checked that the lines are orthogonal
            def ccw(p1, p2, p3):
                # Check if the points p1, p2, p3 are in counter-clockwise order
                return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

            # Return true if line segments AB and CD intersect
            def intersect(a1, a2, b1, b2):
                return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)

            if not intersect(l1p1, l1p2, l2p1, l2p2):
                continue

            # all conditions are met, so we can update the maximum product
            ortho_line_products_max = line_length_product
            max_l1 = l1
            max_l2 = l2

    if max_l1 == -1 or max_l2 == -1:
        if debug: print("No orthogonal lines found")
        return None

    # get the coordinates of the two lines with the maximum product
    ortho_line_max_prod_coords = [line_coords[max_l1, :, :], line_coords[max_l2, :, :]]

    return ortho_line_max_prod_coords


def get_instance_segmentation_by_connected_component_analysis(bin_seg):
    """
    Convert a binary segmentation into a segmentation where each connected component gets a different label > 0.

    Args:
        bin_seg: Binary segmentation with shape W, H, D and labels 0 and 1

    Returns:
        instance_seg: Segmentation with shape W, H, D and labels 0, ..., num_ccs (number of connected components)
    """
    # get connected components (each CC will get a different label)
    instance_seg, num_ccs = measure.label(bin_seg, background=0, return_num=True)
    return instance_seg



def match_instance_segmentations_by_IoU(instance_segs):
    """
    Given a list of instance segmentations, this function relabels the segmentations so that the labels match
    between timepoints. It does so by finding the optimal matching of labels based on the IoU (Intersection over
    Union) of the instances.

    Args:
        instance_segs: list of instance segmentations, where each segmentation is a 3D numpy array with shape (W, H, D) and
        labels 0, ..., num_instances (number of instances)
    Returns:
        matched_instance_segs: list of instance segmentations with matching labels between timepoints

    """
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

    matched_instance_segs = instance_segs

    return matched_instance_segs


def get_max_orthogonal_line_product_coords(seg, valid_axes=(0, 1, 2), center=None, center_tol=5,
                                           opening_radius=None, ijkToWorld=None):
    """
    Given a segmentation, this function returns the two lines that are orthogonal to each other and have the largest
    product of their lengths. It does so by looping over views (axial, coronal, sagittal) and planes and finding the
    two lines with the largest product of their lengths that are orthogonal to each other.

    Args:
        seg: a 3D numpy array where background is 0 and foreground is > 0. The array defines the IJK space.
        valid_axes: the IJK orientations to consider for finding the orthogonal lines
        center: IJK coordinates of a point that the plane must be close to
        center_tol: IJK tolerance for the distance between the plane and the center point
        opening_radius: the radius of the circle opening operation to apply to the slice segmentation before finding
        the orthogonal lines
        ijkToWorld: the transformation matrix to convert from voxel coordinates to world coordinates

    Returns:
        max_ortho_line_coords: a list of two lines, where each line is a list of two points and the product of their
        lengths is maximal. The coordinates are in world space.
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
            all_line_coords_np_3d = np.insert(all_line_coords_np, view, i, axis=2)

            # convert the coordinates to world space
            # swap z and x from numpy to slicer
            all_line_coords_np_3d_swapped = all_line_coords_np_3d[:, :, [2, 1, 0]]
            # append 1 to the coordinates (homogeneous coordinates)
            all_line_coords_np_3d_homo = np.insert(all_line_coords_np_3d_swapped, 3, 1, axis=2)
            # matrix multiplication with einsum
            all_line_coords_np_3d_world = np.einsum('ab,cdb->cda',
                                                    slicer.util.arrayFromVTKMatrix(ijkToWorld),
                                                    all_line_coords_np_3d_homo)[:, :, :3]


            ortho_line_coords = get_max_orthogonal_line_product_coords_plane(all_line_coords_np_3d_world,
                                                                             degree_tol=0.1)

            if not ortho_line_coords:
                continue

            # calculate line length product
            line_length1 = np.sqrt(np.sum((ortho_line_coords[0][1] - ortho_line_coords[0][0]) ** 2))
            line_length2 = np.sqrt(np.sum((ortho_line_coords[1][1] - ortho_line_coords[1][0]) ** 2))
            line_length_product = line_length1 * line_length2

            if line_length_product > max_line_length_product:
                max_line_length_product = line_length_product

                max_ortho_line_coords = ortho_line_coords

    max_ortho_line_coords = np.array(max_ortho_line_coords)

    # print("Max product:", max_line_length_product)
    # print("Max product coords:", max_ortho_line_coords)

    return max_ortho_line_coords



def circle_opening(seg, labels, radius):
    """
    Given a segmentation, this function performs a circle opening operation on the specified labels.
    The opening operation is performed in 3 steps:
    1. A mask is created as the union of the specified labels
    2. The distance map of the mask is calculated and used to threshold the mask such that only the pixels that are
    at least radius away from the boundary of the mask are kept. This mask corresponds to the valid centers of the
    structuring element (the circle).
    3. The mask is dilated by the structuring element (the circle) to get the final segmentation (the circle opened mask).

    Args:
        seg: a 2D numpy array where background is 0 and foreground is > 0
        labels: list of labels to combine for the circle opening operation
        radius: the distance from the boundary of the combined labels (corresponds to the radius of the circle)
    Returns:
        seg_circle_open: a 2D numpy array where background is 0 and foreground is 1
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

    # dilate the mask by the threshold
    seg_circle_open = binary_dilation(dist_thresh_mask, structure)

    return seg_circle_open

    # find the plane in which the line coordinates are placed


def sphere_opening(seg, labels, radius):
    """
    Given a segmentation, this function performs a sphere opening operation on the specified labels.
    The opening operation is performed in 3 steps:
    1. A mask is created as the union of the specified labels
    2. The distance map of the mask is calculated and used to threshold the mask such that only the pixels that are
    at least radius away from the boundary of the mask are kept. This mask corresponds to the valid centers of the
    structuring element (the sphere).
    3. The mask is dilated by the structuring element (the sphere) to get the final segmentation (the sphere opened mask).

    Args:
        seg: a 3D numpy array where background is 0 and foreground is > 0
        labels: list of labels to combine for the sphere opening operation
        radius: the distance from the boundary of the combined labels (corresponds to the radius of the sphere)
    Returns:
        seg_open: a 3D numpy array where background is 0 and foreground is 1
    """
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

    Args:
        coords: a list of two lines, where each line is a list of two points with shape (2 lines x 2 points x 3 coordinates)
    Returns:
        axis_of_const_coordinate_values: the axis that is perpendicular to the plane in which the lines are placeddinate
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
    """
    Given two lines, this function returns the axis of the plane in which the lines are placed.

    Args:
        coords: a list of two lines, where each line is a list of two points (2 lines x 2 points x 3 coordinates)
    Returns:
        axis_idx: the index of the axis that is perpendicular to the plane in which the lines are placed
    """
    l1p1 = coords[0][0]
    l1p2 = coords[0][1]
    l2p1 = coords[1][0]
    l2p2 = coords[1][1]

    v1 = l1p2 - l1p1
    v2 = l2p2 - l2p1

    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # find the axis with the largest component of the normal
    axis_idx = np.argmax(np.abs(normal))
    return axis_idx



def point_closest_to_two_lines(coords):
    """
    Given two lines, this function returns the point that is closest to both lines. This can be used to find a point
    that can be annotated that describes a line pair.

    Args:
        coords: a list of two lines, where each line is a list of two points (2 lines x 2 points x 3 coordinates)
    Returns:
        closest_point: the point that is closest to both lines

    """
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

    closest_point = out
    return closest_point


def circle_opening_on_slices_perpendicular_to_axis(segmentationArray, axes, labels, radius, slice_idx=None):
    """
    Given a 3D segmentation, this function performs a circle opening operation on the specified labels in the slices
    perpendicular to each of the specified axes, then returns the union of segmentations from all axes.

    Args:
        segmentationArray: a 3D numpy array with the segmentation
        axes: the axis perpendicular to the slices where the circle opening operation is performed
        labels: list of labels to combine for the circle opening operation
        radius: radius of the circle for the circle opening operation
        slice_idx: if the slice index is specified, the circle opening operation is only performed on that slice. If
        None, the operation is performed on all slices along the axis. If multiple axes are specified, slice_idx must be
        a list of the same length as axes.

    Returns:
        the circle-opened segmentation along the specified axis
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

    Args:
        node: the segmentation node or volume node

    Returns:
        ijkToWorld: the IJK to world matrix of the binary labelmap
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

    Args:
        ijk: the IJK coordinates (x, y, z)
        ijkToWorld_matrix: the IJK to world matrix

    Returns:
        the world coordinate (x, y, z)
    """
    x_idx, y_idx, z_idx = 2, 1, 0  # swap z and x from numpy to slicer
    return ijkToWorld_matrix.MultiplyPoint([ijk[x_idx], ijk[y_idx], ijk[z_idx], 1])[
           :3]  # [:3] to remove the homogeneous coordinate (1)

@jit(nopython=True)
def transform_ijk_to_world_coord_np(coord, ijkToWorld_np):
    """
    Transform a coordinate from IJK to world space using the ijkToWorld matrix given as a numpy array.

    Args:
        coord: the IJK coordinates (x, y, z)
        ijkToWorld_np: the IJK to world matrix as a numpy array

    Returns:
        the world coordinate (x, y, z)
    """
    x_idx, y_idx, z_idx = 2, 1, 0  # swap z and x from numpy to slicer

    # matrix multiplication replicating np.dot(ijkToWorld_np, np.array([coord[x_idx], coord[y_idx], coord[z_idx], 1]))
    world = np.zeros(3, dtype=np.float32)
    mat = ijkToWorld_np
    vec = np.array([coord[x_idx], coord[y_idx], coord[z_idx], 1], dtype=np.float32)

    for i in range(3):  # 3D coordinates, remove the homogeneous coordinate (1)
        world[i] = mat[i, 0] * vec[0] + mat[i, 1] * vec[1] + mat[i, 2] * vec[2] + mat[i, 3] * vec[3]

    #world = np.dot(ijkToWorld_np, np.array([coord[x_idx], coord[y_idx], coord[z_idx], 1]))[:3]
    return world



def transform_world_to_ijk_coord(world, worldToIJK_matrix):
    """
    Given a world coordinate and the world to IJK matrix, this function returns the IJK coordinate

    Args:
        world: the world coordinates (x, y, z)
        worldToIJK_matrix: the world to IJK matrix

    Returns:
        the IJK coordinate (x, y, z)
    """
    kji = worldToIJK_matrix.MultiplyPoint([world[0], world[1], world[2], 1])[
          :3]  # [:3] to remove the homogeneous coordinate (1)
    return [int(round(kji[2])), int(round(kji[1])), int(round(kji[0]))]  # swap z and x from slicer
