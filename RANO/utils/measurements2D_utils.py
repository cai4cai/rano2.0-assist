"""
This module contains the Measurements2DMixin class, which is used to perform 2D measurements on
lesions in medical images. The class provides methods for calculating, displaying, and managing
lesion measurements, as well as handling user interactions with the GUI.
"""
import functools
from collections import defaultdict

import numpy as np
from nibabel.orientations import aff2axcodes, io_orientation

import qt
import slicer
import vtk
from utils import response_classification_utils

from slicer.util import *
import utils.ui_helper_utils as ui_helper_utils
import utils.results_table_utils as results_table_utils

from utils.rano_utils import get_instance_segmentation_by_connected_component_analysis, \
    match_instance_segmentations_by_IoU, get_max_orthogonal_line_product_coords, \
    get_ijk_to_world_matrix, transform_world_to_ijk_coord, transform_ijk_to_world_coord, \
    point_closest_to_two_lines, circle_opening_on_slices_perpendicular_to_axis, sphere_opening, find_closest_plane

from utils.config import debug


class Measurements2DMixin:
    """
    This mixin class provides methods for performing 2D measurements on lesions in medical images.
    It includes methods for calculating, displaying, and managing lesion measurements, as well as
    handling user interactions with the GUI.
    """

    def __init__(self, parameterNode, ui, lineNodePairs):
        self._parameterNode = parameterNode
        """The parameter node of the module. This node stores all user choices in parameter values, node selections, etc.
        so that when the scene is saved and reloaded, these settings are restored."""

        self.ui = ui
        """The UI elements of the module. This is a dictionary containing all the widgets in the module."""

        self.lineNodePairs = lineNodePairs
        """List of line node pairs used for 2D measurements."""

        self.instance_segmentations_matched = None
        """List of instance segmentations (numpy arrays) with matching labels across time points."""

        self.resampledSegNodes = None
        """List of segmentation nodes (vtkMRMLSegmentationNode) containing the matched instance segmentations."""

        self.resampledVolumeNodes = None
        """List of instance segmentations (vtkMRMLLabelMapVolumeNode) transformed and resampled to the reference input volume space."""

        self.previous_timepoint_orientation = {}
        """Dictionary to store the orientation of the lesions in the previous timepoint to enable consistent 2D measurement orientation across timepoints."""

        self.previous_timepoint_center = {}
        """Dictionary to store the center of the lesions in the previous timepoint to enable consistent 2D measurement slices across timepoints."""

        self.observations = []
        """Store observers for the line nodes to handle user interactions."""

        self.observations2 = []
        """Store observers for the line nodes to handle user interactions."""

    def onCalc2DButton(self):
        """
        This method is called when the "Calculate 2D" button is pressed.

        It performs the following steps:
        1. Get the selected segmentations and instance segmentations.
        2. Match the instance segmentations across timepoints.
        3. Transform and resample the instance segmentations in the original reference image space.
        4. For each timepoint and lesion, place the RANO lines.
        5. Evaluate the 2D measurements and store the results in a dictionary.
        6. Display the results in the UI.
        7. Update the line pair UI list.
        8. Calculate the results table.
        """

        if debug: print("Calc 2D button pressed")

        # determine the method to use for 2D measurements
        method2DmeasComboBox = self.ui.method2DmeasComboBox.currentText

        def create_lineNodePairs(lesion_stats):
            """
            From line coordinates stored in lesion_stats, create lineNodePairs that are used to display the lines in the
            UI views.

            Args:
                lesion_stats: dictionary containing the line coordinates for each lesion and timepoint
            Returns:
                lineNodePairs: LineNodePairList containing the line node pairs for each lesion and timepoint
            """
            lineNodePairs = LineNodePairList()
            for les_idx in lesion_stats:
                for tp in lesion_stats[les_idx]:
                    coords = lesion_stats[les_idx][tp]["coords"]

                    if len(coords) == 0:
                        continue

                    lineNodePair = LineNodePair(lesion_idx=les_idx, timepoint=tp)
                    lineNodePair.set_coords(coords)
                    lineNodePairs.append(lineNodePair)
            return lineNodePairs

        def setLinePairViews(lineNodePairs):
            """
            Set the views for the line node pairs in the UI, i.e. it will make sure that lines of timepoint1 are shown in
            the timepoint1 views and lines of timepoint2 are shown in the timepoint2 views.

            It will also center the views of each timepoint on the first available line node pair in the list.
            Args:
                lineNodePairs: LineNodePairList containing the line node pairs for each lesion and timepoint

            """
            tp_view_set = []  # keep track of which timepoint views have already been set
            for pair in lineNodePairs:
                les_idx = pair.lesion_idx
                tp = pair.timepoint

                # make sure the line node pair is displayed in the correct view
                self.setViews(pair, tp)

                # center on the first available line node pair in the list
                if tp not in tp_view_set:  # to set views for this timepoint only once
                    tp_view_set.append(tp)
                    self.centerTimepointViewsOnCenterPoint(pair, tp)

        def get_binary_semantic_segmentations(segNodes, segmentId):
            """
            Extract the binary semantic segmentations of a segment from the segmentation nodes as numpy arrays.

            Args:
                segNodes: list of segmentation nodes for each time point
                segmentId: ID of the segment to extract
            Returns:
                binary_semantic_segmentations: list of binary semantic segmentations of the segment (numpy arrays)
            """
            binary_semantic_segmentations = []
            for node in segNodes:
                node.CreateBinaryLabelmapRepresentation()
                refVol = node.GetNodeReference(slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole())

                if not segmentId in node.GetSegmentation().GetSegmentIDs():
                    # if the segmentId does not exist, we want an empty binary segmentation (all zeros)
                    bin_sem_seg = np.zeros_like(arrayFromVolume(refVol))
                else:
                    bin_sem_seg = arrayFromVolume(refVol) == int(segmentId)

                binary_semantic_segmentations.append(bin_sem_seg)
            return binary_semantic_segmentations

        def get_instance_segmentations(binary_segmentations):
            """
            Convert the binary segmentations to instance segmentations.
            Currently, this is done by connected component analysis.
            Args:
                binary_segmentations: list of binary segmentations (numpy arrays)
            Returns:
                instance_segmentations: list of instance segmentations (numpy arrays) for each time point. Note that at
                the moment, the instance segmentations are not matched across time points, i.e., the labels do not
                correspond to the same instance across time points.
            """
            instance_segmentations = []
            for seg in binary_segmentations:
                instance_seg = get_instance_segmentation_by_connected_component_analysis(seg)
                instance_segmentations.append(instance_seg)
            return instance_segmentations

        def matched_instances_across_timepoints(instance_segmentations):
            """
            Match the instance segmentations across time points. For example, as an output, if the first timepoint
            has instances 1, 2, 3 and the second timepoint can have instance 1, 3, 4 then instance 2 is missing in
            the second timepoint (disappeared) and instance 4 is a new instance in the second timepoint.

            Args:
                instance_segmentations: list of instance segmentations (numpy arrays) for each time point
            Returns:
                instance_segmentations_matched: list of instance segmentations (numpy arrays) with matching labels across
                time points.
            """
            instance_segmentations_matched = match_instance_segmentations_by_IoU(instance_segmentations)
            return instance_segmentations_matched

        def transform_to_and_resample_in_original_img_space(instance_segmentations_matched, segNodes):
            """
            The instance_segmentations_matched need to be transformed and resampled in the reference spaces, i.e.,
            seg1 to the reference space of timepoint1 and seg2 to the reference space of timepoint2. This is so that the
            RANO lines can be placed on slices of the original input volumes.
            Currently, the reference space are given by the channel1 input volumes, but this can be changed in the future.
            This means, the RANO lines are placed on slices of the channel1 input volumes (reference space).

            Steps:
            1. Create a segmentation node for each instance segmentation and place instance segments in there.
               Then apply the transform to the segmentation node. This will give a segmentation node in the original space,
               but with spacing 1x1x1.
            2. Resample the image to the original reference input volume space.

            Args:
                instance_segmentations_matched: list of instance segmentations (numpy arrays) with matching labels across time points
                segNodes: list of segmentation nodes for each time point

            Returns:
                resampledVolumeNodes: list of vtkMRMLLabelMapVolumeNode objects with the resampled instance segmentations
            """

            assert(len(instance_segmentations_matched) == len(segNodes)), "Number of instance segmentations and segmentation nodes must be equal"
            num_timepoints = len(segNodes)

            # get the transforms and reference volumes for each timepoint (currently only channel1)
            transformNodes = [slicer.util.getNode(f"Transform_timepoint{i + 1}_channel1 (-)") for i in range(num_timepoints)]
            referenceVolumeNodes = [self._parameterNode.GetNodeReference(f"InputVolume_channel1_t{i + 1}") for i in range(num_timepoints)]

            resampledSegNodes = []
            resampledVolumeNodes = []
            for i, (seg, segNode, transformNode, referenceVolumeNode) in enumerate(zip(instance_segmentations_matched, segNodes, transformNodes, referenceVolumeNodes)):
                # new segmentation node to store the matched instance segmentations
                newSegNodeName = f"matched_instance_segmentation_t{i + 1}"
                # if the segmentation node already exists, remove it
                if slicer.mrmlScene.GetFirstNodeByName(newSegNodeName):
                    slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName(newSegNodeName))
                newSegNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode",
                                                                f"matched_instance_segmentation_t{i + 1}")

                referenceImageVolumeNode = segNode.GetNodeReference('referenceImageGeometryRef')
                newSegNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceImageVolumeNode)

                # add the segmentations to the new segmentation node
                for lab in np.unique(seg):
                    if lab == 0:
                        continue

                    slicer.util.updateSegmentBinaryLabelmapFromArray(narray=np.array(seg == lab).astype(np.uint8),
                                                                     segmentationNode=newSegNode,
                                                                     segmentId=str(lab),
                                                                     referenceVolumeNode=referenceImageVolumeNode
                                                                     )
                    # set the name of the new segment as Les 1, Les 2, etc.
                    newSegNode.GetSegmentation().GetSegment(str(lab)).SetName(f"Les {lab}")

                # render in 3D
                newSegNode.CreateClosedSurfaceRepresentation()

                # make sure there is a displayNode
                newSegNode.CreateDefaultDisplayNodes()

                # display the segmentations on the corresponding views only
                self.setViews(newSegNode, f"timepoint{i + 1}")

                # hide the old and new segmentation nodes since we want to show the resampled labelmap volumes instead
                segNode.GetDisplayNode().SetVisibility(False)
                newSegNode.GetDisplayNode().SetVisibility(False)

                # apply the transform to the new segmentation node
                if transformNode:
                    newSegNode.SetAndObserveTransformNodeID(transformNode.GetID())

                # 2. resample the segmentation to the original input volume space
                # get all segmentIds from the new segmentation node
                segmentIds = newSegNode.GetSegmentation().GetSegmentIDs()
                if segmentIds:
                    # create new labelmap volume node
                    newLabelMapVolumeName = f"matched_instance_segmentation_t{i + 1}_resampled"
                    # if it already exists, remove it
                    if slicer.mrmlScene.GetFirstNodeByName(newLabelMapVolumeName):
                        slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName(newLabelMapVolumeName))
                    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode",
                                                                            newLabelMapVolumeName)

                    slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(newSegNode,
                                                                                          segmentIds,
                                                                                          labelmapVolumeNode,
                                                                                          referenceVolumeNode)

                    # set the views
                    ui_helper_utils.UIHelperMixin.setLabelVolumes(labelmapVolumeNode, f"timepoint{i + 1}")

                else:
                    print(f"No segments found in {newSegNode.GetName()}")
                    labelmapVolumeNode = None

                resampledSegNodes.append(newSegNode)
                resampledVolumeNodes.append(labelmapVolumeNode)

            self.resampledSegNodes = resampledSegNodes
            return resampledVolumeNodes

        def open_instance_segmentation(instance_seg, radius=3):
            """
            Open the instance segmentation using the method specified in the method2DmeasComboBox.

            Args:
                instance_seg: instance segmentation (numpy array)
                radius: radius of the opening operation
            Returns:
                seg_open: opened instance segmentation (numpy array)
            """
            seg_open = np.zeros_like(instance_seg)
            for lab in np.unique(instance_seg):
                if lab == 0:
                    continue

                if method2DmeasComboBox == "RANO_open2D":
                    seg_open += lab * circle_opening_on_slices_perpendicular_to_axis(instance_seg,
                                                                                     axes=[0, 1, 2],
                                                                                     labels=[lab],
                                                                                     radius=radius,
                                                                                    )
                elif method2DmeasComboBox == "RANO_open3D":
                    seg_open += lab * sphere_opening(instance_seg, labels=[lab], radius=radius)
                else:
                    raise ValueError(
                        f"No implementation available for opening the segmentations with method {method2DmeasComboBox}")

            return seg_open

        def open_instance_segmentations(instance_segmentations, opening_radius):
            """
            Loop over the instance segmentations and perform morphological opening on each segmentation.

            Args:
                instance_segmentations: list of instance segmentations (numpy arrays)
                opening_radius: radius of the opening operation
            Returns:
                opened_segmentations: list of opened instance segmentations (numpy arrays)
            """
            # open the segmentations
            opened_segmentations = []
            for seg in instance_segmentations:
                opened_seg = open_instance_segmentation(seg, radius=opening_radius)
                opened_segmentations.append(opened_seg.astype(np.uint8))
            return opened_segmentations

        def evaluate_instance_segmentation(resampledVolumeNode):
            """
            Evaluate the instance segmentation. Specifically, retrieve the coordinates of the 2D measurements considering the
            specified options (orientation, slice consistency, etc.) and the method selected in the UI.
            Also, calculate the volume of the lesions.

            Args:
                resampledVolumeNode: vtkMRMLLabelMapVolumeNode containing the matched instance segmentation.

            Returns:
                lesion_dict: dictionary containing the line pair coordinates and volume of each lesion for the current timepoint.

            """
            lesion_dict = defaultdict(lambda: {"coords": [], "volume": np.nan})
            if not resampledVolumeNode:
                print("resampledVolumeNode is None. Assuming no lesions for this timepoint.")
                return lesion_dict

            instance_segmentation = arrayFromVolume(resampledVolumeNode)
            for lab in np.unique(instance_segmentation):
                if lab == 0:  # skip background label
                    continue
                bin_seg = np.array(instance_segmentation == lab).astype(np.uint8)

                if method2DmeasComboBox in ["RANO", "RANO_open2D", "RANO_open3D"]:

                    # store the orientation of the lesion in the current timepoint
                    orientation_consistency_across_timepoints = self._parameterNode.GetParameter("orient_cons_tp") == "true"
                    previous_timepoint_orientation_current_lesion = self.previous_timepoint_orientation.get(lab, None)

                    if orientation_consistency_across_timepoints and previous_timepoint_orientation_current_lesion:
                        valid_orientations = [previous_timepoint_orientation_current_lesion]
                    else:
                        valid_orientations = [orien for orien in ["sagittal", "coronal", "axial"] if
                                              self._parameterNode.GetParameter(orien) == "true"]

                    # store the slice number of the lesion in the current timepoint
                    force_close_slice_across_timepoints = self._parameterNode.GetParameter("same_slc_tp") == "true"
                    center_IJK = None
                    if force_close_slice_across_timepoints and previous_timepoint_orientation_current_lesion:
                        previous_timepoint_center_current_lesion = self.previous_timepoint_center.get(lab, None)
                        center_world = previous_timepoint_center_current_lesion
                        # need to convert world center point to IJK point
                        worldToIJK = get_ijk_to_world_matrix(resampledVolumeNode)
                        worldToIJK.Invert()

                        center_IJK = transform_world_to_ijk_coord(center_world, worldToIJK)

                    def convert_to_IJK_axis(orientation, resampledVolumeNode):
                        """
                        Convert the anatomical orientation to the corresponding IJK axis index using the IJK to RAS matrix.

                        Args:
                            orientation: anatomical orientation ("sagittal", "coronal", "axial")
                            resampledVolumeNode: vtkMRMLLabelMapVolumeNode containing the matched instance segmentations
                        Returns:
                            ijk_axis_idx: IJK axis index corresponding to the anatomical orientation
                        """

                        # mapping from anatomical orientations to anatomical axis indices in 3D slicer (RAS)
                        orientation_to_world_axis_idx = {"sagittal": 0, "coronal": 1, "axial": 2}
                        world_axis_idx = orientation_to_world_axis_idx[orientation]  # get the current world axis index

                        # get the RAS to IJK matrix
                        ijkToWorld = get_ijk_to_world_matrix(resampledVolumeNode)
                        ijkToWorld = slicer.util.arrayFromVTKMatrix(ijkToWorld)
                        worldToIJK = np.linalg.inv(ijkToWorld)

                        # convert from world axis index to IJK axis index
                        ornt = io_orientation(worldToIJK)  # mapping from RAS axes to IJK axes
                        ijk_axis_idx = int(ornt[world_axis_idx][0])  # [0] picks the corresponding axis in the IJK space ([1] picks the direction)

                        return ijk_axis_idx

                    # need to convert world orientation to IJK axis
                    valid_axes_IJK = [convert_to_IJK_axis(orien, resampledVolumeNode) for orien in valid_orientations]
                    # need to go back to numpy order (KJI)
                    ijk_axis_idx_to_kji = {0: 2, 1: 1, 2: 0}  # convert 0 to 2 and 2 to 0 and leave 1 as is
                    valid_axes_IJK = [ijk_axis_idx_to_kji[idx] for idx in valid_axes_IJK]

                    # get the line pair coordinates for the current lesion
                    coords_world = get_max_orthogonal_line_product_coords(bin_seg, valid_axes_IJK, center_IJK, ijkToWorld=get_ijk_to_world_matrix(resampledVolumeNode))

                    if len(coords_world) > 0:
                        world_normal_axis_idx = find_closest_plane(coords_world)
                        this_timepoint_orientation_curr_lesion = {0: 'sagittal', 1: 'coronal', 2: 'axial'}[world_normal_axis_idx]

                        # get the center of the line pair in IJK space to allow for consistent slice selection across timepoints
                        center_world = point_closest_to_two_lines(coords_world)
                        this_timepoint_center_world_curr_lesion = center_world

                        # store the orientation and center of the lesion in the current timepoint
                        self.previous_timepoint_orientation[lab] = this_timepoint_orientation_curr_lesion
                        self.previous_timepoint_center[lab] = this_timepoint_center_world_curr_lesion

                    volume = np.sum(bin_seg)

                elif method2DmeasComboBox == "Random":
                    coords_world = np.random.randint(0, 100, (2, 2, 3))  # 2 lines, 2 points, 3 coordinates
                    volume = np.random.rand(2, 2, 3) * 100
                else:
                    raise ValueError(f"Method {method2DmeasComboBox} not recognized")

                lesion_dict[lab] = {"coords": coords_world, "volume": volume}

            return lesion_dict


        def display_opened_segmentations(opened_segmentations, segNodes):
            """
            The  opened segmentations (numpy arrays) are added to the segmentation nodes as new segments.

            Args:
                opened_segmentations: list of opened segmentations (numpy arrays)
                segNodes: list of segmentation nodes for each time point
            """
            for i, (seg, segNode) in enumerate(zip(opened_segmentations, segNodes)):

                nextSegmentId = str(int(max(segNode.GetSegmentation().GetSegmentIDs())) + 1)

                # temporarily apply transform to the reference image of the segmentation node
                # this is required because segNode itself has been transformed to the original image space, so the reference image
                # needs to be transformed to the original image space as well
                transformNode = slicer.util.getNode(f"Transform_timepoint{i + 1}_channel1 (-)")
                if transformNode:
                    referenceVolumeNode = segNode.GetNodeReference('referenceImageGeometryRef')
                    referenceVolumeNode.SetAndObserveTransformNodeID(transformNode.GetID())

                slicer.util.updateSegmentBinaryLabelmapFromArray(narray=seg,
                                                                 segmentationNode=segNode,
                                                                 segmentId=nextSegmentId,
                                                                 referenceVolumeNode=segNode.GetNodeReference(
                                                                     'referenceImageGeometryRef'))

                # undo the transform of the reference image (we want to keep it in the segmentation model space to
                # avoid unnecessary resampling)
                if transformNode:
                    referenceVolumeNode.SetAndObserveTransformNodeID(None)

                # set the name of the new segment
                segmentIdSelectedForOpening = self.ui.SegmentSelectorWidget.currentSegmentID()
                segmentNameSelectedForOpening = segNode.GetSegmentation().GetSegment(segmentIdSelectedForOpening).GetName()
                segNode.GetSegmentation().GetSegment(nextSegmentId).SetName(f"{segmentNameSelectedForOpening}_opened")


        def get_lesion_stats(resampledVolumeNodes):
            """
            For each timepoint, evaluate the instance segmentation and store the line pair coordinates and volume of the
            lesions in a dictionary.

            Args:
                resampledVolumeNodes: list of vtkMRMLLabelMapVolumeNode containing the matched instance segmentations
            Returns:
                lesion_stats: dictionary of dictionaries containing the line pair coordinates and volume of each lesion
                (key=lesion_idx) for each timepoint (key=timepoint).
            """
            lesion_stats = defaultdict(lambda: defaultdict(lambda: {}))

            # reset the previous timepoint orientation and center, so the previous run does not affect the current run
            self.previous_timepoint_orientation = {}
            self.previous_timepoint_center = {}

            for tp, resampledVolumeNode in enumerate(resampledVolumeNodes):
                tp = f"timepoint{tp + 1}"
                lesion_dict_tp = evaluate_instance_segmentation(resampledVolumeNode)
                for les_idx in lesion_dict_tp:
                    lesion_stats[les_idx][tp] = lesion_dict_tp[les_idx]
            return lesion_stats


        segNodes = [self.ui.outputSelector.currentNode(), self.ui.outputSelector_t2.currentNode()]

        binary_semantic_segmentations = get_binary_semantic_segmentations(segNodes,
                                                                          segmentId=self.ui.SegmentSelectorWidget.currentSegmentID())

        instance_segmentations = get_instance_segmentations(binary_semantic_segmentations)

        instance_segmentations_matched = matched_instances_across_timepoints(instance_segmentations)

        if method2DmeasComboBox in ["RANO_open2D", "RANO_open3D"]:
            opened_segmentations = open_instance_segmentations(instance_segmentations_matched,
                                                               opening_radius=int(self.ui.radius_spinbox.value))
            display_opened_segmentations(opened_segmentations, segNodes)
            instance_segmentations_matched = opened_segmentations

        self.instance_segmentations_matched = instance_segmentations_matched

        # transform the instance segmentations to the original reference image space
        self.resampledVolumeNodes = transform_to_and_resample_in_original_img_space(instance_segmentations_matched,
                                                                                    segNodes)

        # hide the resampled labelmap volumes
        self.ui.toggleShowInstanceSegPushButton.setChecked(False)
        self.onToggleShowInstanceSegButton()

        lesion_stats = get_lesion_stats(self.resampledVolumeNodes)

        # remove the previous lines
        del self.lineNodePairs[:]

        self.lineNodePairs = create_lineNodePairs(lesion_stats)
        setLinePairViews(self.lineNodePairs)

        # center views on first volume (reference volume)
        self.onShowChannelButton(True, timepoint='timepoint1', inputSelector=self.ui.inputSelector_channel1_t1)
        self.onShowChannelButton(True, timepoint='timepoint2', inputSelector=self.ui.inputSelector_channel1_t2)

        self.update_linepair_ui_list()

        # algorithms to set enhancing, measurable, target lesions
        self.lineNodePairs.decide_enhancing()
        self.lineNodePairs.decide_measurable()
        self.lineNodePairs.decide_target()

        # calculate the results table
        results_table_utils.ResultsTableMixin.calculate_results_table(self.lineNodePairs)

    def onToggleShowInstanceSegButton(self):
        """
        This method is called when the "Show/Hide Lesions" button is pressed. It shows or hides the lesions in the
        slice views based on the resampled labelmap volumes. In the 3D views they are shown based on the resampled
        segmentation nodes (because the resampled labelmap volumes are not displayed smoothly in 3D, but voxelized).
        """
        # show or hide the label volumes via the slicecontrollerwidget
        for timepoint in ["timepoint1", "timepoint2"]:
            if timepoint == 'timepoint1':
                viewnames = ["Red", "Yellow", "Green"]
                tp = 0
            elif timepoint == 'timepoint2':
                viewnames = ["Red_2", "Yellow_2", "Green_2"]
                tp = 1
            else:
                raise ValueError("timepoint must be 'timepoint1' or 'timepoint2'")

            checked = self.ui.toggleShowInstanceSegPushButton.checked
            for viewname in viewnames:
                compositeNode = slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().GetSliceCompositeNode()
                controller = slicer.app.layoutManager().sliceWidget(viewname).sliceController()
                if compositeNode.GetLabelVolumeID() not in [node.GetID() for node in self.resampledVolumeNodes if node]:
                    pass
                    # controller.setLabelMapHidden(True)
                else:
                    compositeNode.SetLabelOpacity(0.5)
                    if checked:
                        controller.setLabelMapHidden(False)
                    else:
                        controller.setLabelMapHidden(True)

            # show or hide the 3D view
            node = self.resampledSegNodes[tp] if self.resampledSegNodes else None
            if node:
                if checked:
                    node.GetDisplayNode().SetVisibility(True)
                    node.GetDisplayNode().SetVisibility2D(False)
                    node.GetDisplayNode().SetVisibility3D(True)
                    # set opacity
                    node.GetDisplayNode().SetOpacity(0.5)
                else:
                    node.GetDisplayNode().SetVisibility(False)


    def onAddLinePairButton(self, timepoint):
        """
        This method is called when the "Add Lines t1" or "Add Lines t2" button is pressed.
        It allows the user to add a new line pair for the selected timepoint by placing two lines in the slice views of
        the corresponding timepoint. The lines are added to the lineNodePairs list and displayed in the UI.
        Args:
            timepoint: the timepoint for which the line pair is added (e.g., "timepoint1" or "timepoint2")
        """
        if debug: print("Add line pair button pressed")
        lesion_idx = int(self.ui.add_line_lesidx_spinBox.value)

        # check if the lesion index and timepoint already exist in the lineNodePairs
        for pair in self.lineNodePairs:
            if pair.lesion_idx == lesion_idx and pair.timepoint == timepoint:
                # show a message box
                msgBox = qt.QMessageBox()
                msgBox.setText(f"Line pair for Lesion Index {lesion_idx} and {timepoint} already exists")
                msgBox.exec()
                return

        newLineNodePair = LineNodePair(lesion_idx=lesion_idx, timepoint=timepoint)
        newLineNode1, newLineNode2 = newLineNodePair

        self.setViews(newLineNodePair, timepoint)

        # go into placement mode for the new line
        persistentPlaceMode = 1  # need to be in persistent mode to place the second line after first
        slicer.modules.markups.logic().StartPlaceMode(persistentPlaceMode)
        slicer.modules.markups.logic().SetActiveListID(newLineNode1)

        def place_line2(lineNode1, arg2):
            if lineNode1.GetNumberOfControlPoints() == 2:
                # go into placement mode for the new line

                slicer.modules.markups.logic().SetActiveListID(newLineNode2)

                def end_placement(lineNode2, arg2):
                    if int(lineNode2.GetNumberOfControlPoints()) == 2:
                        slicer.modules.markups.logic().StartPlaceMode(0)  # exit placement mode

                    def add_linePair(lineNode2):
                        if lineNode2.GetNumberOfControlPoints() == 2:
                            line_lenghts = newLineNodePair.get_line_lengths()
                            newLineNodePair.measurable = True if all(
                                [l > 10 for l in line_lenghts]) else False
                            self.lineNodePairs.append(newLineNodePair)
                            self.update_linepair_ui_list()
                            # set the views again after both lines have been defined so they can be removed from the views in which they are not orthogonal
                            self.setViews(newLineNodePair, timepoint)

                            for observedNode, observer in self.observations2:
                                observedNode.RemoveObserver(observer)

                    add_linePair(lineNode2)

                # add callback to end placement mode AFTER the second line is placed
                self.observations2.append([newLineNode2, newLineNode2.AddObserver(newLineNode2.PointPositionDefinedEvent, end_placement)])

        def onPointRemovedEvent(lineNode, event):
            if slicer.app.applicationLogic().GetInteractionNode().GetCurrentInteractionMode() == 2:
                # schedule removal the line pair if the user cancels the placement of either line
                # can't remove the lines immediately because one of the lines is the observer caller

                # remove observers
                for observedNode, observer in self.observations2:
                    observedNode.RemoveObserver(observer)

                qt.QTimer.singleShot(0, lambda: slicer.mrmlScene.RemoveNode(newLineNode1))
                qt.QTimer.singleShot(0, lambda: slicer.mrmlScene.RemoveNode(newLineNode2))

        # add callback to place the second line when the first line is placed
        # first remove previous observers
        for observedNode, observer in self.observations2:
            observedNode.RemoveObserver(observer)

        self.observations2.append([newLineNode1, newLineNode1.AddObserver(newLineNode1.PointPositionDefinedEvent, place_line2)])
        #self.observations2.append([newLineNode2, newLineNode2.AddObserver(newLineNode2.PointPositionDefinedEvent, add_linePair)])

        # add callback to remove the line pair if the user cancels the placement of either line
        self.observations2.append([newLineNode1, newLineNode1.AddObserver(newLineNode1.PointRemovedEvent, onPointRemovedEvent)])
        self.observations2.append([newLineNode2, newLineNode2.AddObserver(newLineNode2.PointRemovedEvent, onPointRemovedEvent)])

    def update_linepair_ui_list(self):
        """
        This method updates the UI list of line pairs by populating the table with the lesion index, timepoint, and
        whether the lesion is enhancing, measurable, and target. It also sets the background color of the rows such that
        rows of the same lesion index are grouped together for better readability.
        """
        def onCellClicked(row, col):
            """Called when a cell is clicked."""
            les_idx = int(self.ui.tableWidget.item(row, 0).text())
            tp = self.ui.tableWidget.item(row, 1).text()

            # center timepoint's views on the clicked lesion
            for pair in self.lineNodePairs:
                if pair.lesion_idx == les_idx and pair.timepoint == tp:
                    self.centerTimepointViewsOnCenterPoint(pair, tp)

        # update the menu table
        col_name_to_idx = {"Lesion Index": 0, "Timepoint": 1, "Enhancing": 2, "Measurable": 3, "Target": 4, " ": 5}

        tableWidget = self.ui.tableWidget
        tableWidget.setColumnCount(len(col_name_to_idx))
        tableWidget.setHorizontalHeaderLabels(list(col_name_to_idx.keys()))
        tableWidget.setRowCount(0)
        tableWidget.clearContents()
        tableWidget.verticalHeader().setVisible(False)
        tableWidget.cellClicked.connect(onCellClicked)
        prev_les_idx = None
        color1 = qt.QColor(240, 240, 240)
        color2 = qt.QColor(255, 255, 255)
        prev_color = color1

        # sort the lineNodePairs by lesion index and timepoint
        self.lineNodePairs = self.lineNodePairs.custom_sort(key=lambda x: (int(x.lesion_idx), x.timepoint))

        for pair_idx, pair in enumerate(self.lineNodePairs):
            les_idx = int(pair.lesion_idx)
            tp = pair.timepoint
            # insert new row
            row_count = tableWidget.rowCount
            assert row_count == pair_idx, f"Row count: {row_count}, pair_idx: {pair_idx}, should be equal"
            tableWidget.insertRow(row_count)

            def setRowColor(rowIdx, color):
                """Set the background color of all cells in the row."""
                for col in range(tableWidget.columnCount):
                    item = tableWidget.item(rowIdx, col)
                    if item:
                        item.setBackground(qt.QBrush(color))
                        item.setTextAlignment(qt.Qt.AlignCenter)

            # define the contents of the columns
            tableWidget.setItem(row_count, col_name_to_idx["Lesion Index"], qt.QTableWidgetItem(str(les_idx)))
            tableWidget.setItem(row_count, col_name_to_idx["Timepoint"], qt.QTableWidgetItem(tp))

            # place tick boxes in the "Enhancing", "Measurable", "Target" columns
            for col_name in ["Enhancing", "Measurable", "Target"]:
                checkbox = qt.QCheckBox()

                def is_valid_target_selection(checkbox, les_idx, tp, _):

                    # make sure selected target lesion is measurable
                    for pair in self.lineNodePairs:
                        if pair.lesion_idx == les_idx and pair.timepoint == tp:
                            if not pair.measurable and checkbox.isChecked():
                                msgBox = qt.QMessageBox()
                                msgBox.setText("Target lesion must be measurable")
                                msgBox.exec()
                                checkbox.setChecked(False)
                                return False

                    # count the number of enhancing and non-enhancing target lesions for current timepoint
                    counter_enhancing = 0
                    counter_non_enhancing = 0

                    for pair in self.lineNodePairs:
                        if not pair.timepoint == tp:
                            continue
                        if pair.enhancing and pair.target:
                            counter_enhancing += 1
                        elif not pair.enhancing and pair.target:
                            counter_non_enhancing += 1

                    err = None
                    if counter_enhancing > 0 and counter_non_enhancing > 0:
                        if counter_enhancing > 2 or counter_non_enhancing > 2:
                            err = (f"Only 2 enhancing and 2 non-enhancing target lesions are allowed, but tried to add "
                                   f"{counter_enhancing} enhancing and {counter_non_enhancing} non-enhancing target lesions "
                                   f"for timepoint {tp}")
                    elif counter_enhancing > 3:
                        err = (f"Only 3 enhancing target lesions are allowed, but tried to add {counter_enhancing} "
                               f"for timepoint {tp}")
                    elif counter_non_enhancing > 3:
                        err = (f"Only 3 non-enhancing target lesions are allowed, but tried to add {counter_non_enhancing} "
                               f"for timepoint {tp}")

                    if err:
                        msgBox = qt.QMessageBox()
                        msgBox.setText(err)
                        msgBox.exec()
                        checkbox.setChecked(False)
                        return False

                    return True

                if col_name == "Target":
                    checkbox.clicked.connect(functools.partial(is_valid_target_selection, checkbox, les_idx, tp))

                # add callback to update the lineNodePair when the checkbox is clicked
                def onCheckboxToggled(checkbox, les_idx, tp, col_name, _):
                    if debug: print(f"Checkbox toggled to {checkbox.checked} for line pair of lesion index {les_idx}, timepoint {tp}, and column {col_name}")
                    for pair in self.lineNodePairs:
                        if pair.lesion_idx == les_idx and pair.timepoint == tp:
                            setattr(pair, col_name.lower(), checkbox.checked)

                    response_classification_utils.ResponseClassificationMixin.update_response_assessment(self.ui, self.lineNodePairs)

                checkbox.toggled.connect(functools.partial(onCheckboxToggled, checkbox, les_idx, tp, col_name))

                #tableWidget.setCellWidget(row_count, col_name_to_idx[col_name], checkbox)
                # make sure the checkbox is centered# make sure the checkbox is centered
                widget = qt.QWidget()
                layout = qt.QHBoxLayout(widget)
                layout.addWidget(checkbox)
                layout.setAlignment(qt.Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                tableWidget.setCellWidget(row_count, col_name_to_idx[col_name], widget)

                checkbox.setChecked(getattr(pair, col_name.lower()))

            # place push button in the "Delete" column
            deleteButton = qt.QPushButton()
            deleteButton.setIcon(qt.QIcon(self.resourcePath('Icons/trash.png')))

            def deleteLinePair(index, _):
                if debug: print(f"Deleting line pair at index {index}")
                self.lineNodePairs.pop(index)
                self.update_linepair_ui_list()

            deleteButton.clicked.connect(functools.partial(deleteLinePair, pair_idx))

            tableWidget.setCellWidget(row_count, col_name_to_idx[" "], deleteButton)
            # make the row narrow
            tableWidget.horizontalHeader().resizeSection(col_name_to_idx[" "], 10)

            other_color = color1 if prev_color == color2 else color2
            selected_color = prev_color if (not prev_les_idx or les_idx == prev_les_idx) else other_color
            prev_color = selected_color
            prev_les_idx = les_idx

            setRowColor(row_count, selected_color)

            # make all rows narrow
            row_height = 20
            for row in range(tableWidget.rowCount):
                tableWidget.setRowHeight(row, row_height)

                # if the text font is too large, the actual row height will be larger than the set row height
                row_height = tableWidget.rowHeight(0)

            # make the columns fit the content
            tableWidget.resizeColumnsToContents()

            self.ui.tableWidget.setFixedHeight((row_height + 1) * (tableWidget.rowCount+1) + 2)

        response_classification_utils.ResponseClassificationMixin.update_response_assessment(self.ui, self.lineNodePairs)
        response_classification_utils.ResponseClassificationMixin.update_overall_response_params(self.ui)
        response_classification_utils.ResponseClassificationMixin.update_overall_response_status(self.ui)


    def coords_ijk_to_world(self, coords_ijk, node):
        """
        Convert the coordinates from IJK to world coordinates.
        """
        if len(coords_ijk) == 0:
            return []

        ijkToWorld = get_ijk_to_world_matrix(node)

        l1p1, l1p2, l2p1, l2p2 = coords_ijk[0][0], coords_ijk[0][1], coords_ijk[1][0], coords_ijk[1][1]

        controlPointsLine1 = np.array([transform_ijk_to_world_coord(l1p1, ijkToWorld),
                                       transform_ijk_to_world_coord(l1p2, ijkToWorld)])
        controlPointsLine2 = np.array([transform_ijk_to_world_coord(l2p1, ijkToWorld),
                                       transform_ijk_to_world_coord(l2p2, ijkToWorld)])

        coords_world = np.array([controlPointsLine1, controlPointsLine2])  # 2 lines, 2 points, 3 coordinates

        return coords_world

    @staticmethod
    def setViews(node, timepoint):
        """
        Set the views for the line node pair such that the lines are only shown in the views corresponding to the
        timepoint and the line orientation (sagittal, coronal, axial) of the line pair.
        """
        if timepoint == 'timepoint1':
            viewnames = ["Red", "Yellow", "Green"]
            viewname_3D = "view3d_1"
        elif timepoint == 'timepoint2':
            viewnames = ["Red_2", "Yellow_2", "Green_2"]
            viewname_3D = "view3d_2"
        else:
            raise ValueError(f"timepoint must be 'timepoint1' or 'timepoint2' but is {timepoint}")

        # if the node is a line node, show the line only in the views in which the line lies
        if isinstance(node, LineNodePair):
            # get the control point positions
            coords = node.get_coords()
            # check that there are no nan values in the coords
            if not np.isnan(coords).any():
                axis = find_closest_plane(coords)
                if axis == 0:
                    viewnames = [viewnames[1]]
                elif axis == 1:
                    viewnames = [viewnames[2]]
                elif axis == 2:
                    viewnames = [viewnames[0]]

        viewNodeIDs = []
        for viewname in viewnames:
            viewNodeIDs.append(
                slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().GetSliceNode().GetID())
        # add the 3D view
        # loop over all 3D views to find the correct one
        for i in range(slicer.app.layoutManager().threeDViewCount):
            if slicer.app.layoutManager().threeDWidget(
                    i).threeDView().mrmlViewNode().GetSingletonTag() == viewname_3D:
                threeDWidget_idx = i
                break
        else:
            raise ValueError(f"Could not find the 3D view with singleton tag {viewname_3D}")

        viewNodeIDs.append(slicer.app.layoutManager().threeDWidget(
            threeDWidget_idx).threeDView().mrmlViewNode().GetID())

        if isinstance(node, LineNodePair):
            for lineNode in node:
                lineNode.GetDisplayNode().SetViewNodeIDs(viewNodeIDs)
            if node.fiducialNodeForText:
                node.fiducialNodeForText.GetDisplayNode().SetViewNodeIDs(viewNodeIDs)
        else:
            node.GetDisplayNode().SetViewNodeIDs(viewNodeIDs)

    @staticmethod
    def centerTimepointViewsOnFirstMarkupPoint(lineNode, tp):
        """
        Center the slice views and cameras on the first markup point of the line node.
        Args:
            lineNode: the line node to center on
            tp: the timepoint to center on (e.g., "timepoint1" or "timepoint2")
        """
        # center view group on the first markup point
        # Center slice views and cameras on this position
        position = lineNode.GetNthControlPointPositionWorld(0)
        viewgroup_to_set = 0 if tp == 'timepoint1' else 1
        for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
            viewgroup = sliceNode.GetViewGroup()
            if viewgroup == viewgroup_to_set:
                sliceNode.JumpSliceByCentering(*position)
        for viewNode in slicer.util.getNodesByClass('vtkMRMLViewNode'):
            view_group = viewNode.GetViewGroup()
            if view_group == viewgroup_to_set:
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                camera.SetFocalPoint(*position)

    @staticmethod
    def centerTimepointViewsOnCenterPoint(lineNodePair, tp):
        """
        Center the slice views and cameras on the center point of the line node pair.
        Args:
            lineNodePair: the line node pair to center on
            tp: the timepoint to center on (e.g., "timepoint1" or "timepoint2")
        """
        coords = lineNodePair.get_coords()
        center = point_closest_to_two_lines(coords)
        position = center
        viewgroup_to_set = 0 if tp == 'timepoint1' else 1
        for sliceNode in slicer.util.getNodesByClass('vtkMRMLSliceNode'):
            viewgroup = sliceNode.GetViewGroup()
            if viewgroup == viewgroup_to_set:
                sliceNode.JumpSliceByCentering(*position)
        for viewNode in slicer.util.getNodesByClass('vtkMRMLViewNode'):
            view_group = viewNode.GetViewGroup()
            if view_group == viewgroup_to_set:
                camera = slicer.modules.cameras.logic().GetViewActiveCameraNode(viewNode)
                camera.SetFocalPoint(*position)


class LineNodePair(list):
    """
    A class that represents a pair of line nodes for a lesion in a timepoint. The class inherits from list to allow
    easy access to the line nodes. The class also contains methods to create the line nodes, set and get their
    coordinates, and set the views for the line nodes. The class also contains methods to create a fiducial node for
    the text label of the line nodes and to handle events when the line nodes are modified.
    The class also contains methods to set the enhancing, measurable, and target properties of the line nodes.

    Args:
        lesion_idx: the index of the lesion
        timepoint: the timepoint for which the line nodes are created (e.g., "timepoint1" or "timepoint2")
        enhancing: whether the lesion is enhancing or not (default: True)
        measurable: whether the lesion is measurable or not (default: True)
        target: whether the lesion is a target lesion or not (default: False)
    """
    def __init__(self, lesion_idx, timepoint, enhancing=True, measurable=True, target=False):
        lineNode1, lineNode2 = self.create_twoLineNodes(lesion_idx, timepoint)
        super().__init__([lineNode1, lineNode2])
        self.lesion_idx = lesion_idx
        """Lesion index"""

        self.timepoint = timepoint
        """Timepoint for which the line node pair is created"""

        self.enhancing = enhancing
        """Whether the lesion is enhancing or not"""

        self.measurable = measurable
        """Whether the lesion is measurable or not"""

        self.target = target
        """Whether the lesion is a target lesion or not"""

        self.fiducialNodeForText = self.create_fiducialNodeFor_text()
        """Fiducial node for the text label of the line nodes"""

        self.observations = []
        """List of observers for the line nodes"""

        # add the observers for the lines that trigger when the line nodes are modified
        self.observations.append([lineNode1, lineNode1.AddObserver(vtk.vtkCommand.ModifiedEvent, self.uponLineNodeModifiedEvent)])
        self.observations.append([lineNode2, lineNode2.AddObserver(vtk.vtkCommand.ModifiedEvent, self.uponLineNodeModifiedEvent)])

    def set_coords(self, coords):
        """
        Set the coordinates of the line nodes to the given world coordinates.
        """
        for lineNode, coord in zip(self, coords):  # note: looping over self returns the line nodes because self is a list of line nodes
            if not isinstance(coord, np.ndarray):
                coord = np.array(coord)
            slicer.util.updateMarkupsControlPointsFromArray(lineNode, coord)

        self.uponLineNodeModifiedEvent(n=None, e=None)

    def get_coords(self):
        """
        Get the coordinates of the line nodes in world coordinates.
        """
        coords = np.zeros((2, 2, 3)) * np.nan
        for i, lineNode in enumerate(self):
            # check if both control points exist
            controlpoint1_exists = lineNode.ControlPointExists(0)
            controlpoint2_exists = lineNode.ControlPointExists(1)

            if not controlpoint1_exists or not controlpoint2_exists:
                continue  # leave the coords nan

            coords[i] = np.array([lineNode.GetNthControlPointPositionWorld(j) for j in range(2)])
        return coords

    def get_line_lengths(self):
        """
        Get the lengths of the lines in world coordinates.
        """
        len1 = self[0].GetLineLengthWorld()
        len2 = self[1].GetLineLengthWorld()
        return len1, len2

    def get_line_length_product(self):
        """
        Get the product of the lengths of the lines in world coordinates.
        """
        len1, len2 = self.get_line_lengths()
        return len1 * len2

    @staticmethod
    def create_twoLineNodes(les_idx, timepoint):
        """
        Create two line nodes for the line pair and set their properties.
        Args:
            les_idx: the index of the lesion
            timepoint: the timepoint for which the line nodes are created (e.g., "timepoint1" or "timepoint2")
        """
        # get the line nodes
        lineNode1Name = f"l1_les{int(les_idx)}_{timepoint.replace('timepoint', 't')}"
        lineNode2Name = f"l2_les{int(les_idx)}_{timepoint.replace('timepoint', 't')}"

        # check if the line already exists
        lineNode1 = slicer.mrmlScene.GetFirstNodeByName(lineNode1Name)
        lineNode2 = slicer.mrmlScene.GetFirstNodeByName(lineNode2Name)

        # remove the line nodes if they already exist and create new ones
        if lineNode1:
            slicer.mrmlScene.RemoveNode(lineNode1)
        if lineNode2:
            slicer.mrmlScene.RemoveNode(lineNode2)

        lineNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNode1Name)
        lineNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode", lineNode2Name)

        # make the line thicker
        lineNode1.GetDisplayNode().SetLineThickness(0.35)
        lineNode2.GetDisplayNode().SetLineThickness(0.35)

        # make the control points smaller
        lineNode1.GetDisplayNode().SetGlyphScale(1)
        lineNode2.GetDisplayNode().SetGlyphScale(1)

        # hide the text of the line nodes
        lineNode1.GetDisplayNode().SetPropertiesLabelVisibility(0)
        lineNode2.GetDisplayNode().SetPropertiesLabelVisibility(0)

        return lineNode1, lineNode2

    def create_fiducialNodeFor_text(self):
        """
        Create a fiducial node for the text label of the line nodes.
        """
        # create an extra fiducial point that is only used to annotate the linePair (hiding the ctrlPoint point itself)
        fiducialNodeName = f"text_fiducial_les{int(self.lesion_idx)}_{self.timepoint.replace('timepoint', 't')}"
        fiducialNode = slicer.mrmlScene.GetFirstNodeByName(fiducialNodeName)
        if fiducialNode:
            slicer.mrmlScene.RemoveNode(fiducialNode)
        fiducialNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", fiducialNodeName)

        # hide the control point glyph
        fiducialNode.GetDisplayNode().SetGlyphScale(0)  # used for relative size
        fiducialNode.GetDisplayNode().SetGlyphSize(0)  # used for absolute size

        # add a ctrlPoint point to the scene
        fiducialNode.AddControlPoint(0, 0, 0)

        # for now, set the text to empty
        fiducialNode.SetNthControlPointLabel(0, "")

        # make text size smaller
        fiducialNode.GetDisplayNode().SetTextScale(3)

        # turn the text shadow off
        fiducialNode.GetDisplayNode().GetTextProperty().ShadowOff()

        return fiducialNode

    @staticmethod
    def set_color_depending_on_orthogonality(n, e, lineNode1, lineNode2, fiducialNodeForText=None):
        """
        Set the color of the lines depending on whether they are orthogonal or not.
        Args:
            n: the event name
            e: the event object
            lineNode1: the first line node
            lineNode2: the second line node
            fiducialNodeForText: the fiducial node for the text label of the line nodes
        """
        # set the color of the lines depending on whether they are orthogonal
        # get the two lines

        if debug: print("Setting color depending on orthogonality")
        line1 = np.array(
            [lineNode1.GetNthControlPointPositionWorld(i) for i in range(lineNode1.GetNumberOfControlPoints())])
        line2 = np.array(
            [lineNode2.GetNthControlPointPositionWorld(i) for i in range(lineNode2.GetNumberOfControlPoints())])

        # get the direction vectors of the lines
        if not (len(line1) == 2 and len(line2) == 2):
            return

        dir1 = line1[-1] - line1[0]
        dir2 = line2[-1] - line2[0]

        # normalize the direction vectors
        dir1 /= np.linalg.norm(dir1)
        dir2 /= np.linalg.norm(dir2)

        # calculate the dot product of the direction vectors
        dot_product = np.dot(dir1, dir2)

        # set the color of the lines depending on the dot product
        tolerance_deg = 1
        min_deg = 90 - tolerance_deg
        max_deg = 90 + tolerance_deg
        min_rad = np.deg2rad(min_deg)
        max_rad = np.deg2rad(max_deg)

        if min_rad < np.arccos(dot_product) < max_rad:
            color = (0, 1, 0)  # green
        else:
            color = (1, 0, 0)  # red

        lineNode1.GetDisplayNode().SetSelectedColor(color)
        lineNode2.GetDisplayNode().SetSelectedColor(color)

        if fiducialNodeForText:
            fiducialNodeForText.GetDisplayNode().SetSelectedColor(tuple([v / 1.2 for v in color]))

    def annotate_with_text(self):
        """
        Annotate the line nodes with the length of the lines in world coordinates.
        """
        # set the location of the text to the middle of the line
        coords = self.get_coords()

        # check if coords has nans
        if not np.isnan(coords).any():  # get the intersection point of the two lines
            center_point = point_closest_to_two_lines(coords)
        elif not np.isnan(coords[0]).any():  # get the middle point of the first line
            center_point = coords[0].mean(axis=0)
        else:
            center_point = [0., 0., 0.]

        # set the location of the ctrlPoint point to the middle of the line
        self.fiducialNodeForText.SetNthControlPointPosition(0, center_point[0], center_point[1], center_point[2])

        controlPoint_l1_0_exists = self[0].ControlPointExists(0)
        controlPoint_l1_1_exists = self[0].ControlPointExists(1)
        controlPoint_l2_0_exists = self[1].ControlPointExists(0)
        controlPoint_l2_1_exists = self[1].ControlPointExists(1)

        if not controlPoint_l1_0_exists or not controlPoint_l1_1_exists:
            line1LengthWorld = 0.0
        else:
            line1LengthWorld = self[0].GetLineLengthWorld()

        if not controlPoint_l2_0_exists or not controlPoint_l2_1_exists:
            line2LengthWorld = 0.0
        else:
            line2LengthWorld = self[1].GetLineLengthWorld()

        if line1LengthWorld or line2LengthWorld:
            self.fiducialNodeForText.SetNthControlPointLabel(0,
                                                             f"Les {int(self.lesion_idx)}: {line1LengthWorld:.1f} x {line2LengthWorld:.1f}")

    def uponLineNodeModifiedEvent(self, n, e):
        """
        This method is called when the line nodes are modified.
        It sets the color of the lines depending on whether they are orthogonal or not and updates the text label
        of the line nodes with the length of the lines in world coordinates.
        Args:
            n: the event name
            e: the event object
        """
        # print("LineNode modified event")
        self.set_color_depending_on_orthogonality(n, e, self[0], self[1], self.fiducialNodeForText)
        slicer.modules.RANOWidget.calculate_results_table(slicer.modules.RANOWidget.lineNodePairs)
        self.annotate_with_text()

    def cleanup(self):
        """
        Cleanup the line node pair by removing the observers and the fiducial node.
        """
        # upon deletion of the object, remove the observers
        for observedNode, observer in self.observations:
            observedNode.RemoveObserver(observer)

        # remove the fiducial node
        slicer.mrmlScene.RemoveNode(self.fiducialNodeForText)

        # remove the line nodes
        slicer.mrmlScene.RemoveNode(self[0])
        slicer.mrmlScene.RemoveNode(self[1])


    def __repr__(self):
        return (f"LineNodePair(lesion_idx={self.lesion_idx}, timepoint={self.timepoint}, "
                f"lineNode1={self[0]}, lineNode2={self[1]})")


class LineNodePairList(list):
    """
    A list of LineNodePair objects. This class inherits from list to allow easy access to the line node pairs.
    The class also contains methods to add, remove, and modify the line node pairs. The class also contains methods
    to update the UI and the response assessment based on the line node pairs.
    """

    def __delitem__(self, index):
        """
        Makes sure that line nodes contained in the LineNodePairList are removed from the scene when removed from the list
        """
        if isinstance(index, slice):
            # Get all items that will be deleted
            items = self[index]
            for item in items:
                item.cleanup()
        else:
            self[index].cleanup()

        super().__delitem__(index)
        self.uponModified()

    def pop(self, index):
        """
        Makes sure that line nodes contained in the LineNodePairList are removed from the scene when popped from the list
        """
        self[index].cleanup()
        out = super().pop(index)
        self.uponModified()
        return out

    def uponModified(self):
        """
        This method is called when the line node pairs are modified. It updates the UI and the response assessment
        based on the line node pairs.
        """
        slicer.modules.RANOWidget.calculate_results_table(self)
        slicer.modules.RANOWidget.update_response_assessment(slicer.modules.RANOWidget.ui, self)
        slicer.modules.RANOWidget.update_linepair_ui_list()

    # make sure that the list is returned as a LineNodePairList when sorted
    def custom_sort(self, *args, **kwargs):
        """
        Sort the list of LineNodePair objects and return a new LineNodePairList object.
        """
        sorted_items = sorted(self, *args, **kwargs)
        return LineNodePairList(sorted_items)

    def decide_enhancing(self):
        """
        Logic to decide whether the lesion is enhancing or not. For now, all lesions are considered enhancing initially.
        """
        for pair in self:
            pair.enhancing = True
        self.uponModified()

    def decide_measurable(self):
        """
        Decide whether the lesion is measurable or not based on the orthogonal lines.
        """
        # for now if both lines are more than 10 pixels long, the lesion is measurable
        for pair in self:
            if pair[0].GetLineLengthWorld() > 10 and pair[1].GetLineLengthWorld() > 10:
                pair.measurable = True
            else:
                pair.measurable = False
        self.uponModified()

    def decide_target(self, strategy="two_largest_enhancing"):
        """
        Logic to decide whether the lesion is a target lesion or not. The strategy can be one of the following:
        - "two_largest_enhancing": select the two largest enhancing lesions from the baseline
        - "three_largest_enhancing": select the three largest enhancing lesions from the baseline
        - "two_largest_enhancing_and_two_largest_non_enhancing": select the two largest enhancing lesions and the two
            largest non-enhancing lesions from the baseline

        Args:
            strategy: the strategy to use for selecting the target lesions
        """

        # set all target flags to False
        for pair in self:
            pair.target = False

        # sort the lesions by the product of the orthogonal lines
        sorted_list = self.custom_sort(key=lambda x: x[0].GetLineLengthWorld() * x[1].GetLineLengthWorld(),
                                       reverse=True)

        sorted_list_t1 = [pair for pair in sorted_list if pair.timepoint == "timepoint1"]
        sorted_list_t2 = [pair for pair in sorted_list if pair.timepoint == "timepoint2"]

        if strategy == "two_largest_enhancing" or strategy == "three_largest_enhancing":
            # sort the lesions by the product of the orthogonal lines

            counter_target_les = 0
            for pair in sorted_list_t1:
                num_max = 2 if strategy == "two_largest_enhancing" else 3
                if pair.measurable and counter_target_les < num_max:
                    pair.target = True
                    counter_target_les += 1

        elif strategy == "two_largest_enhancing_and_two_largest_non_enhancing":
            counter_enhancing = 0
            counter_non_enhancing = 0
            for pair in sorted_list_t1:
                if pair.enhancing:
                    if counter_enhancing < 2:
                        pair.target = True
                        counter_enhancing += 1
                else:
                    if counter_non_enhancing < 2:
                        pair.target = True
                        counter_non_enhancing += 1

        # also set corresponding lesions in timepoint2 as targets
        for pair_t2 in sorted_list_t2:
            for pair in sorted_list_t1:
                if pair_t2.lesion_idx == pair.lesion_idx:
                    pair_t2.target = pair.target

        self.uponModified()

    def get_number_of_targets(self):
        """
        Get the number of target lesions, but don't count the same lesion twice it is in the list twice (for both timepoints)
        """
        target_les_idcs = [pair.lesion_idx for pair in self if pair.target]
        num_target_lesions = len(set(target_les_idcs))  # set to remove duplicates
        return num_target_lesions

    def get_number_of_new_target_lesions(self):
        """
        Get the number of new target lesions that were not target lesions at the first timepoint but appeared at the second
        """
        target_les_idcs_tp1 = [pair.lesion_idx for pair in self if pair.target and pair.timepoint == 'timepoint1']
        target_les_idcs_tp2 = [pair.lesion_idx for pair in self if pair.target and pair.timepoint == 'timepoint2']
        num_new_target_lesions = len(set(target_les_idcs_tp2) - set(target_les_idcs_tp1))
        return num_new_target_lesions

    def get_number_of_disappeared_target_lesions(self):
        """
        Get the number of target lesions that were target lesions at the previous timepoints but are not target lesions
        because they disappeared
        """
        target_les_idcs_tp1 = [pair.lesion_idx for pair in self if pair.target and pair.timepoint == 'timepoint1']
        target_les_idcs_tp2 = [pair.lesion_idx for pair in self if pair.target and pair.timepoint == 'timepoint2']
        num_disappeared_target_lesions = len(set(target_les_idcs_tp1) - set(target_les_idcs_tp2))
        return num_disappeared_target_lesions

    def get_number_of_new_measurable_lesions(self):
        """
        Get the number of new measurable lesions that were not measurable at the first timepoint but appeared at the second
        """
        measurable_les_idcs_tp1 = [pair.lesion_idx for pair in self if
                                   pair.measurable and pair.timepoint == 'timepoint1']
        measurable_les_idcs_tp2 = [pair.lesion_idx for pair in self if
                                   pair.measurable and pair.timepoint == 'timepoint2']
        num_new_measurable_lesions = len(set(measurable_les_idcs_tp2) - set(measurable_les_idcs_tp1))
        return num_new_measurable_lesions

    def get_sum_of_bidimensional_products(self, timepoint):
        """
        Given a list of line node pairs, this function returns the sum of bidimensional products of the orthogonal lines
        of all lesions at the given timepoint.
        """
        sum_prod = 0
        for pair in self:
            if pair.target and pair.timepoint == timepoint:
                sum_prod += pair.get_line_length_product()
        return sum_prod

    def get_rel_area_change(self):
        """
        Given a list of line node pairs, this function returns the relative change of the sum of bidimensional products of the
        orthogonal lines of all lesions at timepoint 2 relative to the sum of the bidimensional products of the orthogonal
        lines of all lesions at timepoint 1.
        """
        sum_prod_t1 = self.get_sum_of_bidimensional_products("timepoint1")
        sum_prod_t2 = self.get_sum_of_bidimensional_products("timepoint2")

        return (sum_prod_t2 - sum_prod_t1) / sum_prod_t1