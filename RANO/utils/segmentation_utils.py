import ctk
from utils import ui_helper_utils, measurements2D_utils
from utils.config import module_path
import time
import os
import traceback
import json
import slicer
import numpy as np

from utils.rano_utils import run_segmentation
from utils.config import debug


class SegmentationMixin:
    """
    Mixin class for the segmentation functionality of the RANO module.
    This class handles the segmentation of the input volumes and the loading of the results into Slicer.
    It also handles the progress bar and the cancellation of the segmentation process.
    """
    def __init__(self, parameterNode, ui):
        self.start_time_t1 = None
        """Start time for the first segmentation process"""

        self.start_time_t2 = None
        """Start time for the second segmentation process"""

        self._parameterNode = parameterNode
        """Parameter node for the RANO module"""

        self.ui = ui
        """UI for the RANO module"""

    def onCalcSegmentationsButton(self):
        """
        Run processing when user clicks the "Calculate Segmentations" button.
        """
        # unset start times for the next segmentation run
        self.start_time_t1 = None
        self.start_time_t2 = None

        original_log_level = slicer.app.pythonConsoleLogLevel()

        # timepoint 1
        try:
            # assemble input channel nodes
            input_node_list = [n for n in [self._parameterNode.GetNodeReference("InputVolume_channel1_t1"),
                                           self._parameterNode.GetNodeReference("InputVolume_channel2_t1"),
                                           self._parameterNode.GetNodeReference("InputVolume_channel3_t1"),
                                           self._parameterNode.GetNodeReference("InputVolume_channel4_t1")] if n]

            if debug: print("Passing " + str(len(input_node_list)) + " input channels")

            # Compute output
            model_key = self.ui.modelComboBox.currentText
            task_dir = self.get_task_dir(model_key, self._parameterNode)
            tmp_path_in = os.path.join(module_path, "timepoint1", "tmp_in")
            tmp_path_out = os.path.join(module_path, "timepoint1", "tmp_out")

            # set the parameter node to indicate that the segmentation has not yet been loaded
            self._parameterNode.SetParameter("segmentation_loaded_timepoint1", "false")

            progressBar = slicer.util.createProgressDialog(slicer.util.mainWindow(), 0, 100,
                                                           windowTitle="Segmentation timepoint 1")
            # set progress to 0
            progressBar.setValue(0)
            # make the progress bar wider
            slicer.app.processEvents()  # to create the progress bar before the resizing
            progressBar.resize(progressBar.width * 2, progressBar.height)
            time.sleep(0.1)
            slicer.app.processEvents()

            # a process is started in a separate thread, so the GUI remains responsive
            cliNode = run_segmentation(input_node_list,
                                       self.ui.affineregCheckBox.checked,
                                       self.ui.inputisbetCheckBox.checked,
                                       task_dir,
                                       tmp_path_in,
                                       tmp_path_out,
                                       self.ui.lineEdit_python_path.text,
                                       )

            output_segmentation = self.ui.outputSelector.currentNode()

            # clear the output segmentation node
            output_segmentation.GetSegmentation().RemoveAllSegments()

            # remove any transforms from the output segmentation node
            transformNodeID = output_segmentation.GetTransformNodeID()
            if transformNodeID:
                transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
                if transformNode:
                    slicer.mrmlScene.RemoveNode(transformNode)


            # observe the cliNode to get notified when the cli module makes progress (to update progress bar) and
            # finishes execution to load the results from drive into Slicer
            for event in ['ModifiedEvent']:
                cliNode.AddObserver(event,
                                    lambda c, e: self.onCliNodeStatusUpdate(c, e, progressBar, task_dir, tmp_path_out,
                                                                            output_segmentation, input_node_list,
                                                                            timepoint='timepoint1',
                                                                            original_log_level=original_log_level))
            progressBar.canceled.connect(lambda: self.onCancel(cliNode, progressBar))

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute segmentation: " + str(e))
            traceback.print_exc()

        # timepoint 2
        try:
            # assemble input channel nodes
            input_node_list_t2 = [n for n in [self._parameterNode.GetNodeReference("InputVolume_channel1_t2"),
                                              self._parameterNode.GetNodeReference("InputVolume_channel2_t2"),
                                              self._parameterNode.GetNodeReference("InputVolume_channel3_t2"),
                                              self._parameterNode.GetNodeReference("InputVolume_channel4_t2")] if n]

            if debug: print("Passing " + str(len(input_node_list_t2)) + " input channels")

            # Compute output
            model_key = self.ui.modelComboBox.currentText
            task_dir_t2 = self.get_task_dir(model_key, self._parameterNode)
            tmp_path_in_t2 = os.path.join(module_path, "timepoint2", "tmp_in")
            tmp_path_out_t2 = os.path.join(module_path, "timepoint2", "tmp_out")

            # set the parameter node to indicate that the segmentation has not yet been loaded
            self._parameterNode.SetParameter("segmentation_loaded_timepoint2", "false")

            progressBar_t2 = slicer.util.createProgressDialog(slicer.util.mainWindow(), 0, 100,
                                                              windowTitle="Segmentation timepoint 2")
            slicer.app.processEvents()  # to render the progress bar before the resizing
            # move the progress bar below the other one, so it doesn't overlap
            progressBar_t2.move(progressBar.x, progressBar.y + progressBar.height * 2)
            # make the progress bar wider
            progressBar_t2.resize(progressBar_t2.width * 2, progressBar_t2.height)
            time.sleep(0.1)
            slicer.app.processEvents()  # to create the progress bar before the resizing

            cliNode_t2 = run_segmentation(input_node_list_t2,
                                          self.ui.affineregCheckBox_t2.checked,
                                          self.ui.inputisbetCheckBox_t2.checked,
                                          task_dir_t2,
                                          tmp_path_in_t2,
                                          tmp_path_out_t2,
                                          self.ui.lineEdit_python_path.text,
                                          )

            output_segmentation_t2 = self.ui.outputSelector_t2.currentNode()

            # clear the output segmentation node
            output_segmentation_t2.GetSegmentation().RemoveAllSegments()

            # remove any transforms from the output segmentation node
            transformNodeID = output_segmentation_t2.GetTransformNodeID()
            if transformNodeID:
                transformNode = slicer.mrmlScene.GetNodeByID(transformNodeID)
                if transformNode:
                    slicer.mrmlScene.RemoveNode(transformNode)

            for event in ['ModifiedEvent']:
                cliNode_t2.AddObserver(event,
                                       lambda c, e: self.onCliNodeStatusUpdate(c, e, progressBar_t2, task_dir_t2,
                                                                               tmp_path_out_t2, output_segmentation_t2,
                                                                               input_node_list_t2, timepoint='timepoint2',
                                                                               original_log_level=original_log_level))

            progressBar_t2.canceled.connect(lambda: self.onCancel(cliNode_t2, progressBar_t2))

        except Exception as e:
            slicer.util.errorDisplay("Failed to compute segmentation: " + str(e))
            traceback.print_exc()

        return cliNode, cliNode_t2


    def onCliNodeStatusUpdate(self, cliNode, event, progressBar, task_dir, tmp_path_out, output_segmentation,
                              input_volume_list, timepoint, original_log_level=32):
        """
        Callback function to handle the status update of the CLI node.

        Args:
            cliNode: The CLI node that is being observed.
            event: The event that triggered the callback.
            progressBar: The progress bar to update.
            task_dir: The task directory for the segmentation.
            tmp_path_out: The temporary output path for the segmentation.
            output_segmentation: The output segmentation node.
            input_volume_list: The list of input volume nodes.
            timepoint: The timepoint for the segmentation ('timepoint1' or 'timepoint2').
            original_log_level: The original log level for the Slicer application.
        """


        if debug: print("Received an event '%s' from class '%s'" % (event, cliNode.GetClassName()))
        # print(f"Status of CLI for {timepoint} is {cliNode.GetStatusString()}")
        if cliNode.IsA('vtkMRMLCommandLineModuleNode') and not cliNode.GetStatus() == cliNode.Completed:  # do this when the cli module sends a progress update (not when it is done)
            if timepoint == 'timepoint1' and not self.start_time_t1:
                self.start_time_t1 = time.time()
            elif timepoint == 'timepoint2' and not self.start_time_t2:
                self.start_time_t2 = time.time()
            progressBar.setValue(cliNode.GetProgress())
            # print("Status is %s" % cliNode.GetStatusString())
            # print("cli update: progressBar value is %s" % cliNode.GetProgress(), flush=True)
            if cliNode.GetProgress() == 100:
                # disable stderr output from VTK etc that come from the segmentation CLI
                logLevelNone = getattr(ctk.ctkErrorLogLevel, "None")
                if not debug: slicer.app.setPythonConsoleLogLevel(logLevelNone)

        if cliNode.GetStatus() & cliNode.Completed:  # do this when the cli module is done
            # print("Status is %s" % cliNode.GetStatusString())

            # if cliNode.GetStatus() & cliNode.ErrorsMask:  # upon error
            #     outputText = cliNode.GetOutputText()
            #     errorText = cliNode.GetErrorText()
            #     print("CLI output text: \n" + outputText, flush=True)
            #     print("CLI execution failed: \n" + errorText, flush=True)
            # else:  # upon success
            progressBar.setValue(100)
            start_time = self.start_time_t1 if timepoint == 'timepoint1' else self.start_time_t2
            print(f"Segmentation for {timepoint} completed after {int(time.time() - start_time)} seconds.")
            if debug: print("CLI output text: \n" + cliNode.GetOutputText(), flush=True)
            # print("CLI error text: \n" + cliNode.GetErrorText(), flush=True)  # commented out here, because stderr is printed by VTK to the console already
            self.onSegmentationCliNodeSuccess(input_volume_list, output_segmentation,
                                              task_dir, timepoint, tmp_path_out)
            slicer.app.setPythonConsoleLogLevel(original_log_level)  # restore original log level


    def onSegmentationCliNodeSuccess(self, input_volume_list, output_segmentation, task_dir,
                                     timepoint, tmp_path_out):
        """
        Callback function to handle the success of the CLI node.
        This function loads the output segmentation into Slicer and applies the transformation if required.

        Args:
            input_volume_list: The list of input volume nodes.
            output_segmentation: The output segmentation node.
            task_dir: The task directory for the segmentation.
            timepoint: The timepoint for the segmentation ('timepoint1' or 'timepoint2').
            tmp_path_out: The temporary output path for the segmentation.
        """
        # get output files created by external inference process
        # depending on whether registration is required or not, the output file is in a different location
        # if no registration is required
        affine_checkbox = self.ui.affineregCheckBox if timepoint == 'timepoint1' else self.ui.affineregCheckBox_t2
        if not affine_checkbox.checked:
            tmp_file_path_out = os.path.join(tmp_path_out, "output.nii.gz")
        else:  # want to load the segmentation in the segmentation input template space
            tmp_file_path_out = os.path.join(tmp_path_out, "preprocessed", "registered", "output.nii.gz")
        tmp_transform_file_img0 = os.path.join(tmp_path_out, "preprocessed", "registered", "image_0000",
                                               "img_tmp_0000_ANTsregistered_0GenericAffine.mat")

        # check if the output file exists
        if not os.path.exists(tmp_file_path_out):
            slicer.util.errorDisplay("Output segmentation file not found: " + tmp_file_path_out)
            return

        # load the transformation file
        if not affine_checkbox.checked:
            #slicer.util.errorDisplay("Transformation file not found: " + tmp_transform_file_img0)
            print("Transformation file not found: " + tmp_transform_file_img0)
            print("Creating identity transform instead")
            # create a transform node with the identity transform
            transformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")
        else:
            # load the transform
            transformNode = slicer.util.loadTransform(tmp_transform_file_img0)

        # set the transform name
        transformNode.SetName("Transform_" + timepoint + "_channel1")
        # invert the transform
        transformNode.Inverse()
        transformNode.InverseName()

        # get previous reference image for the output segmentation
        referenceImageVolumeNode = output_segmentation.GetNodeReference('referenceImageGeometryRef')
        # delete the previous reference image
        if referenceImageVolumeNode:
            slicer.mrmlScene.RemoveNode(referenceImageVolumeNode)
        # load the file as a labelVolumeNode
        loadedLabelVolumeNode = slicer.util.loadLabelVolume(tmp_file_path_out, properties={"show": False})
        loadedLabelVolumeNode.SetHideFromEditors(1)  # hide the volume from the subject hierarchy

        segmentation = self.ImportLabelmapToSegmentationNodeWithBackgroundSegment(loadedLabelVolumeNode,
                                                                                  output_segmentation)

        # set labelVolumeNode as the reference image for the output segmentation
        output_segmentation.SetReferenceImageGeometryParameterFromVolumeNode(loadedLabelVolumeNode)

        # rename the segments
        # which structures were predicted by the neural network (including background (0))
        predictedStructureVals = list(
            set(np.unique(slicer.util.arrayFromVolume(loadedLabelVolumeNode))))
        # load the label dictionary
        label_names_path = os.path.join(task_dir, "config", "label_names.json")
        with open(label_names_path) as jsonfile:
            label_dict = json.load(jsonfile)
        for seg_idx in range(segmentation.GetNumberOfSegments()):
            slicer.app.processEvents()  # to keep the GUI responsive
            segment = segmentation.GetNthSegment(seg_idx)
            orig_idx = int(predictedStructureVals[seg_idx])
            segment.SetName(label_dict[str(orig_idx)])

        # render in 3D
        output_segmentation.CreateClosedSurfaceRepresentation()

        # make some segments invisible
        # make sure there is a displayNode
        if not output_segmentation.GetDisplayNode():
            output_segmentation.CreateDefaultDisplayNodes()
        displayNode = output_segmentation.GetDisplayNode()
        ids_to_make_invisible = ['0']  # ['1', '2', '3', '4']
        for id in ids_to_make_invisible:
            displayNode.SetSegmentVisibility(id, False)

        # set opacity
        displayNode.SetOpacity3D(0.5)

        # show the first volume in the 2D views of the "timepoint" row
        first_input_volume = input_volume_list[0]
        ui_helper_utils.UIHelperMixin.setBackgroundVolumes(first_input_volume, timepoint)
        measurements2D_utils.Measurements2DMixin.setViews(output_segmentation, timepoint)
        # don't show the output volume in the 2D views, since the segmentation will be shown. The output volume
        # is just kept as a reference image for the segmentation
        slicer.util.setSliceViewerLayers(label=None)

        # set flag to indicate that the segmentation was successfully computed and loaded
        self._parameterNode.SetParameter(f"segmentation_loaded_{timepoint}", "true")

        # apply the transform to the output segmentation
        if self.ui.affineregCheckBox.checked:
            output_segmentation.SetAndObserveTransformNodeID(transformNode.GetID())

        self.setDefaultSegmentFor2DMeasurements("ETC")


    # add an observer to the progress bar cancel button
    def onCancel(self, cliNode, progressBar):
        cliNode.Cancel()
        progressBar.close()


    @staticmethod
    def ImportLabelmapToSegmentationNodeWithBackgroundSegment(loadedLabelVolumeNode, output_segmentation):
        """
        Import the labelVolumeNode into the segmentation node. The labels in the labelVolumeNode are increased by 1
        temporarily to include the background label (0) and then decreased by 1 again to have the original labels.
        (This is because slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode ignores the background label)
        Finally, the segmentIDs are reduced by 1 to have the correct segmentIDs in the segmentation node.

        Args:
        loadedLabelVolumeNode: The label volume node to import.
        output_segmentation: The output segmentation node to import the label volume into.
        """

        # increase values by 1
        labelArray = slicer.util.arrayFromVolume(loadedLabelVolumeNode)
        labelArray += 1
        slicer.util.arrayFromVolumeModified(loadedLabelVolumeNode)
        # convert the labels in the labelVolumeNode into segments of the output segmentation node
        segmentation = output_segmentation.GetSegmentation()
        segmentation.RemoveAllSegments()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(loadedLabelVolumeNode,
                                                                              output_segmentation)
        # decrease values by 1 again
        labelArray -= 1
        slicer.util.arrayFromVolumeModified(loadedLabelVolumeNode)

        # reduce the segmentIDs by 1
        for segID in segmentation.GetSegmentIDs():
            old_segment = segmentation.GetSegment(segID)
            old_segName = old_segment.GetName()
            new_segID = str(int(old_segName) - 1)

            # first change the name of the segment to the new ID
            old_segment.SetName(new_segID)

            # add segment with new ID (and new name)
            segmentation.AddSegment(old_segment, new_segID)
            # remove segment with old ID
            segmentation.RemoveSegment(segID)

        return segmentation


    def setDefaultSegmentFor2DMeasurements(self, defaultSegmentName="ETC"):
        """
        Set the default segment for 2D measurements in the segment selector widget.
        This function checks if the default segment exists in either of the segmentations
        and sets it as the current segment in the segment selector widget.

        Args:
            defaultSegmentName: The name of the default segment to set.
        """
        # check if the default segment exists in either of the segmentations
        segNode1 = self._parameterNode.GetNodeReference("outputSegmentation")
        segNode2 = self._parameterNode.GetNodeReference("outputSegmentation_t2")

        for segNode in [segNode1, segNode2]:
            if segNode:
                segmentId = segNode.GetSegmentation().GetSegmentIdBySegmentName(defaultSegmentName)
                if segmentId:
                    self.ui.SegmentSelectorWidget.setCurrentNode(segNode)
                    self.ui.SegmentSelectorWidget.setCurrentSegmentID(segmentId)
                    break

    @staticmethod
    def get_task_dir(model_key, parameterNode):
        """
        Get the task directory for the given model key.

        Args:
            model_key: The key of the model to get the task directory for.
            parameterNode: The parameter node for the RANO module.
        Returns:
            The task directory for the given model key.
        """
        if model_key == "":
            return ""
        else:
            model_info = json.loads(parameterNode.GetParameter("ModelInfo"))
            task_dir = model_info[str(model_key)]
            return task_dir