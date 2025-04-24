import os
import shutil
from datetime import datetime
import json
import ScreenCapture
import slicer
import qt
from utils.config import module_path

import utils.measurements2D_utils as measurements2D_utils
import utils.segmentation_utils as segmentation_utils

from utils.rano_utils import find_closest_plane


class ReportCreationMixin:
    def __init__(self, parameterNode, ui, lineNodePairs):
        self._parameterNode = parameterNode
        self.ui = ui
        self.lineNodePairs = lineNodePairs

    def onCreateReportButton(self):
        # create a report
        self.create_report()

    def create_report(self):


        suggested_report_dir = self.get_report_dir_from_node(default_report_dir=os.path.join(module_path, "Resources", "Reports"),
                                                   node=self._parameterNode.GetNodeReference("InputVolume_channel1_t1"))

        # ask the user if they want to use the suggested_report_dir or create a new report directory via QFileDialog
        msg = qt.QMessageBox()
        msg.setText(f"Suggested report directory: {suggested_report_dir}")
        if os.path.exists(suggested_report_dir):
            msg.setInformativeText("Do you want to overwrite the files in the directory?")
        else:
            msg.setInformativeText("Do you want to create a new report directory?")
        msg.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        msg.setDefaultButton(qt.QMessageBox.No)
        ret = msg.exec_()
        if ret == qt.QMessageBox.Yes:
            report_dir = suggested_report_dir
        else:
            qfiledialog = qt.QFileDialog()
            qfiledialog.setFileMode(qt.QFileDialog.Directory)
            qfiledialog.setWindowTitle("Select or create a folder")
            qfiledialog.setDirectory(suggested_report_dir)
            report_dir = qfiledialog.getExistingDirectory()
            if not report_dir:
                print("No directory selected - report creation cancelled")
                return

        print(f"Selected directory: {report_dir}")

        os.makedirs(report_dir, exist_ok=True)

        self.create_json_file(report_dir)
        self.table_to_csv(report_dir)
        self.create_images(report_dir)

    @staticmethod
    def get_report_dir_from_node(default_report_dir, node):
        if not node:
            print("Node is None - use default report directory")
            return default_report_dir

        elif hasattr(node.GetStorageNode(), "GetFileName") and node.GetStorageNode().GetFileName():
            input_file_path = node.GetStorageNode().GetFileName()
            if "BraTS" in input_file_path:
                subfolder_name = os.path.basename(os.path.dirname(input_file_path)).replace(".nii.gz", "")
            elif "TimePoint" in input_file_path:  # KCH data
                subfolder_name = "KCH_" + os.path.basename(
                    os.path.split(os.path.dirname(input_file_path))[-2])
            else:
                print(f"No report directory specified for {input_file_path} - use default report directory")
                return default_report_dir

        else:  # probably dicom data
            # assemble the subfolder name from the dicom tags
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            dataNodeItemID = shNode.GetItemByDataNode(node)
            rootID = shNode.GetItemParent(shNode.GetItemParent(dataNodeItemID))
            patientID = shNode.GetItemAttribute(rootID, "DICOM.PatientID")
            subfolder_name = patientID

            if not subfolder_name:
                print(f"Could not determine subfolder name from input node {node.GetName()} - use default report directory")
                return default_report_dir

        report_dir = os.path.join(default_report_dir, subfolder_name)
        return report_dir


    def create_images(self, report_dir):
        # create one image for each line pair
        for pair in self.lineNodePairs:
            timepoint = pair.timepoint
            # focus the view on the line pair
            measurements2D_utils.Measurements2DMixin.centerTimepointViewsOnCenterPoint(pair, timepoint)

            # decide which orientation to save

            if timepoint == 'timepoint1':
                viewnames = ["Red", "Yellow", "Green"]
            elif timepoint == 'timepoint2':
                viewnames = ["Red_2", "Yellow_2", "Green_2"]
            else:
                raise ValueError(f"timepoint must be 'timepoint1' or 'timepoint2' but is {timepoint}")
            coords = pair.get_coords()
            if coords.size == 12:
                axis = find_closest_plane(coords)
                if axis == 0:
                    viewname = viewnames[1]
                elif axis == 1:
                    viewname = viewnames[2]
                elif axis == 2:
                    viewname = viewnames[0]
                else:
                    raise ValueError(f"Axis {axis} not recognized")
            else:
                raise ValueError(
                    f"not all 12 coordinates are present for lesion {pair.lesion_idx} at timepoint {pair.timepoint}, but {coords.size}")

            # save the view as an image
            viewNodeID = slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().GetSliceNode().GetID()
            cap = ScreenCapture.ScreenCaptureLogic()
            view = cap.viewFromNode(slicer.mrmlScene.GetNodeByID(viewNodeID))
            save_path = os.path.join(report_dir, f"LinePair_{pair.lesion_idx}_{pair.timepoint}.png")
            cap.captureImageFromView(view, save_path)


    @staticmethod
    def table_to_csv(report_dir):
        csv_path = os.path.join(report_dir, "ResultsTable.csv")
        tableName = "Results"
        resultTableNode = slicer.mrmlScene.GetFirstNodeByName(tableName)
        if resultTableNode:
            slicer.util.saveNode(resultTableNode, csv_path)
        else:
            print("No table found or table does not have GetID method")


    def create_json_file(self, report_dir):
        report_path = os.path.join(report_dir, "report.json")

        print(f"Creating report at {report_path}")

        report_dict = {}

        report_dict["ReportTime"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # segmentation models
        model_key = self.ui.modelComboBox.currentText
        task_dir_t1 = segmentation_utils.SegmentationMixin.get_task_dir(model_key, self._parameterNode)
        report_dict["SegmentationTaskDir_t1"] = task_dir_t1

        model_key = self.ui.modelComboBox_t2.currentText
        task_dir_t2 = segmentation_utils.SegmentationMixin.get_task_dir(model_key, self._parameterNode)
        report_dict["SegmentationTaskDir_t2"] = task_dir_t2

        # 2D measurement options
        if self.ui.SegmentSelectorWidget.currentNode():
            currentSegmentID = self.ui.SegmentSelectorWidget.currentSegmentID()
            currentSegment = self.ui.SegmentSelectorWidget.currentNode().GetSegmentation().GetSegment(currentSegmentID)
            currentSegmentName = currentSegment.GetName() if currentSegment else "None"
            report_dict["Segment2DMeasurement"] = currentSegmentName
            report_dict["Method2DMeasurement"] = self.ui.method2DmeasComboBox.currentText

        # save all parameter node parameters
        parameter_node_dict = {}
        for param in self._parameterNode.GetParameterNames():
            parameter_node_dict[param] = self._parameterNode.GetParameter(param)
        report_dict["ParameterNode"] = parameter_node_dict

        # write the report to a json file
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
