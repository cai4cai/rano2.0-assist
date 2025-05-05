import os
import shutil
from datetime import datetime
import json
import ScreenCapture
import slicer
import qt
from utils.config import reports_path

import utils.measurements2D_utils as measurements2D_utils
import utils.segmentation_utils as segmentation_utils

from utils.rano_utils import find_closest_plane


class ReportCreationMixin:
    """
    Mixin class for creating a report for the RANO module.
    """
    def __init__(self, parameterNode, ui, lineNodePairs):
        self._parameterNode = parameterNode
        """Parameter node for the RANO module"""

        self.ui = ui
        """UI for the RANO module"""

        self.lineNodePairs = lineNodePairs
        """List of line node pairs for the RANO module"""

    def onCreateReportButton(self):
        """
        Triggered when the user clicks the "Create Report" button.
        """
        # create a report
        self.create_report()

    def create_report(self):
        """
        Create a report for the RANO module.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_report_dir = self.get_report_dir_from_node(default_report_dir=reports_path,
                                                             node1=self._parameterNode.GetNodeReference("InputVolume_channel1_t1"),
                                                             node2=self._parameterNode.GetNodeReference("InputVolume_channel1_t2"),
                                                             timestamp=timestamp)

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

        report_dir = os.path.normpath(report_dir)
        print(f"Selected directory: {report_dir}")

        os.makedirs(report_dir, exist_ok=True)

        self.create_json_file(report_dir, timestamp)
        self.table_to_csv(report_dir)
        self.create_images(report_dir)

    @staticmethod
    def get_report_dir_from_node(default_report_dir, node1, node2, timestamp):

        fallback_report_dir = os.path.join(default_report_dir, "RANO_Report_" + timestamp)
        if not node1:
            print(f"Node {node1} is None - use default report directory")
            return fallback_report_dir
        if not node2:
            print(f"Node {node2} is None - use default report directory")
            return fallback_report_dir

        elif hasattr(node1.GetStorageNode(), "GetFileName") and node1.GetStorageNode().GetFileName() \
            and hasattr(node2.GetStorageNode(), "GetFileName") and node2.GetStorageNode().GetFileName():
            input_file_path_1 = node1.GetStorageNode().GetFileName()
            input_file_path_2 = node2.GetStorageNode().GetFileName()
            if "BraTS" in input_file_path_1 and "BraTS" in input_file_path_2:  # BraTS data
                id1 = os.path.basename(os.path.dirname(input_file_path_1)).replace(".nii.gz", "")
                id2 = os.path.basename(os.path.dirname(input_file_path_2)).replace(".nii.gz", "")
                subfolder_name = f"BraTS_{id1}_{id2}"
            elif "TimePoint" in input_file_path_1 and "TimePoint" in input_file_path_2:  # KCH data
                id1 = os.path.basename(os.path.split(os.path.dirname(input_file_path_1))[-2])
                id2 = os.path.basename(os.path.split(os.path.dirname(input_file_path_2))[-2])
                subfolder_name = f"KCH_{id1}_{id2}"
            else:
                print(f"No report directory specified for {input_file_path_1} and {input_file_path_2} - use default report directory")
                return fallback_report_dir

        else:  # probably dicom data
            # assemble the subfolder name from the dicom tags
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

            dataNodeItemID_1 = shNode.GetItemByDataNode(node1)
            rootID_1 = shNode.GetItemParent(shNode.GetItemParent(dataNodeItemID_1))
            patientID_1 = shNode.GetItemAttribute(rootID_1, "DICOM.PatientID")  # is actually ScanID for KCH dataset

            dataNodeItemID_2 = shNode.GetItemByDataNode(node2)
            rootID_2 = shNode.GetItemParent(shNode.GetItemParent(dataNodeItemID_2))
            patientID_2 = shNode.GetItemAttribute(rootID_2, "DICOM.PatientID")  # is actually ScanID for KCH dataset

            subfolder_name = f"KCH_{patientID_1}_{patientID_2}"

            if not subfolder_name:
                print(f"Could not determine subfolder name from input nodes {node1.GetName()} and {node2.GetName()} - use default report directory")
                return fallback_report_dir

        report_dir = os.path.normpath(os.path.join(default_report_dir, subfolder_name))
        return report_dir


    def create_images(self, report_dir):
        """
        Create images for the report.

        Args:
            report_dir (str): Directory to save the images.
        """
        # create one image for each line pair
        for pair in self.lineNodePairs:
            timepoint = pair.timepoint
            # select the reference input image so that line pairs are in the plane

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
        """
        Save the results table to a CSV file.

        Args:
            report_dir (str): Directory to save the CSV file.
        """
        csv_path = os.path.join(report_dir, "ResultsTable.csv")
        tableName = "Results"
        resultTableNode = slicer.mrmlScene.GetFirstNodeByName(tableName)
        if resultTableNode:
            slicer.util.saveNode(resultTableNode, csv_path)
        else:
            print("No table found or table does not have GetID method")


    def create_json_file(self, report_dir, timestamp):
        """
        Create a JSON file with the report information.

        Args:
            report_dir (str): Directory to save the JSON file.
        """
        report_path = os.path.join(report_dir, "report.json")

        print(f"Creating report at {report_path}")

        report_dict = {}

        report_dict["ReportTime"] = timestamp

        report_dict["Segmentation"] = {}
        # segmentation models
        model_key = self.ui.modelComboBox.currentText
        task_dir_t1 = segmentation_utils.SegmentationMixin.get_task_dir(model_key, self._parameterNode)
        report_dict["Segmentation"]["SegmentationTaskDir_t1"] = task_dir_t1

        model_key = self.ui.modelComboBox_t2.currentText
        task_dir_t2 = segmentation_utils.SegmentationMixin.get_task_dir(model_key, self._parameterNode)
        report_dict["Segmentation"]["SegmentationTaskDir_t2"] = task_dir_t2

        # 2D measurement options
        report_dict["2DMeasurement"] = {}
        if self.ui.SegmentSelectorWidget.currentNode():
            currentSegmentID = self.ui.SegmentSelectorWidget.currentSegmentID()
            currentSegment = self.ui.SegmentSelectorWidget.currentNode().GetSegmentation().GetSegment(currentSegmentID)
            currentSegmentName = currentSegment.GetName() if currentSegment else "None"
            report_dict["2DMeasurement"]["Segment2DMeasurement"] = currentSegmentName
            report_dict["2DMeasurement"]["Method2DMeasurement"] = self.ui.method2DmeasComboBox.currentText


        # Response Status (Lesion based)
        report_dict["ResponseStatus"] = {}

        report_dict["ResponseStatus"]["Number of indep. target lesions"] = self.ui.numTargetLesSpinBox.value
        report_dict["ResponseStatus"]["Number of new target lesions"] = self.ui.numNewLesSpinBox.value
        report_dict["ResponseStatus"]["Number of disapp. target lesions"] = self.ui.numDisappLesSpinBox.value
        report_dict["ResponseStatus"]["Number of new measurable lesions"] = self.ui.numNewMeasLesSpinBox.value

        report_dict["ResponseStatus"]["Sum of bidim. products timepoint 1"] = self.ui.sum_lineprods_t1_spinbox.value
        report_dict["ResponseStatus"]["Sum of bidim. products timepoint 2"] = self.ui.sum_lineprods_t2_spinbox.value

        report_dict["ResponseStatus"]["Relative change"] = self.ui.sum_lineprods_relchange_spinbox.value
        report_dict["ResponseStatus"]["Lesion Response Status"] = self.ui.responseStatusComboBox.currentText

        # Overall Response Status
        report_dict["OverallResponseStatus"] = {}
        report_dict["OverallResponseStatus"]["Type of Tumor Component to Evaluate"] = self.ui.ceOrNonCeComboBox.currentText
        report_dict["OverallResponseStatus"]["Confirmation required for PD"] = self.ui.confirmationRequiredForPdCheckBox.isChecked()
        report_dict["OverallResponseStatus"]["Reference scan"] = self.ui.referenceScanComboBox.currentText
        report_dict["OverallResponseStatus"]["Curr. scan lesion response"] = self.ui.currScanComboBox.currentText
        report_dict["OverallResponseStatus"]["New Measurable Disease"] = self.ui.newMeasLesCheckBox.isChecked()
        report_dict["OverallResponseStatus"]["Nontarget and nonmeasurable Lesion(s)"] = self.ui.nonTargetNonMeasComboBox.currentText
        report_dict["OverallResponseStatus"]["Clinical Status"] = self.ui.clinicalStatusComboBox.currentText
        report_dict["OverallResponseStatus"]["Increased Steroid Use"] = self.ui.increasedSteroidUseCheckBox.isChecked()
        report_dict["OverallResponseStatus"]["Steroid Dose"] = self.ui.steroidDoseComboBox.currentText
        report_dict["OverallResponseStatus"]["Second line medication"] = self.ui.secondLineMedicationCheckBox.isChecked()
        report_dict["OverallResponseStatus"]["Overall Response Status"] = self.ui.overallResponseStatusComboBox.currentText


        # save all parameter node parameters
        parameter_node_dict = {}
        for param in self._parameterNode.GetParameterNames():
            parameter_node_dict[param] = self._parameterNode.GetParameter(param)
        report_dict["ParameterNode"] = parameter_node_dict

        # write the report to a json file
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=4)
