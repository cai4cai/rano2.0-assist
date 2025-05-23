import os
from datetime import datetime
import json
import ScreenCapture
import numpy as np
from collections import defaultdict

from reportlab.lib import utils, colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether, Image, Preformatted
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import slicer
import qt

from utils.config import reports_path, debug
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

        self.table_to_csv(report_dir)
        self.create_images(report_dir)
        self.create_json_file(report_dir, timestamp)
        self.create_report_pdf(os.path.join(report_dir, "report.json"))

    @staticmethod
    def get_report_dir_from_node(default_report_dir, node1, node2, timestamp):

        try:
            fallback_report_dir = os.path.join(default_report_dir, "RANO_Report_" + timestamp)
            if not node1:
                print(f"Node {node1} is None - use default report directory")
                return fallback_report_dir
            if not node2:
                print(f"Node {node2} is None - use default report directory")
                return fallback_report_dir

            if hasattr(node1.GetStorageNode(), "GetFileName") and node1.GetStorageNode().GetFileName() \
                and hasattr(node2.GetStorageNode(), "GetFileName") and node2.GetStorageNode().GetFileName():
                input_file_path_1 = node1.GetStorageNode().GetFileName()
                input_file_path_2 = node2.GetStorageNode().GetFileName()
                if "BraTS" in input_file_path_1 and "BraTS" in input_file_path_2:  # BraTS data
                    id1 = os.path.basename(os.path.dirname(input_file_path_1)).replace(".nii.gz", "")
                    id2 = os.path.basename(os.path.dirname(input_file_path_2)).replace(".nii.gz", "")
                    subfolder_name = f"BraTS_{id1}_{id2}"
                elif "KCH" in input_file_path_1 and "TimePoint" in input_file_path_2:  # KCH data
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
                if shNode.GetItemAttribute(dataNodeItemID_1, "DICOM.ContentDate"):
                    date_1 = shNode.GetItemAttribute(dataNodeItemID_1, "DICOM.ContentDate")
                elif shNode.GetItemAttribute(dataNodeItemID_1, "DICOM.SeriesDate"):
                    date_1 = shNode.GetItemAttribute(dataNodeItemID_1, "DICOM.SeriesDate")
                else:
                    date_1 = ""

                dataNodeItemID_2 = shNode.GetItemByDataNode(node2)
                rootID_2 = shNode.GetItemParent(shNode.GetItemParent(dataNodeItemID_2))
                patientID_2 = shNode.GetItemAttribute(rootID_2, "DICOM.PatientID")  # is actually ScanID for KCH dataset
                if shNode.GetItemAttribute(dataNodeItemID_2, "DICOM.ContentDate"):
                    date_2 = shNode.GetItemAttribute(dataNodeItemID_2, "DICOM.ContentDate")
                elif shNode.GetItemAttribute(dataNodeItemID_2, "DICOM.SeriesDate"):
                    date_2 = shNode.GetItemAttribute(dataNodeItemID_2, "DICOM.SeriesDate")
                else:
                    date_2 = ""

                # assemble the subfolder name
                patient_1_info = f"{patientID_1}"
                if date_1 and not date_1 in patient_1_info:
                    patient_1_info += f"-{date_1}"

                patient_2_info = f"{patientID_2}"
                if date_2 and not date_2 in patient_2_info:
                    patient_2_info += f"-{date_2}"

                subfolder_name = f"Report_{patient_1_info}_{patient_2_info}"

                if not subfolder_name:
                    print(f"Could not determine subfolder name from input nodes {node1.GetName()} and {node2.GetName()} - use default report directory")
                    return fallback_report_dir

            report_dir = os.path.normpath(os.path.join(default_report_dir, subfolder_name))
            return report_dir
        except Exception as e:
            print(f"Could not determine report directory from input nodes {node1.GetName()} and {node2.GetName()} - use default report directory")
            return fallback_report_dir


    def create_images(self, report_dir):
        """
        Create images for the report.

        Args:
            report_dir (str): Directory to save the images.
        """
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

            # add the image path to the lineNodePair
            pair.image_path = save_path


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

        # Line pairs
        report_dict["LinePairs"] = {}
        for pair in self.lineNodePairs:
            line_pair_dict = {}
            line_pair_dict["LesionIndex"] = str(pair.lesion_idx)
            line_pair_dict["Timepoint"] = str(pair.timepoint)
            line_pair_dict["Enhancing"] = pair.enhancing
            line_pair_dict["Measurable"] = pair.measurable
            line_pair_dict["Target"] = pair.target
            line_pair_dict["Coordinates"] = np.array(pair.get_coords()).tolist()
            line_pair_dict["LineLengths"] = np.array(pair.get_line_lengths()).tolist()
            line_pair_dict["LineLengthProd"] = pair.get_line_length_product()

            if hasattr(pair, "image_path") and pair.image_path:
                line_pair_dict["ImagePath"] = pair.image_path

            report_dict["LinePairs"][f"Les{pair.lesion_idx}_{pair.timepoint}"] = line_pair_dict


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

    def create_report_pdf(self, report_json_path):
        """
        Create a PDF report from the JSON file.

        Args:
            report_json_path (str): Path to the JSON file.
        """

        # Read the JSON file

        with open(report_json_path, "r") as f:
            report_data = json.load(f)
         
        styles = getSampleStyleSheet()

        header_style = ParagraphStyle(
            name="CenteredHeader",
            fontSize=10,
            textColor=colors.black,
            alignment=TA_CENTER,
            parent=styles['Normal'],
        )

        subtitle_style = ParagraphStyle(
            name="Subtitle",
            fontSize=14,
            leading=18,
            textColor=colors.black,
            alignment=TA_LEFT,
            spaceAfter=6
        )

        table_style_header = TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgrey),  # Color for header (row 0)
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Color for all rows after the header (rows 1 to end)
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Horizontal alignment
        ])

        table_style_noheader = TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),  # Color for all rows after the header (rows 1 to end)
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Horizontal alignment
        ])

        spacer = KeepTogether(Spacer(1, 0.25 * inch))
        spacer2 = KeepTogether(Spacer(1, 0.5 * inch))
         
        # Set up document
        doc = SimpleDocTemplate(report_json_path.replace(".json", ".pdf"), pagesize=letter)

        # Get styles
        title_style = styles['Title']  # or use 'Heading1', 'Heading2', etc.

        # Create a title paragraph
        title = Paragraph("RANO Report", title_style)
         
        subtitle_target = Paragraph("Target Lesions", subtitle_style)
        subtitle_nontarget = Paragraph("Non-target Lesions", subtitle_style)

        linePairs = report_data["LinePairs"]

        # sort by key
        linePairs = dict(sorted(linePairs.items()))

        linePairs_dict = defaultdict(dict)
        for pair in linePairs:
            les_idx = linePairs[pair]["LesionIndex"]
            timepoint = linePairs[pair]["Timepoint"]
            linePairs_dict[les_idx][timepoint] = linePairs[pair]

        # sort the dictionary by the keys
        linePairs_dict = dict(sorted(linePairs_dict.items()))

        img_width = 200  # width in pixels
        img_height = 200

        def get_image(path, width=2 * inch):
            img = utils.ImageReader(path)
            iw, ih = img.getSize()
            aspect = ih / float(iw)
            return Image(path, width=width, height=(width * aspect))

        target_data = []
        nontarget_data = []
        for i, les_idx in enumerate(linePairs_dict):

            # check if target lesion
            if 'timepoint1' in linePairs_dict[les_idx]:
                timepoint1_is_target = linePairs_dict[les_idx]['timepoint1']['Target']
                timepoint1_img = get_image(linePairs_dict[les_idx]['timepoint1']['ImagePath'], width=img_width)
            else:
                timepoint1_is_target = False
                timepoint1_img = None

            if 'timepoint2' in linePairs_dict[les_idx]:
                timepoint2_is_target = linePairs_dict[les_idx]['timepoint2']['Target']
                timepoint2_img = get_image(linePairs_dict[les_idx]['timepoint2']['ImagePath'], width=img_width)
            else:
                timepoint2_is_target = False
                timepoint2_img = None

            if timepoint1_is_target or timepoint2_is_target:
                target_data.append([timepoint1_img, timepoint2_img])
            else:
                nontarget_data.append([timepoint1_img, timepoint2_img])

        # insert headers if there are images
        if target_data:
            target_data.insert(0, [Paragraph("Timepoint 1", header_style), Paragraph("Timepoint 2", header_style)])
        if nontarget_data:
            nontarget_data.insert(0, [Paragraph("Timepoint 1", header_style), Paragraph("Timepoint 2", header_style)])
         
        # Create table with header repeat across pages
        colWidths = [img_width * 1.05, img_width * 1.05]

        if target_data:
            target_data_table = Table(target_data, colWidths=colWidths, repeatRows=1, hAlign="LEFT")
            target_data_table.setStyle(table_style_header)

        if nontarget_data:
            nontarget_data_table = Table(nontarget_data, colWidths=colWidths, repeatRows=1, hAlign="LEFT")
            nontarget_data_table.setStyle(table_style_header)
         
        # Add a table with the lesion info
        subtitle_lesion_info = Paragraph("Lesion Information", subtitle_style)

        lesion_info = []
        for pair_name, pair in linePairs.items():
            new_row = [pair["LesionIndex"],
                       pair["Timepoint"].replace("timepoint", ""),
                       "✓" if pair["Target"] else "✗",
                       "✓" if pair["Measurable"] else "✗",
                       "✓" if pair["Enhancing"] else "✗",
                       f"{pair['LineLengthProd']:.0f} mm²",  # square mm
                       ]

            lesion_info.append(new_row)

        if lesion_info:
            lesion_info.insert(0, ["Lesion Index", "Timepoint", "Target", "Measurable", "Enhancing", "Product"])

            lesion_info_table = Table(lesion_info, repeatRows=1, hAlign="LEFT")
            lesion_info_table.setStyle(table_style_header)
         
        # Lesion-based response classification

        subtitle_lesion_response = Paragraph("Lesion-based Response Classification", subtitle_style)

        lesion_response_status = report_data["ResponseStatus"]

        lesion_response_data = []
        for key, val in lesion_response_status.items():
            if "lesion" in key:
                new_row = [key, f"{val:.0f}"]
            elif "product" in key:
                new_row = [key, f"{val:.0f} mm²"]
            elif "change" in key:
                new_row = [key, f"{val:.1f} %"]
            else:
                new_row = [key, val]

            lesion_response_data.append(new_row)

        if lesion_response_data:
            lesion_response_data.insert(0, ["Lesion Index", "Response Classification"])

        lesion_response_table = Table(lesion_response_data, repeatRows=1, hAlign="LEFT")
        lesion_response_table.setStyle(table_style_noheader)
         
        # Overall response classification
        subtitle_overall_response = Paragraph("Overall Response Classification", subtitle_style)

        overall_response_status = report_data["OverallResponseStatus"]
        overall_response_data = []
        for key, val in overall_response_status.items():
            new_row = [key, val]

            overall_response_data.append(new_row)

        if overall_response_data:
            overall_response_data.insert(0, ["Lesion Index", "Response Classification"])

        overall_response_table = Table(overall_response_data, repeatRows=1, hAlign="LEFT")
        overall_response_table.setStyle(table_style_noheader)
         
        # Settings
        subtitle_settings = Paragraph("Settings", subtitle_style)

        settings = []
        settings.append(["Report Time",
                         datetime.strptime(report_data["ReportTime"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")])
        settings.append(["Affine registration t1", report_data["ParameterNode"]["AffineReg"]])
        settings.append(["Affine registration t2", report_data["ParameterNode"]["AffineReg_t2"]])
        settings.append(["Input is skull-stripped t1", report_data["ParameterNode"]["InputIsBET"]])
        settings.append(["Input is skull-stripped t2", report_data["ParameterNode"]["InputIsBET_t2"]])
        settings.append(["Segmentation model timepoint 1", report_data["ParameterNode"]["model_key"]])
        settings.append(["Segmentation model timepoint 2", report_data["ParameterNode"]["model_key_t2"]])
        settings.append(["Segment for 2D measurements", report_data["2DMeasurement"]["Segment2DMeasurement"]])
        settings.append(["Method 2D  measurements", report_data["2DMeasurement"]["Method2DMeasurement"]])
        settings.append(["Allow axial orientation", report_data["ParameterNode"]["axial"]])
        settings.append(["Allow sagittal orientation", report_data["ParameterNode"]["sagittal"]])
        settings.append(["Allow coronal orientation", report_data["ParameterNode"]["coronal"]])
        settings.append(["Force same orientation", report_data["ParameterNode"]["orient_cons_tp"]])
        settings.append(["Force approx same slice", report_data["ParameterNode"]["same_slc_tp"]])

        if settings:
            settings.insert(0, ["Setting", "Value"])
            settings_table = Table(settings, repeatRows=1, hAlign="LEFT")
            settings_table.setStyle(table_style_header)

        if debug:
            # add the json data to the end of the document
            json_style = ParagraphStyle(
                name="JsonStyle",
                fontSize=10,
                textColor=colors.black,
                alignment=TA_LEFT,
                parent=styles['Normal'],
            )
            json_str = json.dumps(report_data, indent=4)

            # Use Preformatted to preserve line breaks and spacing
            json_block = Preformatted(json_str, style=styles['Code'])
         
        # Build the PDF with title, spacer, and table
        components = [title, spacer]
        if 'target_data' in locals() and target_data:
            components.append(subtitle_target)
            components.append(target_data_table)
            components.append(spacer2)
        if 'nontarget_data' in locals() and nontarget_data:
            components.append(subtitle_nontarget)
            components.append(nontarget_data_table)
            components.append(spacer2)
        if 'lesion_info_table' in locals() and lesion_info_table:
            components.append(subtitle_lesion_info)
            components.append(lesion_info_table)
            components.append(spacer2)
        if 'lesion_response_table' in locals() and lesion_response_table:
            components.append(subtitle_lesion_response)
            components.append(lesion_response_table)
            components.append(spacer2)
        if 'overall_response_table' in locals() and overall_response_table:
            components.append(subtitle_overall_response)
            components.append(overall_response_table)
            components.append(spacer2)
        if 'settings_table' in locals() and settings_table:
            components.append(subtitle_settings)
            components.append(settings_table)
            components.append(spacer2)
        if 'json_block' in locals() and json_block:
            components.append(json_block)
            components.append(spacer2)
         
        # Build the document
        doc.build(components)
