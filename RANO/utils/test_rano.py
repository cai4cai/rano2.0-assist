import os
import time
from glob import glob
import json

import qt
import slicer
from slicer.ScriptedLoadableModule import *
from DICOMLib import DICOMUtils

from utils.config import module_path, test_data_path


#
# RANOTest
#
class RANOTest(ScriptedLoadableModuleTest):
    """
    This is the test class for the RANO module.
    It is used to test the RANO module and its functionality.
    Upon clicking the "Reload and Test" button in the module (only shown when the module is loaded in developer mode),
    this classes runTest() method is called.
    """
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        #self.test_RANO_dicom()
        #self.test_RANO_nifti()
        self.test_eano()

    def test_eano(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''

        base_path = "/home/aaron/KCL_data/RANO/Input_Data"

        kch_dcm_timepoint_pairs =   [
                                     (['Patient_003', 'TimePoint_001'], ['Patient_003', 'TimePoint_002']),
                                     (['Patient_003', 'TimePoint_001'], ['Patient_003', 'TimePoint_003']),
                                     (['Patient_010', 'TimePoint_001'], ['Patient_010', 'TimePoint_002']),
                                     (['Patient_010', 'TimePoint_001'], ['Patient_010', 'TimePoint_003']),
                                     (['Patient_014', 'TimePoint_002'], ['Patient_014', 'TimePoint_003']),
                                     (['Patient_019', 'TimePoint_001'], ['Patient_019', 'TimePoint_002']),
                                     (['Patient_023', 'TimePoint_003'], ['Patient_023', 'TimePoint_004']),
                                     (['Patient_023', 'TimePoint_003'], ['Patient_023', 'TimePoint_005']),
                                     (['Patient_026', 'TimePoint_001'], ['Patient_026', 'TimePoint_002']),
                                     (['Patient_026', 'TimePoint_001'], ['Patient_026', 'TimePoint_003']),
                                     (['Patient_026', 'TimePoint_001'], ['Patient_026', 'TimePoint_004']),
                                     (['Patient_026', 'TimePoint_001'], ['Patient_026', 'TimePoint_005']),
                                     # (['Patient_030', 'TimePoint_003'], ['Patient_030', 'TimePoint_004']),  # this case didn't run through. it didn't load the t2w into the input selector. had to use the gui manually
                                     (['Patient_031', 'TimePoint_001'], ['Patient_031', 'TimePoint_002']),
                                     (['Patient_032', 'TimePoint_002'], ['Patient_032', 'TimePoint_003']),
                                     (['Patient_032', 'TimePoint_002'], ['Patient_032', 'TimePoint_005']),
                                     (['Patient_036', 'TimePoint_001'], ['Patient_036', 'TimePoint_004']),
                                     (['Patient_037', 'TimePoint_004'], ['Patient_037', 'TimePoint_005']),
                                     (['Patient_038', 'TimePoint_003'], ['Patient_038', 'TimePoint_004']),
                                     (['Patient_038', 'TimePoint_003'], ['Patient_038', 'TimePoint_005']),
                                    ]

        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels
        for test_case_idx in range(len(kch_dcm_timepoint_pairs)):
            patient = kch_dcm_timepoint_pairs[test_case_idx][0][0]
            timepoint_t1 = kch_dcm_timepoint_pairs[test_case_idx][0][1]
            timepoint_t2 = kch_dcm_timepoint_pairs[test_case_idx][1][1]

            p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1c")
            p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1n")
            p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2f")
            p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2w")

            p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1c")
            p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1n")
            p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2f")
            p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2w")
            #
            # if not "006" in p_t1c_tp1:
            #     continue

            # check if all files exist
            if not all([os.path.isdir(p) for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]]):
                print(f"Not all files exist for test case {test_case_idx}")
                continue

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])


        '''
        run test for each test case
        '''

        print(f"Running {len(cases_of_paths_2tp_times_4channels)} test cases")
        for test_case_idx, curr_paths in enumerate(cases_of_paths_2tp_times_4channels):
            # print(f"Paths t1: {curr_paths[0]}")
            # print(f"Paths t2: {curr_paths[1]}")

            paths_t1 = curr_paths[0]
            paths_t2 = curr_paths[1]

            # clear the scene
            slicer.mrmlScene.Clear()

            # load dicoms into slicer using DICOMutils

            def dcm_dir_to_node(dcm_dir):
                with DICOMUtils.TemporaryDICOMDatabase() as db:
                    DICOMUtils.importDicom(dcm_dir, db)
                    patientUIDs = db.patients()
                    loadedNodeIDs = []
                    for patientUID in patientUIDs:
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

                    if not len(loadedNodeIDs) == 1:
                        print(f"Expected 1 loaded node from importing DICOM files in {dcm_dir}"
                                                     f" but got {len(loadedNodeIDs)}: {loadedNodeIDs}")

                    loadedNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0])
                return loadedNode

            inputVolumes = [dcm_dir_to_node(p) for p in paths_t1]
            inputVolumes_t2 = [dcm_dir_to_node(p) for p in paths_t2]

            self.delayDisplay('Loaded test data set')

            slicer.modules.RANOWidget.ui.checkBox_axial.setChecked(True)
            slicer.modules.RANOWidget.ui.checkBox_coronal.setChecked(False)
            slicer.modules.RANOWidget.ui.checkBox_sagittal.setChecked(False)
            slicer.modules.RANOWidget.ui.checkBox_same_slc_tp.setChecked(False)

            self.test_pipeline(inputVolumes,
                               inputVolumes_t2,
                               do_affinereg=True,
                               input_is_bet=False,
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)

    def test_RANO_dicom(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''

        base_path = os.path.join(test_data_path, 'KCH-internal')

        kch_dcm_timepoint_pairs = [((f"Patient_{id:03d}", f"TimePoint_001"), (f"Patient_{id:03d}", f"TimePoint_002"))
                                  for id in range(1, 40)]

        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels
        for test_case_idx in range(len(kch_dcm_timepoint_pairs)):
            patient = kch_dcm_timepoint_pairs[test_case_idx][0][0]
            timepoint_t1 = kch_dcm_timepoint_pairs[test_case_idx][0][1]
            timepoint_t2 = kch_dcm_timepoint_pairs[test_case_idx][1][1]

            p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1c")
            p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1n")
            p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2f")
            p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2w")

            p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1c")
            p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1n")
            p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2f")
            p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2w")

            if not "006" in p_t1c_tp1:
                continue

            # check if all files exist
            if not all([os.path.isdir(p) for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]]):
                print(f"Not all files exist for test case {test_case_idx}")
                continue

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])


        '''
        run test for each test case
        '''

        print(f"Running {len(cases_of_paths_2tp_times_4channels)} test cases")
        for test_case_idx, curr_paths in enumerate(cases_of_paths_2tp_times_4channels):
            # print(f"Paths t1: {curr_paths[0]}")
            # print(f"Paths t2: {curr_paths[1]}")

            paths_t1 = curr_paths[0]
            paths_t2 = curr_paths[1]

            # clear the scene
            slicer.mrmlScene.Clear()

            # load dicoms into slicer using DICOMutils

            def dcm_dir_to_node(dcm_dir):
                with DICOMUtils.TemporaryDICOMDatabase() as db:
                    DICOMUtils.importDicom(dcm_dir, db)
                    patientUIDs = db.patients()
                    loadedNodeIDs = []
                    for patientUID in patientUIDs:
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

                    if not len(loadedNodeIDs) == 1:
                        print(f"Expected 1 loaded node from importing DICOM files in {dcm_dir}"
                                                     f" but got {len(loadedNodeIDs)}: {loadedNodeIDs}")

                    loadedNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0])
                return loadedNode

            inputVolumes = [dcm_dir_to_node(p) for p in paths_t1]
            inputVolumes_t2 = [dcm_dir_to_node(p) for p in paths_t2]

            self.delayDisplay('Loaded test data set')

            self.test_pipeline(inputVolumes,
                               inputVolumes_t2,
                               do_affinereg=True,
                               input_is_bet=False,
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)


    def test_RANO_nifti(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''
        # to store all the paths for all test cases
        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels

        base_path = os.path.join(test_data_path, 'MU-Glioma-Post')

        timepoint_pairs = [
            ("PatientID_0003/Timepoint_1/PatientID_0003_Timepoint_1_brain_", "PatientID_0003/Timepoint_2/PatientID_0003_Timepoint_2_brain_"),

        ]

        # check all files exist
        for brats_case_t1, brats_case_t2 in timepoint_pairs:
            assert(os.path.isfile(os.path.join(base_path, brats_case_t1 + "t1c.nii.gz")))
            assert(os.path.isfile(os.path.join(base_path, brats_case_t1 + "t1n.nii.gz")))
            assert(os.path.isfile(os.path.join(base_path, brats_case_t1 + "t2f.nii.gz")))
            assert(os.path.isfile(os.path.join(base_path, brats_case_t1 + "t2w.nii.gz")))

        for test_case_idx in range(len(timepoint_pairs)):
            p_t1c_tp1 = os.path.join(base_path, timepoint_pairs[test_case_idx][0] + "t1c.nii.gz")
            p_t1n_tp1 = os.path.join(base_path, timepoint_pairs[test_case_idx][0] + "t1n.nii.gz")
            p_t2f_tp1 = os.path.join(base_path, timepoint_pairs[test_case_idx][0] + "t2f.nii.gz")
            p_t2w_tp1 = os.path.join(base_path, timepoint_pairs[test_case_idx][0] + "t2w.nii.gz")

            p_t1c_tp2 = os.path.join(base_path, timepoint_pairs[test_case_idx][1] + "t1c.nii.gz")
            p_t1n_tp2 = os.path.join(base_path, timepoint_pairs[test_case_idx][1] + "t1n.nii.gz")
            p_t2f_tp2 = os.path.join(base_path, timepoint_pairs[test_case_idx][1] + "t2f.nii.gz")
            p_t2w_tp2 = os.path.join(base_path, timepoint_pairs[test_case_idx][1] + "t2w.nii.gz")

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])


        '''
        run test for each test case
        '''

        print(f"Running {len(cases_of_paths_2tp_times_4channels)} test cases")
        for test_case_idx, curr_paths in enumerate(cases_of_paths_2tp_times_4channels):
            # print(f"Paths t1: {curr_paths[0]}")
            # print(f"Paths t2: {curr_paths[1]}")

            paths_t1 = curr_paths[0]
            paths_t2 = curr_paths[1]

            # clear the scene
            slicer.mrmlScene.Clear()

            # load volumes into slicer
            inputVolumes = [slicer.util.loadVolume(p) for p in paths_t1]
            inputVolumes_t2 = [slicer.util.loadVolume(p) for p in paths_t2]

            self.delayDisplay('Loaded test data set')

            self.test_pipeline(inputVolumes,
                               inputVolumes_t2,
                               do_affinereg=False,
                               input_is_bet=True,
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)

    def test_pipeline(self,
                      inputVolumes,
                      inputVolumes_t2,
                      do_affinereg,
                      input_is_bet,
                      seg_model_key,
                      automatic_segmentation,
                      line_placement,
                      report_creation,):
        # set the UI

        # print(f"input volumes = {inputVolumes}")
        # print(f"input volumes t2 = {inputVolumes_t2}")

        # set the input volumes
        slicer.modules.RANOWidget.ui.inputSelector_channel1_t1.setCurrentNode(inputVolumes[0])
        slicer.modules.RANOWidget.ui.inputSelector_channel2_t1.setCurrentNode(inputVolumes[1])
        slicer.modules.RANOWidget.ui.inputSelector_channel3_t1.setCurrentNode(inputVolumes[2])
        slicer.modules.RANOWidget.ui.inputSelector_channel4_t1.setCurrentNode(inputVolumes[3])

        slicer.modules.RANOWidget.ui.inputSelector_channel1_t2.setCurrentNode(inputVolumes_t2[0])
        slicer.modules.RANOWidget.ui.inputSelector_channel2_t2.setCurrentNode(inputVolumes_t2[1])
        slicer.modules.RANOWidget.ui.inputSelector_channel3_t2.setCurrentNode(inputVolumes_t2[2])
        slicer.modules.RANOWidget.ui.inputSelector_channel4_t2.setCurrentNode(inputVolumes_t2[3])

        # set the output segmentation
        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        outputSegmentation.SetName("outputSegmentation_t1")
        slicer.modules.RANOWidget.ui.outputSelector.setCurrentNode(outputSegmentation)

        outputSegmentation_t2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        outputSegmentation_t2.SetName("outputSegmentation_t2")
        slicer.modules.RANOWidget.ui.outputSelector_t2.setCurrentNode(outputSegmentation_t2)

        # set the segmentation model
        model_info = json.loads(slicer.modules.RANOWidget._parameterNode.GetParameter("ModelInfo"))
        model_index = list(model_info.keys()).index(seg_model_key)
        slicer.modules.RANOWidget.ui.modelComboBox.setCurrentIndex(model_index)
        slicer.modules.RANOWidget.ui.modelComboBox_t2.setCurrentIndex(model_index)

        # set the affine registration and input is bet checkboxes
        slicer.modules.RANOWidget.ui.affineregCheckBox.checked = do_affinereg
        slicer.modules.RANOWidget.ui.inputisbetCheckBox.checked = input_is_bet

        slicer.modules.RANOWidget.ui.affineregCheckBox_t2.checked = do_affinereg
        slicer.modules.RANOWidget.ui.inputisbetCheckBox_t2.checked = input_is_bet

        if automatic_segmentation:
            # press the run segmentation button
            print("Starting the automatic segmentation")
            cliNode, cliNode_t2 = slicer.modules.RANOWidget.onCalcSegmentationsButton()

            # block execution here until all segments have been loaded into the segmentation nodes
            # check if segmentations have been loaded
            parameterNode = slicer.modules.RANOWidget._parameterNode
            while (parameterNode.GetParameter("segmentation_loaded_timepoint1") != "true"
                   or parameterNode.GetParameter("segmentation_loaded_timepoint2") != "true"):
                slicer.app.processEvents()  # Keep the GUI responsive and run the events that create the segmentation after the CLIs have finished
                time.sleep(0.1)
                # print("waiting for the segmentation to be loaded")
        else:
            print("Skipping the automatic segmentation")

        if line_placement:
            print("Starting the automatic 2D measurements")
            # select the segment in the segment selector
            slicer.modules.RANOWidget.ui.SegmentSelectorWidget.setCurrentNode(outputSegmentation)

            segmentID = "3"  # Enhancing tumor
            predicted_segmentIDs = slicer.modules.RANOWidget.ui.SegmentSelectorWidget.currentNode().GetSegmentation().GetSegmentIDs()
            # select the first segment in the segment selector
            if segmentID in predicted_segmentIDs:
                slicer.modules.RANOWidget.ui.SegmentSelectorWidget.setCurrentSegmentID(segmentID)
            else:
                slicer.modules.RANOWidget.ui.SegmentSelectorWidget.setCurrentSegmentID(None)

            # set the method
            slicer.modules.RANOWidget.ui.method2DmeasComboBox.setCurrentIndex(
                0)  # 0 for RANO, 1 for RANO_open2D, 2 for RANO_open3D, 3 for Random

            # press the calc 2D button
            slicer.modules.RANOWidget.onCalc2DButton()
        else:
            print("Skipping the automatic 2D measurements")

        if report_creation:
            print("Starting the report creation")

            def auto_click_yes():
                widgets = qt.QApplication.topLevelWidgets()
                for w in widgets:
                    if isinstance(w, qt.QMessageBox):
                        yes_button = w.button(qt.QMessageBox.Yes)
                        yes_button.click()

            qt.QTimer.singleShot(0, auto_click_yes)

            # press the calc report button
            slicer.modules.RANOWidget.onCreateReportButton()
        else:
            print("Skipping the report creation")

        self.delayDisplay('Test passed')


