import os
import time
from glob import glob

import slicer
from slicer.ScriptedLoadableModule import *
from DICOMLib import DICOMUtils

from utils.config import module_path


#
# RANOTest
#
class RANOTest(ScriptedLoadableModuleTest):
    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_RANO1()

    def test_RANO1(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''
        automatic_segmentation = True
        line_placement = True

        brats_cases = False
        kch_cases = False
        kch_dcm_cases = False
        cropped_case = True

        # to store all the paths for all test cases
        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels

        if cropped_case:
            base_path = os.path.join(module_path, "..", "input_data", "test_data_cropped")

            cropped_timepoint_pairs = [
                (("Patient_999", "TimePoint_001"), ("Patient_999", "TimePoint_002")),
            ]

            for test_case_idx in range(len(cropped_timepoint_pairs)):
                patient = cropped_timepoint_pairs[test_case_idx][0][0]
                timepoint_t1 = cropped_timepoint_pairs[test_case_idx][0][1]
                timepoint_t2 = cropped_timepoint_pairs[test_case_idx][1][1]

                p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t1c_cropped.nii.gz")
                p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t1n_cropped.nii.gz")
                p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t2f_cropped.nii.gz")
                p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t2w_cropped.nii.gz")

                p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t1c_cropped.nii.gz")
                p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t1n_cropped.nii.gz")
                p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t2f_cropped.nii.gz")
                p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t2w_cropped.nii.gz")


                # check if all files exist
                if not all([os.path.isfile(p) for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]]):
                    print(f"Not all files exist for test case {test_case_idx}")
                    print(f"Paths:")
                    print(p_t1c_tp1)
                    continue

                cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                           [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])


        if brats_cases:
            base_path = "/home/aaron/Dropbox/KCL/Projects/RANO/RANO_pipeline/dynunet_pipeline/data/tasks/task4001_brats24allmod/input/dataset/training_data1_v2"

            brats_timepoint_pairs = [
                ("BraTS-GLI-00020-100", "BraTS-GLI-00020-101"),
                # ("BraTS-GLI-00005-100", "BraTS-GLI-00005-101"),
                # ("BraTS-GLI-00006-100", "BraTS-GLI-00006-101"),
                # ("BraTS-GLI-00008-100", "BraTS-GLI-00008-101"),
                # ("BraTS-GLI-00008-100", "BraTS-GLI-00008-102"),
                # ("BraTS-GLI-00008-100", "BraTS-GLI-00008-103"),
                # ("BraTS-GLI-00009-100", "BraTS-GLI-00009-101"),
                # ("BraTS-GLI-00020-100", "BraTS-GLI-00020-101"),
            ]

            # check all files exist
            for brats_case_t1, brats_case_t2 in brats_timepoint_pairs:
                assert(os.path.isfile(os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t1c.nii.gz")))
                assert(os.path.isfile(os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t1n.nii.gz")))

            for test_case_idx in range(len(brats_timepoint_pairs)):
                brats_case_t1 = brats_timepoint_pairs[test_case_idx][0]
                brats_case_t2 = brats_timepoint_pairs[test_case_idx][1]

                p_t1c_tp1 = os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t1c.nii.gz")
                p_t1n_tp1 = os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t1n.nii.gz")
                p_t2f_tp1 = os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t2f.nii.gz")
                p_t2w_tp1 = os.path.join(base_path, brats_case_t1, f"{brats_case_t1}-t2w.nii.gz")

                p_t1c_tp2 = os.path.join(base_path, brats_case_t2, f"{brats_case_t2}-t1c.nii.gz")
                p_t1n_tp2 = os.path.join(base_path, brats_case_t2, f"{brats_case_t2}-t1n.nii.gz")
                p_t2f_tp2 = os.path.join(base_path, brats_case_t2, f"{brats_case_t2}-t2f.nii.gz")
                p_t2w_tp2 = os.path.join(base_path, brats_case_t2, f"{brats_case_t2}-t2w.nii.gz")

                cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                           [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])

        if kch_cases: # e.g. /home/aaron/KCL_data/RANO/data/Patient_001/TimePoint_001/Patient_001_TimePoint_001_brain_t1c.nii.gz
                           # /home/aaron/KCL_data/RANO/data/Patient_001_TimePoint_001/Patient_001_TimePoint_001_brain_t1c.nii.gz
            base_path = "/home/aaron/KCL_data/RANO/data"


            # kch_timepoint_pairs = [
            #     (("Patient_001", "TimePoint_001"), ("Patient_001", "TimePoint_002")),
            #     (("Patient_002", "TimePoint_001"), ("Patient_002", "TimePoint_002")),
            #     (("Patient_003", "TimePoint_001"), ("Patient_003", "TimePoint_002")),
            #     ]

            kch_timepoint_pairs = [((f"Patient_{id:03d}", f"TimePoint_001"), (f"Patient_{id:03d}", f"TimePoint_002"))
                                   for id in range(1, 40)]

            for test_case_idx in range(len(kch_timepoint_pairs)):

                patient = kch_timepoint_pairs[test_case_idx][0][0]
                timepoint_t1 = kch_timepoint_pairs[test_case_idx][0][1]
                timepoint_t2 = kch_timepoint_pairs[test_case_idx][1][1]

                p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t1c.nii.gz")
                p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t1n.nii.gz")
                p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t2f.nii.gz")
                p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, f"{patient}_{timepoint_t1}_brain_t2w.nii.gz")

                p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t1c.nii.gz")
                p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t1n.nii.gz")
                p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t2f.nii.gz")
                p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, f"{patient}_{timepoint_t2}_brain_t2w.nii.gz")

                if not "006" in p_t1c_tp1:
                    continue

                # check if all files exist
                if not all([os.path.isfile(p) for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]]):
                    print(f"Not all files exist for test case {test_case_idx}")
                    continue

                cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                           [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])

        if kch_dcm_cases:
            # e.g. /home/aaron/KCL_data/RANO/Input_Data/Patient_001/TimePoint_001/t1c/image_001.dcm

            base_path = "/home/aaron/KCL_data/RANO/Input_Data"

            kch_dcm_timepoint_pairs = [((f"Patient_{id:03d}", f"TimePoint_001"), (f"Patient_{id:03d}", f"TimePoint_002"))
                                      for id in range(1, 40)]

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

            # load volumes into slicer
            if ".nii.gz" in paths_t1[0]:  # load nifti
                inputVolumes = [slicer.util.loadVolume(p) for p in paths_t1]
                inputVolumes_t2 = [slicer.util.loadVolume(p) for p in paths_t2]
            else:  # assume current paths are dicom folders
                # load dicoms into slicer using DICOMutils

                def dcm_dir_to_node(dcm_dir):
                    with DICOMUtils.TemporaryDICOMDatabase() as db:
                        DICOMUtils.importDicom(dcm_dir, db)
                        patientUIDs = db.patients()
                        loadedNodeIDs = []
                        for patientUID in patientUIDs:
                            loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

                        assert len(loadedNodeIDs) == 1, (f"Expected 1 loaded node from importing DICOM files in {dcm_dir}"
                                                         f" but got {len(loadedNodeIDs)}: {loadedNodeIDs}")

                        loadedNode = slicer.mrmlScene.GetNodeByID(loadedNodeIDs[0])
                    return loadedNode

                inputVolumes = [dcm_dir_to_node(p) for p in paths_t1]
                inputVolumes_t2 = [dcm_dir_to_node(p) for p in paths_t2]

            self.delayDisplay('Loaded test data set')

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

            # set the model key
            slicer.modules.RANOWidget.ui.modelComboBox.setCurrentIndex(1)
            slicer.modules.RANOWidget.ui.modelComboBox_t2.setCurrentIndex(1)


            # set the checkboxes
            if "999" in paths_t1[0]:  # load nifti test case
                do_affinereg = True
                input_is_bet = True
            elif ".nii.gz" in paths_t1[0]:  # load nifti
                do_affinereg = True
                input_is_bet = True
            elif len(glob(os.path.join(paths_t1[0], "*.dcm"))) > 0:  # load dicom
                do_affinereg = True
                input_is_bet = False
            else:
                raise ValueError("Unknown test case")

            slicer.modules.RANOWidget.ui.affineregCheckBox.checked = do_affinereg
            slicer.modules.RANOWidget.ui.inputisbetCheckBox.checked = input_is_bet

            slicer.modules.RANOWidget.ui.affineregCheckBox_t2.checked = do_affinereg
            slicer.modules.RANOWidget.ui.inputisbetCheckBox_t2.checked = input_is_bet


            if automatic_segmentation:
                # press the run segmentation button
                print("Starting the automatic segmentation")
                cliNode, cliNode_t2 = slicer.modules.RANOWidget.onCalcSegmentationsButton()

            if line_placement:
                # block execution here until all segments have been loaded into the segmentation nodes
                # check if segmentations have been loaded
                parameterNode = slicer.modules.RANOWidget._parameterNode
                while (parameterNode.GetParameter("segmentation_loaded_timepoint1") != "true"
                       or parameterNode.GetParameter("segmentation_loaded_timepoint2") != "true"):
                    slicer.app.processEvents()  # Keep the GUI responsive and run the events that create the segmentation after the CLIs have finished
                    time.sleep(0.1)
                    # print("waiting for the segmentation to be loaded")

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

                # press the calc report button
                slicer.modules.RANOWidget.onCreateReportButton()

            self.delayDisplay('Test passed')