import os
import shutil
import sys
import tempfile
import time
from glob import glob
import json

import numpy as np
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

        slicer.mrmlScene.Clear()

        # check which tests were specified in the command line
        tests_to_run = []
        for arg in sys.argv:
            if arg.startswith("test_"):
                tests_to_run.append(arg)

        if len(tests_to_run) > 0:  # if tests were specified in the command line
            for test in tests_to_run:
                if hasattr(self, test):
                    print(f"Running test {test}")
                    # run the test
                    try:
                        getattr(self, test)()
                    except Exception as e:
                        print(f"Test {test} failed with error: {e}")
                        sys.exit(1)
                else:
                    print(f"Test {test} not found")
        else:  # if no tests were specified run all tests
            self.test_export_preserves_matched_labels()
            self.test_export_preserves_matched_labels_RANO_open2D()
            self.test_export_labels_consistent_for_repeated_runs()
            self.test_RANO_dicom_KCL()
            #self.test_RANO_dicom_CPTAC()
            self.test_RANO_nifti_MU()

        if any(['.py' in arg for arg in sys.argv]):
            print('Tests executed from command line. Quitting Slicer...')
            slicer.app.quit()

    # axial cylinders (k range, j center, i center) of the enhancing tumor label. The connected component analysis
    # labels them in this order (raster order), so a, b, c, ... get the instance labels 1, 2, 3, ... at the baseline.
    # They are large enough to persist after the morphological opening of the RANO_open2D method.
    lesion_shapes = {"a": (4, 20, 16, 16), "b": (4, 20, 16, 48), "c": (4, 20, 16, 80), "d": (4, 20, 48, 16),
                     "e": (4, 20, 48, 48), "f": (4, 20, 48, 80), "g": (4, 20, 80, 16),
                     "c_low": (2, 20, 16, 80)}  # lesion c extending to lower slices, so it is labelled first

    @classmethod
    def _lesion_mask(cls, lesions, shape=(24, 96, 96), radius=7, label=3):
        """
        Create a segmentation mask (k, j, i) containing the given lesions as axial cylinders with the given label.
        """
        mask = np.zeros(shape, dtype=np.int16)
        jj, ii = np.ogrid[:shape[1], :shape[2]]
        for lesion in lesions:
            k0, k1, j, i = cls.lesion_shapes[lesion]
            mask[k0:k1, (jj - j) ** 2 + (ii - i) ** 2 <= radius ** 2] = label
        return mask

    def _run_2D_measurements_on_masks(self, masks, method2Dmeas="RANO", seg_model_key="t1c, t1n, t2f, t2w: task4001"):
        """
        Run the automatic 2D measurements on the given segmentation masks of timepoint 1 and 2. Instead of running the
        segmentation model, the masks are loaded with the same callback that loads the model output (no registration).
        """
        widget = slicer.modules.RANOWidget
        slicer.mrmlScene.Clear()

        model_info = json.loads(widget._parameterNode.GetParameter("ModelInfo"))
        model_index = list(model_info.keys()).index(seg_model_key)
        task_dir = widget.get_task_dir(seg_model_key, widget._parameterNode)

        tmp_dir = tempfile.mkdtemp()
        try:
            segNodes = []
            for tp, (mask, sfx) in enumerate(zip(masks, ["", "_t2"]), start=1):
                timepoint = f"timepoint{tp}"
                inputVolume = slicer.util.addVolumeFromArray(np.where(mask > 0, 300, 50).astype(np.int16),
                                                             name=f"input_t{tp}")
                for channel in range(1, 5):
                    getattr(widget.ui, f"inputSelector_channel{channel}_t{tp}").setCurrentNode(inputVolume)

                segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", f"outputSegmentation_t{tp}")
                getattr(widget.ui, f"outputSelector{sfx}").setCurrentNode(segNode)
                getattr(widget.ui, f"modelComboBox{sfx}").setCurrentIndex(model_index)
                getattr(widget.ui, f"affineregCheckBox{sfx}").checked = False

                # write the mask as the model output and load it
                tmp_path_out = os.path.join(tmp_dir, timepoint)
                os.makedirs(tmp_path_out)
                maskNode = slicer.util.addVolumeFromArray(mask, nodeClassName="vtkMRMLLabelMapVolumeNode")
                slicer.util.saveNode(maskNode, os.path.join(tmp_path_out, "output.nii.gz"))
                slicer.mrmlScene.RemoveNode(maskNode)
                widget.onSegmentationCliNodeSuccess([inputVolume] * 4, segNode, task_dir, timepoint, tmp_path_out)
                segNodes.append(segNode)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        widget.ui.SegmentSelectorWidget.setCurrentNode(segNodes[0])
        widget.ui.SegmentSelectorWidget.setCurrentSegmentID("3")  # Enhancing tumor
        widget.ui.method2DmeasComboBox.setCurrentIndex(widget.ui.method2DmeasComboBox.findText(method2Dmeas))
        widget.onCalc2DButton()
        return widget

    def _assert_exported_labels(self, widget, expected_labels):
        """
        Check that the matched instance labels of each timepoint are the expected ones and that the exported
        (resampled) labelmap volumes and the line pairs keep these labels.
        """
        for tp, (matched, labelmapVolumeNode, expected) in enumerate(zip(widget.instance_segmentations_matched,
                                                                         widget.resampledVolumeNodes,
                                                                         expected_labels), start=1):
            self.assertEqual([int(lab) for lab in np.unique(matched) if lab != 0], expected,
                             f"Unexpected matched labels at timepoint {tp}")
            line_pair_labels = sorted({int(pair.lesion_idx) for pair in widget.lineNodePairs
                                       if pair.timepoint == f"timepoint{tp}"})
            self.assertEqual(line_pair_labels, expected, f"Unexpected line pair lesion labels at timepoint {tp}")
            if not expected:
                self.assertIsNone(labelmapVolumeNode)
                continue
            exported = slicer.util.arrayFromVolume(labelmapVolumeNode)
            self.assertEqual([int(lab) for lab in np.unique(exported)], [0] + expected,
                             f"Unexpected exported labels at timepoint {tp}")
            # same grid and identity transform, so the exported labelmap must equal the matched instance segmentation
            np.testing.assert_array_equal(exported, matched)
            self.assertIsNotNone(labelmapVolumeNode.GetDisplayNode().GetColorNode())

    def test_export_preserves_matched_labels(self):
        cases = [
            ("ab", "b", [[1, 2], [2]]),  # lesion 1 disappears: lesion 2 keeps label 2 (not 1)
            ("abc", ["b", "c_low"], [[1, 2, 3], [2, 3]]),  # lesion 3 is labelled first at the follow-up
            ("abcdefg", "acg", [[1, 2, 3, 4, 5, 6, 7], [1, 3, 7]]),  # nonconsecutive labels
            ("ab", "abd", [[1, 2], [1, 2, 3]]),  # new lesion
            ("ab", "", [[1, 2], []]),  # all lesions disappear: nothing is exported at the follow-up
        ]
        for baseline, follow_up, expected in cases:
            with self.subTest(baseline=baseline, follow_up=follow_up):
                masks = [self._lesion_mask(baseline), self._lesion_mask(follow_up)]
                widget = self._run_2D_measurements_on_masks(masks)
                self._assert_exported_labels(widget, expected)

    def test_export_preserves_matched_labels_RANO_open2D(self):
        # lesion 1 disappears at the follow-up: lesion 2 must keep label 2 after the opening of RANO_open2D
        widget = self._run_2D_measurements_on_masks([self._lesion_mask("ab"), self._lesion_mask("b")],
                                                    method2Dmeas="RANO_open2D")
        self._assert_exported_labels(widget, [[1, 2], [2]])

    def test_export_labels_consistent_for_repeated_runs(self):
        masks = [self._lesion_mask("abc"), self._lesion_mask("bc")]
        widget = self._run_2D_measurements_on_masks(masks)
        first = [slicer.util.arrayFromVolume(node).copy() for node in widget.resampledVolumeNodes]
        widget.onCalc2DButton()
        self._assert_exported_labels(widget, [[1, 2, 3], [2, 3]])
        for exported_first, labelmapVolumeNode in zip(first, widget.resampledVolumeNodes):
            np.testing.assert_array_equal(slicer.util.arrayFromVolume(labelmapVolumeNode), exported_first)

    def test_RANO_dicom_KCL(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''

        base_path = os.path.join(test_data_path, 'KCL')

        patients = sorted(glob(os.path.join(base_path, "Patient_*")))

        kcl_dcm_timepoint_pairs = [((patient, f"TimePoint_001"), (patient, f"TimePoint_002"))
                                  for patient in patients]

        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels
        for test_case_idx in range(len(kcl_dcm_timepoint_pairs)):
            patient = kcl_dcm_timepoint_pairs[test_case_idx][0][0]
            timepoint_t1 = kcl_dcm_timepoint_pairs[test_case_idx][0][1]
            timepoint_t2 = kcl_dcm_timepoint_pairs[test_case_idx][1][1]

            p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1c")
            p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1n")
            p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2f")
            p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2w")

            p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1c")
            p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1n")
            p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2f")
            p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2w")

            # check if all files exist
            for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]:
                if not os.path.isdir(p):
                    raise NotADirectoryError(f"The directory {p} does not exist...")

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])

        assert(len(cases_of_paths_2tp_times_4channels) > 0), f"No test cases found for KCL patients"


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
                               method2Dmeas="RANO",
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)


    def test_RANO_dicom_CPTAC(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''

        base_path = os.path.join(test_data_path, 'CPTAC-GBM')

        patients = ["C3L-00016"]


        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels
        for patient in patients:

            if patient == "C3L-00016":
                timepoint1 = "11-15-1999-NA-MR BRAIN WOW CONTRAST-47088"
                timepoint2 = "11-17-1999-NA-MR BRAIN WOW CONTRAST-15329"

                p_t1c_tp1 = os.path.join(base_path, patient, timepoint1, "13.000000-AX T1C-70939")
                p_t1n_tp1 = os.path.join(base_path, patient, timepoint1, "9.000000-AX T1-06604")
                p_t2f_tp1 = os.path.join(base_path, patient, timepoint1, "7.000000-Ax Flair irFSE H-84835")
                p_t2w_tp1 = os.path.join(base_path, patient, timepoint1, "10.000000-Prop T2 TRF-43669")

                p_t1c_tp2 = os.path.join(base_path, patient, timepoint2, "12.000000-AX T1C-87798")
                p_t1n_tp2 = os.path.join(base_path, patient, timepoint2, "5.000000-AX T1-46920")
                p_t2f_tp2 = os.path.join(base_path, patient, timepoint2, "6.000000-Ax Flair irFSE H-49772")
                p_t2w_tp2 = os.path.join(base_path, patient, timepoint2, "8.000000-Prop T2 TRF-56708")
            else:
                raise ValueError(f"No paths defined for patient {patient}")

            # check if all directories exist
            for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]:
                if not os.path.isdir(p):
                    raise NotADirectoryError(f"The directory {p} does not exist...")

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])

        assert(len(cases_of_paths_2tp_times_4channels) > 0), f"No test cases found for CPTAC-GBM patients"


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
                               method2Dmeas="RANO_open3D",
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)


    def test_RANO_nifti_MU(self):
        slicer.mrmlScene.Clear()
        self.delayDisplay("Starting the test")

        '''
        define the test cases
        '''
        # to store all the paths for all test cases
        cases_of_paths_2tp_times_4channels = []  # n cases, 2 timepoints, 4 channels

        base_path = os.path.join(test_data_path, 'MU-Glioma-Post')

        patient_dirs = sorted(glob(os.path.join(base_path, "PatientID_*")))
        patients = [os.path.basename(p) for p in patient_dirs]

        for patient, patient_dir in zip(patients, patient_dirs):
            timepoints = [os.path.basename(p) for p in sorted(glob(os.path.join(patient_dir, "Timepoint_*")))]
            timepoint_1_dir = os.path.join(patient_dir, timepoints[0])
            timepoint_2_dir = os.path.join(patient_dir, timepoints[1])

            p_t1c_tp1 = os.path.join(timepoint_1_dir, patient + f"_{timepoints[0]}_brain_t1c.nii.gz")
            p_t1n_tp1 = os.path.join(timepoint_1_dir, patient + f"_{timepoints[0]}_brain_t1n.nii.gz")
            p_t2f_tp1 = os.path.join(timepoint_1_dir, patient + f"_{timepoints[0]}_brain_t2f.nii.gz")
            p_t2w_tp1 = os.path.join(timepoint_1_dir, patient + f"_{timepoints[0]}_brain_t2w.nii.gz")

            p_t1c_tp2 = os.path.join(timepoint_2_dir, patient + f"_{timepoints[1]}_brain_t1c.nii.gz")
            p_t1n_tp2 = os.path.join(timepoint_2_dir, patient + f"_{timepoints[1]}_brain_t1n.nii.gz")
            p_t2f_tp2 = os.path.join(timepoint_2_dir, patient + f"_{timepoints[1]}_brain_t2f.nii.gz")
            p_t2w_tp2 = os.path.join(timepoint_2_dir, patient + f"_{timepoints[1]}_brain_t2w.nii.gz")

            # check if all files exist
            for p in [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1, p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"The file {p} does not exist...")

            cases_of_paths_2tp_times_4channels.append([[p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1],
                                                       [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]])

        assert(len(cases_of_paths_2tp_times_4channels) > 0), f"No test cases found for MU-Glioma-Post patients"


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
                               method2Dmeas="RANO",
                               seg_model_key="t1c, t1n, t2f, t2w: task4001",
                               automatic_segmentation=True,
                               line_placement=True,
                               report_creation=True)

    def test_pipeline(self,
                      inputVolumes,
                      inputVolumes_t2,
                      do_affinereg,
                      input_is_bet,
                      method2Dmeas,
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
            method_idx = slicer.modules.RANOWidget.ui.method2DmeasComboBox.findText(method2Dmeas)
            slicer.modules.RANOWidget.ui.method2DmeasComboBox.setCurrentIndex(method_idx)

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


