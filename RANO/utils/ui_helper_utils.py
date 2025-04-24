import functools
import json

import qt
import slicer
from glob import glob
import os

from DICOMLib import DICOMUtils
from utils import measurements2D_utils
from utils.enums import response_description_detailed, Response, tumorComponentsForEval_description, TumorComponentsForEval, \
    refScanRole_description, RefScanRole, currScanRole_description, CurrScanRole, nonTargetOrNonMeasLes_description, \
    NonTargetOrNonMeasLes, clinicalStatus_description, ClinicalStatus, steroidDose_description, SteroidDose, \
    overallResponse_description, OverallResponse
import utils.segmentation_utils as segmentation_utils
import utils.response_classification_utils as response_classification_utils
from utils.config import debug, module_path


class UIHelperMixin:
    def __init__(self, parameterNode, ui):
        self._parameterNode = parameterNode
        self.ui = ui

    def setup_test_cases(self):
        # test cases
        base_path = "/home/aaron/KCL_data/RANO/Input_Data"

        patients = glob(os.path.join(base_path, "Patient_*"))

        timepoints = {}

        for p in sorted(patients):
            timepoints_this_patient = glob(os.path.join(base_path, p, "TimePoint*"))
            if len(timepoints_this_patient) > 1:
                timepoints[p] = timepoints_this_patient


        # make sure the timepoint comboboxes show only the available timepoints
        def onUpdateTestCaseComboBox(*args, **kwargs):
            if debug: print(f"Triggered onUpdateTestCaseComboBox with args {args} and kwargs {kwargs} ")
            p = args[0]
            tps = sorted([p.split(os.sep)[-1] for p in glob(os.path.join(base_path, p, "TimePoint*"))])

            # remove previous items and add found timepoints
            self.ui.testCaseTp1ComboBox.clear()
            self.ui.testCaseTp2ComboBox.clear()
            for tp in tps:
                self.ui.testCaseTp1ComboBox.addItem(tp)
                self.ui.testCaseTp2ComboBox.addItem(tp)

            # select first and second instance respectively
            self.ui.testCaseTp1ComboBox.setCurrentIndex(0)
            self.ui.testCaseTp2ComboBox.setCurrentIndex(1)


        self.ui.testCaseComboBox.connect("currentIndexChanged(const QString &)", onUpdateTestCaseComboBox)

        for p in timepoints.keys():
            self.ui.testCaseComboBox.addItem(p.split(os.sep)[-1])


        # test cases push button
        def onTestCaseLoadButton(*args, **kwargs):
            if debug: print(f"Pushed onTestCaseLoadButton, with args {args} ,  and kwargs {kwargs}")
            # clear the scene
            slicer.mrmlScene.Clear()

            patient = self.ui.testCaseComboBox.currentText
            timepoint_t1 = self.ui.testCaseTp1ComboBox.currentText
            timepoint_t2 = self.ui.testCaseTp2ComboBox.currentText

            p_t1c_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1c")
            p_t1n_tp1 = os.path.join(base_path, patient, timepoint_t1, "t1n")
            p_t2f_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2f")
            p_t2w_tp1 = os.path.join(base_path, patient, timepoint_t1, "t2w")

            p_t1c_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1c")
            p_t1n_tp2 = os.path.join(base_path, patient, timepoint_t2, "t1n")
            p_t2f_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2f")
            p_t2w_tp2 = os.path.join(base_path, patient, timepoint_t2, "t2w")

            paths_t1 = [p_t1c_tp1, p_t1n_tp1, p_t2f_tp1, p_t2w_tp1]
            paths_t2 = [p_t1c_tp2, p_t1n_tp2, p_t2f_tp2, p_t2w_tp2]

            def dcm_dir_to_node(dcm_dir):
                with DICOMUtils.TemporaryDICOMDatabase() as db:
                    DICOMUtils.importDicom(dcm_dir, db)
                    patientUIDs = db.patients()
                    loadedNodeIDs = []
                    for patientUID in patientUIDs:
                        loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

                    loadedVolumeNodeIDs = [id for id in loadedNodeIDs if
                                           "VolumeNode" in slicer.mrmlScene.GetNodeByID(id).GetClassName()]

                    assert len(loadedVolumeNodeIDs) == 1, (
                        f"Expected 1 loaded volume node from importing DICOM files in {dcm_dir}"
                        f" but got {len(loadedVolumeNodeIDs)}: {loadedVolumeNodeIDs}")

                    loadedNode = slicer.mrmlScene.GetNodeByID(loadedVolumeNodeIDs[0])
                return loadedNode

            inputVolumes = [dcm_dir_to_node(p) for p in paths_t1]
            inputVolumes_t2 = [dcm_dir_to_node(p) for p in paths_t2]

            # set the input volumes
            slicer.modules.RANOWidget.ui.inputSelector_channel1_t1.setCurrentNode(inputVolumes[0])
            slicer.modules.RANOWidget.ui.inputSelector_channel2_t1.setCurrentNode(inputVolumes[1])
            slicer.modules.RANOWidget.ui.inputSelector_channel3_t1.setCurrentNode(inputVolumes[2])
            slicer.modules.RANOWidget.ui.inputSelector_channel4_t1.setCurrentNode(inputVolumes[3])

            slicer.modules.RANOWidget.ui.inputSelector_channel1_t2.setCurrentNode(inputVolumes_t2[0])
            slicer.modules.RANOWidget.ui.inputSelector_channel2_t2.setCurrentNode(inputVolumes_t2[1])
            slicer.modules.RANOWidget.ui.inputSelector_channel3_t2.setCurrentNode(inputVolumes_t2[2])
            slicer.modules.RANOWidget.ui.inputSelector_channel4_t2.setCurrentNode(inputVolumes_t2[3])

            # show the first channel in the views
            self.onShowChannelButton(checked=False, timepoint='timepoint1', inputSelector=self.ui.inputSelector_channel1_t1)
            self.onShowChannelButton(checked=False, timepoint='timepoint2', inputSelector=self.ui.inputSelector_channel1_t2)

            # center all views
            slicer.util.resetSliceViews()

        self.ui.testCasePushButton.connect('clicked(bool)', onTestCaseLoadButton)

    def setup_add_data_box(self):
        def onAddDataButton(*args, **kwargs):
            slicer.util.openAddDataDialog()

        def onAddDicomDataButton(*args, **kwargs):
            # switch to DICOM module
            slicer.util.selectModule("DICOM")


        self.ui.addDataPushButton.connect('clicked(bool)', onAddDataButton)
        self.ui.addDicomDataPushButton.connect('clicked(bool)', onAddDicomDataButton)


    def setup_input_box(self):
        # timepoint 1
        self.ui.inputSelector_channel1_t1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel2_t1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel3_t1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel4_t1.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # timepoint 2
        self.ui.inputSelector_channel1_t2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel2_t2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel3_t2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSelector_channel4_t2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # show channel buttons
        for pushbutton, timepoint, inputSelector in [
            (self.ui.showChannel1PushButton_t1, 'timepoint1', self.ui.inputSelector_channel1_t1),
            (self.ui.showChannel2PushButton_t1, 'timepoint1', self.ui.inputSelector_channel2_t1),
            (self.ui.showChannel3PushButton_t1, 'timepoint1', self.ui.inputSelector_channel3_t1),
            (self.ui.showChannel4PushButton_t1, 'timepoint1', self.ui.inputSelector_channel4_t1),
            (self.ui.showChannel1PushButton_t2, 'timepoint2', self.ui.inputSelector_channel1_t2),
            (self.ui.showChannel2PushButton_t2, 'timepoint2', self.ui.inputSelector_channel2_t2),
            (self.ui.showChannel3PushButton_t2, 'timepoint2', self.ui.inputSelector_channel3_t2),
            (self.ui.showChannel4PushButton_t2, 'timepoint2', self.ui.inputSelector_channel4_t2)]:
            pushbutton.connect('clicked(bool)', functools.partial(self.onShowChannelButton, timepoint=timepoint,
                                                                  inputSelector=inputSelector))
            # add icons to the show channel buttons
            pushbutton.setIcon(qt.QIcon(self.resourcePath('Icons/eye.png')))
            # make the icon size smaller
            pushbutton.setIconSize(qt.QSize(16, 16))

    def update_ui_input_channel_selectors(self, modalities_path, timepoint):
        if timepoint == 1:
            postfix = "_t1"
        elif timepoint == 2:
            postfix = "_t2"
        else:
            raise ValueError("timepoint must be 1 or 2")

        if os.path.isfile(modalities_path):
            with open(modalities_path) as jsonfile:
                modalities = json.load(jsonfile)

            # make channel input selectors visible/invisible
            for i in range(4):
                vis = True if i < len(modalities) else False
                self.ui.__getattribute__(f"inputSelector_channel{i + 1}" + postfix).setVisible(vis)
                self.ui.__getattribute__(f"channel{i + 1}_label" + postfix).setVisible(vis)
                # also make the push button visible/invisible
                self.ui.__getattribute__(f"showChannel{i + 1}PushButton" + postfix).setVisible(vis)

            # set the channel labels
            for i in range(4):
                if i < len(modalities):
                    self.ui.__getattribute__(f"channel{i + 1}_label" + postfix).setText(modalities[str(i)] + ":")
                else:
                    self.ui.__getattribute__(f"channel{i + 1}_label" + postfix).setText(f"Channel {i + 1}:")
        else:
            if debug: print(f"Could not find modalities.json at {modalities_path}")
            for i in range(4):
                self.ui.__getattribute__(f"inputSelector_channel{i + 1}" + postfix).setVisible(True)
                self.ui.__getattribute__(f"channel{i + 1}_label" + postfix).setVisible(True)
                self.ui.__getattribute__(f"channel{i + 1}_label" + postfix).setText(f"Channel {i + 1}:")

    def setup_autosegmentation_box(self):

        models_dict = json.loads(self._parameterNode.GetParameter("ModelInfo"))
        defaultModelIndex = self._parameterNode.GetParameter("DefaultModelIndex")

        # timepoint 1
        self.ui.affineregCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.inputisbetCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.ui.modelComboBox.connect("currentIndexChanged(const QString &)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # add models to the combobox
        self.ui.modelComboBox.clear()
        self.ui.modelComboBox.addItems(list(models_dict.keys()))
        self.ui.modelComboBox.setCurrentIndex(int(defaultModelIndex))

        # timepoint 2
        self.ui.affineregCheckBox_t2.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.inputisbetCheckBox_t2.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.ui.modelComboBox_t2.connect("currentIndexChanged(const QString &)", self.updateParameterNodeFromGUI)
        self.ui.outputSelector_t2.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # add models to the combobox
        self.ui.modelComboBox_t2.clear()
        self.ui.modelComboBox_t2.addItems(list(models_dict.keys()))
        self.ui.modelComboBox_t2.setCurrentIndex(int(defaultModelIndex))


        def onCalcSegmentationsButton(*args, **kwargs):
            if debug: print(f"Pushed onCalcSegmentationsButton, with args {args} ,  and kwargs {kwargs}")
            segmentation_utils.SegmentationMixin.onCalcSegmentationsButton(slicer.modules.RANOWidget)
        # calculate segmentations button
        self.ui.applyButton.connect('clicked(bool)', onCalcSegmentationsButton)


    def setup_auto_2D_measurements(self):
        self.ui.SegmentSelectorWidget.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.SegmentSelectorWidget.connect("currentSegmentChanged(QString)", self.updateParameterNodeFromGUI)
        self.ui.SegmentSelectorWidget.connect("segmentSelectionChanged(QStringList)", self.updateParameterNodeFromGUI)

        # method selection (RANO, RANOopen, etc)
        self.ui.method2DmeasComboBox.connect("currentIndexChanged(const QString &)", self.updateParameterNodeFromGUI)

        self.ui.checkBox_axial.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_coronal.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_sagittal.connect("toggled(bool)", self.updateParameterNodeFromGUI)
        self.ui.checkBox_orient_cons_tp.connect("toggled(bool)",
                                                self.updateParameterNodeFromGUI)  # orientation consistency between timepoints
        self.ui.checkBox_same_slc_tp.connect("toggled(bool)",
                                             self.updateParameterNodeFromGUI)  # same slice between timepoints


        def onCalc2DButton(*args, **kwargs):
            if debug: print(f"Pushed onCalc2DButton, with args {args} ,  and kwargs {kwargs}")
            measurements2D_utils.Measurements2DMixin.onCalc2DButton(slicer.modules.RANOWidget)
        # calculate 2D lines button
        self.ui.calc2DButton.connect('clicked(bool)', onCalc2DButton)


    def setup_manual_2D_measurements(self):

        def onToggleShowInstanceSegButton(*args, **kwargs):
            if debug: print(f"Pushed onToggleShowInstanceSegButton, with args {args} ,  and kwargs {kwargs}")
            measurements2D_utils.Measurements2DMixin.onToggleShowInstanceSegButton(slicer.modules.RANOWidget)

        self.ui.toggleShowInstanceSegPushButton.connect('clicked(bool)', onToggleShowInstanceSegButton)
        self.ui.toggleShowInstanceSegPushButton.setCheckable(True)
        self.ui.toggleShowInstanceSegPushButton.setChecked(False)

        def onAddLinePairButton(*args, **kwargs):
            if debug: print(f"Pushed onAddLinePairButton, with args {args} ,  and kwargs {kwargs}")
            timepoint = kwargs['timepoint']
            measurements2D_utils.Measurements2DMixin.onAddLinePairButton(slicer.modules.RANOWidget, timepoint=timepoint)

        # add linePair buttons
        self.ui.addLinePairTp1_pushButton.connect('clicked(bool)', functools.partial(onAddLinePairButton, timepoint='timepoint1'))
        self.ui.addLinePairTp2_pushButton.connect('clicked(bool)', functools.partial(onAddLinePairButton, timepoint='timepoint2'))

        self.ui.tableWidget.cellChanged.connect(self.updateParameterNodeFromGUI)


    def setup_layout(self):
        customLayout = """
                <layout type="vertical" split="true">
                 <item splitSize="1000">
                  <layout type="horizontal">
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Red">
                     <property name="orientation" action="default">Axial</property>
                     <property name="viewlabel" action="default">R</property>
                     <property name="viewcolor" action="default">#F34A33</property>
                     <property name="viewgroup" action="default">0</property>
                    </view>
                   </item>
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Yellow">
                     <property name="orientation" action="default">Sagittal</property>
                     <property name="viewlabel" action="default">Y</property>
                     <property name="viewcolor" action="default">#EDD54C</property>
                     <property name="viewgroup" action="default">0</property>
                    </view>
                   </item>
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Green">
                     <property name="orientation" action="default">Coronal</property>
                     <property name="viewlabel" action="default">G</property>
                     <property name="viewcolor" action="default">#6EB04B</property>
                     <property name="viewgroup" action="default">0</property>
                    </view>
                   </item>
                   <item>
                   <view class="vtkMRMLViewNode" singletontag="view3d_1">
                    <property name="viewlabel" action="default">1</property>
                    <property name="viewgroup" action="default">0</property>
                   </view>
                   </item>
                  </layout>
                 </item>
                 <item splitSize="1000">
                  <layout type="horizontal">
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Red_2">
                     <property name="orientation" action="default">Axial</property>
                     <property name="viewlabel" action="default">R</property>
                     <property name="viewcolor" action="default">#F34A33</property>
                     <property name="viewgroup" action="default">1</property>
                    </view>
                   </item>
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Yellow_2">
                     <property name="orientation" action="default">Sagittal</property>
                     <property name="viewlabel" action="default">Y</property>
                     <property name="viewcolor" action="default">#EDD54C</property>
                     <property name="viewgroup" action="default">1</property>
                    </view>
                   </item>
                   <item>
                    <view class="vtkMRMLSliceNode" singletontag="Green_2">
                     <property name="orientation" action="default">Coronal</property>
                     <property name="viewlabel" action="default">G</property>
                     <property name="viewcolor" action="default">#6EB04B</property>
                     <property name="viewgroup" action="default">1</property>
                    </view>
                   </item>
                   <item>
                   <view class="vtkMRMLViewNode" singletontag="view3d_2">
                    <property name="viewlabel" action="default">1</property>
                    <property name="viewgroup" action="default">1</property>
                   </view>
                   </item>
                  </layout>
                 </item>
                 <item splitSize="50">
                  <view class="vtkMRMLTableViewNode" singletontag="TableView1">
                   <property name="viewlabel" action="default">T</property>
                  </view>
                 </item>
                </layout>
                """

        # Built-in layout IDs are all below 100, so you can choose any large random number
        # for your custom layout ID.
        customLayoutId = 501

        layoutManager = slicer.app.layoutManager()
        layoutManager.layoutLogic().GetLayoutNode().AddLayoutDescription(customLayoutId,
                                                                         customLayout)  # Add a new layout, does nothing if already exists
        layoutManager.layoutLogic().GetLayoutNode().SetLayoutDescription(customLayoutId,
                                                                         customLayout)  # Update existing layout, does nothing if it does not exist

        # Switch to the new custom layout
        layoutManager.setLayout(customLayoutId)


    def setup_lesion_based_response_status_box(self):

        def onUpdateOverallResponseParams(*args, **kwargs):
            if debug: print(f"Triggered onUpdateOverallResponseParams with args {args} and kwargs {kwargs} ")
            response_classification_utils.ResponseClassificationMixin.update_overall_response_params(self.ui)

        # clear any previous items
        self.ui.responseStatusComboBox.clear()
        response_enum_values = [response_description_detailed[response] for response in Response]
        self.ui.responseStatusComboBox.addItems(response_enum_values)
        # connect the response status combo box
        self.ui.responseStatusComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseParams)


    def setup_overall_response_status_box(self):

        def onUpdateOverallResponseStatus(*args, **kwargs):
            if debug: print(f"Triggered onUpdateOverallResponseStatus with args {args} and kwargs {kwargs} ")
            response_classification_utils.ResponseClassificationMixin.update_overall_response_status(self.ui)

        # connect the check boxes
        self.ui.confirmationRequiredForPdCheckBox.connect("toggled(bool)", onUpdateOverallResponseStatus)
        self.ui.newMeasLesCheckBox.connect("toggled(bool)", onUpdateOverallResponseStatus)
        self.ui.increasedSteroidUseCheckBox.connect("toggled(bool)", onUpdateOverallResponseStatus)
        self.ui.secondLineMedicationCheckBox.connect("toggled(bool)", onUpdateOverallResponseStatus)

        # fill and connect the combo boxes

        ###
        self.ui.ceOrNonCeComboBox.clear()
        tumor_components_for_eval_enum_values = [tumorComponentsForEval_description[tumor_component] for tumor_component in
                                                 TumorComponentsForEval]
        self.ui.ceOrNonCeComboBox.addItems(tumor_components_for_eval_enum_values)
        self.ui.ceOrNonCeComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseStatus)


        ###
        self.ui.referenceScanComboBox.clear()
        ref_scan_role_enum_values = [refScanRole_description[ref_scan_role] for ref_scan_role in RefScanRole]
        self.ui.referenceScanComboBox.addItems(ref_scan_role_enum_values)
        self.ui.referenceScanComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseStatus)


        ###
        self.ui.currScanComboBox.clear()
        curr_scan_role_enum_values = [currScanRole_description[curr_scan_role] for curr_scan_role in CurrScanRole]
        self.ui.currScanComboBox.addItems(curr_scan_role_enum_values)
        self.ui.currScanComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseStatus)


        ###
        self.ui.nonTargetNonMeasComboBox.clear()
        non_target_or_non_meas_les_enum_values = [nonTargetOrNonMeasLes_description[non_target_or_non_meas_les] for
                                                  non_target_or_non_meas_les in NonTargetOrNonMeasLes]
        self.ui.nonTargetNonMeasComboBox.addItems(non_target_or_non_meas_les_enum_values)
        self.ui.nonTargetNonMeasComboBox.connect("currentIndexChanged(const QString &)",
                                                 onUpdateOverallResponseStatus)


        ###
        self.ui.clinicalStatusComboBox.clear()
        clinical_status_enum_values = [clinicalStatus_description[clinical_status] for clinical_status in ClinicalStatus]
        self.ui.clinicalStatusComboBox.addItems(clinical_status_enum_values)
        self.ui.clinicalStatusComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseStatus)


        ###
        self.ui.steroidDoseComboBox.clear()
        steroid_dose_enum_values = [steroidDose_description[steroidDose] for steroidDose in SteroidDose]
        self.ui.steroidDoseComboBox.addItems(steroid_dose_enum_values)
        self.ui.steroidDoseComboBox.connect("currentIndexChanged(const QString &)", onUpdateOverallResponseStatus)


        ###
        self.ui.overallResponseStatusComboBox.clear()
        response_assessment_overall_enum_values = [overallResponse_description[overall_response] for overall_response in
                                                   OverallResponse]
        self.ui.overallResponseStatusComboBox.addItems(response_assessment_overall_enum_values)
        self.ui.overallResponseStatusComboBox.connect("currentIndexChanged(const QString &)",
                                                      onUpdateOverallResponseStatus)


    @staticmethod
    def setBackgroundVolumes(node, timepoint, rotateSliceToLowestVolumeAxes=False):
        if timepoint == 'timepoint1':
            viewnames = ["Red", "Yellow", "Green"]
        elif timepoint == 'timepoint2':
            viewnames = ["Red_2", "Yellow_2", "Green_2"]
        else:
            raise ValueError("timepoint must be 'timepoint1' or 'timepoint2'")

        for viewname in viewnames:
            compositeNode = slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().GetSliceCompositeNode()
            compositeNode.SetBackgroundVolumeID(node.GetID())

            if rotateSliceToLowestVolumeAxes:
                # print(f"Align view {viewname} with closest IJK plane")
                slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().RotateSliceToLowestVolumeAxes()

    @staticmethod
    def setLabelVolumes(node, timepoint):
        if timepoint == 'timepoint1':
            viewnames = ["Red", "Yellow", "Green"]
        elif timepoint == 'timepoint2':
            viewnames = ["Red_2", "Yellow_2", "Green_2"]
        else:
            raise ValueError("timepoint must be 'timepoint1' or 'timepoint2'")

        for viewname in viewnames:
            compositeNode = slicer.app.layoutManager().sliceWidget(viewname).sliceLogic().GetSliceCompositeNode()
            compositeNode.SetLabelVolumeID(node.GetID())

    def onNodeSelected(self, node, timepoint):
        if node:
            self.setBackgroundVolumes(node, timepoint)

    def onShowChannelButton(self, checked, timepoint, inputSelector):
        if debug: print(f"Triggered onShowChannelButton with checked {checked}, timepoint {timepoint}, inputSelector {inputSelector}")
        # show that node
        node = inputSelector.currentNode()
        if node:
            self.setBackgroundVolumes(node, timepoint, rotateSliceToLowestVolumeAxes=True)
            # rotate to volume plane

        else:
            if debug: print("No node selected")

