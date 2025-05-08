"""
RANO Module

This module is part of a 3D Slicer extension and  provides tools for Response Assessment in Neuro-Oncology (RANO)
based on the RANO 2.0 guidelines. It includes functionality for segmentation, 2D measurements,
response classification, and report generation.
"""

import os
import sys
from importlib import reload

import slicer, vtk
from slicer.ScriptedLoadableModule import *
from slicer.util import *

from utils.config import debug

# reload the submodules to ensure that the latest version of the code is used upon pressing the "Reload" button in the
# module GUI (development mode)
for mod in ['utils.enums',  # must be first to ensure that the subsequent modules are reloaded with the new enums
            'utils.config',
            'utils.rano_utils',
            'utils.RANO_Logic',
            'utils.ui_helper_utils',
            'utils.segmentation_utils',
            'utils.measurements2D_utils',
            'utils.response_classification_utils',
            'utils.report_creation_utils',
            'utils.results_table_utils',
            'utils.test_rano',
            ]:
    if debug: print(f"Reloading {mod}")
    if mod in sys.modules:
        reload(sys.modules[mod])

from utils.ui_helper_utils import UIHelperMixin
from utils.RANOLogic import RANOLogic
from utils.segmentation_utils import SegmentationMixin
from utils.measurements2D_utils import Measurements2DMixin, LineNodePairList
from utils.response_classification_utils import ResponseClassificationMixin
from utils.report_creation_utils import ReportCreationMixin
from utils.results_table_utils import ResultsTableMixin

from utils.test_rano import RANOTest  # tests run in developer mode


class RANO(ScriptedLoadableModule):
    """
    Required class for 3D Slicer module.
    Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "RANO"
        self.parent.categories = ["Tools"]
        self.parent.dependencies = []
        self.parent.contributors = ["Aaron Kujawa (King's College London)"]
        self.parent.helpText = ""
        self.parent.acknowledgementText = """
This file is based on a file developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. which was partially funded by NIH grant 3P41RR013218-12S1.
"""

#
# RANOWidget
#
class RANOWidget(SegmentationMixin, UIHelperMixin, Measurements2DMixin, ResponseClassificationMixin,
                 ReportCreationMixin, ResultsTableMixin,
                 ScriptedLoadableModuleWidget, VTKObservationMixin):  # these two classes have to be last because of the super() calls in their __init__ methods (MRO), otherwise the __init__ parameters are incompatible
    """
    Required class for 3D Slicer module.
    UI elements can be accessed as follows from the Slicer python console:

      `slicer.modules.RANOWidget.ui`

    For example, to access the text of the line edit widget:

      `slicer.modules.RANOWidget.ui.lineEdit.text`
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation

        self.ui = None
        """The UI elements of the module. This is a dictionary containing all the widgets in the module."""

        self.logic = None
        """The logic class of the module. This class implements all computations that should be possible to run
        in batch mode, without a graphical user interface."""

        self._parameterNode = None
        """The parameter node of the module. This node stores all user choices in parameter values, node selections, etc.
        so that when the scene is saved and reloaded, these settings are restored."""

        self._updatingGUIFromParameterNode = False
        """ Flag to indicate if the GUI is being updated from the parameter node. This is used to prevent infinite loops
        when the parameter node is changed by a script or any other module. """

        self.lineNodePairs = LineNodePairList()
        """List of line node pairs used for 2D measurements."""

        SegmentationMixin.__init__(self, self._parameterNode, self.ui)
        UIHelperMixin.__init__(self, self._parameterNode, self.ui)
        Measurements2DMixin.__init__(self, self._parameterNode, self.ui, self.lineNodePairs)
        ResponseClassificationMixin.__init__(self, self._parameterNode, self.ui, self.lineNodePairs)
        ReportCreationMixin.__init__(self, self._parameterNode, self.ui, self.lineNodePairs)
        ResultsTableMixin.__init__(self, self._parameterNode, self.ui, self.lineNodePairs)


    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/RANO.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RANOLogic()

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # set up the 3D slicer layout for the module
        self.setup_layout()

        # set up the test cases box
        self.setup_test_cases()

        # set up the directory load box
        self.setup_add_data_box()

        # set up the box for the input files
        self.setup_input_box()

        # set up the box for the auto segmentation
        self.setup_autosegmentation_box()

        # set up automatic 2D measurements
        self.setup_auto_2D_measurements()

        # set up manual 2D measurements
        self.setup_manual_2D_measurements()

        # set up response status box (lesion based)
        self.setup_lesion_based_response_status_box()

        # set up overall response status box
        self.setup_overall_response_status_box()

        # create report button
        self.ui.createReportPushButton.connect('clicked(bool)', self.onCreateReportButton)

        # external python path line edit
        self.ui.lineEdit_python_path.connect("textChanged(const QString &)", self.updateParameterNodeFromGUI)

        self.updateParameterNodeFromGUI()

        # center views on first volume
        self.onShowChannelButton(True, timepoint='timepoint1', inputSelector=self.ui.inputSelector_channel1_t1)
        self.onShowChannelButton(True, timepoint='timepoint2', inputSelector=self.ui.inputSelector_channel1_t2)

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.

        Args:
            caller: The object that triggered the event.
            event: The event that occurred.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.

        Args:
            caller: The object that triggered the event.
            event: The event that occurred.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.

        Args:
            inputParameterNode: The parameter node to set.
        """

        if inputParameterNode:
            if not inputParameterNode.GetParameter("DefaultParamsSet"):  # set default parameters only if they are not set yet
                self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()


    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        From slicer python interface, you can access the variables like this:
        slicer.modules.RANOWidget.ui.radius_spinbox

        Args:
            caller: The object that triggered the event.
            event: The event that occurred.
        """
        if debug: print("updateGUIFromParameterNode")

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # timepoint 1
        # get the model info
        model_key = self.ui.modelComboBox.currentText
        task_dir = self.get_task_dir(model_key, self._parameterNode)
        # load the modality info
        modalities_path = os.path.abspath(os.path.join(task_dir, "config", "modalities.json"))

        self.update_ui_input_channel_selectors(modalities_path, timepoint=1)
        # timepoint 2
        model_key_t2 = self.ui.modelComboBox_t2.currentText
        task_dir_t2 = self.get_task_dir(model_key_t2, self._parameterNode)
        # load the modality info
        modalities_path_t2 = os.path.abspath(os.path.join(task_dir_t2, "config", "modalities.json"))

        self.update_ui_input_channel_selectors(modalities_path_t2, timepoint=2)

        # Update node selectors and sliders
        # timepoint 1
        self.ui.inputSelector_channel1_t1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel1_t1"))
        self.ui.inputSelector_channel2_t1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel2_t1"))
        self.ui.inputSelector_channel3_t1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel3_t1"))
        self.ui.inputSelector_channel4_t1.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel4_t1"))

        self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("outputSegmentation"))
        self.ui.affineregCheckBox.checked = (self._parameterNode.GetParameter("AffineReg") == "true")
        self.ui.inputisbetCheckBox.checked = (self._parameterNode.GetParameter("InputIsBET") == "true")

        # timepoint 2
        self.ui.inputSelector_channel1_t2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel1_t2"))
        self.ui.inputSelector_channel2_t2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel2_t2"))
        self.ui.inputSelector_channel3_t2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel3_t2"))
        self.ui.inputSelector_channel4_t2.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume_channel4_t2"))

        self.ui.outputSelector_t2.setCurrentNode(self._parameterNode.GetNodeReference("outputSegmentation_t2"))
        self.ui.affineregCheckBox_t2.checked = (self._parameterNode.GetParameter("AffineReg_t2") == "true")
        self.ui.inputisbetCheckBox_t2.checked = (self._parameterNode.GetParameter("InputIsBET_t2") == "true")

        # update the 2D measurements widgets
        # update the segment selector widget current node with the segmentation node stored in the parameter node
        self.ui.SegmentSelectorWidget.setCurrentNode(self._parameterNode.GetNodeReference("meas2D_segmentation"))

        # update the same widget with the segmentID stored in the parameter node
        self.ui.SegmentSelectorWidget.setCurrentSegmentID(self._parameterNode.GetParameter("segmentID"))

        # toggle visibility of the opening radius label and spinbox depending on the selected method
        if self.ui.method2DmeasComboBox.currentText in ["RANO_open2D", "RANO_open3D"]:
            self.ui.radius_spinbox.show()
            self.ui.radius_label.show()
        else:
            self.ui.radius_spinbox.hide()
            self.ui.radius_label.hide()

        # orientation checkboxes
        self.ui.checkBox_axial.checked = (self._parameterNode.GetParameter("axial") == "true")
        self.ui.checkBox_coronal.checked = (self._parameterNode.GetParameter("coronal") == "true")
        self.ui.checkBox_sagittal.checked = (self._parameterNode.GetParameter("sagittal") == "true")
        self.ui.checkBox_orient_cons_tp.checked = (self._parameterNode.GetParameter("orient_cons_tp") == "true")
        self.ui.checkBox_same_slc_tp.checked = (self._parameterNode.GetParameter("same_slc_tp") == "true")

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume_channel1_t1") and self._parameterNode.GetNodeReference(
                "outputSegmentation"):
            self.ui.applyButton.toolTip = "Compute output volume"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input and output volume nodes"
            self.ui.applyButton.enabled = False

        if self._parameterNode.GetNodeReference("meas2D_segmentation") and self._parameterNode.GetParameter(
                "segmentID"):
            self.ui.calc2DButton.toolTip = "Compute 2D lines"
            self.ui.calc2DButton.enabled = True
        else:
            self.ui.calc2DButton.toolTip = "Select segment"
            self.ui.calc2DButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).

        Args:
            caller: The object that triggered the event.
            event: The event that occurred.
        """
        if debug: print("updateParameterNodeFromGUI")
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        # timepoint 1
        self._parameterNode.SetParameter("model_key", str(self.ui.modelComboBox.currentText))

        self._parameterNode.SetNodeReferenceID("InputVolume_channel1_t1", self.ui.inputSelector_channel1_t1.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel2_t1", self.ui.inputSelector_channel2_t1.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel3_t1", self.ui.inputSelector_channel3_t1.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel4_t1", self.ui.inputSelector_channel4_t1.currentNodeID)

        self._parameterNode.SetNodeReferenceID("outputSegmentation", self.ui.outputSelector.currentNodeID)
        self._parameterNode.SetParameter("AffineReg", "true" if self.ui.affineregCheckBox.checked else "false")
        self._parameterNode.SetParameter("InputIsBET", "true" if self.ui.inputisbetCheckBox.checked else "false")

        # timepoint 2
        self._parameterNode.SetParameter("model_key_t2", str(self.ui.modelComboBox_t2.currentText))

        self._parameterNode.SetNodeReferenceID("InputVolume_channel1_t2", self.ui.inputSelector_channel1_t2.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel2_t2", self.ui.inputSelector_channel2_t2.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel3_t2", self.ui.inputSelector_channel3_t2.currentNodeID)
        self._parameterNode.SetNodeReferenceID("InputVolume_channel4_t2", self.ui.inputSelector_channel4_t2.currentNodeID)

        self._parameterNode.SetNodeReferenceID("outputSegmentation_t2", self.ui.outputSelector_t2.currentNodeID)
        self._parameterNode.SetParameter("AffineReg_t2", "true" if self.ui.affineregCheckBox_t2.checked else "false")
        self._parameterNode.SetParameter("InputIsBET_t2", "true" if self.ui.inputisbetCheckBox_t2.checked else "false")


        # 2D measurements
        self._parameterNode.SetNodeReferenceID("meas2D_segmentation", self.ui.SegmentSelectorWidget.currentNodeID())
        self._parameterNode.SetParameter("segmentID",
                                         self.ui.SegmentSelectorWidget.currentSegmentID() if self.ui.SegmentSelectorWidget.currentNodeID() else "")

        self._parameterNode.SetParameter("method2Dmeas", self.ui.method2DmeasComboBox.currentText)

        # orientation checkboxes
        self._parameterNode.SetParameter("axial", "true" if self.ui.checkBox_axial.checked else "false")
        self._parameterNode.SetParameter("coronal", "true" if self.ui.checkBox_coronal.checked else "false")
        self._parameterNode.SetParameter("sagittal", "true" if self.ui.checkBox_sagittal.checked else "false")
        self._parameterNode.SetParameter("orient_cons_tp", "true" if self.ui.checkBox_orient_cons_tp.checked else "false")
        self._parameterNode.SetParameter("same_slc_tp", "true" if self.ui.checkBox_same_slc_tp.checked else "false")

        self._parameterNode.EndModify(wasModified)
