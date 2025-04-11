import json
import os
import slicer
from slicer.ScriptedLoadableModule import *

debug = False

from utils.config import module_path, debug


class RANOLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def setDefaultParameters(parameterNode):
        """
        Initialize parameter node with default settings.
        """

        parameterNode.SetParameter("DefaultParamsSet", "true")

        # load the model information and store it in the parameter node
        model_info_path = os.path.join(module_path, "Resources", "model_info.json")
        parameterNode.SetParameter("model_info_path", model_info_path)
        if debug: print(f"Loading model info from ", model_info_path)
        with open(model_info_path) as jsonfile:
            modelInfo = json.load(jsonfile)
            parameterNode.SetParameter("ModelInfo", json.dumps(modelInfo))
            parameterNode.SetParameter("DefaultModelIndex", "1")

        parameterNode.SetParameter("AffineReg", "true")
        parameterNode.SetParameter("InputIsBET", "false")

        parameterNode.SetParameter("AffineReg_t2", "true")
        parameterNode.SetParameter("InputIsBET_t2", "false")

        parameterNode.SetParameter("axial", "true")
        parameterNode.SetParameter("coronal", "true")
        parameterNode.SetParameter("sagittal", "true")
        parameterNode.SetParameter("orient_cons_tp", "true")
        parameterNode.SetParameter("same_slc_tp", "true")

        # Create a new Segmentation node if none is selected
        if not parameterNode.GetNodeReference("outputSegmentation"):
            newSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            parameterNode.SetNodeReferenceID("outputSegmentation", newSegmentationNode.GetID())

        if not parameterNode.GetNodeReference("outputSegmentation_t2"):
            newSegmentationNode_t2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            parameterNode.SetNodeReferenceID("outputSegmentation_t2", newSegmentationNode_t2.GetID())
