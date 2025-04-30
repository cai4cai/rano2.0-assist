import json
import os
import slicer
from slicer.ScriptedLoadableModule import *

debug = False

from utils.config import module_path, debug, dynunet_pipeline_path


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
        tasks_dir = os.path.join(dynunet_pipeline_path, 'data', 'tasks')
        model_dirs = [os.path.normpath(os.path.join(tasks_dir, p)) for p in os.listdir(tasks_dir) if p.startswith('task')]
        modelInfo = {}
        for model_dir in model_dirs:
            with open(os.path.join(model_dir, 'config', 'modalities.json'), 'r') as jsonfile:
                modalities = json.load(jsonfile).values()
            key = ", ".join(modalities) + ": " + os.path.basename(model_dir).split('_')[0]
            value = model_dir
            modelInfo[key] = value

        parameterNode.SetParameter("ModelInfo", json.dumps(modelInfo))
        parameterNode.SetParameter("DefaultModelIndex", "0")

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
