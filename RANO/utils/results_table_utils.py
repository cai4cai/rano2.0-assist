from collections import defaultdict

import numpy as np

import slicer
import vtk


class ResultsTableMixin:
    def __init__(self, parameterNode, ui, lineNodePairs):
        self._parameterNode = parameterNode
        self.ui = ui
        self.lineNodePairs = lineNodePairs

    @staticmethod
    def calculate_results_table(lineNodePairs):
        instance_segmentations_matched = slicer.modules.RANOWidget.instance_segmentations_matched

        # calculate the perpendicular product for each line pair
        default_row = {"⊥ Prod t1": np.nan, "⊥ Prod t2": np.nan, "δ (⊥ Prod) [%]": np.nan, "Vol t1": np.nan, "Vol t2": np.nan, "δ (Vol) [%]": np.nan}
        lesion_info = defaultdict(lambda: default_row.copy())

        for pair in lineNodePairs:
            l1, l2 = pair
            prod = l1.GetLineLengthWorld() * l2.GetLineLengthWorld()
            les_idx = pair.lesion_idx
            tp = pair.timepoint
            tp_idx = int(tp.replace("timepoint", ""))-1
            if tp == 'timepoint1':
                lesion_info[les_idx].update({f"⊥ Prod t1": prod})
                if instance_segmentations_matched:
                    lesion_info[les_idx].update({f"Vol t1": np.sum(instance_segmentations_matched[tp_idx] == les_idx)})
            elif tp == 'timepoint2':
                lesion_info[les_idx].update({f"⊥ Prod t2": prod})
                if instance_segmentations_matched:
                    lesion_info[les_idx].update({f"Vol t2": np.sum(instance_segmentations_matched[tp_idx] == les_idx)})
            else:
                raise ValueError(f"timepoint must be 'timepoint1' or 'timepoint2' but is {tp}")

        # add another key for the relative change in perpendicular product and relative change in volume
        for les_idx, les_dict in lesion_info.items():
            if not np.isnan(les_dict["⊥ Prod t1"]) and not np.isnan(les_dict["⊥ Prod t2"]):
                les_dict["δ (⊥ Prod) [%]"] = (les_dict["⊥ Prod t2"] - les_dict["⊥ Prod t1"]) / les_dict["⊥ Prod t1"] * 100
            if not np.isnan(les_dict["Vol t1"]) and not np.isnan(les_dict["Vol t2"]):
                les_dict["δ (Vol) [%]"] = (les_dict["Vol t2"] - les_dict["Vol t1"]) / les_dict["Vol t1"] * 100


        # convert to table_dict
        table_dict = {"Lesion Index": [str(key) for key in lesion_info.keys()]}
        for key in default_row.keys():
            table_dict[key] = [f"{les_dict[key]:.1f}" if not isinstance(les_dict[key], str) else les_dict[key] for les_dict in lesion_info.values()]

        # show the table in the table view
        ResultsTableMixin.present_table(table_dict, delete_existing=True)


    @staticmethod
    def present_table(table_dict, delete_existing=False):
        # show table based on table_dict
        # Create a table from result arrays
        # add new node if it doesn't exist
        tableName = "Results"
        if not tableName in [n.GetName() for n in slicer.mrmlScene.GetNodesByClass("vtkMRMLTableNode")]:
            resultTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", tableName)
        else:
            if delete_existing:
                slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetFirstNodeByName(tableName))
                slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", tableName)
            resultTableNode = slicer.mrmlScene.GetFirstNodeByName(tableName)
        # add columns to the table
        for key, values in table_dict.items():
            if len(values) == 0:
                continue

            if isinstance(values[0], str):
                col = vtk.vtkStringArray()
            elif isinstance(values[0], int) or isinstance(values[0], float):
                col = vtk.vtkDoubleArray()
            else:
                raise ValueError("Table values must be of type str, int or float")
            col.SetName(key)
            # add values to the columns
            for value in values:
                col.InsertNextValue(value)

            # add column to the table
            resultTableNode.AddColumn(col)

        # Show table in view layout
        slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveTableID(resultTableNode.GetID())
        slicer.app.applicationLogic().PropagateTableSelection()
