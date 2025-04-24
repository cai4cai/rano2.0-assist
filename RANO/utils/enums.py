"""
This module contains enumerations and descriptions for various response types and clinical statuses used in the RANO module.
"""
from enum import Enum


class Response(Enum):
    """
    Enumeration for different response types based only on the target lesions' 2D measurements.
    """
    CR = 0
    PR = 1
    SD = 2
    PD = 3

    def description(self):
        return {
            self.CR: "Complete Response",
            self.PR: "Partial Response",
            self.SD: "Stable Disease",
            self.PD: "Progressive Disease"
        }[self]

    def description_detailed(self):
        return {
            self.CR: "Complete Response (CR):   100% Decrease",
            self.PR: "Partial Response (PR):    >= 50% Decrease",
            self.SD: "Stable Disease (SD):      < 50% Decrease to < 25% Increase",
            self.PD: "Progressive Disease (PD): >= 25% Increase"
        }[self]


class TumorComponentsForEval(Enum):
    """
    Enumeration for different tumor components used for evaluation.
    """
    CE = 0
    NonCE = 1
    Mixed = 2

    def description(self):
        return {
            self.CE: "CE",
            self.NonCE: "Non-CE",
            self.Mixed: "Mixed"
        }[self]


class RefScanRole(Enum):
    """
    Enumeration for different reference scan roles.
    """
    CR = 0
    PR = 1
    SD = 2
    PD = 3
    PPD = 4  # preliminary progressive disease
    Baseline = 5

    def description(self):
        return {
            self.CR: "Complete Response (CR)",
            self.PR: "Partial Response (PR)",
            self.SD: "Stable Disease (SD)",
            self.PD: "Progressive Disease (PD)",
            self.PPD: "Preliminary Progressive Disease (PPD)",
            self.Baseline: "Baseline"
        }[self]


class CurrScanRole(Enum):
    """
    Enumeration for different current scan roles.
    """
    CR = 0
    PR = 1
    SD = 2
    PD = 3

    def description(self):
        return {
            self.CR: "Complete Response (CR)",
            self.PR: "Partial Response (PR)",
            self.SD: "Stable Disease (SD)",
            self.PD: "Progressive Disease (PD)"
        }[self]


class NonTargetOrNonMeasLes(Enum):
    """
    Enumeration for assessment of non-target lesions or non-measurable lesions.
    """
    NoneOrStableOrCR = 0
    Worse = 1

    def description(self):
        return {
            self.NoneOrStableOrCR: "None or Stable or CR",
            self.Worse: "Worse"
        }[self]


class ClinicalStatus(Enum):
    """
    Enumeration for clinical status of the patient.
    """
    StableOrImproved = 0
    Worse = 1

    def description(self):
        return {
            self.StableOrImproved: "Stable or Improved",
            self.Worse: "Worse"
        }[self]


class SteroidDose(Enum):
    """
    Enumeration for steroid dose administered to the patient.
    """
    No = 0
    Yes = 1

    def description(self):
        return {
            self.No: "No",
            self.Yes: "Yes"
        }[self]


class OverallResponse(Enum):
    """
    Enumeration for overall response assessment based on all RANO criteria.
    """
    CR = 0
    PR = 1
    SD = 2
    PD = 3
    PCR = 4  # preliminary complete response
    PPR = 5  # preliminary partial response
    PSD = 6  # preliminary stable disease
    PPD = 7  # preliminary progressive disease
    CCR = 8  # confirmed complete response
    CPR = 9  # confirmed partial response
    CSD = 10  # confirmed stable disease
    CPD = 11  # confirmed progressive disease

    def description(self):
        return {
            self.CR: "Complete Response (CR)",
            self.PR: "Partial Response (PR)",
            self.SD: "Stable Disease (SD)",
            self.PD: "Progressive Disease (PD)",
            self.PCR: "Preliminary Complete Response (PCR)",
            self.PPR: "Preliminary Partial Response (PPR)",
            self.PSD: "Preliminary Stable Disease (PSD)",
            self.PPD: "Preliminary Progressive Disease (PPD)",
            self.CCR: "Confirmed Complete Response (CCR)",
            self.CPR: "Confirmed Partial Response (CPR)",
            self.CSD: "Confirmed Stable Disease (CSD)",
            self.CPD: "Confirmed Progressive Disease (CPD)"
        }[self]