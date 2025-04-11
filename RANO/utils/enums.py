from enum import Enum


class Response(Enum):
    CR = 0
    PR = 1
    SD = 2
    PD = 3


# response description
response_description = {
    Response.CR: "Complete Response",
    Response.PR: "Partial Response",
    Response.SD: "Stable Disease",
    Response.PD: "Progressive Disease"
}

response_description_detailed = {
    Response.CR: "Complete Response (CR):   100% Decrease",
    Response.PR: "Partial Response (PR):    >= 50% Decrease",
    Response.SD: "Stable Disease (SD):      < 50% Decrease to < 25% Increase",
    Response.PD: "Progressive Disease (PD): >= 25% Increase"
}


class TumorComponentsForEval(Enum):
    CE = 0
    NonCE = 1
    Mixed = 2


tumorComponentsForEval_description = {
    TumorComponentsForEval.CE: "CE",
    TumorComponentsForEval.NonCE: "Non-CE",
    TumorComponentsForEval.Mixed: "Mixed",
}


class RefScanRole(Enum):
    CR = 0
    PR = 1
    SD = 2
    PD = 3
    PPD = 4  # preliminary progressive disease
    Baseline = 5


refScanRole_description = {
    RefScanRole.CR: "Complete Response (CR)",
    RefScanRole.PR: "Partial Response (PR)",
    RefScanRole.SD: "Stable Disease (SD)",
    RefScanRole.PD: "Progressive Disease (PD)",
    RefScanRole.PPD: "Preliminary Progressive Disease (PPD)",
    RefScanRole.Baseline: "Baseline"
}


class CurrScanRole(Enum):
    CR = 0
    PR = 1
    SD = 2
    PD = 3


currScanRole_description = {
    CurrScanRole.CR: "Complete Response (CR)",
    CurrScanRole.PR: "Partial Response (PR)",
    CurrScanRole.SD: "Stable Disease (SD)",
    CurrScanRole.PD: "Progressive Disease (PD)"
}


class NonTargetOrNonMeasLes(Enum):
    NoneOrStableOrCR = 0
    Worse = 1


nonTargetOrNonMeasLes_description = {
    NonTargetOrNonMeasLes.NoneOrStableOrCR: "None or Stable or CR",
    NonTargetOrNonMeasLes.Worse: "Worse"
}


class ClinicalStatus(Enum):
    StableOrImproved = 0
    Worse = 1


clinicalStatus_description = {
    ClinicalStatus.StableOrImproved: "Stable or Improved",
    ClinicalStatus.Worse: "Worse"
}


class SteroidDose(Enum):
    No = 0
    Yes = 1


steroidDose_description = {
    SteroidDose.No: "No",
    SteroidDose.Yes: "Yes"
}


class OverallResponse(Enum):
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


overallResponse_description = {
    OverallResponse.CR: "Complete Response (CR)",
    OverallResponse.PR: "Partial Response (PR)",
    OverallResponse.SD: "Stable Disease (SD)",
    OverallResponse.PD: "Progressive Disease (PD)",
    OverallResponse.PCR: "Preliminary Complete Response (PCR)",
    OverallResponse.PPR: "Preliminary Partial Response (PPR)",
    OverallResponse.PSD: "Preliminary Stable Disease (PSD)",
    OverallResponse.PPD: "Preliminary Progressive Disease (PPD)",
    OverallResponse.CCR: "Confirmed Complete Response (CCR)",
    OverallResponse.CPR: "Confirmed Partial Response (CPR)",
    OverallResponse.CSD: "Confirmed Stable Disease (CSD)",
    OverallResponse.CPD: "Confirmed Progressive Disease (CPD)"
}
