import numpy as np

from utils.enums import Response, OverallResponse, RefScanRole, CurrScanRole, NonTargetOrNonMeasLes, ClinicalStatus, \
    SteroidDose, TumorComponentsForEval

from utils.config import debug


class ResponseClassificationMixin:
    def __init__(self, parameterNode, ui, lineNodePairs):
        self._parameterNode = parameterNode
        self.ui = ui
        self.lineNodePairs = lineNodePairs

    @staticmethod
    def response_assessment_from_rel_area_or_vol(rel_area=None, rel_vol=None):
        """
        Given the relative size of the sum of bidimensional products or sum of volumes of the second timepoint,
        this function returns the response assessment according to the RANO 2.0 criteria.
        :param rel_area: the relative size of bidimensional product. Defined as the sum of the bidimensional products of the
        orthogonal lines of all lesions at timepoint 2 divided by the sum of the bidimensional products of the orthogonal
        lines of all lesions at timepoint 1.
        :param rel_vol: the relative size of volume. Defined as the sum of the volumes of all lesions at timepoint 2 divided
        by the sum of the volumes of all lesions at timepoint 1.
        :return: the response assessment according to the RANO 2.0 criteria
        """
        # make sure only one of the two is provided
        assert (rel_area is None) != (rel_vol is None), "Either del_area or del_vol must be provided, but not both"

        if rel_area is not None:
            if rel_area < 0:  # decrease in size
                rel_decrease = -rel_area
                if rel_decrease == 1:
                    return Response.CR
                elif rel_decrease >= 0.5:
                    return Response.PR
            elif rel_area > 0:  # increase in size
                rel_increase = rel_area
                if rel_increase >= 0.25:
                    return Response.PD

            if -0.5 < rel_area < 0.25:
                return Response.SD

            raise ValueError(f"Relative area {rel_area} does not match any of the RANO 2.0 criteria")

        if rel_vol is not None:
            if rel_vol < 1:
                rel_decrease = 1 - rel_vol
                if rel_decrease == 1:
                    return Response.CR
                elif rel_decrease >= 0.65:
                    return Response.PR
            elif rel_vol > 1:
                rel_increase = rel_vol - 1
                if rel_increase >= 0.4:
                    return Response.PD

            if 0.35 < rel_vol < 1.4:
                return Response.SD

            raise ValueError(f"Relative volume {rel_vol} does not match any of the RANO 2.0 criteria")

    @staticmethod
    def     response_assessment_overall(ref_scan=RefScanRole.Baseline,
                                    curr_scan=CurrScanRole.CR,
                                    newMeasLes=False,
                                    nonTargetOrNonMeasLes=NonTargetOrNonMeasLes.NoneOrStableOrCR,
                                    clinicalStatus=ClinicalStatus.StableOrImproved,
                                    increasedSteroids=False,
                                    steroidDose=SteroidDose.No,
                                    tumorComponentsForEval=TumorComponentsForEval.CE,
                                    confirmPD=True):

        if newMeasLes:
            return OverallResponse.PD

        if nonTargetOrNonMeasLes == NonTargetOrNonMeasLes.Worse:
            return OverallResponse.PD

        if clinicalStatus == ClinicalStatus.Worse:
            return OverallResponse.PD

        if increasedSteroids or steroidDose == SteroidDose.Yes:
            return OverallResponse.SD  # TODO: Increase in corticosteroid dose alone, in the absence of clinical
            # deterioration related to tumor, will not be used as a determinant of progression. Patients with stable
            # imaging studies whose corticosteroid dose was increased for reasons other than clinical deterioration
            # related to tumor do not qualify for stable disease or progression. They should be observed closely. If
            # their corticosteroid dose can be reduced back to baseline, they will be considered as having stable
            # disease; if further clinical deterioration related to tumor becomes apparent, they will be considered to
            # have progression. The date of progression should be the first time point at which corticosteroid increase
            # was necessary

        if tumorComponentsForEval == TumorComponentsForEval.NonCE:
            pass  # TODO: In clinical trials applying the “mixed” tumor criteria, the whole evaluation should be
            # performed in parallel for both the CE and the non-CE tumor burden at each timepoint in order to assign the
            # response category (e.g., PD, SD, PR, . . .), then the overall response category is assigned based on both
            # CE and non-CE categories: PD1SD/MR/PR/CR¼PD; MR/PR1SD¼MR/PR; CR1SD/MR/PR¼SD/MR/PR; SD1SD¼SD (see text for
            # details).

        # in all other cases overall response is the same as the current scan  # TODO: this is not correct
        response = OverallResponse(curr_scan.value)

        return response

    @staticmethod
    def update_response_assessment(ui, lineNodePairs):
        # update the number of target lesions, new lesions, and disappeared lesions in the UI
        num_target_les = lineNodePairs.get_number_of_targets()
        num_new_target_les = lineNodePairs.get_number_of_new_target_lesions()
        num_disapp_target_les = lineNodePairs.get_number_of_disappeared_target_lesions()
        num_new_meas_les = lineNodePairs.get_number_of_new_measurable_lesions()

        num_target_les_spinbox = ui.numTargetLesSpinBox
        num_new_les_spinbox = ui.numNewLesSpinBox
        num_disapp_les_spinbox = ui.numDisappLesSpinBox
        num_new_meas_les_spinbox = ui.numNewMeasLesSpinBox

        num_target_les_spinbox.setValue(num_target_les)
        num_new_les_spinbox.setValue(num_new_target_les)
        num_disapp_les_spinbox.setValue(num_disapp_target_les)
        num_new_meas_les_spinbox.setValue(num_new_meas_les)

        # update the sum of line products and relative change in the UI
        sum_line_products_t1 = lineNodePairs.get_sum_of_bidimensional_products(timepoint='timepoint1')
        sum_line_products_t2 = lineNodePairs.get_sum_of_bidimensional_products(timepoint='timepoint2')

        ui.sum_lineprods_t1_spinbox.setValue(sum_line_products_t1)
        ui.sum_lineprods_t2_spinbox.setValue(sum_line_products_t2)

        if not sum_line_products_t1 == 0:
            relative_change = lineNodePairs.get_rel_area_change()
            ui.sum_lineprods_relchange_spinbox.setValue(relative_change * 100)  # convert to percentage
            ui.sum_lineprods_relchange_spinbox.setVisible(True)

            response = ResponseClassificationMixin.response_assessment_from_rel_area_or_vol(rel_area=relative_change)

        else:
            ui.sum_lineprods_relchange_spinbox.setValue(np.nan)
            # make invisible since the relative change is not defined for division by zero
            ui.sum_lineprods_relchange_spinbox.setVisible(False)
            response = Response.SD

        if debug: print(f"Response assessment: {response}, setting index to {response.value}")
        ui.responseStatusComboBox.setCurrentIndex(response.value)

    @staticmethod
    def update_overall_response_params(ui):
        # parameters
        # ceOrNonCeComboBox = ui.ceOrNonCeComboBox
        # confirmationRequiredForPdCheckBox = ui.confirmationRequiredForPdCheckBox
        # referenceScanComboBox = ui.referenceScanComboBox
        currScanComboBox = ui.currScanComboBox
        newMeasLesCheckBox = ui.newMeasLesCheckBox
        # nonTargetNonMeasComboBox = ui.nonTargetNonMeasComboBox
        # clinicalStatusComboBox = ui.clinicalStatusComboBox
        # increasedSteroidUseCheckBox = ui.increasedSteroidUseCheckBox
        # steroidDoseComboBox = ui.steroidDoseComboBox
        # secondLineMedicationCheckBox = ui.secondLineMedicationCheckBox

        # update the overall response parameters
        currScanComboBox.setCurrentIndex(ui.responseStatusComboBox.currentIndex)
        newMeasLesCheckBox.setChecked(ui.numNewMeasLesSpinBox.value > 0)

    @staticmethod
    def update_overall_response_status(ui):
        # resulting response status combo box
        if debug:
            print("Updating overall response status")
            print("current parameters are as follows:")
            print(f"reference scan: {ui.referenceScanComboBox.currentIndex}")
            print(f"current scan: {ui.currScanComboBox.currentIndex}")
            print(f"new measurable lesions: {ui.newMeasLesCheckBox.isChecked()}")
            print(f"non-target or non-measurable lesions: {ui.nonTargetNonMeasComboBox.currentIndex}")
            print(f"clinical status: {ui.clinicalStatusComboBox.currentIndex}")
            print(f"increased steroid use: {ui.increasedSteroidUseCheckBox.isChecked()}")
            print(f"steroid dose: {ui.steroidDoseComboBox.currentIndex}")
            print(f"tumor components for evaluation: {ui.ceOrNonCeComboBox.currentIndex}")
            print(f"confirmation required for PD: {ui.confirmationRequiredForPdCheckBox.isChecked()}")

        overallResponseStatusComboBox = ui.overallResponseStatusComboBox

        overall_response = ResponseClassificationMixin.response_assessment_overall(
            ref_scan=RefScanRole(ui.referenceScanComboBox.currentIndex),
            curr_scan=CurrScanRole(ui.currScanComboBox.currentIndex),
            newMeasLes=ui.newMeasLesCheckBox.isChecked(),
            nonTargetOrNonMeasLes=NonTargetOrNonMeasLes(
                ui.nonTargetNonMeasComboBox.currentIndex),
            clinicalStatus=ClinicalStatus(
                ui.clinicalStatusComboBox.currentIndex),
            increasedSteroids=ui.increasedSteroidUseCheckBox.isChecked(),
            steroidDose=SteroidDose(ui.steroidDoseComboBox.currentIndex),
            tumorComponentsForEval=TumorComponentsForEval(
                ui.ceOrNonCeComboBox.currentIndex),
            confirmPD=ui.confirmationRequiredForPdCheckBox.isChecked())

        # set the overall response status
        overallResponseStatusComboBox.setCurrentIndex(overall_response.value)
