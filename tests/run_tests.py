from numpy.version import release

import slicer

slicer.util.selectModule('RANO')

# click the reload and test button
reloadTestMenuButton = slicer.modules.RANOWidget.reloadTestMenuButton
reloadTestAction = slicer.modules.RANOWidget.reloadTestAction
reloadTestMenuButton.setDefaultAction(reloadTestAction)
reloadTestMenuButton.click()