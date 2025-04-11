# This file is executed when Slicer starts.
# How this works:
# https://slicer.readthedocs.io/en/latest/user_guide/settings.html#application-startup-file

import slicer

modulePath = "/home/researcher/rano2.0-assist/run_command/run_command.py"
factoryManager = slicer.app.moduleManager().factoryManager()
factoryManager.registerModule(qt.QFileInfo(modulePath))
factoryManager.loadModules(["run_command",])
slicer.util.selectModule("run_command")

modulePath = "/home/researcher/rano2.0-assist/RANO/RANO.py"
factoryManager = slicer.app.moduleManager().factoryManager()
factoryManager.registerModule(qt.QFileInfo(modulePath))
factoryManager.loadModules(["RANO",])
slicer.util.selectModule("RANO")
