cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(RANO2assist)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/RANO2assist")
set(EXTENSION_CONTRIBUTORS "Aaron Kujawa (King's College London)")
set(EXTENSION_DESCRIPTION "Semi-automated interactive response assemessment in neuro-oncology for gliomas")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/RANO2assist.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/RANO2assist/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(RANO)
add_subdirectory(run_command)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
