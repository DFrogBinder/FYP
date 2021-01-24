export DYLD_LIBRARY_PATH='/Users/boyanivanov/Elastix/lib':$DYLD_LIBRARY_PATH
output=$(eval "elastix -m /Users/boyanivanov/Desktop/FYP/DICOM/Leeds_Patient_4128010/19/DICOM/T1_slice_3_MoCoMo/Original/001.mhd -f /Users/boyanivanov/Desktop/FYP/DICOM/Leeds_Patient_4128010/19/DICOM/T1_slice_3_MoCoMo/Fitted/001.mhd -out /Users/boyanivanov/Desktop/FYP/DICOM/Leeds_Patient_4128010/19/DICOM/T1_slice_3_MoCoMo/BSplines_Registered_1 -p /Users/boyanivanov/Desktop/FYP/DICOM/Elastix_Parameters_Files/BSplines_T1.txt") 
echo "$output" 

