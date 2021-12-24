import os
import platform
import tqdm


def getListOfFiles(dirName,PNumber,homeDir):
    # create a list of file and sub directories 
    # names in the given directory 
    if platform.system() == "Windows":
            Parts = dirName.split('\\')
    elif platform.system() == "Darwin":
            Parts = dirName.split('/')
    elif platform.system() == "Linux":
            Parts = dirName.split('/')
    if Parts[-1] == 'DICOM':
        os.chdir(dirName)
        if os.getcwd() == dirName:
            if os.listdir(dirName)[0][-1] == 'm':
                os.system('rm *.dcm')
            elif os.listdir(dirName)[0][-1] == 'A':
                os.system('rm *.IMA')
            else:
                print("Unrecognised file format")
                return
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    if 'Nifti_Export' in listOfFile:
        ExportDir = os.path.join(dirName, 'Nifti_Export')
        os.chdir(ExportDir)
        Slices = os.listdir(ExportDir)
        for sl in Slices:
            if sl == 'Slice_0.mha' or sl == 'Slice_1.mha' or sl == 'Slice_7.mha':
                tpath = os.path.join(ExportDir,sl)
                os.remove(tpath)
            else:
                tName = ['P'+PNumber+'S'+sl.split('_')[-1]]
                os.rename(os.path.join(ExportDir,sl),os.path.join(ExportDir,str(tName[0])))
        os.system('mv *.mha ../../../../')
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath,PNumber,homeDir)
        else:
            allFiles.append(fullPath)
    return allFiles

def ResultCleanup(path):
    global homeDir 
    homeDir = path
    ResultsList = os.listdir(path)
    
    if '.DS_Store' in ResultsList:
        os.remove(os.path.join(path,'.DS_Store'))
    
    for entry in tqdm.tqdm(ResultsList):
        PathToDir = os.path.join(path, entry)
        PNumber = PathToDir.split('/')[-1]
        PNumber = PNumber[-3:]
        if os.path.isdir(PathToDir):
            a = getListOfFiles(PathToDir,PNumber,PathToDir)
            
    return
ResultCleanup('/Users/boyanivanov/Documents/Temp_Data/ML_Data/')