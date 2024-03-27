from os import listdir, rename
from os.path import isfile, join

mypath = "/media/home/mp74207/GitHub/CMB_HU/sphericalharmonics/Big_KLM/"


onlyfiles = [join(mypath, f) for f in sorted(listdir(mypath)) if isfile(join(mypath, f)) if "Big" not in f]

newfilenames = []
for f in onlyfiles:
    oldfilename = f.replace(mypath,"")
    klm = f.replace(mypath,"").replace(".pkl","").split("_")
    newfilename = "%03.0f_%03.0f_%03.0f.pkl" % (int(klm[0]), int(klm[1]), int(klm[2]))
    newfilenames.append(newfilename)
# for f in sorted(newfilenames):
#     print(f)
    # print(oldfilename, newfilename)
    filenameold = join(mypath, oldfilename )
    filenameNew = join(mypath, newfilename)
    rename(filenameold,filenameNew)

