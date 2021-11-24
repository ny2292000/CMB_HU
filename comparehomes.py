from filecmp import dircmp

def print_diff_files(dcmp):
    for name in dcmp.diff_files:
        print("diff_file %s found in %s and %s" % (name, dcmp.left,dcmp.right))
        for sub_dcmp in dcmp.subdirs.values():
            print_diff_files(sub_dcmp)

if __name__=="__main__":
    dcmp=dircmp("/home/mp74207", "/media/home/mp74207")
    print_diff_files(dcmp)