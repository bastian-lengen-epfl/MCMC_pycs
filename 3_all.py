import sys

try :
    execfile('3a_generate_tweakml.py')
except :
    print "Error in script 3a."
    sys.exit()

try :
    execfile('3b_draw_copy_mocks.py')
except :
    print "Error in script 3b."
    sys.exit()

try :
    execfile('3c_optimise_copy_mocks.py')
except :
    print "Error in script 3c."
    sys.exit()

try :
    execfile('3d_check_statistics.py')
except :
    print "Error in script 3d."
    sys.exit()