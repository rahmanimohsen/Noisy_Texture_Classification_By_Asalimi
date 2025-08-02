import numpy as np
import math
import cv2

def set_bit(v, index, x):
    """Set the index:th bit of v to 1 if x is truthy, else to 0, and return the new value."""
    mask = 1 << index   # Compute mask, an integer with just bit 'index' set.
    v &= ~mask          # Clear the bit indicated by the mask (if x is False)
    if x:
        v |= mask         # If x was True, set the bit indicated by the mask.
    return v            # Return the result, we're done.
def get_bit(v,index):
    return ((v&(1<<index))!=0);

def get_bits(v,indexs):
    retValue = []
    for i in indexs:
        retValue.append(int(get_bit(v,i)))
    return retValue;

#GETMAPPING returns a structure containing a mapping table for LBP codes.
#  MAPPING = GETMAPPING(SAMPLES,MAPPINGTYPE) returns a
#  structure containing a mapping table for
#  LBP codes in a neighbourhood of SAMPLES sampling
#  points. Possible values for MAPPINGTYPE are
#       'u2'   for uniform LBP
#       'ri'   for rotation-invariant LBP
#       'riu2' for uniform rotation-invariant LBP.
#


def getmapping(samples,mappingtype): 
    table = np.array(range(0,2**samples))
    newMax  = 0; #number of patterns in the resulting LBP code
    index   = 0;

    if mappingtype == 'u2': #Uniform 2
        newMax = samples*(samples-1) + 3;   ## P=8 -> bins number = 59
        for i in range(0,2**samples):
            j = set_bit(i<<1 & 2**samples-1,0,get_bit(i,samples-1)) #rotate left
            numt = sum(get_bits(i^j,list(range(0,samples)))) #number of 1->0 and 0->1 transitions in binary string x is equal to the number of 1-bits in XOR(x,Rotate left(x)) 
            if numt <= 2:
                table[i] = index
                index = index + 1
            else:
                table[i] = newMax - 1
    if mappingtype =='ri': #Rotation invariant (MinROR)
        tmpMap = np.ones(2**samples) * (-1)  
        for i in range(0,2**samples):
            rm = i;
            r  = i;
            for j in range(1,samples):
                r = set_bit(i<<1 & 2**samples-1,0,get_bit(i,samples-1)) #rotate left
                if r < rm : #find minROR
                    rm = r;
            if tmpMap[rm] < 0:
                tmpMap[rm] = newMax
                newMax = newMax + 1;
            table[i] = tmpMap[rm]

    if mappingtype =='riu2': #Uniform & Rotation invariant
        newMax = samples + 2;   # P+2 
        sampleList = []
        for item in range(0,samples):
            sampleList.append(item)
        for i in range(0,2**samples):
            j = set_bit(i<<1 & 2**samples-1,0,get_bit(i,samples-1)) #rotate left
            numt = sum(get_bits(i^j,list(range(0,samples))))
            if numt <= 2: # U<=2
                table[i] = sum(list(get_bits(i,sampleList))) #count 1s
            else:
                table[i] = samples+1 # count non-uniform pattern

    return {"table":table, "samples":samples, "newMax":newMax}
def clbp(image,R=1,P=8,patternMapping=None,_mode='h'): 
    d_image=image.astype(np.float32)
    if(R==1 and P!=8):
        print("error in set R and P arguments.")
        return -1
    if R==1 and P==8:
        spoints=np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors=8
        mapping=patternMapping
        mode=_mode
    else:
        radius = R
        neighbors = P
        spoints = np.zeros((neighbors,2))
        # Angle step.
        a = 2*math.pi/neighbors
        for i in range(0,neighbors):
            spoints[i,0] = -radius*math.sin((i)*a)
            spoints[i,1] = radius*math.cos((i)*a)
        if patternMapping is not None:
            mapping=patternMapping
            if(mapping["samples"] != neighbors):
                print('Incompatible mapping')
                return -1
        else:
            mapping = None     
            mode=_mode
    # Determine the dimensions of the input image.
    ysize, xsize = image.shape
    miny=min(spoints[:,0])
    maxy=max(spoints[:,0])
    minx=min(spoints[:,1])
    maxx=max(spoints[:,1])

    # Block size, each LBP code is computed within a block of size bsizey*bsizex
    bsizey=math.ceil(max(maxy,0))-math.floor(min(miny,0))+1
    bsizex=math.ceil(max(maxx,0))-math.floor(min(minx,0))+1
    # Coordinates of origin (0,0) in the block
    origy=0-math.floor(min(miny,0))
    origx=0-math.floor(min(minx,0))

    # Minimum allowed size for the input image depends
    # on the radius of the used LBP operator.
    if(xsize < bsizex or ysize < bsizey):
        error('Too small input image. Should be at least (2*radius+1) x (2*radius+1)');

    # Calculate dx and dy;
    dx = xsize - bsizex
    dy = ysize - bsizey

    # Fill the center pixel matrix C.
    C = image[origy:origy+dy+1,origx:origx+dx+1]
    d_C = C.astype(np.float32)
    
    bins = 2**neighbors
    # Initialize the result matrix with zeros.
    CLBP_S=np.zeros((dy+1,dx+1),dtype=np.int32)
    CLBP_M=np.zeros((dy+1,dx+1),dtype=np.int32)
    CLBP_C=np.zeros((dy+1,dx+1),dtype=np.int32)
    #Compute the LBP code image
    D = np.zeros((neighbors,dy+1,dx+1),dtype=np.int32)
    Diff = np.zeros((neighbors,dy+1,dx+1))
    MeanDiff = np.zeros(neighbors)
    for i in range(0,neighbors):
        y = spoints[i,0]+origy
        x = spoints[i,1]+origx
        # Calculate floors, ceils and rounds for the x and y.
        fy = math.floor(y)
        cy = math.ceil(y)
        ry = int(round(y))
        fx = math.floor(x)
        cx = math.ceil(x)
        rx = int(round(x))
        # Check if interpolation is needed.
        if (abs(x - rx) < 1e-6) and (abs(y - ry) < 1e-6):
            # Interpolation is not needed, use original datatypes
            N = d_image[ry:ry+dy+1,rx:rx+dx+1]
            D[i] = np.greater_equal(N,d_C)
            Diff[i] = abs(N-d_C);
            MeanDiff[i] = np.mean(np.mean(Diff[i]));
            #     pause
            
        else:
            # Weight of point(X,Y)=(1-difference to pointX)*( 1-difference to pointY)
            # Interpolation needed, use double type images 
            ty = y - fy;
            tx = x - fx;

            # Calculate the interpolation weights.
            w1 = (1 - tx) * (1 - ty);
            w2 =      tx  * (1 - ty);
            w3 = (1 - tx) *      ty ;
            w4 =      tx  *      ty ;
            # Compute interpolated pixel values
            N = w1*d_image[fy:fy+dy+1,fx:fx+dx+1] + w2*d_image[fy:fy+dy+1,cx:cx+dx+1] + w3*d_image[cy:cy+dy+1,fx:fx+dx+1] + w4*d_image[cy:cy+dy+1,cx:cx+dx+1]
            D[i] = np.greater_equal(N,d_C)      #1 or 0   
            Diff[i] = abs(N-d_C); # magnitude of difference    
            
            

            #mean of differences of all neighbores points
            #it is used for threshold of CLBP_C
            MeanDiff[i] = np.mean(np.mean(Diff[i]));

    # Difference threshold for CLBP_M
    DiffThreshold = np.mean(MeanDiff);

    # compute CLBP_S and CLBP_M
    for i in range(0,neighbors):
        # Update the result matrix.
        v = 2**(i);
        #change 1,0 bits to CLBP values
        CLBP_S = CLBP_S + v*D[i];
        CLBP_M = CLBP_M + v*(Diff[i]>=DiffThreshold);

    # CLBP_C
    CLBP_C = np.greater_equal(d_C,np.ones(d_C.shape)*np.mean(d_image.flatten()))
    CLBP_C = CLBP_C.astype(np.int32)
    #Apply mapping if it is defined
    if mapping is not None:
        bins = mapping["newMax"];
        CLBP_S = CLBP_S.astype(np.int32)
        CLBP_M = CLBP_M.astype(np.int32)
        # # another implementation method
        for i in range(0, CLBP_S.shape[0]):
            for j in range(0, CLBP_S.shape[1]):
                CLBP_S[i,j] = mapping["table"][CLBP_S[i,j]];
                CLBP_M[i,j] = mapping["table"][CLBP_M[i,j]];
    CLBP_C = CLBP_C.astype(np.uint8)
    CLBP_S = CLBP_S.astype(np.uint8)
    CLBP_M = CLBP_M.astype(np.uint8)

    CLBP_MCSum = CLBP_M
    CLBP_MCSum = CLBP_MCSum+CLBP_C * mapping["newMax"]
    Hist3D , xedges, yedges = np.histogram2d(CLBP_S.flatten("F"),CLBP_MCSum.flatten("F"),[mapping["newMax"],mapping["newMax"]*2],[[0,mapping["newMax"]],[0,mapping["newMax"]*2]])
    CLBP_SMCH = Hist3D.flatten("F")
    return CLBP_SMCH
