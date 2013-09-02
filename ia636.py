"""
    Module ia636 -- Toolbox ia636
    -------------------------------------------------------------------
    This is a collection of functions and lessons used in the IA636 course
    (Visao Computacional) taught at University of Campinas.
    -------------------------------------------------------------------
    iaapplylut()       -- Intensity image transform.
    iabwlp()           -- Low-Pass Butterworth frequency filter.
    iacircle()         -- Create a binary circle image.
    iacolorhist()      -- Color-image histogram.
    iacolormap()       -- Create a colormap table.
    iacomb()           -- Create a grid of impulses image.
    iacontour()        -- Contours of binary images.
    iaconv()           -- 2D convolution.
    iacos()            -- Create a cossenoidal image.
    iacrop()           -- Crop an image to find the minimum rectangle.
    iadct()            -- Discrete Cossine Transform.
    iadctmatrix()      -- Kernel matrix for the DCT Transform.
    iadft()            -- Discrete Fourier Transform.
    iadftmatrix()      -- Kernel matrix for the DFT Transform.
    iadftview()        -- Generate optical Fourier Spectrum for display from DFT
                          data.
    iadither()         -- Ordered Dither.
    iaellipse()        -- Generate a 2D ellipse, rectangle or diamond image.
    iaerror()          -- Print message of error.
    iaffine()          -- Affine transform.
    iafftshift()       -- Shifts zero-frequency component to center of spectrum.
    iafloyd()          -- Floyd-Steinberg error diffusion.
    iagaussian()       -- Generate a 2D Gaussian image.
    iageorigid()       -- 2D Rigid body geometric transformation and scaling.
    iagshow()          -- Matrix of the image display.
    iahaarmatrix()     -- Kernel matrix for the Haar Transform.
    iahadamard()       -- Hadamard Transform.
    iahadamardmatrix() -- Kernel matrix for the Hadamard Transform.
    iahistogram()      -- Image histogram.
    iahsv2rgb()        -- Convert HSV to RGB color model.
    iahwt()            -- Haar Wavelet Transform.
    iaidct()           -- Inverse Discrete Cossine Transform.
    iaidft()           -- Inverse Discrete Fourier Transform.
    iaifftshift()      -- Undoes the effects of iafftshift.
    iaihadamard()      -- Inverse Hadamard Transform.
    iaihwt()           -- Inverse Haar Wavelet Transform.
    iaind2sub()        -- Convert linear index to double subscripts.
    iaisdftsym()       -- Check for conjugate symmetry
    iaisolines()       -- Isolines of a grayscale image.
    ialabel()          -- Label a binary image.
    ialblshow()        -- Display a labeled image assigning a random color for
                          each label.
    ialog()            -- Laplacian of Gaussian image.
    ialogfilter()      -- Laplacian of Gaussian filter.
    iameshgrid()       -- Create two 2-D matrices of indexes.
    ianeg()            -- Negate an image.
    ianormalize()      -- Normalize the pixels values between the specified
                          range.
    iaotsu()           -- Thresholding by Otsu.
    iapad()            -- Extend the image inserting a frame around it.
    iapconv()          -- 2D Periodic convolution.
    iaplot()           -- Plot a function.
    iaptrans()         -- Periodic translation.
    iaramp()           -- Create an image with vertical bands of increasing gray
                          values.
    iaread()           -- Read an image file (PBM, PGM and PPM).
    iarec()            -- Reconstruction of a connect component.
    iarectangle()      -- Create a binary rectangle image.
    iaresize()         -- Resizes an image.
    iargb2hsv()        -- Convert RGB to HSV color model.
    iargb2ycbcr()      -- Convert RGB to YCbCr color model.
    iaroi()            -- Cut a rectangle out of an image.
    iashow()           -- Image display.
    iasobel()          -- Sobel edge detection.
    iasplot()          -- Plot a surface.
    iastat()           -- Calculates MSE, PSNR and Pearson correlation between
                          two images.
    iasub2ind()        -- Convert linear double subscripts to linear index.
    iatcrgb2ind()      -- True color RGB to index image and colormap.
    iatile()           -- Replicate the image until reach a new size.
    iatype()           -- Print the source code of a function.
    iaunique()         -- Set unique.
    iavarfilter()      -- Variance filter.
    iawrite()          -- Write an image file (PBM, PGM and PPM).
    iaycbcr2rgb()      -- Convert RGB to YCbCr color model.

"""
#
__version__ = '0.5 all'

__version_string__ = 'Toolbox ia636 V0.5 10dec2002'

__build_date__ = '28jul2003 17:15'

#

#
# =====================================================================
#
#   iabwlp
#
# =====================================================================
def iabwlp(fsize, tc, n, option='circle'):
    """
        - Purpose
            Low-Pass Butterworth frequency filter.
        - Synopsis
            H = iabwlp(fsize, tc, n, option='circle')
        - Input
            fsize:  Gray-scale (uint8 or uint16) or binary image (logical).
                    Filter size: a col vector: first element: rows, second:
                    cols, etc. uses same convention as the return of size.
            tc:     Cutoff period.
            n:      Filter order.
            option: String. Default: 'circle'. Filter options. Possible
                    values: 'circle' or 'square'.
        - Output
            H: Gray-scale (uint8 or uint16) or binary image (logical). DFT
               mask filter, with H(0,0) as (u,v)=(0,0)
        - Description
            This function generates a frequency domain Low Pass Butterworth
            Filter with cutoff period tc and order n . At the cutoff period
            the filter amplitude is about 0.7 of the amplitude at H(0,0).
            This function returns the mask filter with H(0,0). As the larger
            the filter order, sharper will be the amplitude transition at
            cutoff period. The minimum cutoff period is always 2 pixels,
            despite of the size of the frequency filter.
        - Examples
            #
            #   example 1
            #
            H2_10 = iabwlp([100,100],2,10) # cutoff period: 2 pixels, order: 10
            iashow(iadftview(H2_10))
            H4_1 = iabwlp([100,100],4,1) # cutoff period: 4, order: 1
            iashow(iadftview(H4_1))
            H8_100 = iabwlp([100,100],8,100) # cutoff period: 8, order: 100
            iashow(iadftview(H8_100))
            H4_1box = iabwlp([100,100],4,1,'square') # cutoff period: 4, order: 1
            iashow(iadftview(H4_1box))
            #
            #   example 2
            #
            import Numeric
            import FFT
            f = iaread('cookies.pgm')
            iashow(f)
            F = FFT.fft2d(f)
            iashow(iadftview(F))
            H = iabwlp(F.shape,16,6)
            iashow(iadftview(H))
            G = F * H
            iashow(iadftview(G))
            g = FFT.inverse_fft2d(G)
            g = abs(g).astype(Numeric.UnsignedInt8)
            iashow(g)
    """
    from Numeric import arange, sqrt, maximum, ravel, reshape
    import string

    def test_exp(x, y):
        try:
            return x**(2*y)
        except:
            return 1E300 # Infinito!
    rows, cols = fsize[0], fsize[1]
    mh, mw = rows/2, cols/2
    y, x = iameshgrid(arange(-mw,cols-mw), arange(-mh,rows-mh)) # center
    if string.find(string.upper(option), 'SQUARE') != -1:
        H=1./(1.+(sqrt(2)-1)*(maximum(abs(1.*x/rows) , abs(1.*y/cols))*tc)**(2*n))
    else:
        aux1 = ravel(sqrt(((1.*x)/rows)**2 + ((1.*y)/cols)**2)*tc)
        aux2 = 0.*aux1 + n
        aux = reshape(map(test_exp, aux1, aux2), x.shape)
        H=1./(1+(sqrt(2)-1)*aux)
    H=iafftshift(H)
    return H
#
# =====================================================================
#
#   iacircle
#
# =====================================================================
def iacircle(s, r, c):
    """
        - Purpose
            Create a binary circle image.
        - Synopsis
            g = iacircle(s, r, c)
        - Input
            s: Gray-scale (uint8 or uint16) or binary image (logical). [rows
               cols], output image dimensions.
            r: Non-negative integer. radius.
            c: Gray-scale (uint8 or uint16) or binary image (logical). [row0
               col0], center of the circle.
        - Output
            g: Binary image (logical).
        - Description
            Creates a binary image with dimensions given by s, radius given
            by r and center given by c. The pixels inside the circle are one
            and outside zero.
        - Examples
            #
            #   example 1
            #
            F = iacircle([5,7], 2, [2,3])
            print F
            #
            #   example 2
            #
            F = iacircle([200,300], 90, [100,150])
            iashow(F)
    """

    rows, cols = s[0], s[1]
    y0, x0 = c[0], c[1]
    x, y = iameshgrid(range(cols), range(rows))
    g = (x - x0)**2 + (y - y0)**2 <= r**2
    return g
#
# =====================================================================
#
#   iaramp
#
# =====================================================================
def iaramp(s, n, range):
    """
        - Purpose
            Create an image with vertical bands of increasing gray values.
        - Synopsis
            g = iaramp(s, n, range)
        - Input
            s:     Gray-scale (uint8 or uint16) or binary image (logical).
                   [H W], height and width output image dimensions.
            n:     Non-negative integer. number of vertical bands.
            range: Gray-scale (uint8 or uint16) or binary image (logical).
                   [kmin, kmax], minimum and maximum gray scale values.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Creates a gray scale image with dimensions given by s, with n
            increasing gray scale bands from left to right with values
            varying from the specified range.
        - Examples
            #
            #   example 1
            #
            F = iaramp([5,7], 3, [4,10])
            print F
            #
            #   example 2
            #
            F = iaramp([200,300], 10, [0,255])
            iashow(F)
    """
    from Numeric import arange

    rows, cols = s[0], s[1]
    x, y = iameshgrid(arange(cols), arange(rows))
    g = x*n/cols * (range[1]-range[0]) / (n-1) + range[0]
    return g
#
# =====================================================================
#
#   iacos
#
# =====================================================================
def iacos(s, t, theta, phi):
    """
        - Purpose
            Create a cossenoidal image.
        - Synopsis
            f = iacos(s, t, theta, phi)
        - Input
            s:     Gray-scale (uint8 or uint16) or binary image (logical).
                   size: [rows cols].
            t:     Gray-scale (uint8 or uint16) or binary image (logical).
                   Period: in pixels.
            theta: spatial direction of the wave, in radians. 0 is a wave on
                   the horizontal direction.
            phi:   Double. Phase
        - Output
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Generate a cosenosoid image of size s with amplitude 1, period
            T, phase phi and wave direction of theta. The output image is a
            double array.
        - Examples
            #
            import Numeric
            f = iacos([128,256], 100, Numeric.pi/4, 0)
            iashow(ianormalize(f, [0,255]))
    """
    from Numeric import cos, sin, pi

    cols, rows = s[1], s[0]
    x, y = iameshgrid(range(cols),range(rows))
    freq = 1./t
    fcols = freq * cos(theta)
    frows = freq * sin(theta)
    f = cos(2*pi*(fcols*x + frows*y) + phi)
    return f
#
# =====================================================================
#
#   iaroi
#
# =====================================================================
def iaroi(f, p1, p2):
    """
        - Purpose
            Cut a rectangle out of an image.
        - Synopsis
            g = iaroi(f, p1, p2)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image (logical).
                input image.
            p1: Gray-scale (uint8 or uint16) or binary image (logical).
                indices of the coordinates at top-left.
            p2: Gray-scale (uint8 or uint16) or binary image (logical).
                indices of the coordinates at bottom-right.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = iaread('lenina.pgm')
            iashow(f)
            froi = iaroi(f, [90,70], [200,180])
            iashow(froi)
    """
    from Numeric import asarray, NewAxis

    f = asarray(f)
    if len(f.shape) == 1: f = f[NewAxis,:]
    g = f[p1[0]:p2[0]+1, p1[1]:p2[1]+1]
    return g
#
# =====================================================================
#
#   iacrop
#
# =====================================================================
def iacrop(f, side='all', color='black'):
    """
        - Purpose
            Crop an image to find the minimum rectangle.
        - Synopsis
            g = iacrop(f, side='all', color='black')
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image (logical).
                   input image.
            side:  Default: 'all'. side of the edge which will be removed.
                   Possible values: 'all', 'left', 'right', 'top', 'bottom'.
            color: Default: 'black'. color of the edge. Possible values:
                   'black', 'white'.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = iaread('club.pgm')
            iashow(f)
            g = iacrop(f)
            iashow(g)
    """
    from Numeric import asarray, NewAxis, sometrue, nonzero

    ##if len(f.shape) == 1: f = f[NewAxis,:]
    ##aux1, aux2 = sometrue(f,0), sometrue(f,1)
    ##col, row = nonzero(aux1), nonzero(aux2)
    ##g = f[row[0]:row[-1]+1, col[0]:col[-1]+1]
    f = asarray(f)
    if len(f.shape) == 1: f = f[NewAxis,:]
    if color == 'white': f = ianeg(f)
    aux1, aux2 = sometrue(f,0), sometrue(f,1)
    col, row = nonzero(aux1), nonzero(aux2)
    if (not col) and (not row):
        return None
    if   side == 'left':   g = f[:, col[0]::]
    elif side == 'right':  g = f[:, 0:col[-1]+1]
    elif side == 'top':    g = f[row[0]::, :]
    elif side == 'bottom': g = f[0:row[-1]+1, :]
    else:                  g = f[row[0]:row[-1]+1, col[0]:col[-1]+1]
    if color == 'white': g = ianeg(g)
    return g
#
# =====================================================================
#
#   iapad
#
# =====================================================================
def iapad(f, thick=[1,1], value=0):
    """
        - Purpose
            Extend the image inserting a frame around it.
        - Synopsis
            g = iapad(f, thick=[1,1], value=0)
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image (logical).
                   input image.
            thick: Gray-scale (uint8 or uint16) or binary image (logical).
                   Default: [1,1]. [rows cols] to be padded.
            value: Double. Default: 0. value used in the frame around the
                   image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = Numeric.array([[0,1,2],[3,4,5]], Numeric.UnsignedInt8)
            print f
            g1 = iapad(f)
            print g1
            g2 = iapad(f, (1,3), 5)
            print g2
    """
    from Numeric import asarray, ones, array

    f, thick = asarray(f), asarray(thick)
    g = (value * ones(array(f.shape)+2*thick)).astype(f.typecode())
    g[thick[0]:-thick[0], thick[1]:-thick[1]] = f
    return g
#
# =====================================================================
#
#   iatile
#
# =====================================================================
def iatile(f, new_size):
    """
        - Purpose
            Replicate the image until reach a new size.
        - Synopsis
            g = iatile(f, new_size)
        - Input
            f:        Gray-scale (uint8 or uint16) or binary image
                      (logical). input image.
            new_size: Gray-scale (uint8 or uint16) or binary image
                      (logical). [rows cols], output image dimensions.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f=[[1,2],[3,4]]
            print f
            g = iatile(f, (3,6))
            print g
    """
    from Numeric import asarray, NewAxis, resize, transpose

    f = asarray(f)
    if len(f.shape) == 1: f = f[NewAxis,:]
    aux = resize(f, (new_size[0], f.shape[1]))
    aux = transpose(aux)
    aux = resize(aux, (new_size[1], new_size[0]))
    g = transpose(aux)
    return g
#
# =====================================================================
#
#   iaapplylut
#
# =====================================================================
def iaapplylut(fi, it):
    """
        - Purpose
            Intensity image transform.
        - Synopsis
            g = iaapplylut(fi, it)
        - Input
            fi: Gray-scale (uint8 or uint16) or binary image (logical).
                input image, gray scale or index image.
            it: Gray-scale (uint8 or uint16) or binary image (logical).
                Intensity transform. Table of one or three columns.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Apply an intensity image transform to the input image. The input
            image can be seen as an gray scale image or an index image. The
            intensity transform is represented by a table where the input
            (gray scale) color address the table line and its column
            contents indicates the output (gray scale) image color. The
            table can have one or three columns. If it has three columns,
            the output image is a three color band image. This intensity
            image transformation is very powerful and can be use in many
            applications involving gray scale and color images. If the input
            image has an index (gray scale color) that is greater than the
            size of the intensity table, an error is reported.
        - Examples
            #
            #   example 1
            #
            f = [[0,1,2], [3,4,5]]
            it = Numeric.array(range(6)) # identity transform
            print it
            g = iaapplylut(f, it)
            print g
            itn = 5 - it  # negation
            g = iaapplylut(f, itn)
            print g
            #
            #   example 2
            #
            f = iaread('cameraman.pgm')
            it = 255 - Numeric.arange(256)
            g = iaapplylut(f, it)
            iashow(f)
            iashow(g)
            #
            #   example 3
            #
            f = [[0,1,1], [0,0,1]]
            ct = [[255,0,0], [0,255,0]]
            g = iaapplylut(f,ct)
            print g
            #
            #   example 4
            #
            f = iaread('cameraman.pgm')
            aux = Numeric.resize(range(256), (256,1))
            ct = Numeric.concatenate((aux, Numeric.zeros((256,2))), 1)
            g = iaapplylut(f, ct)
            iashow(f)
            iashow(g)
    """
    from Numeric import asarray, product, ravel, NewAxis, reshape, array, transpose

    it = asarray(it)
    if product(it.shape) == max(it.shape): # 1D
        it = ravel(it)
    def lut_map(intens, it=it):
        return it[intens]
    fi = asarray(fi)
    if len(fi.shape) == 1:
        fi = fi[NewAxis,:]
    aux = ravel(fi)
    if len(it.shape) == 1: # 1D
        g = reshape(map(lut_map, aux), fi.shape)
    elif it.shape[1] == 3:
        g = reshape(transpose(map(lut_map, aux)), (3,fi.shape[0],fi.shape[1]))
    else:
        iaerror('error, table should be 1 or 3 columns')
        g = array([])
    return g
#
# =====================================================================
#
#   iacolormap
#
# =====================================================================
def iacolormap(type='gray'):
    """
        - Purpose
            Create a colormap table.
        - Synopsis
            ct = iacolormap(type='gray')
        - Input
            type: Default: 'gray'. Type of the colormap. Options: 'gray',
                  'hsv', 'hot', 'cool', 'bone','copper', 'pink'.
        - Output
            ct: Gray-scale (uint8 or uint16) or binary image (logical).
                Colormap table.
        - Description
            Create a colormap table.
        - Examples
            #
            f = ianormalize(iabwlp([150,150], 4, 1), [0,255]).astype('b')
            cm1 = iacolormap('hsv')
            g_cm1 = iaapplylut(f, cm1)
            cm2 = iacolormap('hot')
            g_cm2 = iaapplylut(f, cm2)
            iashow(f)
            iashow(g_cm1)
            iashow(g_cm2)
    """
    from Numeric import sqrt, transpose, resize, reshape, NewAxis, concatenate, arange, ones, zeros, matrixmultiply
    import colorsys

    if type == 'gray':
        ct = transpose(resize(arange(256), (3,256)))
    elif type == 'hsv':
        h = arange(256)/255.
        s = ones(256)
        v = ones(256)
        ct = ianormalize(reshape(map(colorsys.hsv_to_rgb, h, s, v), (256,3)), [0,255]).astype('b')
    elif type == 'hot':
        n = 1.*int(3./8*256)
        r = concatenate((arange(1,n+1)/n, ones(256-n)), 1)[:,NewAxis]
        g = concatenate((zeros(n), arange(1,n+1)/n, ones(256-2*n)), 1)[:,NewAxis]
        b = concatenate((zeros(2*n), arange(1,256-2*n+1)/(256-2*n)), 1)[:,NewAxis]
        ct = ianormalize(concatenate((r,g,b), 1), [0,255]).astype('b')
    elif type == 'cool':
        r = (arange(256)/255.)[:,NewAxis]
        ct = ianormalize(concatenate((r, 1-r, ones((256,1))), 1), [0,255]).astype('b')
    elif type == 'bone':
        ct = ianormalize((7*iacolormap('gray') + iacolormap('hot')[:,::-1]) / 8., [0,255]).astype('b')
    elif type == 'copper':
        ct = ianormalize(min(1, matrixmultiply(iacolormap('gray')/255., [[1.25,0,0],[0,0.7812,0],[0,0,0.4975]])), [0,255]).astype('b')
    elif type == 'pink':
        ct = ianormalize(sqrt((2*iacolormap('gray') + iacolormap('hot')) / 3), [0,255]).astype('b')
    else:
        ct = zeros((256,3))
    return ct
#
# =====================================================================
#
#   iatcrgb2ind
#
# =====================================================================
def iatcrgb2ind(f):
    """
        - Purpose
            True color RGB to index image and colormap.
        - Synopsis
            fi,cm = iatcrgb2ind(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). True
               color RGB image.
        - Output
            fi,cm: Gray-scale (uint8 or uint16) or binary image (logical).
                   Index image and its colormap.
        - Description
            Converts a true color RGB image to the format of index image and
            a colormap.
        - Examples
            #
            import Numeric
            r = [[4,5,6],[4,2,4]]
            g = [[0,1,2],[0,4,0]]
            b = [[1,0,2],[1,2,2]]
            f = Numeric.zeros((3,2,3))
            f[0,:,:], f[1,:,:], f[2,:,:] = r, g, b
            print f
            (fi, tm) = iatcrgb2ind(f)
            print fi
            print tm
    """
    from Numeric import asarray, reshape, concatenate

    f = asarray(f)
    r, g, b = f[0,:,:], f[1,:,:], f[2,:,:]
    c = r + 256*g + 256*256*b
    (t,i,fi) = iaunique(c)
    siz = len(t)
    rt = reshape(map(lambda k:int(k%256), t), (siz,1))
    gt = reshape(map(lambda k:int((k%(256*256))/256.), t), (siz,1))
    bt = reshape(map(lambda k:int(k), t/(256*256.)), (siz,1))
    cm = concatenate((rt, gt, bt), axis=1)
    fi = reshape(fi, (f.shape[1], f.shape[2]))
    return fi,cm
#
# =====================================================================
#
#   iargb2hsv
#
# =====================================================================
def iargb2hsv(f):
    """
        - Purpose
            Convert RGB to HSV color model.
        - Synopsis
            g = iargb2hsv(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). True
               color RGB image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical). HSV
               color model.
        - Description
            Converts red-green-blue colors to hue-saturation-value.
        - Examples
            #
            import Numeric
            r = [[4,5,6],[4,2,4]]
            g = [[0,1,2],[0,4,0]]
            b = [[1,0,2],[1,2,2]]
            f = Numeric.zeros((3,2,3))
            f[0,:,:], f[1,:,:], f[2,:,:] = r, g, b
            print f[0,:,:]
            print f[1,:,:]
            print f[2,:,:]
            g = iargb2hsv(f)
            print g[0,:,:]
            print g[1,:,:]
            print g[2,:,:]
    """
    from Numeric import ravel, reshape, transpose
    import colorsys

    g = map(colorsys.rgb_to_hsv, ravel(f[0,:,:]/255.), ravel(f[1,:,:]/255.), ravel(f[2,:,:]/255.))
    g = reshape(transpose(g), f.shape)
    return g
#
# =====================================================================
#
#   iahsv2rgb
#
# =====================================================================
def iahsv2rgb(f):
    """
        - Purpose
            Convert HSV to RGB color model.
        - Synopsis
            g = iahsv2rgb(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). HSV
               color model.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical). True
               color RGB image.
        - Description
            Converts hue-saturation-value to red-green-blue colors.
        - Examples
            #
            import Numeric
            r = [[4,5,6],[4,2,4]]
            g = [[0,1,2],[0,4,0]]
            b = [[1,0,2],[1,2,2]]
            f = Numeric.zeros((3,2,3))
            f[0,:,:], f[1,:,:], f[2,:,:] = r, g, b
            print f[0,:,:]
            print f[1,:,:]
            print f[2,:,:]
            g = iargb2hsv(f)
            f_ = iahsv2rgb(g)
            print f_[0,:,:]
            print f_[1,:,:]
            print f_[2,:,:]
    """
    from Numeric import ravel, reshape, transpose
    import colorsys

    g = map(colorsys.hsv_to_rgb, ravel(f[0,:,:]), ravel(f[1,:,:]), ravel(f[2,:,:]))
    g = 255*reshape(transpose(g), f.shape)
    return g
#
# =====================================================================
#
#   iargb2ycbcr
#
# =====================================================================
def iargb2ycbcr(f):
    """
        - Purpose
            Convert RGB to YCbCr color model.
        - Synopsis
            g = iargb2ycbcr(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). True
               color RGB image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical). YCbCr
               image.
        - Description
            Convert RGB values to YCbCr color space.
        - Examples
            #
            import Numeric
            r = [[4,5,6],[4,2,4]]
            g = [[0,1,2],[0,4,0]]
            b = [[1,0,2],[1,2,2]]
            f = Numeric.zeros((3,2,3))
            f[0,:,:], f[1,:,:], f[2,:,:] = r, g, b
            print f[0,:,:]
            print f[1,:,:]
            print f[2,:,:]
            g = iargb2ycbcr(f)
            print g[0,:,:]
            print g[1,:,:]
            print g[2,:,:]
    """

    g = 0.*f
    g[0,:,:] =  f[0,:,:]*0.257 + f[1,:,:]*0.504 + f[2,:,:]*0.098 + 16
    g[1,:,:] = -f[0,:,:]*0.148 - f[1,:,:]*0.291 + f[2,:,:]*0.439 + 128
    g[2,:,:] =  f[0,:,:]*0.439 - f[1,:,:]*0.368 - f[2,:,:]*0.071 + 128
    return g
#
# =====================================================================
#
#   iaycbcr2rgb
#
# =====================================================================
def iaycbcr2rgb(f):
    """
        - Purpose
            Convert RGB to YCbCr color model.
        - Synopsis
            g = iaycbcr2rgb(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). YCbCr
               image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical). True
               color RGB image.
        - Description
            Convert RGB values to YCbCr color space.
        - Examples
            #
            import Numeric
            r = [[4,5,6],[4,2,4]]
            g = [[0,1,2],[0,4,0]]
            b = [[1,0,2],[1,2,2]]
            f = Numeric.zeros((3,2,3))
            f[0,:,:], f[1,:,:], f[2,:,:] = r, g, b
            print f[0,:,:]
            print f[1,:,:]
            print f[2,:,:]
            g = iargb2ycbcr(f)
            f_ = iaycbcr2rgb(g)
            print f_[0,:,:]
            print f_[1,:,:]
            print f_[2,:,:]
    """

    g = 0.*f
    g[0,:,:] = 1.164*(f[0,:,:]-16) + 1.596*(f[2,:,:]-128)
    g[1,:,:] = 1.164*(f[0,:,:]-16) - 0.392*(f[1,:,:]-128) - 0.813*(f[2,:,:]-128)
    g[2,:,:] = 1.164*(f[0,:,:]-16) + 2.017*(f[1,:,:]-128)
    return g
#
# =====================================================================
#
#   iadct
#
# =====================================================================
def iadct(f):
    """
        - Purpose
            Discrete Cossine Transform.
        - Synopsis
            F = iadct(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = 255 * iacircle([256,256], 10, [129,129])
            iashow(f)
            F = iadct(f)
            iashow(Numeric.log(abs(F)+1))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    if (n == 1):
        A = iadctmatrix(m)
        F = matrixmultiply(A, f)
    else:
        A=iadctmatrix(m)
        B=iadctmatrix(n)
        F = matrixmultiply(matrixmultiply(A, f), transpose(B))
    return F
#
# =====================================================================
#
#   iadctmatrix
#
# =====================================================================
def iadctmatrix(N):
    """
        - Purpose
            Kernel matrix for the DCT Transform.
        - Synopsis
            A = iadctmatrix(N)
        - Input
            N: Non-negative integer.
        - Output
            A: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            #   example 1
            #
            A = iadctmatrix(128)
            iashow(A)
            #
            #   example 2
            #
            import Numeric
            A = iadctmatrix(4)
            print Numeric.array2string(A, precision=4, suppress_small=1)
            B = Numeric.matrixmultiply(A,Numeric.transpose(A)) 
            print Numeric.array2string(B, precision=4, suppress_small=1)
    """
    from Numeric import ones, sqrt, cos, pi

    x, u = iameshgrid(range(N), range(N)) # (u,x)
    alpha = ones((N,N)) * sqrt(2./N)
    alpha[0,:] = sqrt(1./N) # alpha(u,x)
    A = alpha * cos((2*x+1)*u*pi / (2.*N)) # Cn(u,x)
    return A
#
# =====================================================================
#
#   iadft
#
# =====================================================================
def iadft(f):
    """
        - Purpose
            Discrete Fourier Transform.
        - Synopsis
            F = iadft(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = 255 * iacircle([256,256], 10, [129,129])
            iashow(f)
            F = iadft(f)
            Fv = iadftview(F)
            iashow(Fv)
    """
    from Numeric import asarray, Float64, NewAxis, sqrt, matrixmultiply

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    if (n == 1):
        A = iadftmatrix(m)
        F = sqrt(m) * matrixmultiply(A, f)
    else:
        A = iadftmatrix(m)
        B = iadftmatrix(n)
        F = sqrt(m * n) * matrixmultiply(matrixmultiply(A, f), B)
    return F
#
# =====================================================================
#
#   iadftmatrix
#
# =====================================================================
def iadftmatrix(N):
    """
        - Purpose
            Kernel matrix for the DFT Transform.
        - Synopsis
            A = iadftmatrix(N)
        - Input
            N: Non-negative integer.
        - Output
            A: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            #   example 1
            #
            import Numeric
            A = iadftmatrix(128)
            iashow(A.imag)
            iashow(A.real)
            #
            #   example 2
            #
            import Numeric
            import LinearAlgebra
            A = iadftmatrix(4)
            print Numeric.array2string(A, precision=4, suppress_small=1)
            print Numeric.array2string(A-Numeric.transpose(A), precision=4, suppress_small=1)
            print abs(LinearAlgebra.inverse(A)-Numeric.conjugate(A)) < 10E-15
    """
    from Numeric import reshape, Float64, exp, pi, sqrt, matrixmultiply, transpose

    x = reshape(range(N),(N,1)).astype(Float64)
    u = x
    Wn = exp(-1j*2*pi/N)
    A = (1./sqrt(N)) * (Wn ** matrixmultiply(u, transpose(x)))
    return A
#
# =====================================================================
#
#   iaisolines
#
# =====================================================================
def iaisolines(f, nc=10, np=1):
    """
        - Purpose
            Isolines of a grayscale image.
        - Synopsis
            g = iaisolines(f, nc=10, np=1)
        - Input
            f:  Gray-scale (uint8 or uint16) or binary image (logical).
                Input image.
            nc: Default: 10. Number of colors.
            np: Default: 1. Number of pixels by isoline.
        - Output
            g: Gray-scale uint8 image. Input image with color isolines.
        - Description
            Shows lines where the pixels have same intensity with a unique
            color.
        - Examples
            #
            f = ianormalize(iabwlp([150,150], 4, 1), [0,255]).astype('b')
            g = iaisolines(f, 10, 3)
            iashow(f)
            iashow(g)
    """
    from Numeric import ravel, ceil, zeros, concatenate

    maxi = max(ravel(f))
    mini = min(ravel(f))
    d = ceil(1.*(maxi-mini)/nc)
    m = zeros((d,1)); m[0:np,:] = 1;
    m = iatile(m, (maxi-mini, 1))
    m = concatenate((zeros((mini,1)), m))
    m = concatenate((m, zeros((256-maxi,1))))
    m = concatenate((m,m,m), 1)
    ct = m*iacolormap('hsv') + (1-m)*iacolormap('gray')
    g = iaapplylut(f, ct)
    return g
#
# =====================================================================
#
#   iadftview
#
# =====================================================================
def iadftview(F):
    """
        - Purpose
            Generate optical Fourier Spectrum for display from DFT data.
        - Synopsis
            G = iadftview(F)
        - Input
            F: Gray-scale (uint8 or uint16) or binary image (logical). DFT
               complex data. F(1,1) is the center of the spectrum
               (u,v)=(0,0)
        - Output
            G: Gray-scale uint8 image. uint8 image suitable for displaying
        - Description
            Generate the logarithm of the magnitude of F, shifted so that
            the (0,0) stays at the center of the image. This is suitable for
            displaying only.
        - Examples
            #
            import FFT
            f = iaread('cameraman.pgm')
            iashow(f)
            F = FFT.fft2d(f)
            Fv = iadftview(F)
            iashow(Fv)
    """
    from Numeric import log, ravel, UnsignedInt8

    FM = iafftshift(log(abs(F)+1))
    G = (FM * 255./max(ravel(FM)))
    G = G.astype(UnsignedInt8)
    return G
#
# =====================================================================
#
#   iaerror
#
# =====================================================================
def iaerror(msg):
    """
        - Purpose
            Print message of error.
        - Synopsis
            iaerror(msg)
        - Input
            msg: Error message.

    """

    print msg
    return

#
# =====================================================================
#
#   iaffine
#
# =====================================================================
def iaffine(f, T):
    """
        - Purpose
            Affine transform.
        - Synopsis
            g = iaffine(f, T)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
            T: Gray-scale (uint8 or uint16) or binary image (logical).
               Affine matrix for the geometric transformation.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Apply the affine transform to the coordinate pixels of image f.
            The resultant image g has the same domain of the input image.
            Any pixel outside this domain is not shown and any pixel that
            does not exist in the original image has the nearest pixel
            value. This method is based on the inverse mapping. An affine
            transform is a geometrical transformation that preserves the
            parallelism of lines but not their lengths and angles. The
            affine transform can be a composition of translation, rotation,
            scaling and shearing. To simplify this composition, these
            transformations are represented in homogeneous coordinates using
            3x3 matrix T. The origin is at coordinate (0,0).
        - Examples
            #
            f = Numeric.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
            print f
            T = Numeric.array([[1,0,0],[0,1,0],[0,0,1]], 'd')
            print T
            g = iaffine(f,T)
            print g
            T[0,0] = 0.5
            print T
            g = iaffine(f,T)
            print g
    """
    from Numeric import asarray, ravel, NewAxis, concatenate, reshape, zeros, matrixmultiply, maximum, minimum, take
    from LinearAlgebra import inverse

    f, T = asarray(f), asarray(T)
    faux = ravel(f)
    if len(f.shape) == 1:
        f = f[NewAxis,:]
    (m, n) = f.shape
    (x, y) = iameshgrid(range(n),range(m))
    aux = concatenate((reshape(x, (1,m*n)), reshape(y, (1,m*n))))
    aux = concatenate((aux, zeros((1, m*n))))
    XY = matrixmultiply(inverse(T), aux)
    X, Y = XY[0,:], XY[1,:]
    X = maximum(0, minimum(n-1, X))
    Y = maximum(0, minimum(m-1, Y))
    XYi = iasub2ind(f.shape, map(round, Y), map(round, X))
    g = take(faux, XYi)
    g = reshape(g, f.shape)
    return g
#
# =====================================================================
#
#   iageorigid
#
# =====================================================================
def iageorigid(f, scale, theta, t):
    """
        - Purpose
            2D Rigid body geometric transformation and scaling.
        - Synopsis
            g = iageorigid(f, scale, theta, t)
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image (logical).
            scale: Gray-scale (uint8 or uint16) or binary image (logical).
                   [srow scol], scale in each dimension
            theta: Double. Rotation
            t:     Gray-scale (uint8 or uint16) or binary image (logical).
                   [trow tcol], translation in each dimension
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('lenina.pgm')
            g = iageorigid(f, [0.5,0.5], Numeric.pi/4, [0,64])
            iashow(g)
    """
    from Numeric import cos, sin, matrixmultiply

    Ts   = [[scale[1],0,0], [0,scale[0],0], [0,0,1]]
    Trot = [[cos(theta),-sin(theta),0], [sin(theta),cos(theta),0], [0,0,1]] 
    Tx   = [[1,0,t[1]], [0,1,t[0]], [0,0,1]]
    g = iaffine(f, matrixmultiply(matrixmultiply(Tx,Trot), Ts))
    return g
#
# =====================================================================
#
#   iafftshift
#
# =====================================================================
def iafftshift(H):
    """
        - Purpose
            Shifts zero-frequency component to center of spectrum.
        - Synopsis
            HS = iafftshift(H)
        - Input
            H: Gray-scale (uint8 or uint16) or binary image (logical). FFT
               image.
        - Output
            HS: Gray-scale (uint8 or uint16) or binary image (logical).
                Shift of the FFT image.
        - Description
            The origen (0,0) of the DFT is normally at top-left corner of
            the image. For visualization purposes, it is common to
            periodicaly translate the origen to the image center. This is
            particularly interesting because of the complex conjugate
            simmetry of the DFT of a real function. Note that as the image
            can have even or odd sizes, to translate back the DFT from the
            center to the corner, there is another correspondent function.
        - Examples
            #
            import Numeric
            import FFT
            f = iarectangle([120,150],[7,10],[60,75])
            F = FFT.fft2d(f)
            Fs = iafftshift(F)
            iashow(Numeric.log(abs(F)+1))
            iashow(Numeric.log(abs(Fs)+1))
    """
    from Numeric import asarray, NewAxis, array

    H = asarray(H)
    if len(H.shape) == 1: H = H[NewAxis,:]
    HS = iaptrans(H, array(H.shape)/2)
    return HS
#
# =====================================================================
#
#   iaifftshift
#
# =====================================================================
def iaifftshift(H):
    """
        - Purpose
            Undoes the effects of iafftshift.
        - Synopsis
            HS = iaifftshift(H)
        - Input
            H: Gray-scale (uint8 or uint16) or binary image (logical). DFT
               image with (0,0) in the center.
        - Output
            HS: Gray-scale (uint8 or uint16) or binary image (logical). DFT
                image with (0,0) in the top-left corner.

    """
    from Numeric import ceil, array, shape

    HS = iaptrans(H, ceil(-array(shape(H))/2))
    return HS
#
# =====================================================================
#
#   iagaussian
#
# =====================================================================
def iagaussian(s, mu, sigma):
    """
        - Purpose
            Generate a 2D Gaussian image.
        - Synopsis
            g = iagaussian(s, mu, sigma)
        - Input
            s:     Gray-scale (uint8 or uint16) or binary image (logical).
                   [rows columns]
            mu:    Gray-scale (uint8 or uint16) or binary image (logical).
                   Mean vector. 2D point (x;y). Point of maximum value.
            sigma: Gray-scale (uint8 or uint16) or binary image (logical).
                   covariance matrix (square). [ Sx^2 Sxy; Syx Sy^2]
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            A 2D Gaussian image is an image with a Gaussian distribution. It
            can be used to generate test patterns or Gaussian filters both
            for spatial and frequency domain. The integral of the gaussian
            function is 1.0.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = iagaussian([8,4], [3,1], [[1,0],[0,1]])
            print Numeric.array2string(f, precision=4, suppress_small=1)
            g = ianormalize(f, [0,255]).astype(Numeric.UnsignedInt8)
            print g
            #
            #   example 2
            #
            f = iagaussian(100, 50, 10*10)
            g = ianormalize(f, [0,1])
            g,d = iaplot(g)
            showfig(g)
            #
            #   example 3
            #
            f = iagaussian([50,50], [25,10], [[10*10,0],[0,20*20]])
            g = ianormalize(f, [0,255]).astype(Numeric.UnsignedInt8)
            iashow(g)
    """
    from Numeric import asarray, product, arange, NewAxis, transpose, matrixmultiply, reshape, concatenate, resize, sum, zeros, Float, ravel, pi, sqrt, exp
    from LinearAlgebra import inverse, determinant

    if type(sigma).__name__ in ['int', 'float', 'complex']: sigma = [sigma]
    s, mu, sigma = asarray(s), asarray(mu), asarray(sigma)
    if (product(s) == max(s)):
        x = arange(product(s))
        d = x - mu
        if len(d.shape) == 1:
            tmp1 = d[:,NewAxis]
            tmp3 = d
        else:
            tmp1 = transpose(d)
            tmp3 = tmp1
        if len(sigma) == 1:
            tmp2 = 1./sigma
        else:
            tmp2 = inverse(sigma)
        k = matrixmultiply(tmp1, tmp2) * tmp3
    else:
        aux = arange(product(s))
        x, y = iaind2sub(s, aux)
        xx = reshape(concatenate((x,y)), (2, product(x.shape)))
        d = transpose(xx) - resize(reshape(mu,(len(mu),1)), (s[0]*s[1],len(mu)))
        if len(sigma) == 1:
            tmp = 1./sigma
        else:
            tmp = inverse(sigma)
        k = matrixmultiply(d, tmp) * d
        k = sum(transpose(k))
    g = zeros(s, Float)
    aux = ravel(g)
    if len(sigma) == 1:
        tmp = sigma
    else:
        tmp = determinant(sigma)
    aux[:] = 1./(2*pi*sqrt(tmp)) * exp(-1./2 * k)
    return g
#
# =====================================================================
#
#   iaellipse
#
# =====================================================================
def iaellipse(r, theta=0, shape='ELLIPSE'):
    """
        - Purpose
            Generate a 2D ellipse, rectangle or diamond image.
        - Synopsis
            g = iaellipse(r, theta=0, shape='ELLIPSE')
        - Input
            r:     Gray-scale (uint8 or uint16) or binary image (logical).
                   vertical and horizontal radius: [rx, ry]
            theta: Double. Default: 0. rotation angle, in degrees.
            shape: String. Default: 'ELLIPSE'. Possible shapes: ELLIPSE,
                   RECTANGLE, or DIAMOND.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Generates a binary 2D ellipse, rectangle or diamond image given
            its maximum and minimum radius and rotation angle.

    """
    import Numeric

    return g
#
# =====================================================================
#
#   iaresize
#
# =====================================================================
def iaresize(f, new_shape):
    """
        - Purpose
            Resizes an image.
        - Synopsis
            g = iaresize(f, new_shape)
        - Input
            f:         Gray-scale (uint8 or uint16) or binary image
                       (logical).
            new_shape: Gray-scale (uint8 or uint16) or binary image
                       (logical). [h w], new image dimensions
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = Numeric.array([[10,20,30],[40,50,60]])
            print f
            g = iaresize(f, [5,7])
            print g
    """
    from Numeric import array, asarray, ravel, reshape, ceil, concatenate, zeros, matrixmultiply, maximum, minimum, take
    from LinearAlgebra import inverse

    def calc(f, new_shape=new_shape):
        from Numeric import array, ravel, reshape, ceil, concatenate, zeros, matrixmultiply, maximum, minimum, take
        from LinearAlgebra import inverse
        Sh = 1.* new_shape[0]/f.shape[0]
        Sw = 1.* new_shape[1]/f.shape[1]
        T = array([[Sw,0,0],[0,Sh,0],[0,0,1]])
        faux = ravel(f)
        if len(f.shape) == 1:
            f = reshape(f, (1, f.shape[0]))
        m, n = ceil(Sh*f.shape[0]), ceil(Sw*f.shape[1])
        x, y = iameshgrid(range(n),range(m))
        aux = concatenate((reshape(x, (1,m*n)), reshape(y, (1,m*n))))
        aux = concatenate((aux, zeros((1, m*n))))
        XY = matrixmultiply(inverse(T), aux+1)
        X, Y = XY[0,:], XY[1,:]

        X = maximum(1, minimum(f.shape[1], X))
        Y = maximum(1, minimum(f.shape[0], Y))

        XYi = iasub2ind(f.shape, map(round, Y-1), map(round, X-1))

        g = take(faux, XYi)
        g = reshape(g, (m,n))
        g = g[0:new_shape[0], 0:new_shape[1]]
        return g
    f = asarray(f)
    if len(f.shape) == 3: # imagem colorida
        g = zeros(concatenate(([3],new_shape)))
        for i in range(f.shape[0]):
            g[i,:,:] = calc(f[i,:,:])
    else:
        g = calc(f)
    return g
#
# =====================================================================
#
#   iahaarmatrix
#
# =====================================================================
def iahaarmatrix(N):
    """
        - Purpose
            Kernel matrix for the Haar Transform.
        - Synopsis
            A = iahaarmatrix(N)
        - Input
            N: Non-negative integer.
        - Output
            A: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            #   example 1
            #
            A = iahaarmatrix(128)
            iashow(A)
            #
            #   example 2
            #
            import Numeric
            A = iahaarmatrix(4)
            print A
            print Numeric.matrixmultiply(A, Numeric.transpose(A))
    """
    from Numeric import floor, log, arange, maximum, sqrt

    n = floor(log(N)/log(2))
    if 2**n != N:
       iaerror('error: size '+str(N)+' is not multiple of power of 2')
       return -1
    z, k = iameshgrid(1.*arange(N)/N, 1.*arange(N))
    p  = floor(log(maximum(1,k))/log(2))
    q  = k - (2**p) + 1
    z1 = (q-1)   / (2**p)
    z2 = (q-0.5) / (2**p)
    z3 = q       / (2**p)
    A  = (1/sqrt(N)) * ((( 2**(p/2.)) * ((z >= z1) & (z < z2))) \
                              + ((-2**(p/2.)) * ((z >= z2) & (z < z3))))
    A[0,:] = 1/sqrt(N)
    return A
#
# =====================================================================
#
#   iahadamard
#
# =====================================================================
def iahadamard(f):
    """
        - Purpose
            Hadamard Transform.
        - Synopsis
            F = iahadamard(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            F = iahadamard(f)
            iashow(Numeric.log(abs(F)+1))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    A = iahadamardmatrix(m)
    if A==-1: return -1
    if (n == 1):
        F = matrixmultiply(A, f)
    else:
        B = iahadamardmatrix(n)
        if B==-1: return -1
        F = matrixmultiply(matrixmultiply(A, f), transpose(B))
    return F
#
# =====================================================================
#
#   iahadamardmatrix
#
# =====================================================================
def iahadamardmatrix(N):
    """
        - Purpose
            Kernel matrix for the Hadamard Transform.
        - Synopsis
            A = iahadamardmatrix(N)
        - Input
            N: Non-negative integer.
        - Output
            A: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            #   example 1
            #
            A = iahadamardmatrix(128)
            iashow(A)
            #
            #   example 2
            #
            import Numeric
            A = iahadamardmatrix(4)
            print A
            print Numeric.matrixmultiply(A, Numeric.transpose(A))
    """
    from Numeric import floor, log, sqrt

    def bitsum(x):
        s = 0
        while x:
            s += x & 1
            x >>= 1
        return s
    n = floor(log(N)/log(2))
    if 2**n != N:
       iaerror('error: size '+str(N)+' is not multiple of power of 2')
       return -1
    u, x = iameshgrid(range(N), range(N))
    A = ((-1)**(bitsum(x & u)))/sqrt(N)
    return A
#
# =====================================================================
#
#   iahistogram
#
# =====================================================================
def iahistogram(f):
    """
        - Purpose
            Image histogram.
        - Synopsis
            h = iahistogram(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            h: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = iaread('woodlog.pgm')
            iashow(f)
            h = iahistogram(f)
            g,d = iaplot(h)
            g('set data style boxes')
            g.plot(d)
            showfig(h)
    """
    from Numeric import asarray, searchsorted, sort, ravel, concatenate, product

    f = asarray(f)
    n = searchsorted(sort(ravel(f)), range(max(ravel(f))+1))
    n = concatenate([n, [product(f.shape)]])
    h = n[1:]-n[:-1]
    return h
#
# =====================================================================
#
#   iastat
#
# =====================================================================
def iastat(f1, f2):
    """
        - Purpose
            Calculates MSE, PSNR and Pearson correlation between two images.
        - Synopsis
            MSE, PSNR, PC = iastat(f1, f2)
        - Input
            f1: Gray-scale (uint8 or uint16) or binary image (logical).
                Input image 1.
            f2: Gray-scale (uint8 or uint16) or binary image (logical).
                Input image 2 (Ex.: input image 1 degraded).
        - Output
            MSE:  Mean Square Error.
            PSNR: Peak Signal Noise Ratio.
            PC:   Correlation of the product of the moments of Pearson (r).
                  If r is +1 then there is a perfect linear relation between
                  f1 and f2. If r is -1 then there is a perfect linear
                  negative relation between f1 and f2.
        - Description
            Calculates the mean square error (MSE), the peak signal noise
            ratio (PSNR), and the correlation of the product of the moments
            of Pearson (PC), between two images.
        - Examples
            #
            f1 = Numeric.array([[2,5,3],[4,1,2]])
            print f1
            f2 = Numeric.array([[4,9,5],[8,3,3]])
            print f2
            (mse, psnr, pc) = iastat(f1, f2)
            print mse
            print psnr
            print pc
    """
    from Numeric import ravel, sum, product, log10, sqrt

    f1_, f2_ = 1.*ravel(f1), 1.*ravel(f2)
    #-- MSE --
    MSE = sum((f1_-f2_)**2)/product(f1.shape[:2])
    #-- PSNR --
    if MSE == 0:
        PSNR = 1E309 # infinito
    else:
        PSNR= 10*log10(255./MSE)
        PSNR = 0.0007816*(PSNR**2) - 0.06953*PSNR + 1.5789
    #-- PC --
    N = len(f1_)
    r1 = sum(f1_*f2_) - sum(f1_)*sum(f2_)/N
    r2 = sqrt((sum(f1_**2)-(sum(f1_)**2)/N)*(sum(f2_**2)-(sum(f2_)**2)/N))
    PC = r1/r2
    return MSE, PSNR, PC
#
# =====================================================================
#
#   iacolorhist
#
# =====================================================================
def iacolorhist(f, mask=None):
    """
        - Purpose
            Color-image histogram.
        - Synopsis
            hc = iacolorhist(f, mask=None)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image (logical).
            mask: Binary image (logical). Default: None.
        - Output
            hc: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Compute the histogram of a color image and return a graphical
            image suitable for visualization with the 3 marginal histograms:
            red-green at top-left, blue-green at top-right and red-blue at
            bottom-left. If the optional mask image is available, the
            histogram is computed only for those pixels under the mask.
        - Examples
            #
            f = iaread('boat.ppm')
            iashow(f)
            hc = iacolorhist(f)
            iashow(hc)
            iashow(Numeric.log(hc+1))
    """
    from Numeric import asarray, nonzero, ravel, zeros, Int32, ones, transpose, put, NewAxis

    WFRAME=5
    f = asarray(f)
    if len(f.shape) == 1: f = f[NewAxis,:]
    if not f.typecode() == 'b':
      iaerror('error, can only process uint8 images')
      return
    if not f.shape[0] == 3:
      iaerror('error, can only process 3-band images')
      return
    r,g,b = 1.*f[0,:,:], 1.*f[1,:,:], 1.*f[2,:,:]
    n_zeros = 0
    if mask:
      n_zeros = f.shape[0]*f.shape[1]-len(nonzero(ravel(mask)))
      r,g,b = mask*r, mask*g, mask*b
    hrg = zeros((256,256), Int32); hbg=hrg+0; hrb=hrg+0
    img = 256*r + g; m1 = max(ravel(img))
    aux = iahistogram(img.astype(Int32)); aux[0] = aux[0] - n_zeros
    put(ravel(hrg), range(m1+1), aux)
    img = 256*b + g; m2 = max(ravel(img))
    aux = iahistogram(img.astype(Int32)); aux[0] = aux[0] - n_zeros
    put(ravel(hbg), range(m2+1), aux)
    img = 256*r + b; m3 = max(ravel(img))
    aux = iahistogram(img.astype(Int32)); aux[0] = aux[0] - n_zeros
    put(ravel(hrb), range(m3+1), aux)
    m=max(max(ravel(hrg)),max(ravel(hbg)),max(ravel(hrb)))
    hc=m*ones((3*WFRAME+2*256,3*WFRAME+2*256))
    hc[WFRAME:WFRAME+256,WFRAME:WFRAME+256] = transpose(hrg)
    hc[WFRAME:WFRAME+256,2*WFRAME+256:2*WFRAME+512] = transpose(hbg)
    hc[2*WFRAME+256:2*WFRAME+512,WFRAME:WFRAME+256] = transpose(hrb)
    return hc
#
# =====================================================================
#
#   iahwt
#
# =====================================================================
def iahwt(f):
    """
        - Purpose
            Haar Wavelet Transform.
        - Synopsis
            F = iahwt(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            F = iahwt(f)
            iashow(Numeric.log(abs(F)+1))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    A = iahaarmatrix(m)
    if A==-1: return -1
    if (n == 1):
        F = matrixmultiply(A, f)
    else:
        B = iahaarmatrix(n)
        if B==-1: return -1
        F = matrixmultiply(matrixmultiply(A, f), transpose(B))
    return F
#
# =====================================================================
#
#   iaidct
#
# =====================================================================
def iaidct(f):
    """
        - Purpose
            Inverse Discrete Cossine Transform.
        - Synopsis
            F = iaidct(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = iaread('cameraman.pgm')
            iashow(f)
            F = iadct(f)
            g = iaidct(F)
            print Numeric.sum(Numeric.sum(abs(f-g)))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    if (n == 1):
        A = iadctmatrix(m)
        F = matrixmultiply(transpose(A), f)
    else:
        A=iadctmatrix(m)
        B=iadctmatrix(n)
        F = matrixmultiply(matrixmultiply(transpose(A), f), B)
    return F
#
# =====================================================================
#
#   iaidft
#
# =====================================================================
def iaidft(F):
    """
        - Purpose
            Inverse Discrete Fourier Transform.
        - Synopsis
            f = iaidft(F)
        - Input
            F: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            f: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('cameraman.pgm')
            F = iadft(f)
            H = iagaussian(F.shape, Numeric.array(F.shape)/2., [[50,0],[0,50]])
            H = ianormalize(H,[0,1])
            FH = F * iaifftshift(H)
            print iaisdftsym(FH)
            g=iaidft(FH)
            iashow(f)
            iashow(iadftview(F))
            iashow(ianormalize(H,[0,255]))
            iashow(iadftview(FH))
            iashow(abs(g))
    """
    from Numeric import asarray, Complex64, NewAxis, matrixmultiply, conjugate, sqrt

    F = asarray(F).astype(Complex64)
    if len(F.shape) == 1: F = F[:,NewAxis]
    (m, n) = F.shape
    if (n == 1):
        A = iadftmatrix(m)
        f = (matrixmultiply(conjugate(A), F))/sqrt(m)
    else:
        A = iadftmatrix(m)
        B = iadftmatrix(n)
        f = (matrixmultiply(matrixmultiply(conjugate(A), F), conjugate(B)))/sqrt(m*n)
    return f
#
# =====================================================================
#
#   iaihadamard
#
# =====================================================================
def iaihadamard(f):
    """
        - Purpose
            Inverse Hadamard Transform.
        - Synopsis
            F = iaihadamard(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            F = iahadamard(f)
            g = iaihadamard(F)
            print Numeric.sum(Numeric.sum(abs(f.astype(Numeric.Float)-g.astype(Numeric.Float))))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    A = iahadamardmatrix(m)
    if A==-1: return -1
    if (n == 1):
        F = matrixmultiply(transpose(A), f)
    else:
        B = iahadamardmatrix(n)
        if B==-1: return -1
        F = matrixmultiply(matrixmultiply(transpose(A), f), B)
    return F
#
# =====================================================================
#
#   iaihwt
#
# =====================================================================
def iaihwt(f):
    """
        - Purpose
            Inverse Haar Wavelet Transform.
        - Synopsis
            F = iaihwt(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
        - Output
            F: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            F = iahwt(f)
            g = iaihwt(F)
            print Numeric.sum(Numeric.sum(abs(f.astype(Numeric.Float)-g.astype(Numeric.Float))))
    """
    from Numeric import asarray, Float64, NewAxis, matrixmultiply, transpose

    f = asarray(f).astype(Float64)
    if len(f.shape) == 1: f = f[:,NewAxis]
    (m, n) = f.shape
    A = iahaarmatrix(m)
    if A==-1: return -1
    if (n == 1):
        F = matrixmultiply(transpose(A), f)
    else:
        B = iahaarmatrix(n)
        if B==-1: return -1
        F = matrixmultiply(matrixmultiply(transpose(A), f), B)
    return F
#
# =====================================================================
#
#   iaind2sub
#
# =====================================================================
def iaind2sub(dim, i):
    """
        - Purpose
            Convert linear index to double subscripts.
        - Synopsis
            x, y = iaind2sub(dim, i)
        - Input
            dim: Gray-scale (uint8 or uint16) or binary image (logical).
                 Dimension.
            i:   Gray-scale (uint8 or uint16) or binary image (logical).
                 Index.
        - Output
            x: Gray-scale (uint8 or uint16) or binary image (logical).
            y: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = Numeric.array([[0,6,0,2],[4,0,1,8],[0,0,3,0]])
            print f
            i = Numeric.nonzero(Numeric.ravel(f))
            (x,y) = iaind2sub(f.shape, i)
            print x
            print y
            print f[x[0],y[0]]
            print f[x[4],y[4]]
    """
    from Numeric import asarray

    i = asarray(i)
    x = i / dim[1]
    y = i % dim[1]
    return x, y
#
# =====================================================================
#
#   iaisdftsym
#
# =====================================================================
def iaisdftsym(F):
    """
        - Purpose
            Check for conjugate symmetry
        - Synopsis
            b = iaisdftsym(F)
        - Input
            F: Gray-scale (uint8 or uint16) or binary image (logical).
               Complex image.
        - Output
            b: Boolean.
        - Description
            Verify if a complex array show the conjugate simmetry. Due to
            numerical precision, this comparison is not exact but with
            within a small tolerance (10E-4). This comparison is useful to
            verify if the result of a filtering in the frequency domain is
            correct. Before taking the inverse DCT, the Fourier transform
            must be conjugate symmetric so that its inverse is a real
            function (an image).
        - Examples
            #
            #   example 1
            #
            import FFT, matplotlib.mlab
            print iaisdftsym(FFT.fft2d(matplotlib.mlab.rand(100,100)))
            print iaisdftsym(FFT.fft2d(matplotlib.mlab.rand(101,100)))
            print iaisdftsym(FFT.fft2d(matplotlib.mlab.rand(101,101)))
            #
            #   example 2
            #
            print iaisdftsym(iabwlp([10,10], 8, 5))
            print iaisdftsym(iabwlp([11,11], 8, 5))
    """
    from Numeric import asarray, Complex64, NewAxis, reshape, take, ravel, conjugate, alltrue

    F = (asarray(F)).astype(Complex64)
    if len(F.shape) == 1: F = F[NewAxis,:]
    global rows, cols
    (rows, cols) = F.shape
    (x, y) = iameshgrid(range(cols), range(rows))
    yx = iasub2ind(F.shape, y, x)
    aux1 = reshape(map(lambda k:k%rows, -ravel(y)), (rows, cols))
    aux2 = reshape(map(lambda k:k%cols, -ravel(x)), (rows, cols))
    myx = iasub2ind(F.shape, aux1, aux2)
    is_ = abs( take(ravel(F), yx) - \
          conjugate(take(ravel(F), myx)) ) < 10E-4
    b = alltrue(ravel(is_))
    return b
#
# =====================================================================
#
#   ialabel
#
# =====================================================================
def ialabel(f):
    """
        - Purpose
            Label a binary image.
        - Synopsis
            g = ialabel(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
        - Output
            g: Image.
        - Description
            Creates an image by labeling the connect components of the input
            binary image. The background pixels (with value 0) are not
            labeled. The maximum label value in the output image gives the
            number of its connected components.
        - Examples
            #
            #   example 1
            #
            f = Numeric.array([[0,1,0,1,1], [1,0,0,1,0]])
            print f
            g = ialabel(f)
            print g
            #
            #   example 2
            #
            f = iaread('blobs.pbm')
            g = ialabel(f);
            nblobs = max(Numeric.ravel(g))
            print nblobs
            iashow(f)
            iashow(ialblshow(g))
    """
    from Numeric import zeros, nonzero, ravel

    faux = 1*f
    g    = zeros(faux.shape)
    gaux = zeros(faux.shape)
    i = nonzero(ravel(faux))
    k = 1
    while len(i):
        aux = iaind2sub(faux.shape, i[0])
        gaux = iarec(faux, (aux[0],aux[1]))
        faux = faux - gaux
        g = g + k*gaux
        k += 1
        i = nonzero(ravel(faux))
    return g
#
# =====================================================================
#
#   ialblshow
#
# =====================================================================
def ialblshow(f):
    """
        - Purpose
            Display a labeled image assigning a random color for each label.
        - Synopsis
            g = ialblshow(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
        - Output
            g: Image.
        - Description
            Displays the labeled input image (uint8 or uint16) with a pseudo
            color where each label appears with a random color.
        - Examples
            #
            f = iaread('blobs.pbm')
            g = ialabel(f);
            iashow(g)
            iashow(ialblshow(g))
    """
    from Numeric import ravel, floor, concatenate
    from matplotlib.mlab import rand

    nblobs = max(ravel(f))
    r = floor(0.5 + 255*rand(nblobs, 1))
    g = floor(0.5 + 255*rand(nblobs, 1))
    b = floor(0.5 + 255*rand(nblobs, 1))
    ct = concatenate((r,g,b), 1)
    ct = concatenate(([[0,0,0]], ct))
    g = iaapplylut(f, ct)
    return g
#
# =====================================================================
#
#   ialog
#
# =====================================================================
def ialog(s, mu, sigma):
    """
        - Purpose
            Laplacian of Gaussian image.
        - Synopsis
            g = ialog(s, mu, sigma)
        - Input
            s:     Gray-scale (uint8 or uint16) or binary image (logical).
                   [rows cols], output image dimensions.
            mu:    Gray-scale (uint8 or uint16) or binary image (logical).
                   [row0 col0], center of the function.
            sigma: Non-negative integer. spread factor.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Creates a Laplacian of Gaussian image with dimensions given by
            s, origin given by c and spreading factor given by sigma. This
            function is used in the Marr-Hildreth filter.
        - Examples
            #
            #   example 1
            #
            F = ialog([5,7], [3,4], 1)
            print Numeric.array2string(F, precision=4, suppress_small=1)
            #
            #   example 2
            #
            h = ialog([200,300], [100,150], 20)
            iashow(h)
            g,d = iaplot(h[100,:])
            showfig(h[100,:])
    """
    from Numeric import array, product, shape, arange, ravel, reshape, pi, exp

    def test_exp(x,sigma):
       from Numeric import exp
       try:
          return (exp(-(x / (2. * sigma**2))))
       except:
          return 0
    mu = array(mu)
    if product(shape(s)) == 1:
       x = arange(s)
       r2 = (x-mu)**2
    else:
       (x, y) = iameshgrid(range(s[1]), range(s[0]))
       r2 = (x-mu[1])**2  + (y-mu[0])**2
    r2_aux = ravel(r2)
    aux = reshape(map(test_exp, r2_aux, 0*r2_aux+sigma), r2.shape)
    g = -(((r2 - 2 * sigma**2) / (sigma**4 * pi)) * aux)
    return g
#
# =====================================================================
#
#   ialogfilter
#
# =====================================================================
def ialogfilter(f, sigma):
    """
        - Purpose
            Laplacian of Gaussian filter.
        - Synopsis
            g = ialogfilter(f, sigma)
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image (logical).
                   input image
            sigma: Non-negative integer. scaling factor
        - Output
            g: Image.
        - Description
            Filters the image f by the Laplacian of Gaussian (LoG) filter
            with parameter sigma. This filter is also known as the
            Marr-Hildreth filter. Obs: to better efficiency, this
            implementation computes the filter in the frequency domain.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            g07 = ialogfilter(f, 0.7)
            iashow(g07)
            iashow(g07 > 0)
            #
            #   example 2
            #
            import Numeric
            g5 = ialogfilter(f, 5)
            iashow(g5)
            iashow(g5 > 0)
            g10 = ialogfilter(f, 10)
            iashow(g10)
            iashow(g10 > 0)
    """
    from Numeric import shape, NewAxis, array
    from FFT import fft2d, inverse_fft2d

    if len(shape(f)) == 1: f = f[NewAxis,:]
    h = ialog(shape(f), map(int, array(shape(f))/2.), sigma)
    h = iaifftshift(h)
    H = fft2d(h)
    if not iaisdftsym(H):
       iaerror("error: log filter is not symetrical")
       return None
    G = fft2d(f) * H
    g = inverse_fft2d(G).real
    return g
#
# =====================================================================
#
#   iameshgrid
#
# =====================================================================
def iameshgrid(vx, vy):
    """
        - Purpose
            Create two 2-D matrices of indexes.
        - Synopsis
            x, y = iameshgrid(vx, vy)
        - Input
            vx: Gray-scale (uint8 or uint16) or binary image (logical).
                Vector of indices of x coordinate.
            vy: Gray-scale (uint8 or uint16) or binary image (logical).
                Vector of indices of y coordinate.
        - Output
            x: Gray-scale (uint8 or uint16) or binary image (logical). 2-D
               matrix of indexes of x coordinate.
            y: Gray-scale (uint8 or uint16) or binary image (logical). 2-D
               matrix of indexes of y coordinate.
        - Description
            This function generates 2-D matrices of indices of the domain
            specified by arange1 and arange2. This is very useful to
            generate 2-D functions. Note that unlike other functions, the
            order of the parameters uses the cartesian coordenate
            convention. arange1 is for x (horizontal), and arange2 is for y
            (vertical).
        - Examples
            #
            #   example 1
            #
            (x, y) = iameshgrid(Numeric.arange(1,3,0.5), Numeric.arange(2,4,0.6))
            print x
            print y
            print x + y
            #
            #   example 2
            #
            (x, y) = iameshgrid(range(256), range(256))
            iashow(x)
            iashow(y)
            iashow(x + y)
            z = Numeric.sqrt((x-127)**2 + (y-127)**2)
            iashow(z)
    """
    from Numeric import resize, transpose

    x = resize(vx, (len(vy), len(vx)))
    y = transpose(resize(vy, (len(vx), len(vy))))
    return x, y
#
# =====================================================================
#
#   ianeg
#
# =====================================================================
def ianeg(f):
    """
        - Purpose
            Negate an image.
        - Synopsis
            g = ianeg(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). Set
               initial.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Returns an image that is the negation (i.e., inverse or
            involution) of the input image.
        - Examples
            #
            c1 = Numeric.array([0, 53, 150, 255], Numeric.UnsignedInt8)
            print c1
            n1 = ianeg(c1)
            print n1
            c2 = Numeric.array([-129, -128, 0, 127, 128], Numeric.Int8)
            print c2
            n2 = ianeg(c2)
            print n2
    """
    from Numeric import asarray, Float

    f = asarray(f)
    if f.typecode() in ['b','???']: # (numarray implementara UnsignedInt16 e boolean)
        k = 2**(8*f.itemsize()) - 1
        g = k - f
    else: # Trata os tipos com sinal
        g = -f.astype(Float)
    return g
#
# =====================================================================
#
#   ianormalize
#
# =====================================================================
def ianormalize(f, range):
    """
        - Purpose
            Normalize the pixels values between the specified range.
        - Synopsis
            g = ianormalize(f, range)
        - Input
            f:     Gray-scale (uint8 or uint16) or binary image (logical).
                   input image.
            range: Gray-scale (uint8 or uint16) or binary image (logical).
                   vector: minimum and maximum values in the output image,
                   respectively.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
               normalized image.
        - Description
            Normalize the input image f. The minimum value of f is assigned
            to the minimum desired value and the maximum value of f, to the
            maximum desired value. The minimum and maximum desired values
            are given by the parameter range.
        - Examples
            #
            import Numeric
            f = Numeric.array([100., 500., 1000.])
            g1 = ianormalize(f, [0,255])
            print g1
            g2 = ianormalize(f, [-1,1])
            print g2
            g3 = ianormalize(f, [0,1])
            print g3
            #
            f = Numeric.array([-100., 0., 100.])
            g4 = ianormalize(f, [0,255])
            print g4
            g5 = ianormalize(f, [-1,1])
            print g5
    """
    from Numeric import asarray, ravel, Float, ones, reshape

    f = asarray(f)
    if f.typecode() in ['D', 'F']:
        iaerror('error: cannot normalize complex data')
        return None
    faux = ravel(f).astype(Float)
    minimum = min(faux)
    maximum = max(faux)
    lower = range[0]
    upper = range[1]
    if upper == lower:
        iaerror('error: image is constant, cannot normalize')
        return f
    if minimum == maximum:
        g = ones(f.shape)*(upper + lower) / 2.
    else:
        g = (faux-minimum)*(upper-lower) / (maximum-minimum) + lower
    g = reshape(g, f.shape)
    T = f.typecode()
    if T == 'b': # UnsignedInt8
        if upper > 255:
            iaerror('ianormalize: warning, upper valuer larger than 255. Cannot fit in uint8 image')
        g = g.astype('b')
    ### Nao ha' uint16 no Numeric (foi implementado no numarray)
    return g
#
# =====================================================================
#
#   iaotsu
#
# =====================================================================
def iaotsu(f):
    """
        - Purpose
            Thresholding by Otsu.
        - Synopsis
            t eta = iaotsu(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image.
        - Output
            t eta: Maximum of this result is the thresholding.
        - Description
            Compute the automatic thresholding level of a gray scale image
            based on the Otsu method.
        - Examples
            #
            #   example 1
            #
            #
            #   example 2
            #
            import Numeric
            f = iaread('cookies.pgm')
            iashow(f)
            (t, eta) = iaotsu(f)
            print 'threshold at %f, goodness=%f\n' %(t, eta)
            iashow(f > t)
    """
    from Numeric import product, shape, arange, cumsum, nonzero, sum

    n = product(shape(f))
    h = 1.*iahistogram(f) / n
    x = arange(product(shape(h)))
    w0 = cumsum(h)
    w1 = 1 - w0
    eps = 1e-10
    m0 = cumsum(x * h) / (w0 + eps)
    mt = m0[-1]
    m1 = (mt - m0[0:-1]*w0[0:-1]) / w1[0:-1]
    sB2 = w0[0:-1] * w1[0:-1] * ((m0[0:-1] - m1)**2)
    v = max(sB2)
    t = nonzero(sB2 == v)[0]
    st2 = sum((x-mt)**2 * h)
    eta = v / st2
    return t, eta
#
# =====================================================================
#
#   iacontour
#
# =====================================================================
def iacontour(f):
    """
        - Purpose
            Contours of binary images.
        - Synopsis
            g = iacontour(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
        - Output
            g: Image.
        - Description
            Contours of binary images.
        - Examples
            #
            f = iaread('blobs.pbm')
            g = iacontour(f)
            iashow(f)
            iashow(g)
    """
    from Numeric import array, zeros, logical_and

    f_ = f > 0
    new_shape = array(f_.shape)+2
    n = zeros(new_shape); n[0:-2,1:-1] = f_
    s = zeros(new_shape); s[2:: ,1:-1] = f_
    w = zeros(new_shape); w[1:-1,0:-2] = f_
    e = zeros(new_shape); e[1:-1,2:: ] = f_
    fi = logical_and(logical_and(logical_and(n,s),w),e)
    fi = fi[1:-1,1:-1]
    g = f_ - fi
    g = g > 0
    return g
#
# =====================================================================
#
#   iaconv
#
# =====================================================================
def iaconv(f, h):
    """
        - Purpose
            2D convolution.
        - Synopsis
            g = iaconv(f, h)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image.
            h: Gray-scale (uint8 or uint16) or binary image (logical). PSF
               (point spread function), or kernel. The origin is at the
               array center.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Perform a 2D discrete convolution. The kernel origin is at the
            center of image h.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = Numeric.zeros((5,5))
            f[2,2] = 1
            print f
            h = Numeric.array([[1,2,3],[4,5,6]])
            print h
            a = iaconv(f,h)
            print a
            #
            #   example 2
            #
            f = Numeric.array([[1,0,0,0],[0,0,0,0]])
            print f
            h = Numeric.array([1,2,3])
            print h
            a = iaconv(f,h)
            print a
            #
            #   example 3
            #
            f = Numeric.array([[1,0,0,0,0,0],[0,0,0,0,0,0]])
            print f
            h = Numeric.array([1,2,3,4])
            print h
            a = iaconv(f,h)
            print a
            #
            #   example 4
            #
            f = iaread('cameraman.pgm')
            h = [[1,2,1],[0,0,0],[-1,-2,-1]]
            g = iaconv(f,h)
            gn = ianormalize(g, [0,255])
            iashow(gn)
    """
    from Numeric import asarray, NewAxis, zeros, array, product

    f, h = asarray(f), asarray(h)
    if len(f.shape) == 1: f = f[NewAxis,:]
    if len(h.shape) == 1: h = h[NewAxis,:]
    if product(f.shape) < product(h.shape):
        f, h = h, f
    g = zeros(array(f.shape) + array(h.shape) - 1)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            g[i:i+f.shape[0], j:j+f.shape[1]] += h[i,j] * f
    return g
#
# =====================================================================
#
#   iapconv
#
# =====================================================================
def iapconv(f, h):
    """
        - Purpose
            2D Periodic convolution.
        - Synopsis
            g = iapconv(f, h)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). Input
               image.
            h: Gray-scale (uint8 or uint16) or binary image (logical). PSF
               (point spread function), or kernel. The origin is at the
               array center.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Perform a 2D discrete periodic convolution. The kernel origin is
            at the center of image h. Both image and kernel are periodic
            with same period. Usually the kernel h is smaller than the image
            f, so h is padded with zero until the size of f.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = Numeric.zeros((5,5))
            f[2,2] = 1
            print f
            h = Numeric.array([[1,2,3],[4,5,6]])
            print h
            a = iapconv(f,h)
            print a
            #
            #   example 2
            #
            f = Numeric.array([[1,0,0,0],[0,0,0,0]])
            print f
            h = Numeric.array([1,2,3])
            print h
            a = iapconv(f,h)
            print a
            #
            #   example 3
            #
            f = Numeric.array([[1,0,0,0,0,0],[0,0,0,0,0,0]])
            print f
            h = Numeric.array([1,2,3,4])
            print h
            a = iapconv(f,h)
            print a
            #
            #   example 4
            #
            f = iaread('cameraman.pgm')
            h = [[1,2,1],[0,0,0],[-1,-2,-1]]
            g = iapconv(f,h)
            gn = ianormalize(g, [0,255])
            iashow(gn)
    """
    from Numeric import asarray, ravel, NewAxis, Float64, concatenate, zeros

    f, h = asarray(f), asarray(h)
    faux, haux = ravel(f), ravel(h)
    if len(f.shape) == 1: f = f[NewAxis,:]
    if len(h.shape) == 1: h = h[NewAxis,:]
    (rows, cols) = f.shape
    (hrows, hcols) = h.shape
    f = f.astype(Float64)
    h = h.astype(Float64)
    dr1 = int((hrows-1)/2.)
    dr2 = hrows-dr1
    dc1 = int((hcols-1)/2.)
    dc2 = hcols-dc1
    p = concatenate((concatenate((f[-dr2+1::,:], f)), f[0:dr1,:])) # Insert lines above and below periodicly
    p = concatenate((concatenate((p[:,-dc2+1::], p), 1), p[:,0:dc1]), 1) # Insert columns at left and right periodcly
    g = zeros((rows,cols))
    for r in range(hrows):
       for c in range(hcols):
          hw = h[hrows-r-1,hcols-c-1]
          if (hw):
            g = g + h[hrows-r-1,hcols-c-1] * p[r:rows+r,c:cols+c]
    return g
#
# =====================================================================
#
#   iacomb
#
# =====================================================================
def iacomb(s, delta, offset):
    """
        - Purpose
            Create a grid of impulses image.
        - Synopsis
            g = iacomb(s, delta, offset)
        - Input
            s:      Gray-scale (uint8 or uint16) or binary image (logical).
                    output image dimensions (1-D, 2-D or 3-D).
            delta:  Gray-scale (uint8 or uint16) or binary image (logical).
                    interval between the impulses in each dimension (1-D,
                    2-D or 3-D).
            offset: Gray-scale (uint8 or uint16) or binary image (logical).
                    offset in each dimension (1-D, 2-D or 3-D).
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            #   example 1
            #
            u1 = iacomb(10, 3, 2)
            print u1
            #
            #   example 2
            #
            u2 = iacomb((7,9), (3,4), (3,2))
            print u2
            #
            #   example 3
            #
            u3 = iacomb((7,9,4), (3,4,2), (2,2,1))
            print u3[:,:,0]
            print u3[:,:,1]
            print u3[:,:,2]
            print u3[:,:,3]
    """
    from Numeric import asarray, product, zeros

    s = asarray(s)
    if product(s.shape) == 1:
        g = zeros(s) 
        g[offset::delta] = 1
    elif len(s) >= 2:
        g = zeros((s[0], s[1]))
        g[offset[0]::delta[0], offset[1]::delta[1]] = 1
    if len(s) == 3:
        aux = zeros(s)
        for i in range(offset[2], s[2], delta[2]):
            aux[:,:,i] = g
        g = aux
    return g
#
# =====================================================================
#
#   iaplot
#
# =====================================================================
def iaplot(x=0, y=None, filename=None):
    """
        - Purpose
            Plot a function.
        - Synopsis
            g, d = iaplot(x=0, y=None, filename=None)
        - Input
            x:        Gray-scale (uint8 or uint16) or binary image
                      (logical). Default: 0. x.
            y:        Gray-scale (uint8 or uint16) or binary image
                      (logical). Default: None. f(x).
            filename: Default: None. Name of the postscript file.
        - Output
            g: Gnuplot pointer.
            d: Gnuplot data.
        - Description
            Plot a 2D function y=f(x).
        - Examples
            #
            import Numeric
            x = Numeric.arange(0, 2*Numeric.pi, 0.1)
            y = Numeric.sin(x)
            g, d = iaplot(x,y)
            g('set data style impulses')
            g('set grid')
            g('set xlabel "X values" -20,0')
            g('set ylabel "Y values"')
            g('set title "Example Plot"')
            g.plot(d, 'cos(x)')
            showfig(d, 'cos(x)')
    """
    import Numeric

    try:
        import Gnuplot
        g = Gnuplot.Gnuplot(debug=1)
        g('set data style lines')
        x = Numeric.ravel(x)
        if not y:
            y = x
            x = Numeric.arange(len(y))
        else:
            y = Numeric.ravel(y)
        d = Gnuplot.Data(x, y)
        g.plot(d)
        if filename:
            g.hardcopy(filename, mode='portrait', enhanced=1, color=1)
    except:
        g,d = None,None
    return g, d
#
# =====================================================================
#
#   iasplot
#
# =====================================================================
def iasplot(x=0, y=None, z=None, filename=None):
    """
        - Purpose
            Plot a surface.
        - Synopsis
            g, d = iasplot(x=0, y=None, z=None, filename=None)
        - Input
            x:        Gray-scale (uint8 or uint16) or binary image
                      (logical). Default: 0. x range.
            y:        Gray-scale (uint8 or uint16) or binary image
                      (logical). Default: None. y range.
            z:        Gray-scale (uint8 or uint16) or binary image
                      (logical). Default: None. object function (z=f(x,y)).
            filename: Default: None. Name of the postscript file.
        - Output
            g: Gnuplot pointer.
            d: Gnuplot data.
        - Description
            Plot a 3D function z=f(x,y).
        - Examples
            #
            #   example 1
            #
            x = Numeric.arange(35)/2.0
            y = Numeric.arange(30)/10.0 - 1.5
            def z(x,y): return 1.0 / (1 + 0.01 * x**2 + 0.5 * y**2)
            g,d = iasplot(x, y, z)
            showfig(z)
            #
            #   example 2
            #
            f = iaread('lenina.pgm')
            k = f[90:195, 70:150]
            iashow(k)
            g,d = iasplot(k)
            g('set view 30,60')
            g.splot(d) # surface
            showfig(Surface of k)
            g('set view 0,90')
            g('set nosurface') # level curves
            g.splot(d)
            showfig(Level curves of k)
    """
    import Numeric

    try:
        import Gnuplot
        import Gnuplot.funcutils
        g = Gnuplot.Gnuplot(debug=1)
        g('set parametric')
        g('set data style lines')
        g('set hidden')
        g('set contour base')
        f = Numeric.asarray(x)
        if y is None:
            x = range(f.shape[0])
            y = range(f.shape[1])
            def z(x, y, f=f):
                return f[int(x),int(y)]
        d = Gnuplot.funcutils.compute_GridData(x, y, z, binary=0)
        g.splot(d)
        if filename:
            g.hardcopy(filename, mode='portrait', enhanced=1, color=1)
    except:
        g,d = None,None
    return g, d
#
# =====================================================================
#
#   iaptrans
#
# =====================================================================
def iaptrans(f, t):
    """
        - Purpose
            Periodic translation.
        - Synopsis
            g = iaptrans(f, t)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
            t: Image. [rows cols] to translate.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Translate the image periodically by t=[r0 c0]. This translation
            can be seen as a window view displacement on an infinite tile
            wall where each tile is a copy of the original image. The
            periodical translation is related to the periodic convolution
            and discrete Fourier transform. Be careful when implementing
            this function using the mod, some mod implementations in C does
            not follow the correct definition when the number is negative.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = Numeric.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
            print f
            print iaptrans(f, [-1,2]).astype(Numeric.UnsignedInt8)
            #
            #   example 2
            #
            import Numeric
            f = iaread('cameraman.pgm')
            iashow(f)
            iashow(iaptrans(f, Numeric.array(f.shape)/2))
    """
    from Numeric import asarray, Int, zeros

    f, t = asarray(f), asarray(t).astype(Int)
    h = zeros(2*abs(t) + 1)
    r0 = abs(t[0])
    c0 = abs(t[1])
    h[t[0]+r0, t[1]+c0] = 1
    g = iapconv(f, h).astype(f.typecode())
    return g
#
# =====================================================================
#
#   iaread
#
# =====================================================================
def iaread(filename):
    """
        - Purpose
            Read an image file (PBM, PGM and PPM).
        - Synopsis
            img = iaread(filename)
        - Input
            filename:
        - Output
            img: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = iaread('boat.ppm')
            iashow(f)
    """
    import Numeric
    import string
    import array
    import os
    import sys

    def type(filename):
        import string
        is_pbm, is_pgm, is_ppm = 0, 0, 0
        try:
            import imghdr
            aux = imghdr.what(filename)
            if   aux == 'pbm': is_pbm = 1
            elif aux == 'pgm': is_pgm = 1
            elif aux == 'ppm': is_ppm = 1
        except:
            if   string.find(string.upper(filename), '.PBM') != -1: is_pbm = 1
            elif string.find(string.upper(filename), '.PGM') != -1: is_pgm = 1
            elif string.find(string.upper(filename), '.PPM') != -1: is_ppm = 1
        return is_pbm, is_pgm, is_ppm
    def token(filed):
        token = ''
        c = filed.read(1)
        while (c == ' ') or (c == '\n'):
            c = filed.read(1)
        while (c != ' ') and (c != '\n') and (c != ''):
            token = token + c
            c = filed.read(1)
        return token
    def dec2bin(decimal_number):
        a = 8*[0]
        q, r, i = decimal_number, 0, -1
        while q > 1:
            r = q % 2   # resto
            q = q / 2   # quociente
            a[i] = r; i = i - 1
        a[i] = q
        return a
    try:
        un = len(os.environ['USER'])
        if filename[0:1+un] == '~' + os.environ['USER']:
             filename = os.environ['HOME'] + filename[1+un:]
        elif filename[0:2] == '~/':
             filename = os.environ['HOME'] + filename[1:]
    except:
        pass
    filename_final = filename
    f = -1
    if os.path.isfile(filename):
         f = open(filename, 'rb')
    else:
         for a in sys.path:
              if os.path.isfile(os.path.join(a, filename)):
                   filename_final = os.path.join(a, filename)
                   f = open(filename_final, 'rb')
                   break
    if f == -1:
         iaerror('error: file is not found!')
         return 0
    img = []
    is_pbm, is_pgm, is_ppm = type(filename_final)
    if is_pbm or is_pgm or is_ppm:
        TIP = string.strip(token(f))
        aux = token(f)
        while aux == '#':
            f.readline()
            aux = token(f)
        w, h = int(aux), int(token(f))
        if not is_pbm:   # se nao for PBM
            MAX = int(token(f))   # intensidade maxima
        if (is_pbm and (TIP == 'P1')) or (is_pgm != -1 and (TIP == 'P2')):   # texto
            IMG = map(int, string.split(f.read()))  # string -> inteiro
            img = Numeric.reshape(Numeric.asarray(IMG), (h, w))    # lista -> array, no tamanho original
            if is_pbm:
                img = 1 - img
        elif (is_ppm and (TIP == 'P3')):    # texto
            IMG = map(int, string.split(f.read()))  # string -> inteiro
            aux = Numeric.reshape(Numeric.asarray(IMG), (h, w, 3))    # lista -> array, no tamanho original
            img = Numeric.zeros((3, h, w))
            img[0,:,:], img[1,:,:], img[2,:,:] = aux[:,:,0], aux[:,:,1], aux[:,:,2]
        elif (is_pbm and (TIP == 'P4')):    # binario
            IMG = array.array('B', f.read()).tolist()
            img = Numeric.reshape(Numeric.asarray(map(dec2bin, IMG)), (h, w))
            img = 1 - img
        elif (is_pgm and (TIP == 'P5')):    # binario
            IMG = array.array('B', f.read()).tolist()
            img = Numeric.reshape(Numeric.asarray(IMG), (h, w))
        elif (is_ppm and (TIP == 'P6')):    # binario
            IMG = array.array('B', f.read()).tolist()            
            aux = Numeric.reshape(Numeric.asarray(IMG), (h, w, 3))    # lista -> array, no tamanho original
            img = Numeric.zeros((3, h, w))
            img[0,:,:], img[1,:,:], img[2,:,:] = aux[:,:,0], aux[:,:,1], aux[:,:,2]
        else:
            iaerror(('Tag not supported',TIP))
            return
        f.close()
        img = img.astype(Numeric.UnsignedInt8)
    else:
        iaerror('error: file format is not supported!')
    return img
#
# =====================================================================
#
#   iawrite
#
# =====================================================================
def iawrite(arrayname, filename, mode='bin'):
    """
        - Purpose
            Write an image file (PBM, PGM and PPM).
        - Synopsis
            iawrite(arrayname, filename, mode='bin')
        - Input
            arrayname: Gray-scale (uint8 or uint16) or binary image
                       (logical).
            filename:  
            mode:      Default: 'bin'.

        - Examples
            #
            f = Numeric.resize(range(256), (256, 256))
            f_color = Numeric.zeros((256, 256, 3))
            f_color[:,:,0] = f; f_color[:,:,1] = 127; f_color[:,:,2] = 255-f
            import os
            file_name = os.tempnam() # Name for a temporary file
            print file_name
            print 'Saving f in '+file_name+'.pgm ...'
            iawrite(f, os.tempnam()+'.pgm')
            print 'Saving f_color in '+file_name+'.ppm ...'
            iawrite(f_color, os.tempnam()+'.ppm')
    """
    import Numeric
    import string
    import array
    import os

    def type(filename):
        import string
        is_pbm, is_pgm, is_ppm = 0, 0, 0
        if string.find(string.upper(filename), '.PBM') != -1:
            is_pbm = 1
        elif string.find(string.upper(filename), '.PGM') != -1:
            is_pgm = 1
        elif string.find(string.upper(filename), '.PPM') != -1:
            is_ppm = 1
        return is_pbm, is_pgm, is_ppm
    def bin2asc(bin_num_list):
        a = ''
        x = [128, 64, 32, 16, 8, 4, 2, 1]
        for i in range(0, len(bin_num_list)-7, 8):
            aux1 = bin_num_list[i:i+8]
            aux2 = x[0:len(aux1)]
            a = a + chr(innerproduct(aux1, aux2))
        return a
    arrayname = Numeric.asarray(arrayname)
    if len(arrayname.shape) == 1:
        arrayname = Numeric.transpose(arrayname[NewAxis,:])
        h, w = arrayname.shape
    elif len(arrayname.shape) == 3:
        h, w = arrayname.shape[2], arrayname.shape[1]
        aux = Numeric.zeros((w, h, 3))
        aux[:,:,0], aux[:,:,1], aux[:,:,2] = arrayname[0,:,:], arrayname[1,:,:], arrayname[2,:,:]
        arrayname = Numeric.transpose(aux, (1,0,2))
    else:
        arrayname = Numeric.transpose(arrayname)
        h, w = arrayname.shape
    is_pbm, is_pgm, is_ppm = type(filename)
    h, w = arrayname.shape[0], arrayname.shape[1]
    if is_pbm or is_pgm or is_ppm:
        try:
            un = len(os.environ['USER'])
            if filename[0:1+un] == '~' + os.environ['USER']:
                 filename = os.environ['HOME'] + filename[1+un:]
            elif filename[0:2] == '~/':
                 filename = os.environ['HOME'] + filename[1:]
        except:
            pass
        f = open(filename, 'w')
        if string.find(string.upper(mode), 'ASC') != -1:
            if is_pbm or is_pgm:
                if is_pbm:
                    f.write('P1\n')
                    arrayname = Numeric.greater(arrayname, Numeric.zeros(arrayname.shape))
                    arrayname = 1 - arrayname
                else:
                    f.write('P2\n')
                f.write('# By ia636 toolbox\n')
                f.write(str(h)+' '+str(w)+'\n')
                if is_pgm:
                    f.write('255\n')
                aux = Numeric.ravel(Numeric.transpose(arrayname))
                img = str(aux)[1:-1]
                f.write(img)
            else:
                f.write('P3\n')
                f.write('# By ia636 toolbox\n')
                f.write(str(h)+' '+str(w)+'\n')
                f.write('255\n')
                aux = Numeric.ravel(Numeric.transpose(arrayname, (1,0,2)))
                img = str(aux)[1:-1]
                f.write(img)
        elif string.find(string.upper(mode), 'BIN') != -1:
            if is_pbm:
                f.write('P4\n')
                f.write('# By ia636 toolbox\n')
                f.write(str(h)+' '+str(w)+'\n')
                arrayname = Numeric.greater(arrayname, Numeric.zeros(arrayname.shape))
                arrayname = 1 - arrayname
                aux = Numeric.ravel(Numeric.transpose(arrayname))
                img = bin2asc(aux.tolist())
                f.write(img)
            elif is_pgm:
                f.write('P5\n')
                f.write('# By ia636 toolbox\n')
                f.write(str(h)+' '+str(w)+'\n')
                f.write('255\n')
                aux = Numeric.ravel(Numeric.transpose(arrayname))
                img = map(chr, aux.tolist())
                img = string.join(img, '')
                f.write(img)
            else:
                f.write('P6\n')
                f.write('# By ia636 toolbox\n')
                f.write(str(h)+' '+str(w)+'\n')
                f.write('255\n')
                aux = Numeric.ravel(Numeric.transpose(arrayname, (1,0,2)))
                img = map(chr, aux.tolist())
                img = string.join(img, '')
                f.write(img)

        f.close()
    else:
        iaerror('error: file format was not specified!')

#
# =====================================================================
#
#   iarec
#
# =====================================================================
def iarec(f, seed):
    """
        - Purpose
            Reconstruction of a connect component.
        - Synopsis
            g = iarec(f, seed)
        - Input
            f:    Gray-scale (uint8 or uint16) or binary image (logical).
                  input image
            seed: Gray-scale (uint8 or uint16) or binary image (logical).
                  seed coordinate
        - Output
            g: Image.
        - Description
            Extracts a connect component of an image by region growing from
            a seed.
        - Examples
            #
            f = Numeric.array([[0,1,0,1,1], [1,0,0,1,0]])
            print f
            g = iarec(f, [0,3])
            print g
    """

    faux = 1*f
    g = 0*f
    S = [seed,]
    while len(S):
        P = S.pop()
        x, y = P[0], P[1]
        if 0 <= x < faux.shape[0] and 0 <= y < faux.shape[1]:
            if faux[x,y]:
                faux[x,y], g[x,y] = 0, 1
                S.append((x+1, y  ))
                S.append((x-1, y  ))
                S.append((x  , y+1))
                S.append((x  , y-1))
    return g
#
# =====================================================================
#
#   iarectangle
#
# =====================================================================
def iarectangle(s, r, c):
    """
        - Purpose
            Create a binary rectangle image.
        - Synopsis
            g = iarectangle(s, r, c)
        - Input
            s: Gray-scale (uint8 or uint16) or binary image (logical). [rows
               cols], output image dimensions.
            r: Non-negative integer. [rrows ccols], rectangle image
               dimensions.
            c: Gray-scale (uint8 or uint16) or binary image (logical). [row0
               col0], center of the rectangle.
        - Output
            g: Binary image (logical).
        - Description
            Creates a binary image with dimensions given by s, rectangle
            dimensions given by r and center given by c. The pixels inside
            the rectangle are one and outside zero.
        - Examples
            #
            #   example 1
            #
            F = iarectangle([7,9], [3,2], [3,4])
            print F
            #
            #   example 2
            #
            F = iarectangle([200,300], [90,120], [70,120])
            iashow(F)
    """

    rows,  cols  = s[0], s[1]
    rrows, rcols = r[0], r[1]
    y0,    x0    = c[0], c[1]
    x, y = iameshgrid(range(cols), range(rows))
    min_row, max_row = y0-rrows/2.0, y0+rrows/2.0
    min_col, max_col = x0-rcols/2.0, x0+rcols/2.0
    g1 = (min_row <= y) & (max_row > y)
    g2 = (min_col <= x) & (max_col > x)
    g = g1 & g2
    return g
#
# =====================================================================
#
#   iashow
#
# =====================================================================
def iashow(f):
    """
        - Purpose
            Image display.
        - Synopsis
            g = iashow(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
               Image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Display an image.
        - Examples
            #
            #   example 1
            #
            f = iaread('boat.ppm')
            iashow(f)
            #
            #   example 2
            #
            f = iaread('club.pgm')
            iashow(f)
            #
            #   example 3
            #
            import Numeric
            f = Numeric.resize(range(256), (256,256))
            iashow(f)
            #
            #   example 4
            #
            import Numeric
            f = Numeric.resize(range(-100,100), (100,200))
            iashow(f)
            #
            #   example 5
            #
            f = iaread('lenina.pgm')
            aux,k = iasobel(f)
            g = aux > 100
            iashow(f)
            iashow(g)
            iashow((f,g))
            #
            #   example 6
            #
            f = iaread('lenina.pgm')
            iashow(f)
            iashow((f,f>100,f>130,f>150,f>180,f>220,f==255))
    """

    x_ = iagshow(f)
    g = None
    return g
#
# =====================================================================
#
#   iagshow
#
# =====================================================================
def iagshow(f):
    """
        - Purpose
            Matrix of the image display.
        - Synopsis
            g = iagshow(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical).
               Image.
        - Output
            g: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Return the matrix of the image display.

    """
    import Numeric
    import matplotlib.mlab
    import View

    if type(f).__name__ == 'tuple':
        it = 0
        d = Numeric.product(f[0].shape[0:2])
        for i in f:
            iaux = Numeric.ravel(i)
            nz = Numeric.nonzero(iaux)
            if 0 <= iaux <= 1:
                iaux = 255*iaux
            if it is 0: # gray
                img = Numeric.zeros((3,i.shape[0],i.shape[1]))
                img[0,:,:] = i
                img[1,:,:] = i
                img[2,:,:] = i
            elif it is 1: # red
                Numeric.put(Numeric.ravel(img), nz    , Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+  d, 0)
                Numeric.put(Numeric.ravel(img), nz+2*d, 0)
            elif it is 2: # green
                Numeric.put(Numeric.ravel(img), nz    , 0)
                Numeric.put(Numeric.ravel(img), nz+  d, Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+2*d, 0)
            elif it is 3: # blue
                Numeric.put(Numeric.ravel(img), nz    , 0)
                Numeric.put(Numeric.ravel(img), nz+  d, 0)
                Numeric.put(Numeric.ravel(img), nz+2*d, Numeric.take(iaux, nz))
            elif it is 4: # cyan
                Numeric.put(Numeric.ravel(img), nz    , 0)
                Numeric.put(Numeric.ravel(img), nz+  d, Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+2*d, Numeric.take(iaux, nz))
            elif it is 5: # magenta
                Numeric.put(Numeric.ravel(img), nz    , Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+  d, 0)
                Numeric.put(Numeric.ravel(img), nz+2*d, Numeric.take(iaux, nz))
            elif it is 6: # yellow
                Numeric.put(Numeric.ravel(img), nz    , Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+  d, Numeric.take(iaux, nz))
                Numeric.put(Numeric.ravel(img), nz+2*d, 0)
            else: # black
                Numeric.put(Numeric.ravel(img), nz    , 0)
                Numeric.put(Numeric.ravel(img), nz+  d, 0)
                Numeric.put(Numeric.ravel(img), nz+2*d, 0)
            it = it + 1
    else:
        img = f
    img = Numeric.asarray(img)
    faux = Numeric.ravel(img)
    mi = min(faux)
    ma = max(faux)
    if abs(mi - ma) < 1E-10:
        print 'Image is constant with value %3f.0\n' %(faux[0])
        me = faux[0]
        st = 0
    else:
        aux = faux.astype('d')
        me = matplotlib.mlab.detrend_mean(aux)
        st = matplotlib.mlab.std(aux)
    print img.shape, 'Min=', mi, 'Max=', ma, 'Mean=%.3f' %(me), 'Std=%.2f' %(st)
    if (mi != 0) or (ma != 255):
      g = ianormalize(img, [0,255])
    else:
      g = img
    g = g.astype('b')
    try:
        View.view(g)
    except:
        iaerror('error: visualization is not possible')
    return g
#
# =====================================================================
#
#   iasobel
#
# =====================================================================
def iasobel(f):
    """
        - Purpose
            Sobel edge detection.
        - Synopsis
            mag theta = iasobel(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
        - Output
            mag theta: Image.
        - Description
            Computes the edge detection by Sobel. Compute magnitude and
            angle.
        - Examples
            #
            #   example 1
            #
            import Numeric
            f = iaread('cameraman.pgm')
            (g,a) = iasobel(f)
            iashow(g)
            iashow(Numeric.log(g+1))
            #
            #   example 2
            #
            iashow(g > 150)
            i = Numeric.nonzero(Numeric.ravel(g > 150))
            z = Numeric.zeros(a.shape)
            Numeric.put(Numeric.ravel(z), i, Numeric.take(Numeric.ravel(a), i))
            iashow(z)
    """
    from Numeric import array, reshape, ravel, arctan2
    from matplotlib.mlab import rot90

    def test_arctan2(x,y):
        from Numeric import arctan2
        try:
            return (arctan2(x,y))
        except:
            return 0
    wx = array([[1,2,1],[0,0,0],[-1,-2,-1]])
    wy = rot90(wx)
    gx = iapconv(f, wx)
    gy = iapconv(f, wy)
    mag = abs(gx + gy*1j)
    theta = reshape(map(test_arctan2, ravel(gy), ravel(gx)), f.shape)
    return mag, theta
#
# =====================================================================
#
#   iasub2ind
#
# =====================================================================
def iasub2ind(dim, x, y):
    """
        - Purpose
            Convert linear double subscripts to linear index.
        - Synopsis
            i = iasub2ind(dim, x, y)
        - Input
            dim: Gray-scale (uint8 or uint16) or binary image (logical).
                 Dimension.
            x:   Gray-scale (uint8 or uint16) or binary image (logical). x
                 index.
            y:   Gray-scale (uint8 or uint16) or binary image (logical). y
                 index.
        - Output
            i: Gray-scale (uint8 or uint16) or binary image (logical).

        - Examples
            #
            f = Numeric.array([[0,6,0,2],[4,0,1,8],[0,0,3,0]])
            print f
            x=[0,0,1,2,2,2]
            y=[0,2,1,0,1,3]
            print x
            print y
            i = iasub2ind(f.shape, x, y)
            print i
            Numeric.put(f, i, 10)
            print f
    """
    from Numeric import asarray, Int

    x, y = asarray(x), asarray(y)
    i = x*dim[1] + y
    i = i.astype(Int)
    return i
#
# =====================================================================
#
#   iavarfilter
#
# =====================================================================
def iavarfilter(f, h):
    """
        - Purpose
            Variance filter.
        - Synopsis
            g = iavarfilter(f, h)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image.
            h: Gray-scale (uint8 or uint16) or binary image (logical).
               scaling factor.
        - Output
            g: Image.
        - Description
            Computes the variance on the neighborhood of the pixel. The
            neighborhood is given by the set marked by the kernel elements.
        - Examples
            #
            f = iaread('cameraman.pgm')
            iashow(f)
            g = iavarfilter(f, [[0,1,0],[1,1,1],[0,1,0]])
            iashow(g)
    """
    from Numeric import asarray, Float64, sum, ravel, sqrt

    f = asarray(f).astype(Float64)
    f = f + 1e-320*(f == 0) # change zero by a very small number (prevent 'math range error') 
    n = sum(ravel(h))
    fm = iapconv(f, h) / n
    f2m = iapconv(f*f, h) / n
    g = sqrt(f2m - (fm*fm)) / fm
    return g
#
# =====================================================================
#
#   iadither
#
# =====================================================================
def iadither(f, n):
    """
        - Purpose
            Ordered Dither.
        - Synopsis
            g = iadither(f, n)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
            n: Gray-scale (uint8 or uint16) or binary image (logical).
               dimension of the base matrix
        - Output
            g: Image.
        - Description
            Ordered dither (Bayer 1973).
        - Examples
            #
            f1 = iaramp([20,150], 256, [0,255])
            f2 = iaresize(iaread('woodlog.pgm'), [150,150])
            f = Numeric.concatenate((f1, f2))
            g_4 = iadither(f, 4)
            g_32 = iadither(f, 32)
            iashow(f)
            iashow(g_4)
            iashow(g_32)
    """
    from Numeric import array, log, ones, concatenate, ravel

    D = 1.*array([[0,2],[3,1]])
    d = 1*D
    k = log(n/2.)/log(2.)
    for i in range(k):
        u = ones(D.shape)
        d1 = 4*D + d[0,0]*u
        d2 = 4*D + d[0,1]*u
        d3 = 4*D + d[1,0]*u
        d4 = 4*D + d[1,1]*u
        D = concatenate((concatenate((d1,d2),1), concatenate((d3,d4),1)))
    D = 255*abs(D/max(ravel(D)))
    g = iatile(D.astype('b'), f.shape)
    g = ianormalize(f,[0,255]) >= g
    return g
#
# =====================================================================
#
#   iafloyd
#
# =====================================================================
def iafloyd(f):
    """
        - Purpose
            Floyd-Steinberg error diffusion.
        - Synopsis
            g = iafloyd(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). input
               image
        - Output
            g: Image.
        - Description
            Floyd-Steinberg error diffusion (1976).
        - Examples
            #
            f1 = iaramp([20,150], 256, [0,255])
            f2 = iaresize(iaread('woodlog.pgm'), [150,150])
            f = Numeric.concatenate((f1, f2))
            g = iafloyd(f)
            iashow(f)
            iashow(g)
    """
    from Numeric import zeros

    f_ = 1.*ianormalize(f, [0,255])
    g = zeros(f_.shape)
    for i in range(f_.shape[0]):
        for j in range(f_.shape[1]):
            if f_[i,j] >= 128:
                g[i,j] = 255
            erro = f_[i,j] - g[i,j]
            if j < f_.shape[1]-1:
                f_[i,j+1] = f_[i,j+1] + 7*erro/16.
            if i < f_.shape[0]-1 and j > 0:
                f_[i+1,j-1] = f_[i+1,j-1] + 3*erro/16.
            if i < f_.shape[0]-1:
                f_[i+1,j] = f_[i+1,j] + 5*erro/16.
            if i < f_.shape[0]-1 and j < f_.shape[1]-1:
                f_[i+1,j+1] = f_[i+1,j+1] + erro/16.
    g = g > 0
    return g
#
# =====================================================================
#
#   iaunique
#
# =====================================================================
def iaunique(f):
    """
        - Purpose
            Set unique.
        - Synopsis
            t, i, j = iaunique(f)
        - Input
            f: Gray-scale (uint8 or uint16) or binary image (logical). Set
               initial.
        - Output
            t: Gray-scale (uint8 or uint16) or binary image (logical).
            i: Gray-scale (uint8 or uint16) or binary image (logical).
            j: Gray-scale (uint8 or uint16) or binary image (logical).
        - Description
            Returns the set unique.
        - Examples
            #
            c = Numeric.array([2, 10, 5, 5, 10, 7, 5])
            print c
            (u,i,j) = iaunique(c)
            print u
            print i
            print j
    """
    from Numeric import ravel, sort, argsort, equal, concatenate, take, nonzero, zeros, put, cumsum

    aux = ravel(f)
    t = sort(aux)
    i = argsort(aux)
    d = equal(t[0:-1], t[1::])
    d = concatenate((d, [0]))
    t = take(t, nonzero(1-d))
    j = zeros(aux.shape)
    put(j, i, cumsum(concatenate(([1], 1-d[0:-1])))-1)
    i = take(i, nonzero(1-d))
    return t, i, j
#
# =====================================================================
#
#   iatype
#
# =====================================================================
def iatype(obj):
    """
        - Purpose
            Print the source code of a function.
        - Synopsis
            iatype(obj)
        - Input
            obj: Object name.

        - Examples
            #
            iatype(iacos)
    """
    import inspect

    print inspect.getsource(obj)
    return

#
#
#
# =====================================================================
#  Adesso -- Generated Mon Jul 28 17:16:03 BRT 2003
# =====================================================================
#

