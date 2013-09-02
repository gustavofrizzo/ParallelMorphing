"""
    Module ia636demo -- Demonstrations
    -------------------------------------------------------------------
    ia636demo is a set of Demonstrations for ia636 package
    Toolbox ia636
    -------------------------------------------------------------------
    iagenimages()      -- Illustrate the generation of different images
    iait()             -- Illustrate the contrast transform function
    iahisteq()         -- Illustrate how to make a histogram equalization
    iacorrdemo()       -- Illustrate the Template Matching technique
    iadftdecompose()   -- Illustrate the decomposition of the image in primitive
                          2-D waves.
    iadftexamples()    -- Demonstrate the DFT spectrum of simple synthetic
                          images.
    iadftmatrixexamples() -- Demonstrate the kernel matrix for the DFT Transform.
    iadftscaleproperty() -- Illustrate the scale property of the Discrete Fourier
                          Transform.
    iaconvteo()        -- Illustrate the convolution theorem
    iahotelling()      -- Illustrate the Hotelling Transform
    iainversefiltering() -- Illustrate the inverse filtering for restoration.
    iamagnify()        -- Illustrate the interpolation of magnified images
    iaotsudemo()       -- Illustrate the Otsu Thresholding Selection Method
"""
from ia63605all.ia636 import *
# =========================================================================
#
#   iagenimages - Illustrate the generation of different images
#
# =========================================================================
def iagenimages():
    from ia636 import iagaussian, ianormalize, iashow, iasplot, iaresize, iacos, iameshgrid, iaisolines
    print
    print '''Illustrate the generation of different images'''
    print
    #
    print '========================================================================='
    print '''
    A gaussian image is controled by its mean and variance.
    '''
    print '========================================================================='
    #1
    print '''
    f1 = iagaussian([256,256], [128,128], [[50*50,0],[0,80*80]])
    fn = ianormalize(f1, [0,255])
    iashow(fn)
    g,d = iasplot(iaresize(fn, [32,32]))
    #showfig(fn)
    iashow(fn > 128)'''
    f1 = iagaussian([256,256], [128,128], [[50*50,0],[0,80*80]])
    fn = ianormalize(f1, [0,255])
    iashow(fn)
    g,d = iasplot(iaresize(fn, [32,32]))
    #showfig(fn)
    iashow(fn > 128)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The difference of two gaussian images gives a Laplacian image.
    '''
    print '========================================================================='
    #1
    print '''
    f2 = iagaussian([256,256], [128,128], [[25*25,0],[0,40*40]])
    fn2 = ianormalize(f2, [0,255])
    f = f1 - f2/5.
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    g,d = iasplot(iaresize(fn, [32,32]))
    #showfig(fn)
    iashow(fn > 230)'''
    f2 = iagaussian([256,256], [128,128], [[25*25,0],[0,40*40]])
    fn2 = ianormalize(f2, [0,255])
    f = f1 - f2/5.
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    g,d = iasplot(iaresize(fn, [32,32]))
    #showfig(fn)
    iashow(fn > 230)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A bidimensional sinusoidal image depends on its period, phase, and
    direction of the wave.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric
    f1 = iacos([128,256], 100, Numeric.pi/4, 0)
    fn = ianormalize(f1, [0,255])
    iashow(fn)
    iashow(fn > 128)'''
    import Numeric
    f1 = iacos([128,256], 100, Numeric.pi/4, 0)
    fn = ianormalize(f1, [0,255])
    iashow(fn)
    iashow(fn > 128)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The multiplication of two sinusoidal images gives a kind of a grid
    of hills and depressions.
    '''
    print '========================================================================='
    #0
    print '''
    f2 = iacos([128,256], 40, Numeric.pi/8, 0)
    fn2 = ianormalize(f2, [0,255])
    f = f1 * f2
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    iashow(fn > 128)'''
    f2 = iacos([128,256], 40, Numeric.pi/8, 0)
    fn2 = ianormalize(f2, [0,255])
    f = f1 * f2
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    iashow(fn > 128)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    In this case, the waves are orthogonal to each other.
    '''
    print '========================================================================='
    #0
    print '''
    f3 = iacos([128,256], 40, (Numeric.pi/8)+(Numeric.pi/2), 0)
    fn2 = ianormalize(f3, [0,255])
    f = f2 * f3
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    iashow(fn > 190)'''
    f3 = iacos([128,256], 40, (Numeric.pi/8)+(Numeric.pi/2), 0)
    fn2 = ianormalize(f3, [0,255])
    f = f2 * f3
    fn = ianormalize(f, [0,255])
    iashow(fn2)
    iashow(fn)
    iashow(fn > 190)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The maximum of two sinusoidal images gives a kind of a surface of
    the union of both waves. They look like pipes
    '''
    print '========================================================================='
    #0
    print '''
    f = Numeric.maximum(f1, f2)
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)'''
    f = Numeric.maximum(f1, f2)
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    If the arguments of the cos are multiplied instead of added as
    before, we have a rather interesting pattern. It is important to
    remember that the proper sinusoidal image that are related to
    Fourier transforms are the bidimensional sinusoide shown earlier
    '''
    print '========================================================================='
    #0
    print '''
    x,y = iameshgrid(range(256), range(256))
    f = Numeric.cos((2*Numeric.pi*x/256) * (2*Numeric.pi*y/256))
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)'''
    x,y = iameshgrid(range(256), range(256))
    f = Numeric.cos((2*Numeric.pi*x/256) * (2*Numeric.pi*y/256))
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    If the arguments of the cos are taking at the power of a number, we
    have a varying period effect
    '''
    print '========================================================================='
    #0
    print '''
    x,y = iameshgrid(range(150), range(150))
    f = Numeric.cos(2*Numeric.pi* (x/80. + y/150.)**3)
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)'''
    x,y = iameshgrid(range(150), range(150))
    f = Numeric.cos(2*Numeric.pi* (x/80. + y/150.)**3)
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    In this case the arguments of the cos are the multiplication of x
    and cos(y)
    '''
    print '========================================================================='
    #0
    print '''
    x,y = iameshgrid(range(150), range(150))
    f = Numeric.cos(2*Numeric.pi*(x/80.) * Numeric.cos(2*Numeric.pi*(y/150.)))
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)'''
    x,y = iameshgrid(range(150), range(150))
    f = Numeric.cos(2*Numeric.pi*(x/80.) * Numeric.cos(2*Numeric.pi*(y/150.)))
    fn = ianormalize(f, [0,255])
    iashow(fn)
    iashow(fn > 200)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The multiplication of x and y gives a surface with saddle point
    '''
    print '========================================================================='
    #0
    print '''
    x,y = iameshgrid(range(-75,75), range(-75,75))
    f = x * y
    fn = ianormalize(f, [0,255]).astype('b')
    iashow(fn)
    iashow(iaisolines(fn,9))'''
    x,y = iameshgrid(range(-75,75), range(-75,75))
    f = x * y
    fn = ianormalize(f, [0,255]).astype('b')
    iashow(fn)
    iashow(iaisolines(fn,9))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iait - Illustrate the contrast transform function
#
# =========================================================================
def iait():
    from ia636 import iaramp, iaapplylut, iashow, iaplot, iaread, ianormalize
    print
    print '''Illustrate the contrast transform function'''
    print
    #
    print '========================================================================='
    print '''
    The simplest intensity function is the identify s=v. This transform
    is a line of 45 degrees. It makes the output image the same as the
    input image.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric
    f = iaramp([100,100], 10, [0,255])
    it = Numeric.arange(256)
    g = iaapplylut(f, it)
    iashow(f)
    iashow(g)'''
    import Numeric
    f = iaramp([100,100], 10, [0,255])
    it = Numeric.arange(256)
    g = iaapplylut(f, it)
    iashow(f)
    iashow(g)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    It is common to visualize the intensity transform function in a
    plot, T(v) x v.
    '''
    print '========================================================================='
    #1
    print '''
    g,d = iaplot(it)
    g('set data style boxes')
    g.plot(d)
    #showfig(it)'''
    g,d = iaplot(it)
    g('set data style boxes')
    g.plot(d)
    #showfig(it)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To change a given gray scale value v1 of the input image to another
    gray scale value s1, we can change the identity function such that
    T(v1)=s1. Suppose we want to change any pixel with value 0 to 255.
    '''
    print '========================================================================='
    #0
    print '''
    it1 = 1*it
    it1[0] = 255
    print it1[0:5] # show the start of the intensity table
    g = iaapplylut(f, it1)
    iashow(g)'''
    it1 = 1*it
    it1[0] = 255
    print it1[0:5] # show the start of the intensity table
    g = iaapplylut(f, it1)
    iashow(g)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    To invert the gray scale of an image, we can apply an intensity
    transform of the form T(v) = 255 - v. This transform will make dark
    pixels light and light pixels dark.
    '''
    print '========================================================================='
    #1
    print '''
    v = Numeric.arange(255)
    Tn = 255 - v
    f = iaread('cameraman.pgm')
    g = iaapplylut(f, Tn)
    iashow(g)
    g,d = iaplot(Tn)
    g('set data style boxes')
    g.plot(d)
    #showfig(Tn)'''
    v = Numeric.arange(255)
    Tn = 255 - v
    f = iaread('cameraman.pgm')
    g = iaapplylut(f, Tn)
    iashow(g)
    g,d = iaplot(Tn)
    g('set data style boxes')
    g.plot(d)
    #showfig(Tn)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A common operation in image processing is called thresholding. It
    assigns value 1 to all pixels equal or above a threshold value and
    assigns zero to the others. The threshold operator converts a gray
    scale image into a binary image. It can be easily implemented using
    the intensity transform. In the example below the threshold value is
    128.
    '''
    print '========================================================================='
    #1
    print '''
    f = iaread('cameraman.pgm')
    thr = Numeric.concatenate((Numeric.zeros(128), Numeric.ones(128)))
    g = iaapplylut(f, thr)
    iashow(g)
    g,d = iaplot(g)
    g('set data style boxes')
    g.plot(d)
    #showfig(thr)'''
    f = iaread('cameraman.pgm')
    thr = Numeric.concatenate((Numeric.zeros(128), Numeric.ones(128)))
    g = iaapplylut(f, thr)
    iashow(g)
    g,d = iaplot(g)
    g('set data style boxes')
    g.plot(d)
    #showfig(thr)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A variation of the thresholding is when the output is one for a
    range of the input gray scale values. In the example below, only
    pixels between values 100 and 120 are turned to one, the others to
    zero.
    '''
    print '========================================================================='
    #1
    print '''
    t1 = Numeric.zeros(256)
    t1[100:121] = 1
    g = iaapplylut(f, t1)
    iashow(g)
    g,d = iaplot(t1)
    g('set data style boxes')
    g.plot(d)
    #showfig(t1)'''
    t1 = Numeric.zeros(256)
    t1[100:121] = 1
    g = iaapplylut(f, t1)
    iashow(g)
    g,d = iaplot(t1)
    g('set data style boxes')
    g.plot(d)
    #showfig(t1)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A generalization of the previous case is to assign different values
    (classes) to different input ranges. This is a typical
    classification decision. For pixels from 0 and t1, assign 1; from t1
    to t2, assign 2, etc. In the example below, the pixels are
    classified in three categories: class 1: dark pixels (between 0 and
    50) corresponding to the cameraman clothing, class 3: medium gray
    pixels (from 51 to 180), and class 2: white pixels (from 181 to
    255), corresponding to the sky.
    '''
    print '========================================================================='
    #1
    print '''
    t2 = Numeric.zeros(256)
    t2[0:51] = 1
    t2[51:181] = 3
    t2[181:256] = 2
    g = iaapplylut(f, t2)
    iashow(g)
    g,d = iaplot(t2)
    g('set data style boxes')
    g.plot(d)
    #showfig(t2)'''
    t2 = Numeric.zeros(256)
    t2[0:51] = 1
    t2[51:181] = 3
    t2[181:256] = 2
    g = iaapplylut(f, t2)
    iashow(g)
    g,d = iaplot(t2)
    g('set data style boxes')
    g.plot(d)
    #showfig(t2)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    When the intensity transform is a crescent function, if v1 >= v1
    then T(v1) >= T(v2). In this class of intensity transforms, the
    order of the gray scale does not change, i.e., if a gray scale value
    is darker than another in the input image, in the transformed image,
    it will still be darker. The intensity order does not change. This
    particular intensity transforms are of special interest to image
    enhancing as our visual system does not feel confortable to
    non-crescent intensity transforms. Note the negation and the
    generalized threshold examples. The identity transform is a crescent
    function with slope 1. If the slope is higher than 1, for a small
    variation of v, there will be a larger variation in T(v), increasing
    the contrast around the gray scale v. If the slope is less than 1,
    the effect is opposite, the constrast around v will be decreased. A
    logarithm function has a higher slope at its beginning and lower
    slope at its end. It is normally used to increase the contrast of
    dark areas of the image.
    '''
    print '========================================================================='
    #1
    print '''
    v = Numeric.arange(256)
    Tlog = (256./Numeric.log(256.)) * Numeric.log(v + 1)
    f = ianormalize(iaread('lenina.pgm'), [0, 255])
    g = iaapplylut(f.astype('b'), Tlog)
    iashow(f)
    iashow(g)
    g,d = iaplot(Tlog)
    g('set data style boxes')
    g.plot(d)
    #showfig(Tlog)'''
    v = Numeric.arange(256)
    Tlog = (256./Numeric.log(256.)) * Numeric.log(v + 1)
    f = ianormalize(iaread('lenina.pgm'), [0, 255])
    g = iaapplylut(f.astype('b'), Tlog)
    iashow(f)
    iashow(g)
    g,d = iaplot(Tlog)
    g('set data style boxes')
    g.plot(d)
    #showfig(Tlog)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Sometimes the input image has ranges in floating point or ranges not
    suitable for displaying or processing. The normalization function is
    used to fit the range of the gray scales, preserving the linear gray
    scale relationship of the input image. The intensity function that
    provides the normalization is a straight line segment which can be
    defined by its two extremities. Suppose the input gray scales ranges
    from m1 to m2 and we want to normalize the image to the range of M1
    to M2. The two extremities points are (m1,M1) and (m2,M2). The
    equation for the intensity normalization function is point-slope
    form of the line equation: T(v) - M1 =(M2-M1)/(m2-m1)*(v - m1). The
    function ianormalize does that.
    '''
    print '========================================================================='
    #0
    print '''
    '''
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iahisteq - Illustrate how to make a histogram equalization
#
# =========================================================================
def iahisteq():
    from Numeric import product
    from MLab import cumsum
    from ia636 import iaread, iashow, iahistogram, iaplot, iaapplylut
    print
    print '''Illustrate how to make a histogram equalization'''
    print
    #
    print '========================================================================='
    print '''
    '''
    print '========================================================================='
    #2
    print '''
    f = iaread('woodlog.pgm')
    iashow(f)
    h = iahistogram(f)
    g,d1 = iaplot(h)
    g('set data style boxes')
    g.plot(d1)
    #showfig(h)
    nch = cumsum(h) / (1.*product(f.shape))
    g,d2 = iaplot(nch)
    g('set data style boxes')
    g.plot(d2)
    #showfig(nch)'''
    f = iaread('woodlog.pgm')
    iashow(f)
    h = iahistogram(f)
    g,d1 = iaplot(h)
    g('set data style boxes')
    g.plot(d1)
    #showfig(h)
    nch = cumsum(h) / (1.*product(f.shape))
    g,d2 = iaplot(nch)
    g('set data style boxes')
    g.plot(d2)
    #showfig(nch)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    '''
    print '========================================================================='
    #2
    print '''
    gn = (255 * iaapplylut(f, nch)).astype('b')
    iashow(gn)
    gh = iahistogram(gn)
    g,d1 = iaplot(gh)
    g('set data style boxes')
    g.plot(d1)
    #showfig(gh)
    gch = cumsum(gh) # cumulative histogram
    g,d2 = iaplot(gch)
    g('set data style boxes')
    g.plot(d2)
    #showfig(gch)'''
    gn = (255 * iaapplylut(f, nch)).astype('b')
    iashow(gn)
    gh = iahistogram(gn)
    g,d1 = iaplot(gh)
    g('set data style boxes')
    g.plot(d1)
    #showfig(gh)
    gch = cumsum(gh) # cumulative histogram
    g,d2 = iaplot(gch)
    g('set data style boxes')
    g.plot(d2)
    #showfig(gch)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iacorrdemo - Illustrate the Template Matching technique
#
# =========================================================================
def iacorrdemo():
    from ia636 import iaread, iashow, iapconv, iaind2sub
    print
    print '''Illustrate the Template Matching technique'''
    print
    #
    print '========================================================================='
    print '''
    We have a gray scale image and a pattern extracting from the image.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric
    f = iaread('cameraman.pgm')
    f = Numeric.asarray(f).astype(Numeric.Float)
    iashow(f)
    w = f[25:25+17,106:106+17]
    iashow(w)'''
    import Numeric
    f = iaread('cameraman.pgm')
    f = Numeric.asarray(f).astype(Numeric.Float)
    iashow(f)
    w = f[25:25+17,106:106+17]
    iashow(w)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Pure image correlation is not good for template matching because it
    depends on the mean gray value in the image, so light regions gives
    higher score than dark regions. A normalization factor can be used
    which improves the template matching.
    '''
    print '========================================================================='
    #0
    print '''
    w1 = w[::-1,::-1]
    iashow(w1)
    g = iapconv(f, w1)
    iashow(g)
    i = Numeric.ones(Numeric.shape(w1))
    fm2 = iapconv(f*f, i)
    g2 = g/Numeric.sqrt(fm2)
    iashow(g2)
    v, pos = max(Numeric.ravel(g2)), Numeric.argmax(Numeric.ravel(g2))
    (row, col) = iaind2sub(g2.shape, pos)
    print 'found best match at (%3.0f,%3.0f)\n' %(col,row)'''
    w1 = w[::-1,::-1]
    iashow(w1)
    g = iapconv(f, w1)
    iashow(g)
    i = Numeric.ones(Numeric.shape(w1))
    fm2 = iapconv(f*f, i)
    g2 = g/Numeric.sqrt(fm2)
    iashow(g2)
    v, pos = max(Numeric.ravel(g2)), Numeric.argmax(Numeric.ravel(g2))
    (row, col) = iaind2sub(g2.shape, pos)
    print 'found best match at (%3.0f,%3.0f)\n' %(col,row)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    A better pattern matching is achievied subtrating the mean value on
    the image application.
    '''
    print '========================================================================='
    #0
    print '''
    import MLab
    n = Numeric.product(w.shape)
    wm = MLab.mean(Numeric.ravel(w))
    fm = 1.*iapconv(f,i)/n
    num = g - (n*fm*wm)
    iashow(num)
    den = Numeric.sqrt(fm2 - (n*fm*fm))
    iashow(den)
    cn = 1.*num/den
    iashow(cn)
    v, pos = max(Numeric.ravel(cn)), Numeric.argmax(Numeric.ravel(cn))
    (row, col) = iaind2sub(g2.shape, pos)
    print 'found best match at (%3.0f,%3.0f)\n' %(col,row)'''
    import MLab
    n = Numeric.product(w.shape)
    wm = MLab.mean(Numeric.ravel(w))
    fm = 1.*iapconv(f,i)/n
    num = g - (n*fm*wm)
    iashow(num)
    den = Numeric.sqrt(fm2 - (n*fm*fm))
    iashow(den)
    cn = 1.*num/den
    iashow(cn)
    v, pos = max(Numeric.ravel(cn)), Numeric.argmax(Numeric.ravel(cn))
    (row, col) = iaind2sub(g2.shape, pos)
    print 'found best match at (%3.0f,%3.0f)\n' %(col,row)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iadftdecompose - Illustrate the decomposition of the image in primitive 2-D waves.
#
# =========================================================================
def iadftdecompose():
    from ia636 import iasub2ind, iashow, iadft, iadftview, iaplot, iafftshift
    print
    print '''Illustrate the decomposition of the image in primitive 2-D waves.'''
    print
    #
    print '========================================================================='
    print '''
    The image is created using.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric
    f = Numeric.ones((128, 128)) * 50
    x, y = [], map(lambda k:k%128, range(-32,32))
    for i in range(128): x = x + (len(y) * [i])
    y = 128 * y
    Numeric.put(f, iasub2ind([128,128], x, y), 200)
    iashow(f)'''
    import Numeric
    f = Numeric.ones((128, 128)) * 50
    x, y = [], map(lambda k:k%128, range(-32,32))
    for i in range(128): x = x + (len(y) * [i])
    y = 128 * y
    Numeric.put(f, iasub2ind([128,128], x, y), 200)
    iashow(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT is computed and displayed
    '''
    print '========================================================================='
    #1
    print '''
    F = iadft(f)
    E = iadftview(F)
    iashow(E)
    g,d = iaplot(iafftshift(F[:,0]).real)
    g('set data style impulses')
    g.plot(d)
    #showfig(central_line)'''
    F = iadft(f)
    E = iadftview(F)
    iashow(E)
    g,d = iaplot(iafftshift(F[:,0]).real)
    g('set data style impulses')
    g.plot(d)
    #showfig(central_line)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iadftexamples - Demonstrate the DFT spectrum of simple synthetic images.
#
# =========================================================================
def iadftexamples():
    from ia636 import iaind2sub, iashow, iadftview, iagaussian, ianormalize, iacomb
    print
    print '''Demonstrate the DFT spectrum of simple synthetic images.'''
    print
    #
    print '========================================================================='
    print '''
    The DFT of a constant image is a single point at F(0,0), which gives
    the sum of all pixels in the image.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric, FFT
    f = 50 * Numeric.ones((10, 20))
    F = FFT.fft2d(f)
    aux = F.real > 1E-5
    r, c = iaind2sub([10, 20], Numeric.nonzero(Numeric.ravel(aux)))
    print r
    print c
    print F[r[0],c[0]]/(10.*20.)'''
    import Numeric, FFT
    f = 50 * Numeric.ones((10, 20))
    F = FFT.fft2d(f)
    aux = F.real > 1E-5
    r, c = iaind2sub([10, 20], Numeric.nonzero(Numeric.ravel(aux)))
    print r
    print c
    print F[r[0],c[0]]/(10.*20.)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of a square image is a digital sync.
    '''
    print '========================================================================='
    #0
    print '''
    f = Numeric.zeros((128, 128))
    f[63:63+5,63:63+5] = 1
    iashow(f)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)'''
    f = Numeric.zeros((128, 128))
    f[63:63+5,63:63+5] = 1
    iashow(f)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of a pyramid is the square of the digital sync.
    '''
    print '========================================================================='
    #0
    print '''
    f = Numeric.zeros((128, 128))
    k = Numeric.array([[1,2,3,4,5,6,5,4,3,2,1]])
    k2 = Numeric.matrixmultiply(Numeric.transpose(k), k)
    f[63:63+k2.shape[0], 63:63+k2.shape[1]] = k2
    iashow(f)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)'''
    f = Numeric.zeros((128, 128))
    k = Numeric.array([[1,2,3,4,5,6,5,4,3,2,1]])
    k2 = Numeric.matrixmultiply(Numeric.transpose(k), k)
    f[63:63+k2.shape[0], 63:63+k2.shape[1]] = k2
    iashow(f)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of a Gaussian image is a Gaussian image.
    '''
    print '========================================================================='
    #0
    print '''
    f = iagaussian([128,128],[65,65],[[3*3,0],[0,5*5]])
    fn = ianormalize(f,[0,255])
    iashow(fn)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)'''
    f = iagaussian([128,128],[65,65],[[3*3,0],[0,5*5]])
    fn = ianormalize(f,[0,255])
    iashow(fn)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of an impulse image is an impulse image.
    '''
    print '========================================================================='
    #0
    print '''
    f = iacomb((128,128), (4,4), (0,0))
    fn = ianormalize(f, (0,255))
    iashow(fn)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)'''
    f = iacomb((128,128), (4,4), (0,0))
    fn = ianormalize(f, (0,255))
    iashow(fn)
    F = FFT.fft2d(f)
    Fv = iadftview(F)
    iashow(Fv)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iadftmatrixexamples - Demonstrate the kernel matrix for the DFT Transform.
#
# =========================================================================
def iadftmatrixexamples():
    from ia636 import iadftmatrix, iashow, iaplot
    print
    print '''Demonstrate the kernel matrix for the DFT Transform.'''
    print
    #
    print '========================================================================='
    print '''
    Imaginary and real parts of the DFT kernel.
    '''
    print '========================================================================='
    #0
    print '''
    A = iadftmatrix(128)
    Aimag, Areal = A.imag, A.real
    iashow(Aimag)
    iashow(Areal)'''
    A = iadftmatrix(128)
    Aimag, Areal = A.imag, A.real
    iashow(Aimag)
    iashow(Areal)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Three first lines from imaginary and real parts of the kernel
    matrix. Observe the increasing frequencies of the senoidals
    (imaginary part) and cossenoidals (real part).
    '''
    print '========================================================================='
    #2
    print '''
    g,i1 = iaplot(Aimag[0,:])
    g,i2 = iaplot(Aimag[1,:])
    g,i3 = iaplot(Aimag[2,:])
    g.plot(i1,i2,i3)
    #showfig(i1,i2,i3)
    g,r1 = iaplot(Areal[0,:])
    g,r2 = iaplot(Areal[1,:])
    g,r3 = iaplot(Areal[2,:])
    g.plot(r1,r2,r3)
    #showfig(r1,r2,r3)'''
    g,i1 = iaplot(Aimag[0,:])
    g,i2 = iaplot(Aimag[1,:])
    g,i3 = iaplot(Aimag[2,:])
    g.plot(i1,i2,i3)
    #showfig(i1,i2,i3)
    g,r1 = iaplot(Areal[0,:])
    g,r2 = iaplot(Areal[1,:])
    g,r3 = iaplot(Areal[2,:])
    g.plot(r1,r2,r3)
    #showfig(r1,r2,r3)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iadftscaleproperty - Illustrate the scale property of the Discrete Fourier Transform.
#
# =========================================================================
def iadftscaleproperty():
    from ia636 import iaread, iashow, iadftview
    print
    print '''Illustrate the scale property of the Discrete Fourier Transform.'''
    print
    #
    print '========================================================================='
    print '''
    The image is read and a small portion (64x64) is selected.
    '''
    print '========================================================================='
    #0
    print '''
    f = iaread('cameraman.pgm')
    froi = f[19:19+64,99:99+64] # ROI selection
    iashow(f)
    iashow(froi)'''
    f = iaread('cameraman.pgm')
    froi = f[19:19+64,99:99+64] # ROI selection
    iashow(f)
    iashow(froi)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of the ROI image is taken and its spectrum is displayed
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric, FFT
    fd = froi.astype(Numeric.Float)
    F = FFT.fft2d(fd) # F is the DFT of f
    iashow(froi)
    iashow(iadftview(F))'''
    import Numeric, FFT
    fd = froi.astype(Numeric.Float)
    F = FFT.fft2d(fd) # F is the DFT of f
    iashow(froi)
    iashow(iadftview(F))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image is expanded by 4, but filling the new pixels with 0
    '''
    print '========================================================================='
    #0
    print '''
    fx4 = Numeric.zeros(4*Numeric.array(froi.shape)) # size is 4 times larger
    fx4[::4,::4] = froi                              # filling the expanded image
    iashow(froi)
    iashow(fx4)'''
    fx4 = Numeric.zeros(4*Numeric.array(froi.shape)) # size is 4 times larger
    fx4[::4,::4] = froi                              # filling the expanded image
    iashow(froi)
    iashow(fx4)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    the resulting DFT is a periodical replication of the original DFT.
    '''
    print '========================================================================='
    #0
    print '''
    fdx4 = fx4.astype(Numeric.Float)
    Fx4 = FFT.fft2d(fdx4) # Fx4 is the DFT of fx4 (expanded f)
    iashow(iadftview(F))
    iashow(iadftview(Fx4))'''
    fdx4 = fx4.astype(Numeric.Float)
    Fx4 = FFT.fft2d(fdx4) # Fx4 is the DFT of fx4 (expanded f)
    iashow(iadftview(F))
    iashow(iadftview(Fx4))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Alternatively, the original DFT (F) is replicated by 4 in each
    direction and compared with the DFT of the expanded image. For
    quantitative comparison, both the sum of the absolute errors of all
    pixels is computed and displayed.
    '''
    print '========================================================================='
    #0
    print '''
    aux = Numeric.concatenate((F,F,F,F))
    FFx4 = Numeric.concatenate((aux,aux,aux,aux), 1) # replicate the DFT of f
    iashow(iadftview(FFx4))
    diff = abs(FFx4 - Fx4)                           # compare the replicated DFT with DFT of expanded f
    print Numeric.sum(Numeric.ravel(diff))           # print the error signal power'''
    aux = Numeric.concatenate((F,F,F,F))
    FFx4 = Numeric.concatenate((aux,aux,aux,aux), 1) # replicate the DFT of f
    iashow(iadftview(FFx4))
    diff = abs(FFx4 - Fx4)                           # compare the replicated DFT with DFT of expanded f
    print Numeric.sum(Numeric.ravel(diff))           # print the error signal power
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    '''
    print '========================================================================='
    #0
    print '''
    ffdx4 = FFT.inverse_fft2d(FFx4)
    fimag = ffdx4.imag
    print Numeric.sum(Numeric.ravel(fimag))
    ffdx4 = Numeric.floor(0.5 + ffdx4.real) # round
    iashow(ffdx4.astype(Numeric.Int))
    error = abs(fdx4 - ffdx4)
    print Numeric.sum(Numeric.ravel(error))'''
    ffdx4 = FFT.inverse_fft2d(FFx4)
    fimag = ffdx4.imag
    print Numeric.sum(Numeric.ravel(fimag))
    ffdx4 = Numeric.floor(0.5 + ffdx4.real) # round
    iashow(ffdx4.astype(Numeric.Int))
    error = abs(fdx4 - ffdx4)
    print Numeric.sum(Numeric.ravel(error))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iaconvteo - Illustrate the convolution theorem
#
# =========================================================================
def iaconvteo():
    from ia636 import iaread, iashow, iaroi, iapconv, iaptrans, iadft, iadftview, iaisdftsym, iaidft
    print
    print '''Illustrate the convolution theorem'''
    print
    #
    print '========================================================================='
    print '''
    The image is read and displayed
    '''
    print '========================================================================='
    #0
    print '''
    fin = iaread('lenina.pgm')
    iashow(fin)
    froi = iaroi(fin, (90,70), (200,180))
    iashow(froi)'''
    fin = iaread('lenina.pgm')
    iashow(fin)
    froi = iaroi(fin, (90,70), (200,180))
    iashow(froi)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image is convolved (periodicaly) with the 3x3 Laplacian kernel
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric
    fd = froi.astype(Numeric.Float)
    h = Numeric.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    g = iapconv(fd,h)
    iashow(g)'''
    import Numeric
    fd = froi.astype(Numeric.Float)
    h = Numeric.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    g = iapconv(fd,h)
    iashow(g)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The 3x3 kernel is zero padded to the size of the input image and
    periodicaly translated so that the center of the kernel stays at the
    top-left image corner. Its spectrum is visualized.
    '''
    print '========================================================================='
    #0
    print '''
    hx = Numeric.zeros(Numeric.array(froi.shape))
    hx[:h.shape[0],:h.shape[1]] = h
    hx = iaptrans(hx,-Numeric.floor((Numeric.array(h.shape)-1)/2).astype(Numeric.Int))
    H = iadft(hx);
    iashow(iadftview(H))'''
    hx = Numeric.zeros(Numeric.array(froi.shape))
    hx[:h.shape[0],:h.shape[1]] = h
    hx = iaptrans(hx,-Numeric.floor((Numeric.array(h.shape)-1)/2).astype(Numeric.Int))
    H = iadft(hx);
    iashow(iadftview(H))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image is filtered by multiplying its DFT by the frequency mask
    computed in the previous step.
    '''
    print '========================================================================='
    #0
    print '''
    F = iadft(fd)
    G = F * H
    print "Is symmetrical:", iaisdftsym(G)
    iashow(iadftview(G))
    g_aux = iaidft(G).real
    iashow(g_aux)'''
    F = iadft(fd)
    G = F * H
    print "Is symmetrical:", iaisdftsym(G)
    iashow(iadftview(G))
    g_aux = iaidft(G).real
    iashow(g_aux)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Both images, filtered by the convolution and filtered in the
    frequency domain are compared to see that they are the same. The
    small differencies are due to numerical precision errors.
    '''
    print '========================================================================='
    #0
    print '''
    e = abs(g - g_aux)
    print "Max error:", max(Numeric.ravel(e))'''
    e = abs(g - g_aux)
    print "Max error:", max(Numeric.ravel(e))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iahotelling - Illustrate the Hotelling Transform
#
# =========================================================================
def iahotelling():
    from ia636 import iagaussian, iashow, iaind2sub, iacontour, iaplot, iasub2ind, iaread
    print
    print '''Illustrate the Hotelling Transform'''
    print
    #
    print '========================================================================='
    print '''
    A binary image with an ellipsis
    '''
    print '========================================================================='
    #0
    print '''
    f = iagaussian([100,100], [45,50], [[20,5],[5,10]]) > 0.0000001
    iashow(f)'''
    f = iagaussian([100,100], [45,50], [[20,5],[5,10]]) > 0.0000001
    iashow(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The coordinates of each 1 pixel are the features: cols and rows
    coordinates. The mean value is the centroid.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric, MLab
    (rows, cols) = iaind2sub(f.shape, Numeric.nonzero(Numeric.ravel(f)))
    x = Numeric.concatenate((cols[:,Numeric.NewAxis], rows[:,Numeric.NewAxis]), 1)
    mx = MLab.mean(x)
    print mx'''
    import Numeric, MLab
    (rows, cols) = iaind2sub(f.shape, Numeric.nonzero(Numeric.ravel(f)))
    x = Numeric.concatenate((cols[:,Numeric.NewAxis], rows[:,Numeric.NewAxis]), 1)
    mx = MLab.mean(x)
    print mx
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The eigenvalues and eigenvectors are computed from the covariance.
    The eigenvalues are sorted in decrescent order
    '''
    print '========================================================================='
    #0
    print '''
    Cx = MLab.cov(x)
    print Cx
    [aval, avec] = MLab.eig(Cx)
    aux = Numeric.argsort(aval)[::-1]
    aval = MLab.diag(Numeric.take(aval, aux))
    print aval
    avec = Numeric.take(avec, aux)
    print avec'''
    Cx = MLab.cov(x)
    print Cx
    [aval, avec] = MLab.eig(Cx)
    aux = Numeric.argsort(aval)[::-1]
    aval = MLab.diag(Numeric.take(aval, aux))
    print aval
    avec = Numeric.take(avec, aux)
    print avec
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The direction of the eigenvector of the largest eigenvalue gives the
    inclination of the elongated figure
    '''
    print '========================================================================='
    #0
    print '''
    a1 = Numeric.sqrt(aval[0,0])
    a2 = Numeric.sqrt(aval[1,1])
    vec1 = avec[:,0]
    vec2 = avec[:,1]
    theta = Numeric.arctan(1.*vec1[1]/vec1[0])*180/Numeric.pi
    print 'angle is %3f degrees' % (theta)'''
    a1 = Numeric.sqrt(aval[0,0])
    a2 = Numeric.sqrt(aval[1,1])
    vec1 = avec[:,0]
    vec2 = avec[:,1]
    theta = Numeric.arctan(1.*vec1[1]/vec1[0])*180/Numeric.pi
    print 'angle is %3f degrees' % (theta)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The eigenvectors are placed at the centroid and scaled by the square
    root of its correspondent eigenvalue
    '''
    print '========================================================================='
    #1
    print '''
    iashow(f)
    x_,y_ = iaind2sub(f.shape, Numeric.nonzero(Numeric.ravel(iacontour(f))))
    g,d0 = iaplot(y_, 100-x_)
    # esboça os autovetores na imagem
    x1, y1 = Numeric.arange(mx[0], 100-mx[0]+a1*vec1[0], -0.1), Numeric.arange(mx[1], mx[1]-a1*vec1[1],0.235)
    g,d1 = iaplot(x1, 100-y1) # largest in green
    x2, y2 = Numeric.arange(mx[0], mx[0]-a2*vec2[0], 0.1), Numeric.arange(mx[1], mx[1]+a2*vec2[1],0.042)
    g,d2 = iaplot(x2, 100-y2) # smaller in blue
    g,d3 = iaplot(mx[0], 100-mx[1]) # centroid in magenta
    g('set data style points')
    g('set xrange [0:100]')
    g('set yrange [0:100]')
    g.plot(d0, d1, d2, d3)
    #showfig(mx)'''
    iashow(f)
    x_,y_ = iaind2sub(f.shape, Numeric.nonzero(Numeric.ravel(iacontour(f))))
    g,d0 = iaplot(y_, 100-x_)
    # esboça os autovetores na imagem
    x1, y1 = Numeric.arange(mx[0], 100-mx[0]+a1*vec1[0], -0.1), Numeric.arange(mx[1], mx[1]-a1*vec1[1],0.235)
    g,d1 = iaplot(x1, 100-y1) # largest in green
    x2, y2 = Numeric.arange(mx[0], mx[0]-a2*vec2[0], 0.1), Numeric.arange(mx[1], mx[1]+a2*vec2[1],0.042)
    g,d2 = iaplot(x2, 100-y2) # smaller in blue
    g,d3 = iaplot(mx[0], 100-mx[1]) # centroid in magenta
    g('set data style points')
    g('set xrange [0:100]')
    g('set yrange [0:100]')
    g.plot(d0, d1, d2, d3)
    #showfig(mx)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The Hotelling transform, also called Karhunen-Loeve (K-L) transform,
    or the method of principal components, is computed below
    '''
    print '========================================================================='
    #0
    print '''
    y = Numeric.transpose(Numeric.matrixmultiply(avec, Numeric.transpose(x-mx)))
    my = MLab.mean(y)
    print my
    Cy = MLab.cov(y)
    print Cy
    print Numeric.floor(0.5 + Numeric.sqrt(Cy))'''
    y = Numeric.transpose(Numeric.matrixmultiply(avec, Numeric.transpose(x-mx)))
    my = MLab.mean(y)
    print my
    Cy = MLab.cov(y)
    print Cy
    print Numeric.floor(0.5 + Numeric.sqrt(Cy))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The centroid of the transformed data is zero (0,0). To visualize it
    as an image, the features are translated by the centroid of the
    original data, so that only the rotation effect of the Hotelling
    transform is visualized.
    '''
    print '========================================================================='
    #0
    print '''
    ytrans = Numeric.floor(0.5 + y + Numeric.resize(mx, (x.shape[0], 2)))
    g = Numeric.zeros(f.shape)
    i = iasub2ind(f.shape, ytrans[:,1], ytrans[:,0])
    Numeric.put(g, i, 1)
    iashow(g)'''
    ytrans = Numeric.floor(0.5 + y + Numeric.resize(mx, (x.shape[0], 2)))
    g = Numeric.zeros(f.shape)
    i = iasub2ind(f.shape, ytrans[:,1], ytrans[:,0])
    Numeric.put(g, i, 1)
    iashow(g)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The RGB color image is read and displayed
    '''
    print '========================================================================='
    #0
    print '''
    f = iaread('boat.ppm')
    iashow(f)'''
    f = iaread('boat.ppm')
    iashow(f)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The color components are stored in the third dimension of the image
    array
    '''
    print '========================================================================='
    #0
    print '''
    r = f[0,:,:]
    g = f[1,:,:]
    b = f[2,:,:]
    iashow(r)
    iashow(g)
    iashow(b)'''
    r = f[0,:,:]
    g = f[1,:,:]
    b = f[2,:,:]
    iashow(r)
    iashow(g)
    iashow(b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The features are the red, green and blue components. The mean vector
    is the average color in the image. The eigenvalues and eigenvectors
    are computed. The dimension of the covariance matrix is 3x3 as there
    are 3 features in use
    '''
    print '========================================================================='
    #0
    print '''
    x = 1.*Numeric.concatenate((Numeric.ravel(r)[:,Numeric.NewAxis], Numeric.ravel(g)[:,Numeric.NewAxis], Numeric.ravel(b)[:,Numeric.NewAxis]), 1)
    mx = MLab.mean(x)
    print mx
    Cx = MLab.cov(x)
    print Cx
    [aval, avec] = MLab.eig(Cx)
    aux = Numeric.argsort(aval)[::-1]
    aval = MLab.diag(Numeric.take(aval, aux))
    print aval
    avec = Numeric.take(avec, aux)
    print avec'''
    x = 1.*Numeric.concatenate((Numeric.ravel(r)[:,Numeric.NewAxis], Numeric.ravel(g)[:,Numeric.NewAxis], Numeric.ravel(b)[:,Numeric.NewAxis]), 1)
    mx = MLab.mean(x)
    print mx
    Cx = MLab.cov(x)
    print Cx
    [aval, avec] = MLab.eig(Cx)
    aux = Numeric.argsort(aval)[::-1]
    aval = MLab.diag(Numeric.take(aval, aux))
    print aval
    avec = Numeric.take(avec, aux)
    print avec
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The K-L transform is computed. The mean vector is zero and the
    covariance matrix is decorrelated. We can see the values of the
    standard deviation of the first, second and third components
    '''
    print '========================================================================='
    #0
    print '''
    y = Numeric.transpose(Numeric.matrixmultiply(avec, Numeric.transpose(x-mx)))
    my = MLab.mean(y)
    print my
    Cy = MLab.cov(y)
    print Cy
    Cy = Cy * (1 - (-1e-10 < Cy < 1e-10))
    print Numeric.floor(0.5 + Numeric.sqrt(Cy))'''
    y = Numeric.transpose(Numeric.matrixmultiply(avec, Numeric.transpose(x-mx)))
    my = MLab.mean(y)
    print my
    Cy = MLab.cov(y)
    print Cy
    Cy = Cy * (1 - (-1e-10 < Cy < 1e-10))
    print Numeric.floor(0.5 + Numeric.sqrt(Cy))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The transformed features are put back in three different images with
    the g1 with the first component (larger variance) and g3 with the
    smaller variance
    '''
    print '========================================================================='
    #0
    print '''
    g1 = y[:,0]
    g2 = y[:,1]
    g3 = y[:,2]
    g1 = Numeric.reshape(g1, r.shape)
    g2 = Numeric.reshape(g2, r.shape)
    g3 = Numeric.reshape(g3, r.shape)
    iashow(g1)
    iashow(g2)
    iashow(g3)'''
    g1 = y[:,0]
    g2 = y[:,1]
    g3 = y[:,2]
    g1 = Numeric.reshape(g1, r.shape)
    g2 = Numeric.reshape(g2, r.shape)
    g3 = Numeric.reshape(g3, r.shape)
    iashow(g1)
    iashow(g2)
    iashow(g3)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iainversefiltering - Illustrate the inverse filtering for restoration.
#
# =========================================================================
def iainversefiltering():
    from ia636 import iaread, iabwlp, iashow, ianormalize, iadftview
    print
    print '''Illustrate the inverse filtering for restoration.'''
    print
    #
    print '========================================================================='
    print '''
    The original image is corrupted using a low pass Butterworth filter
    with cutoff period of 8 pixels and order 4. This has the a similar
    effect of an out of focus image.
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric, FFT
    f = iaread('keyb.pgm').astype(Numeric.Float)
    F = FFT.fft2d(f)                # Discrete Fourier Transform of f
    H = iabwlp(f.shape, 16, 4)      # Butterworth filter, cutoff period 16, order 4
    G = F*H                         # Filtering in frequency domain
    g = FFT.inverse_fft2d(G).real   # inverse DFT
    iashow(ianormalize(g, [0,255])) # display distorted image'''
    import Numeric, FFT
    f = iaread('keyb.pgm').astype(Numeric.Float)
    F = FFT.fft2d(f)                # Discrete Fourier Transform of f
    H = iabwlp(f.shape, 16, 4)      # Butterworth filter, cutoff period 16, order 4
    G = F*H                         # Filtering in frequency domain
    g = FFT.inverse_fft2d(G).real   # inverse DFT
    iashow(ianormalize(g, [0,255])) # display distorted image
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    '''
    print '========================================================================='
    #0
    print '''
    iashow(iadftview(G))  # Spectrum of the corrupted image
    IH = 1.0/H            # The inverse filter
    iashow(iadftview(IH)) # Spectrum of the inverse filter'''
    iashow(iadftview(G))  # Spectrum of the corrupted image
    IH = 1.0/H            # The inverse filter
    iashow(iadftview(IH)) # Spectrum of the inverse filter
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    '''
    print '========================================================================='
    #0
    print '''
    FR = G*IH             # Inverse filtering
    iashow(iadftview(FR)) # Spectrum of the restored image
    fr = FFT.inverse_fft2d(FR).real
    iashow(fr)            # display the restored image'''
    FR = G*IH             # Inverse filtering
    iashow(iadftview(FR)) # Spectrum of the restored image
    fr = FFT.inverse_fft2d(FR).real
    iashow(fr)            # display the restored image
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The previous example is rather didactical. Just as an experience,
    instead of using the corrupted image with pixels values represented
    in double, we will use a integer truncation to the pixel values.
    '''
    print '========================================================================='
    #0
    print '''
    gfix = Numeric.floor(g)                                                 # truncate pixel values
    Gfix = FFT.fft2d(gfix)                                                  # DFT of truncated filted image
    FRfix = Gfix * IH                                                       # applying the inverse filtering
    frfix = FFT.inverse_fft2d(FRfix).real
    iashow(gfix)
    iashow(ianormalize(frfix, [0,255]))                                     # display the restored image
    iashow(iadftview(FRfix))                                                # spectrum of the restored image'''
    gfix = Numeric.floor(g)                                                 # truncate pixel values
    Gfix = FFT.fft2d(gfix)                                                  # DFT of truncated filted image
    FRfix = Gfix * IH                                                       # applying the inverse filtering
    frfix = FFT.inverse_fft2d(FRfix).real
    iashow(gfix)
    iashow(ianormalize(frfix, [0,255]))                                     # display the restored image
    iashow(iadftview(FRfix))                                                # spectrum of the restored image
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    When the distorted image is rounded, it is equivalent of subtracting
    a noise from the distorted image. This noise has uniform
    distribution with mean 0.5 pixel intensity. The inverted filter is
    very high pass frequency, and at high frequency, the noise outcomes
    the power of the signal. The result is a magnification of high
    frequency noise.
    '''
    print '========================================================================='
    #0
    print '''
    import MLab
    fn = g - gfix                                                      # noise that was added
    print [MLab.mean(Numeric.ravel(fn)), MLab.min(Numeric.ravel(fn)), MLab.max(Numeric.ravel(fn))] # mean, minumum and maximum values
    iashow(ianormalize(fn, [0,255]))                                   # display noise
    FN = FFT.fft2d(fn)
    iashow(iadftview(FN))                                              # spectrum of the noise
    FNI = FN*IH                                                        # inverse filtering in the noise
    fni = FFT.inverse_fft2d(FNI).real
    print [MLab.min(Numeric.ravel(fni)), MLab.max(Numeric.ravel(fni))] # min and max of restored noise
    iashow(ianormalize(fni, [0,255]))                                  # display restoration of noise'''
    import MLab
    fn = g - gfix                                                      # noise that was added
    print [MLab.mean(Numeric.ravel(fn)), MLab.min(Numeric.ravel(fn)), MLab.max(Numeric.ravel(fn))] # mean, minumum and maximum values
    iashow(ianormalize(fn, [0,255]))                                   # display noise
    FN = FFT.fft2d(fn)
    iashow(iadftview(FN))                                              # spectrum of the noise
    FNI = FN*IH                                                        # inverse filtering in the noise
    fni = FFT.inverse_fft2d(FNI).real
    print [MLab.min(Numeric.ravel(fni)), MLab.max(Numeric.ravel(fni))] # min and max of restored noise
    iashow(ianormalize(fni, [0,255]))                                  # display restoration of noise
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iamagnify - Illustrate the interpolation of magnified images
#
# =========================================================================
def iamagnify():
    from ia636 import iaread, iashow, iadftview, iapconv, iabwlp, ianormalize
    print
    print '''Illustrate the interpolation of magnified images'''
    print
    #
    print '========================================================================='
    print '''
    The image is read and a 64x64 ROI is selected and displayed
    '''
    print '========================================================================='
    #0
    print '''
    fin = iaread('lenina.pgm')
    iashow(fin)
    froi = fin[137:137+64,157:157+64]
    iashow(froi)
    print froi.shape'''
    fin = iaread('lenina.pgm')
    iashow(fin)
    froi = fin[137:137+64,157:157+64]
    iashow(froi)
    print froi.shape
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The DFT of the small image is taken and its spectrum displayed
    '''
    print '========================================================================='
    #0
    print '''
    import Numeric, FFT
    fd = froi.astype(Numeric.Float)
    F = FFT.fft2d(fd)
    iashow(froi)
    iashow(iadftview(F))'''
    import Numeric, FFT
    fd = froi.astype(Numeric.Float)
    F = FFT.fft2d(fd)
    iashow(froi)
    iashow(iadftview(F))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The image is expanded by 4, but filling the new pixels with 0
    '''
    print '========================================================================='
    #0
    print '''
    fx4 = Numeric.zeros(4*Numeric.array(froi.shape))
    fx4[::4,::4] = froi
    iashow(froi)
    iashow(fx4)'''
    fx4 = Numeric.zeros(4*Numeric.array(froi.shape))
    fx4[::4,::4] = froi
    iashow(froi)
    iashow(fx4)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Using the expansion propertie of the DFT (only valid for the
    discrete case), the resulting DFT is a periodical replication of the
    original DFT.
    '''
    print '========================================================================='
    #0
    print '''
    fdx4 = fx4.astype(Numeric.Float)
    Fx4 = FFT.fft2d(fdx4)
    iashow(fx4)
    iashow(iadftview(Fx4))'''
    fdx4 = fx4.astype(Numeric.Float)
    Fx4 = FFT.fft2d(fdx4)
    iashow(fx4)
    iashow(iadftview(Fx4))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering the expanded image using an average filter of size 4x4 is
    equivalent of applying a nearest neighbor interpolator. The zero
    pixels are replaced by the nearest non-zero pixel. This is
    equivalent to interpolation by pixel replication.
    '''
    print '========================================================================='
    #0
    print '''
    k = Numeric.ones((4,4))
    fx4nn = iapconv(fdx4, k)
    iashow(fx4)
    iashow(fx4nn.astype(Numeric.Int))'''
    k = Numeric.ones((4,4))
    fx4nn = iapconv(fdx4, k)
    iashow(fx4)
    iashow(fx4nn.astype(Numeric.Int))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering by the average filter in space domain is equivalent to
    filter in the frequency domain by the sync filter.
    '''
    print '========================================================================='
    #0
    print '''
    kzero = Numeric.zeros(fx4.shape)
    kzero[0:4,0:4] = k
    K = FFT.fft2d(kzero)
    iashow(iadftview(K))
    Fx4nn = K * Fx4
    iashow(iadftview(Fx4nn))'''
    kzero = Numeric.zeros(fx4.shape)
    kzero[0:4,0:4] = k
    K = FFT.fft2d(kzero)
    iashow(iadftview(K))
    Fx4nn = K * Fx4
    iashow(iadftview(Fx4nn))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering by a pyramidal kernel in space domain is equivalent to
    make a bi-linear interpolation. The zero pixels are replaced by a
    weighted sum of the neighbor pixels, the weight is inversely
    proportional to the non-zero pixel distance.
    '''
    print '========================================================================='
    #0
    print '''
    klinear = Numeric.array([1,2,3,4,3,2,1])/4.
    k2dlinear = Numeric.matrixmultiply(Numeric.reshape(klinear, (7,1)), Numeric.reshape(klinear, (1,7)))
    fx4li = iapconv(fdx4, k2dlinear)
    iashow(fx4)
    iashow(fx4li.astype(Numeric.Int))'''
    klinear = Numeric.array([1,2,3,4,3,2,1])/4.
    k2dlinear = Numeric.matrixmultiply(Numeric.reshape(klinear, (7,1)), Numeric.reshape(klinear, (1,7)))
    fx4li = iapconv(fdx4, k2dlinear)
    iashow(fx4)
    iashow(fx4li.astype(Numeric.Int))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering by the pyramid filter in space domain is equivalent to
    filter in the frequency domain by the square of the sync filter.
    '''
    print '========================================================================='
    #0
    print '''
    klizero = Numeric.zeros(fx4.shape).astype(Numeric.Float)
    klizero[0:7,0:7] = k2dlinear
    Klinear = FFT.fft2d(klizero)
    iashow(iadftview(Klinear))
    Fx4li = Klinear * Fx4
    iashow(iadftview(Fx4li))'''
    klizero = Numeric.zeros(fx4.shape).astype(Numeric.Float)
    klizero[0:7,0:7] = k2dlinear
    Klinear = FFT.fft2d(klizero)
    iashow(iadftview(Klinear))
    Fx4li = Klinear * Fx4
    iashow(iadftview(Fx4li))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering by cutoff period of 8
    '''
    print '========================================================================='
    #0
    print '''
    H8 = iabwlp(fx4.shape, 8, 10000)
    iashow(iadftview(H8))
    G8 = Fx4 * H8
    iashow(iadftview(G8))
    g_ideal = FFT.inverse_fft2d(G8)
    print max(Numeric.ravel(g_ideal.imag))
    g_ideal = ianormalize(g_ideal.real, [0,255])
    iashow(g_ideal)'''
    H8 = iabwlp(fx4.shape, 8, 10000)
    iashow(iadftview(H8))
    G8 = Fx4 * H8
    iashow(iadftview(G8))
    g_ideal = FFT.inverse_fft2d(G8)
    print max(Numeric.ravel(g_ideal.imag))
    g_ideal = ianormalize(g_ideal.real, [0,255])
    iashow(g_ideal)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Filtering by cutoff period of 8
    '''
    print '========================================================================='
    #0
    print '''
    HB8 = iabwlp(fx4.shape, 8, 5)
    iashow(iadftview(HB8))
    GB = Fx4 * HB8
    iashow(iadftview(GB))
    g_b = FFT.inverse_fft2d(GB)
    print max(Numeric.ravel(g_b).imag)
    g_b = ianormalize(g_b.real, [0,255])
    iashow(g_b)'''
    HB8 = iabwlp(fx4.shape, 8, 5)
    iashow(iadftview(HB8))
    GB = Fx4 * HB8
    iashow(iadftview(GB))
    g_b = FFT.inverse_fft2d(GB)
    print max(Numeric.ravel(g_b).imag)
    g_b = ianormalize(g_b.real, [0,255])
    iashow(g_b)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Top-left: nearest neighbor, Top-right: linear, Bottom-left: ideal,
    Bottom-right: Butterworth
    '''
    print '========================================================================='
    #0
    print '''
    aux1 = Numeric.concatenate((fx4nn[0:256,0:256], fx4li[0:256,0:256]), 1)
    aux2 = Numeric.concatenate((g_ideal, g_b), 1)
    iashow(Numeric.concatenate((aux1, aux2)))'''
    aux1 = Numeric.concatenate((fx4nn[0:256,0:256], fx4li[0:256,0:256]), 1)
    aux2 = Numeric.concatenate((g_ideal, g_b), 1)
    iashow(Numeric.concatenate((aux1, aux2)))
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return
# =========================================================================
#
#   iaotsudemo - Illustrate the Otsu Thresholding Selection Method
#
# =========================================================================
def iaotsudemo():
    from ia636 import iaread, iashow, iahistogram, iaplot, iaerror, iaplot
    print
    print '''Illustrate the Otsu Thresholding Selection Method'''
    print
    #
    print '========================================================================='
    print '''
    Gray scale image and its histogram
    '''
    print '========================================================================='
    #1
    print '''
    import Numeric
    f = iaread('woodlog.pgm');
    iashow(f)
    H = iahistogram(f)
    x = Numeric.arange(len(H))
    k = x[0:-1]
    g,d = iaplot(x, H)
    #showfig(H)'''
    import Numeric
    f = iaread('woodlog.pgm');
    iashow(f)
    H = iahistogram(f)
    x = Numeric.arange(len(H))
    k = x[0:-1]
    g,d = iaplot(x, H)
    #showfig(H)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    If the histogram is divided by the number of pixels in the image, it
    can be seen as a probability distribution. The sum of each values
    gives one. The mean gray level can be computed from the normalized
    histogram.
    '''
    print '========================================================================='
    #1
    print '''
    import MLab
    h = 1.*H/Numeric.product(f.shape)
    print Numeric.sum(h)
    mt = Numeric.sum(x * h)
    st2 = Numeric.sum((x-mt)**2 * h)
    if abs(mt - MLab.mean(Numeric.ravel(f))) > 0.01: iaerror('error in computing mean')
    if abs(st2 - MLab.std(Numeric.ravel(f))**2) > 0.0001: iaerror('error in computing var')
    print 'mean is %2.2f, var2 is %2.2f' % (mt, st2)
    maux = 0 * h
    maux[int(mt)] = max(h)
    g,d1 = iaplot(x,h)
    g,d2 = iaplot(x,maux)
    g.plot(d1, d2)
    #showfig(h)'''
    import MLab
    h = 1.*H/Numeric.product(f.shape)
    print Numeric.sum(h)
    mt = Numeric.sum(x * h)
    st2 = Numeric.sum((x-mt)**2 * h)
    if abs(mt - MLab.mean(Numeric.ravel(f))) > 0.01: iaerror('error in computing mean')
    if abs(st2 - MLab.std(Numeric.ravel(f))**2) > 0.0001: iaerror('error in computing var')
    print 'mean is %2.2f, var2 is %2.2f' % (mt, st2)
    maux = 0 * h
    maux[int(mt)] = max(h)
    g,d1 = iaplot(x,h)
    g,d2 = iaplot(x,maux)
    g.plot(d1, d2)
    #showfig(h)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    Suppose the pixels are categorized in two classes: smaller or equal
    than the gray scale t and larger than t. The probability of the
    first class occurrence is the cummulative normalized histogram. The
    other class is the complementary class.
    '''
    print '========================================================================='
    #1
    print '''
    w0 = MLab.cumsum(h[0:-1])
    aux = h[1::]
    w1aux = MLab.cumsum(aux[::-1])[::-1]
    w1 = 1 - w0
    if max(abs(w1-w1aux)) > 0.0001: iaerror('error in computing w1')
    g,d1 = iaplot(k,w0)
    g,d2 = iaplot(k,w1)
    g.plot(d1, d2)
    #showfig(w0,w1)'''
    w0 = MLab.cumsum(h[0:-1])
    aux = h[1::]
    w1aux = MLab.cumsum(aux[::-1])[::-1]
    w1 = 1 - w0
    if max(abs(w1-w1aux)) > 0.0001: iaerror('error in computing w1')
    g,d1 = iaplot(k,w0)
    g,d2 = iaplot(k,w1)
    g.plot(d1, d2)
    #showfig(w0,w1)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The mean gray level as a function of the thresholding t is computed
    and displayed below.
    '''
    print '========================================================================='
    #1
    print '''
    m0 = MLab.cumsum(k * h[0:-1]) / (1.*w0)
    m1 = (mt - m0*w0)/w1
    aux = (k+1) * h[1::]
    m1x = MLab.cumsum(aux[::-1])[::-1] / (1.*w1)
    mm = w0 * m0 + w1 * m1
    if max(abs(m1-m1x)) > 0.0001: iaerror('error in computing m1')
    g,d1 = iaplot(k,m0)
    g,d2 = iaplot(k,m1)
    g,d3 = iaplot(k,mm)
    g.plot(d1,d2,d3)
    #showfig(m0,m1)'''
    m0 = MLab.cumsum(k * h[0:-1]) / (1.*w0)
    m1 = (mt - m0*w0)/w1
    aux = (k+1) * h[1::]
    m1x = MLab.cumsum(aux[::-1])[::-1] / (1.*w1)
    mm = w0 * m0 + w1 * m1
    if max(abs(m1-m1x)) > 0.0001: iaerror('error in computing m1')
    g,d1 = iaplot(k,m0)
    g,d2 = iaplot(k,m1)
    g,d3 = iaplot(k,mm)
    g.plot(d1,d2,d3)
    #showfig(m0,m1)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The gray level variance as a function of the thresholding t is
    computed and displayed below.
    '''
    print '========================================================================='
    #1
    print '''
    s02 = MLab.cumsum((k-m0)**2 * h[0:-1]) / (1.*w0)
    aux = ((k+1)-m1)**2 * h[1::]
    s12 = MLab.cumsum(aux[::-1])[::-1] / (1.*w1)
    g,d1 = iaplot(k, s02)
    g,d2 = iaplot(k, s12)
    g.plot(d1, d2)
    #showfig(s02)'''
    s02 = MLab.cumsum((k-m0)**2 * h[0:-1]) / (1.*w0)
    aux = ((k+1)-m1)**2 * h[1::]
    s12 = MLab.cumsum(aux[::-1])[::-1] / (1.*w1)
    g,d1 = iaplot(k, s02)
    g,d2 = iaplot(k, s12)
    g.plot(d1, d2)
    #showfig(s02)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The variance between class is a good measure of class separability.
    As higher this variance, the better the class clustering.
    '''
    print '========================================================================='
    #1
    print '''
    sB2 = w0 * ((m0 - mt)**2) + w1 * ((m1 - mt)**2)
    sBaux2 = w0 * w1 * ((m0 - m1)**2)
    if max(sB2-sBaux2) > 0.0001: iaerror('error in computing sB')
    v = max(sB2)
    t = (sB2.tolist()).index(v)
    eta = 1.*sBaux2[t]/st2
    print 'Optimum threshold at %f quality factor %f' % (t, eta)
    g,d1 = iaplot(k, sB2)
    g,d2 = iaplot(k, H[0:-1])
    g,d3 = iaplot(k, s02)
    g,d4 = iaplot(k, s12)
    g.plot(d1, d2, d3, d4)
    #showfig(sB)'''
    sB2 = w0 * ((m0 - mt)**2) + w1 * ((m1 - mt)**2)
    sBaux2 = w0 * w1 * ((m0 - m1)**2)
    if max(sB2-sBaux2) > 0.0001: iaerror('error in computing sB')
    v = max(sB2)
    t = (sB2.tolist()).index(v)
    eta = 1.*sBaux2[t]/st2
    print 'Optimum threshold at %f quality factor %f' % (t, eta)
    g,d1 = iaplot(k, sB2)
    g,d2 = iaplot(k, H[0:-1])
    g,d3 = iaplot(k, s02)
    g,d4 = iaplot(k, s12)
    g.plot(d1, d2, d3, d4)
    #showfig(sB)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    ##
    print '========================================================================='
    print '''
    The thresholded image is displayed to illustrate the result of the
    binarization using the Otsu method.
    '''
    print '========================================================================='
    #0
    print '''
    x_ = iashow(f > t)'''
    x_ = iashow(f > t)
    print
    raw_input(4*' '+'Please press return to continue...')
    print
    print
    #
    return

# =====================================================================
#
#   script execution
#
# =====================================================================

#all demonstrations - initialization
_alldemos = []

_alldemos.append('iagenimages') 
_alldemos.append('iait') 
_alldemos.append('iahisteq') 
_alldemos.append('iacorrdemo') 
_alldemos.append('iadftdecompose') 
_alldemos.append('iadftexamples') 
_alldemos.append('iadftmatrixexamples') 
_alldemos.append('iadftscaleproperty') 
_alldemos.append('iaconvteo') 
_alldemos.append('iahotelling') 
_alldemos.append('iainversefiltering') 
_alldemos.append('iamagnify') 
_alldemos.append('iaotsudemo') 

#main execution
print '\nia636 Demonstrations -- Toolbox ia636\n'
print 'Available Demonstrations: \n' + str(_alldemos) + '\n'
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        for demo in sys.argv[1:]:
            if _alldemos.count(demo) > 0:
                eval(demo + '()')
            else:
                print "Demonstration " + demo + " is not in this package. Please use help for details\n"
    else:
        print "\nUsage: python ia636.py <demo_name>\n\n"
else:
    print 'Please use help(ia636demo) for details\n'

