#! /usr/bin/env python
"""
Displaying image files in a separate thread on Tk+thread, w/ xv in
forked & execv'ed processes otherwise.

view(array):  will spawn a displaying program for arrays which are
              either NxM or NxMx3.  does the 'min/max' and conversion
              to char.

array2ppm(array): given an NxM or NxMx3 array, returns a ppm string
                  which is a valid thing to put in a PPM file.  (or
                  PGM file if NxM file).

TODO:
  - automatic scaling for small images
  - accept rank-1 arrays


"""
PPMVIEWER = 'xv' # configure
DEFAULT_HEIGHT = 255
MINSIZE = 150
CONTROLE = 1
import os
from Numeric import *
import tempfile, time
try:
    import Tkinter
    _have_tkinter = 1
    try:
        import Image
        _have_PIL = 1
        _have_tkImaging = 1# assume until proven wrong
    except ImportError:
        _have_PIL = 0
        _have_tkImaging = 0
except ImportError:
    _have_tkinter = 0
try:
    from threading import *
    _have_threads = 1
except ImportError:
    _have_threads = 0

def save_ppm(ppm, fname=None):
    if fname == None:
        fname = tempfile.mktemp('.ppm')
    f = open(fname, 'wb')
    f.write(ppm)
    f.close()
    return fname


def array2ppm(image):
    # scaling
    if len(image.shape) == 2:
        # B&W:
        image = transpose(image)
        return "P5\n#PPM version of array\n%d %d\n255\n%s" % \
               (image.shape[1], image.shape[0], ravel(image).tostring())
    else:
        # color
        image = transpose(image, (1, 0, 2))
        return "P6\n%d %d\n255\n%s" % \
               (image.shape[1], image.shape[0], ravel(image).tostring())

def preprocess(image, (scalex,scaley)):
    assert len(image.shape) in (1, 2) or \
           len(image.shape) == 3 and image.shape[2] == 3, \
           "image not correct format"
    themin = float(minimum.reduce(ravel(image)))
    themax = float(maximum.reduce(ravel(image)))
    if len(image.shape) == 1:
        len_x = image.shape[0]
        ys = ((image - themin)/(themax-themin)*(DEFAULT_HEIGHT-1)).astype('b')
        image = (zeros((DEFAULT_HEIGHT, len_x))+255).astype('b')
        for x in range(len_x):
            image[DEFAULT_HEIGHT-1-ys[x],len_x-x-1] = 0
        image = transpose(image)
    elif image.typecode() != 'b':
        image = (image - themin) / (themax-themin) * 255
        image = image.astype('b')

    len_x, len_y = image.shape[:2]
    if scalex is None:
        if len_x < MINSIZE:
            scalex = int(float(MINSIZE) / len_x) + 1
        else:
            scalex = 1
    if scaley is None:
        if len_y < MINSIZE:
            scaley = int(float(MINSIZE) / len_y) + 1
        else:
            scaley = 1
    return image, (scalex, scaley)

#----
# threaded stuff starts here
#----

import sys

_inidle = type(sys.stdin) == types.InstanceType and \
	  sys.stdin.__class__.__name__ == 'PyShell'

if _have_tkinter and (_have_threads or _inidle):

    def tk_root():
	if Tkinter._default_root is None:
	    root = Tkinter.Tk()
	    Tkinter._default_root.withdraw()
	else:
	    root = Tkinter._default_root
	return root

    # Modificacao feita no "view" do "NumTut"
    # Inclusao de um teste com "try" pois nao
    # eh possivel abrir uma janela Tkinter
    # dentro do interpretador Tcl do Adesso
    try:
        _root = tk_root()
    except:
        CONTROLE = 0
        pass

    if CONTROLE:

        class PILImage(Tkinter.Label):
            def __init__(self, master, data, (scalex, scaley)):
                width, height = data.shape[:2]
                if len(data.shape) == 3:
                    mode = rawmode = 'RGB'
                    bits = transpose(data, (1,0,2)).tostring()
                else:
                    mode = rawmode = 'L'
                    bits = transpose(data, (1,0)).tostring()
                self.image2 = Image.fromstring(mode, (width, height),
                                              bits, "raw", rawmode)
                import ImageTk
                self.image = ImageTk.PhotoImage(self.image2)
                Tkinter.Label.__init__(self, master, image=self.image,
                                       bg='black', bd=0)

        class PPMImage(Tkinter.Label):
            def __init__(self, master, ppm, (scalex, scaley)):
                self.image = Tkinter.PhotoImage(file=save_ppm(ppm))
                w, h = self.image.width(), self.image.height()
                self.image = self.image.zoom(scalex, scaley)
                self.image.configure(width=w*scalex, height=h*scaley)
                Tkinter.Label.__init__(self, master, image=self.image,
                                       bg="black", bd=0)

                self.pack()


        class ThreadedTk(Thread):

            def __init__(self, *args, **kw):
                self._done = 0

                apply(Thread.__init__, (self,)+args, kw)

            def done(self):
                self._done = 1

            def run(self):
                global _have_tkImaging

                while not self._done:
                    _pendinglock.acquire()
                    if len(_pendingarrays):       # there are files to process
                        for image, scales in _pendingarrays:
                            tl = Tkinter.Toplevel(background='black')
                            if _have_tkImaging:
                                try:
                                    u = PILImage(tl, image, scales)
                                except Tkinter.TclError:
                                    print "Error loading tkImaging"
                                    _have_tkImaging = 0
                                    u = PPMImage(tl, array2ppm(image), scales)
                            else:
                                u = PPMImage(tl, array2ppm(image), scales)
                            u.pack(fill='both', expand=1)
                            u.tkraise()
                        del _pendingarrays[:]   # we're done
                    _pendinglock.release()
                    _root.update()  # do your thing
                    time.sleep(0.01)   # go to sleep little baby

        def view(image, scale=(None,None)):
            global _have_tkImaging

            if len(image.shape) == 3: ### --> alexgs
                aux = zeros((image.shape[1],image.shape[2],3))
                aux[:,:,0], aux[:,:,1], aux[:,:,2] = image[0,:,:], image[1,:,:], image[2,:,:]
                image = transpose(aux, (1,0,2))
            elif len(image.shape) == 2:
                image = transpose(image) ### <---

            image, scales = preprocess(image, scale)
            if _inidle:
                tl = Tkinter.Toplevel()
                if _have_tkImaging:
                    try:
                        u = PILImage(tl, image, scales)
                    except Tkinter.TclError:
                        print "Error loading tkImaging"
                        _have_tkImaging = 0
                        u = PPMImage(tl, array2ppm(image), scales)
                else:
                    u = PPMImage(tl, array2ppm(image), scales)
                u.pack(fill='both', expand=1)
                u.tkraise()
            else:
                _pendinglock.acquire()
                _pendingarrays.append((image, scales))
                _pendinglock.release()
                while len(_pendingarrays):
                    time.sleep(0.01)

        if _inidle:
            def done(*args): pass
        else:
            _pendingarrays = []
            _pendinglock = Lock()
            _t = ThreadedTk() # this starts a Tk interpreter in a separate thread
            _t.start()
            done = _t.done
else:
    if CONTROLE:
        import sys
        # we're either w/o tk or w/o threads, so we hope we're on unix.
        if sys.platform == 'win32':
            if not _have_tkinter:
                if not _threads:
                    raise 'ConfigurationError', "view needs Tkinter on Win32, and either threads or the IDLE editor"
                elif not _inidle:
                    raise 'ConfigurationError', "view needs either threads or the IDLE editor to be enabled"
        children = []
        def view(image):
            global children
            image, scales = preprocess(image, (None,None))
            try:
                pid = os.fork()
                if pid == 0:
                    ppm = array2ppm(image)
                    try:
                        ret = os.system('%s %s' % (PPMVIEWER, save_ppm(ppm)))
                        if ret != 0:
                            raise 'ConfigurationError', "PPM image viewer '%s' not found" %PPMVIEWER
                    except:
                        raise 'ConfigurationError', "PPM image viewer '%s' not found" %PPMVIEWER
                else:
                    children.append(pid)
            except:
                print "Your system has neither threads nor 'fork'."
                print "As a result, this program can't run interactively."
                print "We'll spawn the image viewer and exit."
                ppm = array2ppm(image)
                os.system('%s %s' % (PPMVIEWER, save_ppm(ppm)))
        def done():
            import signal
            for pid in children:
                os.kill(pid, signal.SIGQUIT)

if CONTROLE:
    # this little bit cleans up
    import sys
    if hasattr(sys, 'exitfunc'):
        oldexitfunc = sys.exitfunc
    else:
        oldexitfunc = None
    def cleanup():
        done()
        if oldexitfunc is not None:
            oldexitfunc()
    sys.exitfunc = cleanup


