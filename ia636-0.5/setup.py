#
#   Setup script for toolbox ia636
#
import os, sys, re, shutil, zipfile
from distutils.core import setup, Extension
from distutils.util import get_platform
#
itdir = 'adtools'
mydir = os.path.join(itdir, 'ia63605all')
mypth = itdir + '.pth'
#
_plat = get_platform()
_ver = 'py' + sys.version[0] + sys.version[2]
if re.match('win', _plat):
    _os = 'windows'
    _lib = '_ia63605all.lib'
    _prefix = os.path.join(sys.prefix, 'lib', 'site-packages')
    if os.path.isdir(_prefix):
        _headers = os.path.join(_prefix, mydir)
        _install = os.path.join(_prefix, mydir)
        _rpath = [itdir + '\n', mydir + '\n', os.path.join(mydir, 'legacy') + '\n', os.path.join(mydir, 'data') + '\n']
    else:
        _prefix = sys.prefix
        _headers = os.path.join(_prefix, 'lib', mydir)
        _install = os.path.join(_prefix, 'lib', mydir)
        _rpath = [os.path.join('lib', itdir) + '\n', os.path.join('lib', mydir) + '\n', os.path.join('lib', mydir, 'legacy') + '\n', os.path.join('lib', mydir, 'data') + '\n']
elif re.match('linux', _plat):
    _os = 'linux'
    _lib = None
    _prefix = os.path.join(sys.prefix, 'lib', 'python' + sys.version[:3], 'site-packages')
    _headers = os.path.join(_prefix, mydir)
    _install = os.path.join(_prefix, mydir)
    _rpath = [itdir + '\n', mydir + '\n', os.path.join(mydir, 'legacy') + '\n', os.path.join(mydir, 'data') + '\n']
elif re.match('solaris', _plat):
    _os = 'sunos'
    _lib = None
    _prefix = os.path.join(sys.prefix, 'lib', 'python' + sys.version[:3], 'site-packages')
    _headers = os.path.join(_prefix, mydir)
    _install = os.path.join(_prefix, mydir)
    _rpath = [itdir + '\n', mydir + '\n', os.path.join(mydir, 'legacy') + '\n', os.path.join(mydir, 'data') + '\n']
else:
    print 'Unsupported platform'
    sys.exit()
#
if not os.path.isfile(os.path.join(_prefix, mypth)):
    if not os.path.isfile(mypth):
        _pth = open(mypth, 'w')
        _pth.writelines(_rpath)
        _pth.close()
    my_data_files = [(_prefix, [mypth])]
else:
    _pth = open(os.path.join(_prefix, mypth), 'r')
    _trpath = _pth.readlines()
    _pth.close()
    xsave = 0
    for p in _rpath:
        if p not in _trpath:
            _trpath.append(p)
            xsave = 1
    if xsave:
        _pth = open(mypth, 'w')
        _pth.writelines(_trpath)
        _pth.close()
        my_data_files = [(_prefix, [mypth])]
    else:
        my_data_files = []
#
if not os.path.isfile('__init__.py'):
    _ini = open('__init__.py', 'w')
    _ini.write('# ia636 0.5 all\n')
    _ini.close()
my_data_files += [(os.path.join(_prefix, mydir), ['__init__.py'])]
#
_basedir = 'Build'
_puredir = os.path.join(_basedir, 'lib')
_platdir = os.path.join(_basedir, 'lib.' + _os + '.' + _ver)
_tmpdir = os.path.join(_basedir, 'tmp.' + _os + '.' + _ver)
_bdistdir = os.path.join(_basedir, 'bdist.' + _os + '.' + _ver)
_distdir = os.path.join(_basedir, 'dist')
## if not os.path.isfile('setup.cfg') or (os.path.getmtime('setup.py') > os.path.getmtime('setup.cfg')):
if 1:
    _opts = [
        '[build]\n',
        'build_base = ', _basedir, '\n',
        'build_purelib = ', _puredir, '\n',
        'build_platlib = ', _platdir, '\n',
        'build_temp = ', _tmpdir, '\n',
        '\n'
        '[sdist]\n',
        'dist_dir = ', _distdir, '\n',
        '\n'
        '[bdist]\n',
        'dist_dir = ', _distdir, '\n',
        'bdist_base = ', _bdistdir, '\n',
        '\n'
        '[bdist_dumb]\n',
        'dist_dir = ', _distdir, '\n',
        'bdist_dir = ', _bdistdir, '\n',
        '\n'
        '[bdist_wininst]\n',
        'dist_dir = ', _distdir, '\n',
        'bdist_dir = ', _bdistdir, '\n',
        '\n'
        '[install]\n',
        'install_lib = ', _install, '\n',
        'install_headers = ', _headers, '\n',
        '\n'
    ]
    _cfg = open('setup.cfg', 'w')
    _cfg.writelines(_opts)
    _cfg.close()

#
my_headers = ['ia636.h']
#
my_ext_modules = []
#
#   Pure python:
#
my_py_modules1 = ['legacy/View']
# my_py_modules = ['ia636', 'ia636demo', 'ia636test'] + my_py_modules1
my_py_modules = ['ia636', 'ia636demo'] + my_py_modules1     # nao instala testsuite

#
tbx_data_files = ['data/blobs.pbm', 'data/boat.ppm', 'data/cameraman.pgm', 'data/club.pgm', 'data/cookies.pgm', 'data/gull.pgm', 'data/keyb.pgm', 'data/lenina.pgm', 'data/tvframe.pgm', 'data/woodlog.pgm']
if tbx_data_files:
    my_data_files = my_data_files + [(os.path.join(_install, 'data'), tbx_data_files)]
#
#   -------------------------------------------------------------------
#
#   finally, the command:
#
setup( name = "ia636",
       version = "0.5 all",
       description = """Toolbox ia636
Functions and demonstrations used in the IA636 course.""",
       author = """""",
       py_modules = my_py_modules,
       ext_modules = my_ext_modules,
       headers = my_headers,
       data_files = my_data_files
)
#
#
#
#
def copySetup(dir):
    ff = open('setup.py', 'r')
    txt = ff.read()
    ff.close()
    ff = open(os.path.join(dir, 'setup.py'), 'w')
    ff.write('import sys\n')
    ff.write("sys.argv = ['setup.py', 'install', '--skip-build']\n")
    ff.write(txt)
    ff.write("\nraw_input('\\n>>> ' + 'Please press return to finish ...')\n")
    ff.close()
#
def zipStore(zip, dir):
    for ff in os.listdir(dir):
        ff = os.path.join(dir, ff)
        if os.path.isdir(ff):
            zipStore(zip, ff)
        else:
            print '... appending', ff
            zip.write(ff)

#
def mkZip(ziptop, zipsrc):
    ziparch = ziptop + '.zip'
    ziplib = os.path.join(ziptop, zipsrc)
    zipbuild = os.path.join(ziptop, 'Build')
    print '--'
    print 'Now we will create the zip package:', ziparch
    #
    if os.path.exists(ziptop):
        shutil.rmtree(ziptop)
    os.mkdir(ziptop)
    os.mkdir(zipbuild)
    shutil.copytree(zipsrc, ziplib)
    shutil.copy('ia636.h', ziptop)
    copySetup(ziptop)
    for ff in tbx_data_files:
        datadir = os.path.join(ziptop, os.path.dirname(ff))
        if not os.path.isdir(datadir):
            os.mkdir(datadir)
        shutil.copy(ff, datadir)
    if os.path.isfile(ziparch):
        os.remove(ziparch)
    try:
        zip = zipfile.ZipFile(ziparch, 'w', zipfile.ZIP_DEFLATED)
        zipStore(zip, ziptop)
        zip.close()
        shutil.rmtree(ziptop)
    except:
        print '*** Cannot create compressed binary package'
        print '... Create it by compressing the directory', ziptop
        pass
    print '... SUCCESS!'
    print '... The file %s contains the binary distribution for toolbox ia636.' % ziparch

#
if sys.argv[1] == 'build':
    mkZip('ia6364py', _puredir)

