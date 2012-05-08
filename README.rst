====================================
FlameEd - a GPU fractal flame editor
====================================

FlameEd is a fractal flame editor written in Qt, CUDA and OpenGL.  It was
written for a bit of fun and to present at the Brisbane GPU users group.

The main aim here is to have as little interface as possible between the user
and the fractal: To enable "direct editing" of the fractal without worrying too
much about transformations.  This reasonably lofty goal isn't really achieved
in the current version, but it does allow full screen interactive fractal
editing.

The control scheme is evolving and is therefore a bit of a mess; see
FlameViewWidget::keyPressEvent() in flamed.cpp for details.


Building and running
--------------------

First you need the CUDA, and Qt4 libraries, as well as the OpenGL headers for
your system.  You also need cmake which is used to manage the build system.

To build on linux::

  cd $FLAMED_SOURCE
  mkdir build
  cd build
  cmake ../
  make

Currently you must run from the build directory::

  cd $FLAMED_SOURCE/build
  ./flamed

