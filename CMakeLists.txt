project(flames)

cmake_minimum_required(VERSION 2.8)

find_package(Qt4 4.6.2 COMPONENTS QtCore QtGui QtOpenGL)
find_package(CUDA)

#-----------------------------------
# find GLEW Snippet Pinched from OIIO
#set (GLEW_VERSION 1.5.1)
#find_library (GLEW_LIBRARIES NAMES GLEW)
#find_path (GLEW_INCLUDES
#           NAMES glew.h
#           PATH_SUFFIXES GL)
#if (GLEW_INCLUDES AND GLEW_LIBRARIES)
#    set (GLEW_FOUND TRUE)
#    message (STATUS "GLEW includes = ${GLEW_INCLUDES}")
#    message (STATUS "GLEW library = ${GLEW_LIBRARIES}")
#else ()
#    message (STATUS "GLEW not found")
#endif ()
#-----------------------------------


include_directories(${QT_INCLUDES} ${GLEW_INCLUDES} ${CUDA_INCLUDE_DIRS})

qt4_wrap_cpp(moc_srcs
    flames.h
)
cuda_compile(cuda_srcs compute_flames.cu)
set(srcs
    ${moc_srcs}
    ${cuda_srcs}
    flames.cpp
    compute_flames.cpp
)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
set(CMAKE_CXX_FLAGS "-Wall")

add_executable(flames ${srcs})
target_link_libraries(flames ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} ${QT_QTOPENGL_LIBRARY}
                      ${CUDA_LIBRARIES})
	#${GLEW_LIBRARIES})


