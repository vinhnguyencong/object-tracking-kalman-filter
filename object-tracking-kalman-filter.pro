#-------------------------------------------------
#
# Project created by QtCreator 2018-08-18T15:35:56
#
#-------------------------------------------------

QT       += core gui

INCLUDEPATH += -I/usr/local/include

LIBS += -L/usr/local/lib -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_core \
        -lopencv_highgui -lopencv_tracking

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = object-tracking-kalman-filter
TEMPLATE = app


SOURCES += main.cpp\
        dialog.cpp \
    detectors.cpp

HEADERS  += dialog.h \
    detectors.h

FORMS    += dialog.ui
