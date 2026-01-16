QT += core gui widgets multimedia multimediawidgets
CONFIG += c++17

INCLUDEPATH += C:/opencv-mingw/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/include

LIBS += -LC:/opencv-mingw/OpenCV-MinGW-Build-OpenCV-4.5.5-x64/x64/mingw/lib \
        -lopencv_core455 \
        -lopencv_imgproc455 \
        -lopencv_imgcodecs455 \
        -lopencv_videoio455

SOURCES += \
    main.cpp \
    widget.cpp

HEADERS += \
    widget.h

DISTFILES += \
    .gitignore
