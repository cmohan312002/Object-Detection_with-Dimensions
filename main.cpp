#include <QApplication>
#include "widget.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("A4 Object Digitizer");
    app.setApplicationVersion("2.0");
    app.setOrganizationName("VisionLab");

    Widget w;
    w.show();
    return app.exec();
}
