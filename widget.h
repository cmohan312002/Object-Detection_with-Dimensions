#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QCamera>
#include <QVideoSink>
#include <QMediaCaptureSession>
#include <QPushButton>
#include <QCheckBox>
#include <QImage>
#include <QPainterPath>

#include <opencv2/opencv.hpp>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>

struct PathCommand
{
    QString cmd;   // "M", "L", "C", "Z"
    double x = 0;
    double y = 0;
    double cx1 = 0;
    double cy1 = 0;
    double cx2 = 0;
    double cy2 = 0;
};

struct VectorObject
{
    int id;
    QVector<PathCommand> commands;
};


struct DigitalPathCommand
{
    enum Type { MoveTo, LineTo, CurveTo } type;

    QPointF p;   // end point (in mm)
    QPointF c1;  // control point 1 (mm) – for curves
    QPointF c2;  // control point 2 (mm) – for curves
};

using DigitalObject = QVector<DigitalPathCommand>;


/* ===================== STATE ===================== */
enum ViewState {
    LiveView,
    CapturedView,
    DetectedView,
    PathOnlyView
};

/* ===================== MEASURE STRUCT ===================== */
struct ObjectMeasure {
    double widthMM;
    double heightMM;
    double areaMM2;
};

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent *) override;

private slots:
    void startCamera();
    void onFrameChanged(const QVideoFrame &frame);
    void captureImage();
    void runDetection();
    void resetView();
    void showPathOnlyView();
    void exportPathsToJson();



private:
    /* UI */
    QPushButton *startButton;
    QPushButton *captureButton;
    QPushButton *detectButton;
    QPushButton *resetButton;
    QCheckBox   *showOverlayCheck;
    QPushButton *pathOnlyButton;
    QPushButton *exportButton;


    /* Camera */
    QCamera *camera;
    QVideoSink *videoSink;
    QMediaCaptureSession *captureSession;

    /* Images */
    QImage liveFrame;
    QImage capturedFrame;
    QImage warpedA4;

    /* State */
    ViewState currentState = LiveView;

    /* Detection */
    bool detectAndWarpA4(const QImage &image);
    void detectObjectsInsideA4(const QImage &img);

    void thinning(cv::Mat &img);
    void thinningIteration(cv::Mat &img, int iter);

    void dumpPainterPath(const QPainterPath&, int);

    /* Measurement */
    void computeAveragedResults();
    double pixelsPerMM = 1.0;
    void saveObjectsToJson(const QString &filePath);

    QVector<DigitalObject> digitalObjects;
    DigitalObject convertToDigitalObject(const QPainterPath &path);
    VectorObject convertPathToVector(
        const QPainterPath &path,
        int objectId
        );

    QPainterPath buildPainterPathFromVector(
        const VectorObject &obj
        );

    QVector<VectorObject> vectorObjects;

    std::vector<QPainterPath> objectPaths;
    std::vector<ObjectMeasure> frameMeasures;
    std::vector<std::vector<ObjectMeasure>> measurementHistory;

    /* UI helper */
    void updateUI();
};

#endif // WIDGET_H
