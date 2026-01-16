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

/* ===================== STATE ===================== */
enum ViewState {
    LiveView,
    CapturedView,
    DetectedView
};

/* ===================== MEASURE STRUCT ===================== */
struct ObjectMeasure {
    double widthMM = 0.0;
    double heightMM = 0.0;
    double diameterMM = 0.0;
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

private:
    /* UI */
    QPushButton *startButton;
    QPushButton *captureButton;
    QPushButton *detectButton;
    QPushButton *resetButton;
    QCheckBox   *showOverlayCheck;

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

    std::vector<QPainterPath> objectPaths;
    std::vector<ObjectMeasure> frameMeasures;
    std::vector<std::vector<ObjectMeasure>> measurementHistory;

    /* UI helper */
    void updateUI();
};

#endif // WIDGET_H
