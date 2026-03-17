#pragma once

#include <QWidget>
#include <QPushButton>
#include <QCheckBox>
#include <QLabel>
#include <QComboBox>
#include <QProgressBar>
#include <QCamera>
#include <QMediaDevices>
#include <QVideoSink>
#include <QMediaCaptureSession>
#include <QVideoFrame>
#include <QPainterPath>
#include <QPointF>
#include <QImage>
#include <QTimer>
#include <QElapsedTimer>

#include <opencv2/opencv.hpp>

#include <vector>
#include <deque>
#include <array>
#include <optional>

/* ─────────────────────────────────────────────────────────────────────────
   Physical A4 constants (ISO 216)
   ───────────────────────────────────────────────────────────────────────── */
static constexpr double A4_WIDTH_MM  = 210.0;
static constexpr double A4_HEIGHT_MM = 297.0;

/* Internal warp canvas (high resolution keeps sub-mm accuracy) */
static constexpr int WARP_W = 2100;   // 10 px/mm
static constexpr int WARP_H = 2970;
static constexpr double PX_PER_MM = WARP_W / A4_WIDTH_MM;  // = 10.0

/* Temporal smoothing window */
static constexpr int SMOOTH_FRAMES = 7;

/* ─────────────────────────────────────────────────────────────────────────
   Data structures
   ───────────────────────────────────────────────────────────────────────── */

struct ObjectMeasure {
    double widthMM  = 0;
    double heightMM = 0;
    double areaMM2  = 0;
    double perimMM  = 0;
    double angleDeg = 0;   // orientation of bounding box
    QPointF centerMM;      // center in mm from A4 top-left
    QString shapeLabel;    // "Rectangle", "Circle", "Triangle", etc.
};

struct DetectedObject {
    QPainterPath  pathPx;      // in warp-canvas pixel space
    QPainterPath  pathMM;      // in mm space (for export)
    ObjectMeasure measure;
};

/* ─────────────────────────────────────────────────────────────────────────
   Widget
   ───────────────────────────────────────────────────────────────────────── */
class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget() override = default;

protected:
    void paintEvent(QPaintEvent *) override;
    void resizeEvent(QResizeEvent *) override;

private slots:
    void startCamera();
    void selectCamera(int index);
    void captureImage();
    void runDetection();
    void resetAll();
    void togglePathOnly();
    void exportJson();
    void onFrameChanged(const QVideoFrame &frame);
    void onAutoDetectTimer();

private:
    /* ── UI ── */
    void buildUI();
    void applyDarkTheme();
    void updateUI();
    void setStatus(const QString &msg, const QString &color = "#aaaaaa");

    /* ── Camera helpers ── */
    void buildCameraForIndex(int index);

    /* ── Image processing ── */
    bool     detectAndWarpA4(const QImage &src);
    bool     tryHsvMethod(const cv::Mat &bgr, cv::Mat &warped);
    bool     tryCannyMethod(const cv::Mat &bgr, cv::Mat &warped);
    bool     tryAdaptiveMethod(const cv::Mat &bgr, cv::Mat &warped);
    bool     warpFromQuad(const cv::Mat &src,
                      std::vector<cv::Point2f> quad,
                      cv::Mat &warped);
    std::vector<cv::Point2f> orderQuad(std::vector<cv::Point2f> pts);

    void     detectObjects(const QImage &warpedImg);
    QString  classifyShape(const std::vector<cv::Point> &approx,
                          double circularity);

    /* ── Smoothing ── */
    void     pushAndSmooth();

    /* ── Drawing helpers ── */
    void     drawLiveView(QPainter &p, const QRect &vp);
    void     drawDetectedView(QPainter &p, const QRect &vp);
    void     drawPathOnlyView(QPainter &p, const QRect &vp);

    /* ── Export ── */
    void     exportPathsToJson();

    /* ── State machine ── */
    enum AppState { LiveView, CapturedView, DetectedView, PathOnlyView };
    AppState currentState = LiveView;

    /* ── Camera ── */
    QCamera              *camera        = nullptr;
    QVideoSink           *videoSink     = nullptr;
    QMediaCaptureSession *captureSession= nullptr;
    QList<QCameraDevice>  availCameras;

    /* ── Images ── */
    QImage liveFrame;
    QImage capturedFrame;
    QImage warpedA4;

    /* ── Detection results ── */
    std::vector<DetectedObject>          objects;
    std::deque<std::vector<ObjectMeasure>> history;   // for temporal avg

    /* ── A4 detection state ── */
    std::vector<cv::Point2f> lastQuad;   // last valid A4 quad (in live px)
    int   quadLockCount  = 0;            // consecutive frames with same quad
    bool  a4Locked       = false;
    double pixPerMM      = PX_PER_MM;

    /* ── UI widgets ── */
    QPushButton *btnStart    = nullptr;
    QPushButton *btnCapture  = nullptr;
    QPushButton *btnDetect   = nullptr;
    QPushButton *btnReset    = nullptr;
    QPushButton *btnPathOnly = nullptr;
    QPushButton *btnExport   = nullptr;
    QComboBox   *cameraCombo = nullptr;
    QCheckBox   *chkOverlay  = nullptr;
    QCheckBox   *chkAutoDetect = nullptr;
    QLabel      *lblStatus   = nullptr;
    QLabel      *lblA4Lock   = nullptr;
    QLabel      *lblObjects  = nullptr;

    /* ── Auto-detect timer ── */
    QTimer *autoTimer = nullptr;

    /* ── Selection ── */
    int selectedObject = -1;
};
