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

/* Internal warp canvas  →  10 px / mm  →  exact integer arithmetic */
static constexpr int    WARP_W    = 2100;
static constexpr int    WARP_H    = 2970;
static constexpr double PX_PER_MM = WARP_W / A4_WIDTH_MM;   // 10.0

/* Temporal smoothing window */
static constexpr int SMOOTH_FRAMES = 7;

/* Checkerboard used for lens calibration
   Change to match your printed pattern (inner corners, not squares) */
static constexpr int   CB_COLS       = 9;   // inner corners per row
static constexpr int   CB_ROWS       = 6;   // inner corners per column
static constexpr float CB_SQUARE_MM  = 25.0f; // physical square size in mm

/* Minimum calibration frames before we accept the result */
static constexpr int   CALIB_MIN_FRAMES = 15;

/* ─────────────────────────────────────────────────────────────────────────
   Data structures
   ───────────────────────────────────────────────────────────────────────── */

struct ObjectMeasure {
    double  widthMM  = 0;
    double  heightMM = 0;
    double  areaMM2  = 0;
    double  perimMM  = 0;
    double  angleDeg = 0;
    QPointF centerMM;
    QString shapeLabel;
};

struct DetectedObject {
    QPainterPath  pathPx;   // warp-canvas pixel coords
    QPainterPath  pathMM;   // millimetre coords (for export)
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
    /* camera */
    void startCamera();
    void selectCamera(int index);
    /* main workflow */
    void captureImage();
    void runDetection();
    void resetAll();
    void togglePathOnly();
    void exportJson();
    /* calibration */
    void startCalibration();
    void collectCalibFrame();
    void finishCalibration();
    void clearCalibration();
    /* internal */
    void onFrameChanged(const QVideoFrame &frame);
    void onAutoDetectTimer();

private:
    /* ── UI ── */
    void buildUI();
    void applyTheme();
    void updateUI();
    void setStatus(const QString &msg, const QString &color = "#6e6460");

    /* ── Camera ── */
    void buildCameraForIndex(int index);

    /* ── Calibration ── */
    bool     loadCalibration();
    bool     saveCalibration();
    cv::Mat  undistortFrame(const cv::Mat &src) const;
    bool     isCalibrated() const { return !camMatrix.empty() && !distCoeffs.empty(); }
    void     drawCalibOverlay(QPainter &p, const QRect &vp);

    /* ── A4 detection ── */
    bool     detectAndWarpA4(const QImage &src);
    bool     tryHsvMethod   (const cv::Mat &bgr, cv::Mat &warped);
    bool     tryCannyMethod (const cv::Mat &bgr, cv::Mat &warped);
    bool     tryAdaptiveMethod(const cv::Mat &bgr, cv::Mat &warped);
    bool     warpFromQuad   (const cv::Mat &src,
                      std::vector<cv::Point2f> quad,
                      cv::Mat &warped);
    std::vector<cv::Point2f> orderQuad(std::vector<cv::Point2f> pts);

    /* ── Object detection ── */
    void     detectObjects(const QImage &warpedImg);
    QString  classifyShape(const std::vector<cv::Point> &approx,
                          double circularity);
    void     pushAndSmooth();

    /* ── Drawing ── */
    void     drawLiveView    (QPainter &p, const QRect &vp);
    void     drawDetectedView(QPainter &p, const QRect &vp);
    void     drawPathOnlyView(QPainter &p, const QRect &vp);
    void     drawCalibView   (QPainter &p, const QRect &vp);

    /* ── Export ── */
    void     exportPathsToJson();

    /* ── State machine ── */
    enum AppState {
        LiveView, CapturedView, DetectedView, PathOnlyView,
        Calibrating   // checkerboard collection mode
    };
    AppState currentState = LiveView;

    /* ── Camera ── */
    QCamera              *camera         = nullptr;
    QVideoSink           *videoSink      = nullptr;
    QMediaCaptureSession *captureSession = nullptr;
    QList<QCameraDevice>  availCameras;

    /* ── Images ── */
    QImage liveFrame;
    QImage capturedFrame;
    QImage warpedA4;

    /* ── Lens calibration data ── */
    cv::Mat camMatrix;           // 3×3 intrinsic matrix
    cv::Mat distCoeffs;          // distortion coefficients (k1,k2,p1,p2,k3)
    cv::Mat optimalMatrix;       // getOptimalNewCameraMatrix result
    cv::Rect validROI;           // valid pixel region after undistort
    double  calibRmsError = 0.0; // reprojection error (pixels) – lower = better

    /* Collected calibration samples */
    std::vector<std::vector<cv::Point2f>> calibImagePts;   // detected corners
    std::vector<std::vector<cv::Point3f>> calibObjPts;     // 3-D world pts
    cv::Size calibImageSize;
    QImage   calibPreviewFrame;   // last frame with corners drawn on it

    /* ── Detection results ── */
    std::vector<DetectedObject>            objects;
    std::deque<std::vector<ObjectMeasure>> history;

    /* ── A4 lock state ── */
    std::vector<cv::Point2f> lastQuad;
    int    quadLockCount = 0;
    bool   a4Locked      = false;
    double pixPerMM      = PX_PER_MM;

    /* ── UI widgets ── */
    QPushButton *btnStart       = nullptr;
    QPushButton *btnCapture     = nullptr;
    QPushButton *btnDetect      = nullptr;
    QPushButton *btnReset       = nullptr;
    QPushButton *btnPathOnly    = nullptr;
    QPushButton *btnExport      = nullptr;
    QPushButton *btnCalib       = nullptr;   // "Calibrate Lens"
    QPushButton *btnCollect     = nullptr;   // "Collect Frame"  (calib mode)
    QPushButton *btnFinishCalib = nullptr;   // "Finish & Apply" (calib mode)
    QPushButton *btnClearCalib  = nullptr;   // "Clear Calibration"
    QComboBox   *cameraCombo    = nullptr;
    QCheckBox   *chkOverlay     = nullptr;
    QCheckBox   *chkAutoDetect  = nullptr;
    QLabel      *lblStatus      = nullptr;
    QLabel      *lblA4Lock      = nullptr;
    QLabel      *lblObjects     = nullptr;
    QLabel      *lblCalibStatus = nullptr;   // shows "Calibrated ✓" or "Not calibrated"

    /* ── Timers ── */
    QTimer *autoTimer = nullptr;

    int selectedObject = -1;
};
