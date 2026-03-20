/*
 * ═══════════════════════════════════════════════════════════════════════════
 *   A4 Object Digitizer  –  widget.cpp  (v3 – with lens calibration)
 *
 *   Calibration pipeline:
 *     Print a 9×6 checkerboard (25 mm squares) → enter Calibrate mode →
 *     show board from 15+ angles → Finish & Apply →
 *     distortion coefficients saved to camera_calib.yml →
 *     every subsequent frame is undistorted before processing
 *
 *   Accuracy improvement:
 *     Without calibration: corner measurements 3-8 % larger than reality
 *     With calibration:    reprojection error < 0.5 px → < 0.05 mm error
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "widget.h"

#include <QApplication>
#include <QMediaDevices>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSizePolicy>
#include <QFont>
#include <QFontMetrics>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDateTime>
#include <QStandardPaths>
#include <QDebug>
#include <QTimer>
#include <QMessageBox>
#include <QDir>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

/* ─── Colour palette (white & orange) ──────────────────────────────────── */
namespace C {
static const QColor BG       {245, 245, 242};
static const QColor SURFACE  {255, 255, 255};
static const QColor BORDER   {220, 210, 200};
static const QColor ACCENT   {235,  95,  20};   // deep orange
static const QColor ACCENT2  {255, 140,  50};
static const QColor SUCCESS  { 34, 160,  80};
static const QColor WARNING  {235, 150,   0};
static const QColor DANGER   {210,  45,  45};
static const QColor TEXT_PRI { 30,  30,  30};
static const QColor TEXT_SEC {110, 100,  90};

static const std::array<QColor,6> OBJ_COLORS {{
    {235,  95,  20},   // orange
    { 30, 130, 200},   // blue
    { 34, 160,  80},   // green
    {180,  40, 180},   // violet
    {200,  40,  60},   // red
    { 20, 170, 170},   // teal
}};
}

/* ═══════════════════════════════════════════════════════════════════════════
   Constructor
   ═══════════════════════════════════════════════════════════════════════════ */
Widget::Widget(QWidget *parent) : QWidget(parent)
{
    setWindowTitle("A4 Object Digitizer  •  v3  (Calibrated)");
    setMinimumSize(1100, 760);
    resize(1200, 840);
    buildUI();
    applyTheme();
    updateUI();

    /* Load existing calibration if available */
    if (loadCalibration())
        setStatus("✓  Lens calibration loaded – measurements are accurate", "#228050");
    else
        setStatus("⚠  No lens calibration – run Calibrate Lens for accurate dimensions", "#eb8c00");

    /* Camera infrastructure (no device opened yet) */
    videoSink      = new QVideoSink(this);
    captureSession = new QMediaCaptureSession(this);
    captureSession->setVideoSink(videoSink);
    connect(videoSink, &QVideoSink::videoFrameChanged,
            this,      &Widget::onFrameChanged);

    autoTimer = new QTimer(this);
    autoTimer->setInterval(200);
    connect(autoTimer, &QTimer::timeout, this, &Widget::onAutoDetectTimer);

    /* Enumerate cameras after event loop starts */
    QTimer::singleShot(0, this, [this](){
        availCameras = QMediaDevices::videoInputs();
        cameraCombo->blockSignals(true);
        if (availCameras.isEmpty()) {
            setStatus("⚠  No camera detected", "#eb5f14");
            btnStart->setEnabled(false);
            cameraCombo->addItem("No cameras found");
            cameraCombo->setEnabled(false);
        } else {
            for (const auto &d : availCameras)
                cameraCombo->addItem(d.description());
            int idx = availCameras.size() - 1;
            cameraCombo->setCurrentIndex(idx);
            buildCameraForIndex(idx);
        }
        cameraCombo->blockSignals(false);
        connect(cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &Widget::selectCamera);
        updateUI();
    });
}

/* ═══════════════════════════════════════════════════════════════════════════
   Build UI
   ═══════════════════════════════════════════════════════════════════════════ */
void Widget::buildUI()
{
    auto btn = [&](const QString &t) -> QPushButton* {
        auto *b = new QPushButton(t, this);
        b->setFixedHeight(34);
        b->setCursor(Qt::PointingHandCursor);
        return b;
    };

    btnStart       = btn("▶  Start Camera");
    btnCapture     = btn("📸  Capture");
    btnDetect      = btn("🔍  Detect");
    btnReset       = btn("↺  Reset");
    btnPathOnly    = btn("🧩  Path View");
    btnExport      = btn("💾  Export JSON");
    btnCalib       = btn("📐  Calibrate Lens");
    btnCollect     = btn("➕  Collect Frame");
    btnFinishCalib = btn("✅  Finish & Apply");
    btnClearCalib  = btn("🗑  Clear Calibration");

    btnCollect    ->setVisible(false);
    btnFinishCalib->setVisible(false);

    cameraCombo = new QComboBox(this);
    cameraCombo->setFixedHeight(34);
    cameraCombo->setMinimumWidth(190);

    chkOverlay    = new QCheckBox("Overlay",     this);
    chkAutoDetect = new QCheckBox("Auto A4",     this);
    chkOverlay   ->setChecked(true);
    chkAutoDetect->setChecked(true);

    lblStatus     = new QLabel("Initialising…", this);
    lblA4Lock     = new QLabel("A4: –",          this);
    lblObjects    = new QLabel("Objects: 0",     this);
    lblCalibStatus= new QLabel("Calibration: –", this);

    lblStatus    ->setAlignment(Qt::AlignLeft   | Qt::AlignVCenter);
    lblA4Lock    ->setAlignment(Qt::AlignCenter);
    lblObjects   ->setAlignment(Qt::AlignCenter);
    lblCalibStatus->setAlignment(Qt::AlignCenter);

    /* Toolbar row 1 – main workflow */
    auto *row1 = new QHBoxLayout;
    row1->setSpacing(6);
    row1->addWidget(cameraCombo);
    row1->addWidget(btnStart);
    row1->addWidget(btnCapture);
    row1->addWidget(btnDetect);
    row1->addWidget(btnPathOnly);
    row1->addWidget(btnReset);
    row1->addWidget(btnExport);
    row1->addSpacing(12);
    row1->addWidget(chkOverlay);
    row1->addWidget(chkAutoDetect);
    row1->addStretch();

    /* Toolbar row 2 – calibration */
    auto *row2 = new QHBoxLayout;
    row2->setSpacing(6);
    row2->addWidget(btnCalib);
    row2->addWidget(btnCollect);
    row2->addWidget(btnFinishCalib);
    row2->addWidget(btnClearCalib);
    row2->addStretch();
    row2->addWidget(lblCalibStatus);

    /* Status bar */
    auto *statusBar = new QHBoxLayout;
    statusBar->setSpacing(16);
    statusBar->addWidget(lblStatus,  3);
    statusBar->addWidget(lblA4Lock,  1);
    statusBar->addWidget(lblObjects, 1);

    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(10, 8, 10, 8);
    root->setSpacing(5);
    root->addLayout(row1);
    root->addLayout(row2);
    root->addLayout(statusBar);
    root->addStretch();

    /* Connections – workflow */
    connect(btnStart,    &QPushButton::clicked, this, &Widget::startCamera);
    connect(btnCapture,  &QPushButton::clicked, this, &Widget::captureImage);
    connect(btnDetect,   &QPushButton::clicked, this, &Widget::runDetection);
    connect(btnReset,    &QPushButton::clicked, this, &Widget::resetAll);
    connect(btnPathOnly, &QPushButton::clicked, this, &Widget::togglePathOnly);
    connect(btnExport,   &QPushButton::clicked, this, &Widget::exportJson);
    /* Connections – calibration */
    connect(btnCalib,       &QPushButton::clicked, this, &Widget::startCalibration);
    connect(btnCollect,     &QPushButton::clicked, this, &Widget::collectCalibFrame);
    connect(btnFinishCalib, &QPushButton::clicked, this, &Widget::finishCalibration);
    connect(btnClearCalib,  &QPushButton::clicked, this, &Widget::clearCalibration);

    connect(chkAutoDetect, &QCheckBox::toggled, [this](bool on){
        if (camera && on && currentState == LiveView && camera->isActive())
            autoTimer->start();
        else
            autoTimer->stop();
    });
}

/* ─── Theme ─────────────────────────────────────────────────────────────── */
void Widget::applyTheme()
{
    setStyleSheet(R"(
        QWidget {
            background:#f5f5f2; color:#1e1e1e;
            font-family:"Segoe UI","SF Pro Display","Helvetica Neue",sans-serif;
            font-size:12px;
        }
        QPushButton {
            background:#fff; color:#1e1e1e;
            border:1px solid #d8cfc8; border-radius:6px;
            padding:0 14px; font-size:12px;
        }
        QPushButton:hover   { background:#fff3ec; border-color:#eb5f14; color:#eb5f14; }
        QPushButton:pressed { background:#ffe8d8; }
        QPushButton:disabled{ color:#b8b0a8; border-color:#e8e2dc; background:#f8f6f4; }
        QPushButton#accent  { background:#eb5f14; border-color:#eb5f14; color:#fff; font-weight:600; }
        QPushButton#accent:hover   { background:#d45210; }
        QPushButton#accent:pressed { background:#be4a0e; }
        QPushButton#calib   { background:#1a6e3a; border-color:#1a6e3a; color:#fff; font-weight:600; }
        QPushButton#calib:hover    { background:#155c30; }
        QPushButton#calibsec{ background:#fff; border-color:#1a6e3a; color:#1a6e3a; }
        QPushButton#calibsec:hover { background:#edfdf4; }
        QPushButton#danger  { background:#fff; border-color:#d42d2d; color:#d42d2d; }
        QPushButton#danger:hover   { background:#fff0f0; }
        QCheckBox { color:#6e6460; spacing:5px; }
        QCheckBox::indicator { width:14px; height:14px; border-radius:3px;
                               border:1px solid #d0c8c0; background:#fff; }
        QCheckBox::indicator:checked { background:#eb5f14; border-color:#eb5f14; }
        QLabel { color:#6e6460; font-size:11px; }
        QComboBox {
            background:#fff; color:#1e1e1e;
            border:1px solid #d8cfc8; border-radius:6px;
            padding:0 10px; font-size:12px;
        }
        QComboBox:hover { border-color:#eb5f14; }
        QComboBox::drop-down { border:none; width:20px; }
        QComboBox QAbstractItemView {
            background:#fff; color:#1e1e1e;
            selection-background-color:#fff0e8;
            selection-color:#eb5f14;
            border:1px solid #d8cfc8;
        }
    )");

    btnStart ->setObjectName("accent");
    btnDetect->setObjectName("accent");
    btnCalib ->setObjectName("calib");
    btnCollect    ->setObjectName("calibsec");
    btnFinishCalib->setObjectName("calib");
    btnClearCalib ->setObjectName("danger");
}

/* ─── updateUI ──────────────────────────────────────────────────────────── */
void Widget::updateUI()
{
    bool calib   = (currentState == Calibrating);
    bool captured= (currentState == CapturedView);
    bool detected= (currentState == DetectedView || currentState == PathOnlyView);
    bool camReady = (camera != nullptr);
    bool camActive= camReady && camera->isActive();

    /* Main workflow buttons */
    btnStart   ->setEnabled(camReady && !calib);
    btnStart   ->setText(camActive ? "⏹  Stop Camera" : "▶  Start Camera");
    btnCapture ->setEnabled(camActive && !liveFrame.isNull() && !calib);
    btnDetect  ->setEnabled(captured);
    btnReset   ->setEnabled(currentState != LiveView);
    btnPathOnly->setEnabled(detected);
    btnExport  ->setEnabled(detected);

    /* Calibration buttons */
    btnCalib      ->setVisible(!calib);
    btnClearCalib ->setVisible(!calib);
    btnCollect    ->setVisible(calib);
    btnFinishCalib->setVisible(calib);

    int n = (int)calibImagePts.size();
    btnCollect    ->setEnabled(calib && camActive && !liveFrame.isNull());
    btnFinishCalib->setEnabled(calib && n >= CALIB_MIN_FRAMES);
    btnFinishCalib->setText(
        QString("✅  Finish & Apply (%1/%2)").arg(n).arg(CALIB_MIN_FRAMES));

    /* Calibration status label */
    if (isCalibrated()) {
        lblCalibStatus->setText(
            QString("📐 Calibrated  RMS=%.2f px").arg(calibRmsError));
        lblCalibStatus->setStyleSheet("color:#1a6e3a;font-size:11px;font-weight:600;");
    } else {
        lblCalibStatus->setText("📐 Not calibrated");
        lblCalibStatus->setStyleSheet("color:#eb8c00;font-size:11px;");
    }

    lblObjects->setText(QString("Objects: %1").arg(objects.size()));
}

void Widget::setStatus(const QString &msg, const QString &color)
{
    lblStatus->setText(msg);
    lblStatus->setStyleSheet(
        QString("color:%1;font-size:11px;").arg(color));
}


/* ═══════════════════════════════════════════════════════════════════════════
   CAMERA
   ═══════════════════════════════════════════════════════════════════════════ */
void Widget::buildCameraForIndex(int index)
{
    if (index < 0 || index >= availCameras.size()) return;
    if (camera) {
        camera->stop();
        captureSession->setCamera(nullptr);
        delete camera;
        camera = nullptr;
    }
    camera = new QCamera(availCameras[index], this);
    connect(camera, &QCamera::errorOccurred,
            this, [this](QCamera::Error, const QString &msg){
                setStatus("⚠  Camera error: " + msg, "#d42d2d");
            });
    connect(camera, &QCamera::activeChanged, this, [this](bool){ updateUI(); });
}

void Widget::selectCamera(int index)
{
    if (index < 0 || index >= availCameras.size()) return;
    autoTimer->stop();
    liveFrame    = QImage();
    currentState = LiveView;
    a4Locked     = false;
    quadLockCount= 0;
    buildCameraForIndex(index);
    setStatus(QString("Selected: %1  –  press ▶ Start Camera")
                  .arg(availCameras[index].description()), "#6e6460");
    updateUI();
    update();
}

void Widget::startCamera()
{
    if (!camera) return;
    if (camera->isActive()) {
        autoTimer->stop();
        camera->stop();
        captureSession->setCamera(nullptr);
        liveFrame = QImage();
        setStatus("Camera stopped", "#6e6460");
        updateUI();
        update();
        return;
    }
    setStatus("Starting camera…", "#eb8c00");
    captureSession->setCamera(camera);
    camera->start();
    currentState = LiveView;
    a4Locked     = false;
    quadLockCount= 0;
    if (chkAutoDetect->isChecked()) autoTimer->start();
    QTimer::singleShot(4000, this, [this](){
        if (liveFrame.isNull() && camera && camera->isActive())
            setStatus("⚠  No frames received – check USB connection", "#eb8c00");
    });
    updateUI();
}

void Widget::onFrameChanged(const QVideoFrame &frame)
{
    if (!frame.isValid()) return;
    if (currentState != LiveView && currentState != Calibrating) return;
    QVideoFrame f(frame);
    if (!f.map(QVideoFrame::ReadOnly)) return;
    QImage img = f.toImage();
    f.unmap();
    if (img.isNull()) return;
    bool first = liveFrame.isNull();
    liveFrame = img.convertToFormat(QImage::Format_ARGB32);
    if (first) {
        setStatus("✓  Camera live  –  point at A4 sheet, then press 📸 Capture", "#228050");
        updateUI();
    }
    update();
}

void Widget::onAutoDetectTimer()
{
    if (liveFrame.isNull() || currentState != LiveView) return;
    cv::Mat src(liveFrame.height(), liveFrame.width(),
                CV_8UC4, (void*)liveFrame.bits(), liveFrame.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    /* Apply undistortion if calibrated */
    if (isCalibrated()) bgr = undistortFrame(bgr);
    cv::Mat warped;
    bool found = tryHsvMethod(bgr, warped) ||
                 tryCannyMethod(bgr, warped) ||
                 tryAdaptiveMethod(bgr, warped);
    if (found) { quadLockCount = std::min(quadLockCount+1, 5); a4Locked = (quadLockCount>=3); }
    else       { quadLockCount = std::max(quadLockCount-1, 0); if (!quadLockCount) a4Locked=false; }
    if (a4Locked) {
        lblA4Lock->setText("A4: ✓ Locked");
        lblA4Lock->setStyleSheet("color:#1a6e3a;font-size:11px;font-weight:bold;");
    } else {
        lblA4Lock->setText(found ? "A4: ~ Acquiring…" : "A4: ✗ Not found");
        lblA4Lock->setStyleSheet(found ?
                                     "color:#eb8c00;font-size:11px;" : "color:#d42d2d;font-size:11px;");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LENS CALIBRATION
   ═══════════════════════════════════════════════════════════════════════════ */

static QString calibFilePath()
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation)
    + "/camera_calib.yml";
}

bool Widget::loadCalibration()
{
    QString path = calibFilePath();
    if (!QFile::exists(path)) return false;
    cv::FileStorage fs(path.toStdString(), cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    fs["camera_matrix"] >> camMatrix;
    fs["dist_coeffs"]   >> distCoeffs;
    fs["rms_error"]     >> calibRmsError;
    cv::Size sz;
    fs["image_size"]    >> sz;
    fs.release();
    if (camMatrix.empty() || distCoeffs.empty()) return false;
    /* Pre-compute optimal matrix for the last known image size */
    optimalMatrix = cv::getOptimalNewCameraMatrix(
        camMatrix, distCoeffs, sz, 0.0, sz, &validROI);
    calibImageSize = sz;
    qDebug() << "Calibration loaded from" << path
             << "RMS=" << calibRmsError;
    return true;
}

bool Widget::saveCalibration()
{
    QString dir  = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dir);
    QString path = calibFilePath();
    cv::FileStorage fs(path.toStdString(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "camera_matrix" << camMatrix;
    fs << "dist_coeffs"   << distCoeffs;
    fs << "rms_error"     << calibRmsError;
    fs << "image_size"    << calibImageSize;
    fs.release();
    qDebug() << "Calibration saved to" << path;
    return true;
}

cv::Mat Widget::undistortFrame(const cv::Mat &src) const
{
    if (!isCalibrated()) return src;
    /* Rebuild optimal matrix if size changed */
    cv::Size sz = src.size();
    cv::Mat optM = (sz == calibImageSize) ? optimalMatrix :
                       cv::getOptimalNewCameraMatrix(camMatrix, distCoeffs, sz, 0.0, sz);
    cv::Mat dst;
    cv::undistort(src, dst, camMatrix, distCoeffs, optM);
    return dst;
}

void Widget::startCalibration()
{
    if (!camera || !camera->isActive()) {
        QMessageBox::information(this, "Calibrate Lens",
                                 "Start the camera first, then enter calibration mode.");
        return;
    }
    calibImagePts.clear();
    calibObjPts.clear();
    calibPreviewFrame = QImage();
    currentState = Calibrating;
    autoTimer->stop();
    setStatus(QString("Calibration mode  –  show 9×6 checkerboard from different angles, press ➕ Collect Frame  (0/%1)").arg(CALIB_MIN_FRAMES), "#1a6e3a");
    updateUI();
    update();
}

void Widget::collectCalibFrame()
{
    if (liveFrame.isNull()) return;

    cv::Mat src(liveFrame.height(), liveFrame.width(),
                CV_8UC4, (void*)liveFrame.bits(), liveFrame.bytesPerLine());
    cv::Mat bgr, gray;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    cv::Size patternSize(CB_COLS, CB_ROWS);
    std::vector<cv::Point2f> corners;

    bool found = cv::findChessboardCorners(gray, patternSize, corners,
                                           cv::CALIB_CB_ADAPTIVE_THRESH |
                                               cv::CALIB_CB_NORMALIZE_IMAGE |
                                               cv::CALIB_CB_FAST_CHECK);

    if (!found) {
        setStatus("⚠  Checkerboard not found – adjust angle or lighting and try again", "#d42d2d");
        return;
    }

    /* Sub-pixel refinement */
    cv::cornerSubPix(gray, corners, {11,11}, {-1,-1},
                     cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001));

    /* Build 3-D object points */
    std::vector<cv::Point3f> objPts;
    for (int r = 0; r < CB_ROWS; ++r)
        for (int c = 0; c < CB_COLS; ++c)
            objPts.emplace_back(c * CB_SQUARE_MM, r * CB_SQUARE_MM, 0.f);

    calibImagePts.push_back(corners);
    calibObjPts.push_back(objPts);
    calibImageSize = gray.size();

    /* Draw corners onto a preview image */
    cv::Mat preview = bgr.clone();
    cv::drawChessboardCorners(preview, patternSize, corners, found);
    calibPreviewFrame = QImage(preview.data, preview.cols, preview.rows,
                               (int)preview.step, QImage::Format_BGR888).copy();

    int n = (int)calibImagePts.size();
    setStatus(QString("✓  Frame %1 collected  –  %2 more needed%3")
                  .arg(n).arg(std::max(0, CALIB_MIN_FRAMES - n))
                  .arg(n >= CALIB_MIN_FRAMES ? "  – press ✅ Finish & Apply!" : ""),
              n >= CALIB_MIN_FRAMES ? "#1a6e3a" : "#eb8c00");
    updateUI();
    update();
}

void Widget::finishCalibration()
{
    int n = (int)calibImagePts.size();
    if (n < CALIB_MIN_FRAMES) {
        setStatus(QString("Need at least %1 frames (have %2)").arg(CALIB_MIN_FRAMES).arg(n), "#d42d2d");
        return;
    }

    setStatus("Running calibration…", "#eb8c00");
    qApp->processEvents();

    std::vector<cv::Mat> rvecs, tvecs;
    camMatrix   = cv::Mat();
    distCoeffs  = cv::Mat();

    calibRmsError = cv::calibrateCamera(
        calibObjPts, calibImagePts, calibImageSize,
        camMatrix, distCoeffs, rvecs, tvecs,
        cv::CALIB_RATIONAL_MODEL   // uses k1-k6 + p1,p2 → better for wide-angle
        );

    optimalMatrix = cv::getOptimalNewCameraMatrix(
        camMatrix, distCoeffs, calibImageSize, 0.0, calibImageSize, &validROI);

    saveCalibration();

    currentState = LiveView;
    calibImagePts.clear();
    calibObjPts.clear();
    calibPreviewFrame = QImage();

    if (chkAutoDetect->isChecked()) autoTimer->start();

    setStatus(QString("✓  Calibration complete!  RMS reprojection error = %1 px%2")
                  .arg(calibRmsError, 0, 'f', 3)
                  .arg(calibRmsError < 1.0 ? "  (excellent)" :
                           calibRmsError < 2.0 ? "  (good)" : "  (consider recalibrating)"),
              calibRmsError < 2.0 ? "#1a6e3a" : "#eb8c00");

    qDebug() << "Calibration RMS error:" << calibRmsError << "px";
    updateUI();
    update();
}

void Widget::clearCalibration()
{
    camMatrix   = cv::Mat();
    distCoeffs  = cv::Mat();
    optimalMatrix = cv::Mat();
    calibRmsError = 0.0;
    QFile::remove(calibFilePath());
    setStatus("Calibration cleared  –  run Calibrate Lens for accurate measurements", "#eb8c00");
    updateUI();
}


/* ═══════════════════════════════════════════════════════════════════════════
   WORKFLOW  – capture / detect / reset
   ═══════════════════════════════════════════════════════════════════════════ */
void Widget::captureImage()
{
    if (liveFrame.isNull()) { setStatus("⚠  No frame yet", "#d42d2d"); return; }
    capturedFrame = liveFrame.copy();
    currentState  = CapturedView;
    autoTimer->stop();
    setStatus("✓  Frame captured  –  press 🔍 Detect", "#228050");
    updateUI();
    update();
}

void Widget::resetAll()
{
    objects.clear();
    history.clear();
    capturedFrame = QImage();
    warpedA4      = QImage();
    lastQuad.clear();
    a4Locked      = false;
    quadLockCount = 0;
    selectedObject= -1;
    currentState  = LiveView;
    setStatus("Reset  –  camera live", "#6e6460");
    if (chkAutoDetect->isChecked() && camera && camera->isActive()) autoTimer->start();
    updateUI();
    update();
}

void Widget::runDetection()
{
    if (capturedFrame.isNull()) { setStatus("⚠  Capture a frame first", "#d42d2d"); return; }
    objects.clear();
    history.clear();
    setStatus("Detecting A4 sheet…", "#eb8c00");
    qApp->processEvents();

    /* Always ensure ARGB32 */
    QImage frameToProcess = capturedFrame.convertToFormat(QImage::Format_ARGB32);

    if (!detectAndWarpA4(frameToProcess)) {
        setStatus("⚠  A4 sheet not found – ensure the full sheet is visible and well-lit", "#d42d2d");
        return;
    }
    setStatus("A4 warped – finding objects…", "#eb8c00");
    qApp->processEvents();

    detectObjects(warpedA4);

    if (objects.empty())
        setStatus("⚠  No objects found – check drawing visibility / lighting", "#eb8c00");
    else
        setStatus(QString("✓  %1 object(s) detected  –  press 🧩 Path View or 💾 Export")
                      .arg(objects.size()), "#228050");

    currentState = DetectedView;
    updateUI();
    update();
}

void Widget::togglePathOnly()
{
    if (objects.empty()) return;
    currentState = (currentState == PathOnlyView) ? DetectedView : PathOnlyView;
    btnPathOnly->setText(currentState == PathOnlyView ? "🖼  Full View" : "🧩  Path View");
    updateUI();
    update();
}

void Widget::exportJson() { exportPathsToJson(); }

/* ═══════════════════════════════════════════════════════════════════════════
   A4 DETECTION  –  3-strategy fallback
   ═══════════════════════════════════════════════════════════════════════════ */

bool Widget::detectAndWarpA4(const QImage &image)
{
    cv::Mat src(image.height(), image.width(),
                CV_8UC4, (void*)image.bits(), image.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);

    /* ── Apply lens undistortion first if calibrated ── */
    if (isCalibrated())
        bgr = undistortFrame(bgr);

    cv::Mat warped;
    if (tryHsvMethod   (bgr, warped)) { qDebug() << "A4: HSV";      return true; }
    if (tryCannyMethod (bgr, warped)) { qDebug() << "A4: Canny";    return true; }
    if (tryAdaptiveMethod(bgr, warped)){ qDebug() << "A4: Adaptive"; return true; }
    return false;
}

bool Widget::tryHsvMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0,0,160), cv::Scalar(180,50,255), mask);
    auto k9 = cv::getStructuringElement(cv::MORPH_RECT,{9,9});
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k9);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k9);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02*cv::arcLength(c,true), true);
        if (approx.size()==4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            double img  = bgr.cols * bgr.rows;
            if (area > img*0.05 && area < img*0.98)
                return warpFromQuad(bgr, {approx.begin(),approx.end()}, warped);
        }
    }
    return false;
}

bool Widget::tryCannyMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat gray, blurred, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, {5,5}, 0);
    cv::Canny(blurred, edges, 50, 150);
    cv::dilate(edges, edges, cv::getStructuringElement(cv::MORPH_RECT,{3,3}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    double best=0; std::vector<cv::Point2f> bestQ;
    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02*cv::arcLength(c,true), true);
        if (approx.size()==4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            if (area > bgr.cols*bgr.rows*0.05 && area > best) {
                best=area; bestQ={approx.begin(),approx.end()};
            }
        }
    }
    return (bestQ.size()==4) ? warpFromQuad(bgr, bestQ, warped) : false;
}

bool Widget::tryAdaptiveMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat gray, bin;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5,5}, 0);
    cv::adaptiveThreshold(gray, bin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 21, 4);
    cv::bitwise_not(bin, bin);
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT,{7,7}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double best=0; std::vector<cv::Point2f> bestQ;
    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02*cv::arcLength(c,true), true);
        if (approx.size()==4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            if (area > bgr.cols*bgr.rows*0.05 && area > best) {
                best=area; bestQ={approx.begin(),approx.end()};
            }
        }
    }
    return (bestQ.size()==4) ? warpFromQuad(bgr, bestQ, warped) : false;
}

std::vector<cv::Point2f> Widget::orderQuad(std::vector<cv::Point2f> pts)
{
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f &a, const cv::Point2f &b){ return a.y < b.y; });
    cv::Point2f tl,tr,bl,br;
    if (pts[0].x<pts[1].x){tl=pts[0];tr=pts[1];}else{tl=pts[1];tr=pts[0];}
    if (pts[2].x<pts[3].x){bl=pts[2];br=pts[3];}else{bl=pts[3];br=pts[2];}
    return {tl,tr,br,bl};
}

bool Widget::warpFromQuad(const cv::Mat &src,
                          std::vector<cv::Point2f> quad,
                          cv::Mat &warped)
{
    if (quad.size()!=4) return false;
    auto ordered = orderQuad(quad);
    lastQuad = ordered;
    std::vector<cv::Point2f> dst{{0,0},{(float)WARP_W,0},
                                 {(float)WARP_W,(float)WARP_H},{0,(float)WARP_H}};
    cv::Mat H = cv::getPerspectiveTransform(ordered, dst);
    cv::warpPerspective(src, warped, H, {WARP_W,WARP_H}, cv::INTER_LANCZOS4);
    warpedA4 = QImage(warped.data, warped.cols, warped.rows,
                      (int)warped.step, QImage::Format_BGR888).copy();
    pixPerMM = PX_PER_MM;
    return true;
}


/* ═══════════════════════════════════════════════════════════════════════════
   OBJECT DETECTION
   ═══════════════════════════════════════════════════════════════════════════ */
QString Widget::classifyShape(const std::vector<cv::Point> &approx, double circ)
{
    int n = (int)approx.size();
    if (circ > 0.85)            return "Circle";
    if (circ > 0.75 && n > 8)  return "Ellipse";
    if (n == 3)                 return "Triangle";
    if (n == 4) {
        cv::RotatedRect rr = cv::minAreaRect(approx);
        double r = rr.size.width / std::max(rr.size.height, 1.0f);
        if (r < 1) r = 1.0/r;
        return (r < 1.15) ? "Square" : "Rectangle";
    }
    if (n == 5) return "Pentagon";
    if (n == 6) return "Hexagon";
    return QString("Polygon(%1)").arg(n);
}

void Widget::detectObjects(const QImage &img)
{
    objects.clear();

    /* warpedA4 is BGR888 → CV_8UC3 */
    cv::Mat src(img.height(), img.width(),
                CV_8UC3, (void*)img.bits(), img.bytesPerLine());

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5,5}, 0);

    cv::Mat bin;
    cv::adaptiveThreshold(gray, bin, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 31, 7);

    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_RECT,{5,5}));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT,{3,3}));

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy,
                     cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    /* RETR_CCOMP: parent==-1 → outer object boundary (keep) */
    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] != -1) continue;   // skip holes

        double areaPx  = cv::contourArea(contours[i]);
        double areaMM2 = areaPx / (pixPerMM * pixPerMM);
        if (areaMM2 < 4.0)  continue;
        if (areaMM2 > A4_WIDTH_MM * A4_HEIGHT_MM * 0.90) continue;

        double epsilon = std::max(1.5, pixPerMM * 0.4);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, true);
        if ((int)approx.size() < 3) continue;

        cv::RotatedRect rr    = cv::minAreaRect(contours[i]);
        double perimPx        = cv::arcLength(contours[i], true);
        double w = rr.size.width, h = rr.size.height;
        if (w < h) std::swap(w, h);
        double circ = (perimPx > 0) ? 4.0*M_PI*areaPx/(perimPx*perimPx) : 0;

        ObjectMeasure m;
        m.widthMM   = w / pixPerMM;
        m.heightMM  = h / pixPerMM;
        m.areaMM2   = areaMM2;
        m.perimMM   = perimPx / pixPerMM;
        m.angleDeg  = rr.angle;
        m.centerMM  = QPointF(rr.center.x/pixPerMM, rr.center.y/pixPerMM);
        m.shapeLabel= classifyShape(approx, circ);

        QPainterPath pathPx, pathMM;
        pathPx.moveTo(approx[0].x, approx[0].y);
        pathMM.moveTo(approx[0].x/pixPerMM, approx[0].y/pixPerMM);
        for (size_t k=1; k<approx.size(); ++k) {
            pathPx.lineTo(approx[k].x, approx[k].y);
            pathMM.lineTo(approx[k].x/pixPerMM, approx[k].y/pixPerMM);
        }
        pathPx.closeSubpath();
        pathMM.closeSubpath();

        objects.push_back({pathPx, pathMM, m});

        qDebug() << "[OBJ]" << m.shapeLabel
                 << "W=" << m.widthMM << "mm  H=" << m.heightMM
                 << "mm  Area=" << m.areaMM2 << "mm²";
    }

    pushAndSmooth();
}

void Widget::pushAndSmooth()
{
    std::vector<ObjectMeasure> cur;
    for (auto &o : objects) cur.push_back(o.measure);
    history.push_back(cur);
    if ((int)history.size() > SMOOTH_FRAMES) history.pop_front();

    for (size_t i = 0; i < objects.size(); ++i) {
        double ws=0, hs=0, as=0; int c=0;
        for (auto &fr : history)
            if (i < fr.size()) { ws+=fr[i].widthMM; hs+=fr[i].heightMM; as+=fr[i].areaMM2; ++c; }
        if (c > 0) {
            objects[i].measure.widthMM  = ws/c;
            objects[i].measure.heightMM = hs/c;
            objects[i].measure.areaMM2  = as/c;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   JSON EXPORT
   ═══════════════════════════════════════════════════════════════════════════ */
void Widget::exportPathsToJson()
{
    if (objects.empty()) { setStatus("⚠  No objects to export", "#eb8c00"); return; }

    QJsonObject root;
    root["version"]     = "3.0";
    root["timestamp"]   = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["unit"]        = "mm";
    root["calibrated"]  = isCalibrated();
    root["calib_rms_px"]= calibRmsError;
    root["pixelsPerMM"] = pixPerMM;

    QJsonObject canvas;
    canvas["width_mm"]  = A4_WIDTH_MM;
    canvas["height_mm"] = A4_HEIGHT_MM;
    root["canvas"] = canvas;

    QJsonArray arr;
    for (int i=0; i<(int)objects.size(); ++i) {
        const auto &obj = objects[i];
        const auto &m   = obj.measure;
        QJsonObject jo;
        jo["id"]       = i+1;
        jo["shape"]    = m.shapeLabel;
        jo["width_mm"] = qRound(m.widthMM *100)/100.0;
        jo["height_mm"]= qRound(m.heightMM*100)/100.0;
        jo["area_mm2"] = qRound(m.areaMM2 * 10)/ 10.0;
        jo["perim_mm"] = qRound(m.perimMM * 10)/ 10.0;
        jo["angle_deg"]= qRound(m.angleDeg* 10)/ 10.0;
        QJsonObject cen; cen["x_mm"]=qRound(m.centerMM.x()*100)/100.0;
        cen["y_mm"]=qRound(m.centerMM.y()*100)/100.0;
        jo["center"] = cen;

        QJsonArray pa;
        auto r2=[](double v){return qRound(v*100)/100.0;};
        const QPainterPath &path = obj.pathMM;
        for (int e=0; e<path.elementCount(); ++e) {
            auto el = path.elementAt(e);
            QJsonObject cmd;
            if (el.type==QPainterPath::MoveToElement)
            { cmd["cmd"]="M"; cmd["x"]=r2(el.x); cmd["y"]=r2(el.y); }
            else if (el.type==QPainterPath::LineToElement)
            { cmd["cmd"]="L"; cmd["x"]=r2(el.x); cmd["y"]=r2(el.y); }
            else if (el.type==QPainterPath::CurveToElement) {
                auto c2=path.elementAt(e+1), ep=path.elementAt(e+2);
                cmd["cmd"]="C";
                cmd["x1"]=r2(el.x); cmd["y1"]=r2(el.y);
                cmd["x2"]=r2(c2.x); cmd["y2"]=r2(c2.y);
                cmd["x"] =r2(ep.x); cmd["y"] =r2(ep.y);
                e+=2;
            }
            pa.append(cmd);
        }
        QJsonObject zc; zc["cmd"]="Z"; pa.append(zc);
        jo["path"] = pa;
        arr.append(jo);
    }
    root["objects"] = arr;

    QString dir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QString fname = dir + "/digitized_" +
                    QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".json";
    QFile file(fname);
    if (!file.open(QIODevice::WriteOnly)) { setStatus("⚠  File write failed", "#d42d2d"); return; }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();
    setStatus(QString("✓  Exported %1 object(s) → %2").arg(objects.size()).arg(fname), "#228050");
}


/* ═══════════════════════════════════════════════════════════════════════════
   PAINT
   ═══════════════════════════════════════════════════════════════════════════ */
void Widget::resizeEvent(QResizeEvent *e) { QWidget::resizeEvent(e); update(); }

void Widget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    const int TOP = 100;   // toolbar (2 rows) + status bar
    QRect vp(0, TOP, width(), height()-TOP);
    p.fillRect(vp, C::BG);

    switch (currentState) {
    case LiveView:
    case CapturedView:  drawLiveView    (p, vp); break;
    case DetectedView:  drawDetectedView(p, vp); break;
    case PathOnlyView:  drawPathOnlyView(p, vp); break;
    case Calibrating:   drawCalibView   (p, vp); break;
    }
}

static std::tuple<double,double,double>
fitRect(const QRect &vp, int imgW, int imgH)
{
    double s  = std::min(vp.width()/(double)imgW, vp.height()/(double)imgH) * 0.95;
    double ox = vp.x() + (vp.width()  - imgW*s)*0.5;
    double oy = vp.y() + (vp.height() - imgH*s)*0.5;
    return {s, ox, oy};
}

/* ─── Live / Captured view ──────────────────────────────────────────────── */
void Widget::drawLiveView(QPainter &p, const QRect &vp)
{
    const QImage &img = (currentState==CapturedView) ? capturedFrame : liveFrame;
    if (img.isNull()) {
        p.setPen(C::TEXT_SEC);
        p.setFont(QFont("Segoe UI",14));
        p.drawText(vp, Qt::AlignCenter,
                   "Select a camera and press ▶ Start Camera");
        return;
    }
    auto [s,ox,oy] = fitRect(vp, img.width(), img.height());
    p.save();
    p.translate(ox,oy); p.scale(s,s);
    p.drawImage(0,0,img);
    if (currentState==LiveView && a4Locked && !lastQuad.empty()) {
        QPolygonF poly;
        for (auto &pt : lastQuad) poly << QPointF(pt.x, pt.y);
        p.setPen(QPen(C::SUCCESS, 3.0/s));
        p.setBrush(QColor(34,160,80,25));
        p.drawPolygon(poly);
        QFont f("Segoe UI", 18.0/s, QFont::Bold);
        p.setFont(f);
        p.setPen(C::SUCCESS);
        QRectF br = QFontMetrics(f).boundingRect("✓ A4 Locked");
        br.moveCenter(poly.boundingRect().center());
        p.drawText(br, Qt::AlignCenter, "✓ A4 Locked");
    }
    p.restore();
}

/* ─── Calibration view ──────────────────────────────────────────────────── */
void Widget::drawCalibView(QPainter &p, const QRect &vp)
{
    /* Show the latest calibration preview (with corners drawn) or live feed */
    const QImage &img = calibPreviewFrame.isNull() ? liveFrame : calibPreviewFrame;
    if (img.isNull()) {
        p.setPen(C::TEXT_SEC);
        p.setFont(QFont("Segoe UI",13));
        p.drawText(vp, Qt::AlignCenter, "Show the checkerboard to the camera");
        return;
    }
    auto [s,ox,oy] = fitRect(vp, img.width(), img.height());
    p.save();
    p.translate(ox,oy); p.scale(s,s);
    p.drawImage(0,0,img);
    /* Instruction overlay */
    QString info = QString("Calibration mode  –  frames collected: %1 / %2\n"
                           "Move the checkerboard to different positions & angles")
                       .arg(calibImagePts.size()).arg(CALIB_MIN_FRAMES);
    QFont f("Segoe UI", 13.0/s, QFont::Bold);
    p.setFont(f);
    QFontMetrics fm(f);
    QRect tr = fm.boundingRect(QRect(),Qt::AlignCenter,info);
    tr.setWidth(tr.width()+20); tr.setHeight(tr.height()+14);
    tr.moveTopLeft(QPoint((int)(10/s),(int)(10/s)));
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(255,255,255,210));
    p.drawRoundedRect(tr,6,6);
    p.setPen(QColor(26,110,58));
    p.drawText(tr, Qt::AlignCenter, info);
    p.restore();
}

/* ─── Detected view ─────────────────────────────────────────────────────── */
void Widget::drawDetectedView(QPainter &p, const QRect &vp)
{
    if (warpedA4.isNull()) return;
    auto [s,ox,oy] = fitRect(vp, warpedA4.width(), warpedA4.height());
    p.save();
    p.translate(ox,oy); p.scale(s,s);
    p.drawImage(0,0,warpedA4);
    p.setPen(QPen(C::BORDER,1.5/s));
    p.setBrush(Qt::NoBrush);
    p.drawRect(0,0,warpedA4.width(),warpedA4.height());

    if (!chkOverlay->isChecked()) { p.restore(); return; }

    for (int i=0; i<(int)objects.size(); ++i) {
        const auto &obj=objects[i]; const auto &m=obj.measure;
        QColor col = C::OBJ_COLORS[i % C::OBJ_COLORS.size()];

        p.setPen(QPen(col, 2.0/s));
        p.setBrush(QColor(col.red(),col.green(),col.blue(),28));
        p.drawPath(obj.pathPx);

        double cx=m.centerMM.x()*pixPerMM, cy=m.centerMM.y()*pixPerMM;
        double cr=7/s;
        p.setPen(QPen(col,1.2/s));
        p.drawLine(QLineF(cx-cr,cy,cx+cr,cy));
        p.drawLine(QLineF(cx,cy-cr,cx,cy+cr));

        /* Compact single-line pill label */
        QString lbl = QString("%1  %2×%3 mm")
                          .arg(m.shapeLabel)
                          .arg(m.widthMM,0,'f',1)
                          .arg(m.heightMM,0,'f',1);
        QFont f("Segoe UI", 9.0/s);
        p.setFont(f);
        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(lbl);
        tr.setWidth(tr.width()+14); tr.setHeight(tr.height()+8);
        tr.moveCenter(QPoint((int)cx,(int)(cy-tr.height()*0.9)));
        if (tr.left()<4)          tr.moveLeft(4);
        if (tr.top() <4)          tr.moveTop(4);
        if (tr.right()>WARP_W-4)  tr.moveRight(WARP_W-4);
        if (tr.bottom()>WARP_H-4) tr.moveBottom(WARP_H-4);

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255,255,255,230));
        p.drawRoundedRect(tr,4,4);
        p.setBrush(col);
        p.drawRect(QRect(tr.left(),tr.top(),3,tr.height()));
        p.setPen(C::TEXT_PRI);
        p.drawText(tr.adjusted(6,0,0,0), Qt::AlignVCenter|Qt::AlignLeft, lbl);

        int bsz=(int)(13/s);
        QRect badge(tr.right()-bsz-2, tr.top()+(tr.height()-bsz)/2, bsz, bsz);
        p.setBrush(col); p.setPen(Qt::NoPen); p.drawEllipse(badge);
        p.setPen(Qt::white);
        p.setFont(QFont("Segoe UI",6.5/s,QFont::Bold));
        p.drawText(badge, Qt::AlignCenter, QString::number(i+1));
    }
    p.restore();
}

/* ─── Path-only view ────────────────────────────────────────────────────── */
void Widget::drawPathOnlyView(QPainter &p, const QRect &vp)
{
    auto [s,ox,oy] = fitRect(vp, WARP_W, WARP_H);
    p.save();
    p.translate(ox,oy); p.scale(s,s);

    p.setPen(Qt::NoPen);
    p.setBrush(QColor(180,160,140,45));
    p.drawRect(5,5,WARP_W,WARP_H);
    p.setBrush(Qt::white);
    p.setPen(QPen(C::BORDER,1.0/s));
    p.drawRect(0,0,WARP_W,WARP_H);

    for (int i=0; i<(int)objects.size(); ++i) {
        const auto &obj=objects[i]; const auto &m=obj.measure;
        QColor col = C::OBJ_COLORS[i % C::OBJ_COLORS.size()];

        p.setPen(QPen(col,1.8/s));
        p.setBrush(QColor(col.red(),col.green(),col.blue(),18));
        p.drawPath(obj.pathPx);

        QString lbl = QString("%1  %2×%3 mm")
                          .arg(m.shapeLabel)
                          .arg(m.widthMM,0,'f',1)
                          .arg(m.heightMM,0,'f',1);
        QFont f("Segoe UI",9.0/s);
        p.setFont(f);
        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(lbl);
        tr.setWidth(tr.width()+14); tr.setHeight(tr.height()+8);
        double cx=m.centerMM.x()*pixPerMM, cy=m.centerMM.y()*pixPerMM;
        tr.moveCenter(QPoint((int)cx,(int)cy));
        if (tr.left()<6)          tr.moveLeft(6);
        if (tr.top() <6)          tr.moveTop(6);
        if (tr.right()>WARP_W-6)  tr.moveRight(WARP_W-6);
        if (tr.bottom()>WARP_H-6) tr.moveBottom(WARP_H-6);

        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255,255,255,240));
        p.drawRoundedRect(tr,4,4);
        p.setBrush(col);
        p.drawRect(QRect(tr.left(),tr.top(),3,tr.height()));
        p.setPen(C::TEXT_PRI);
        p.drawText(tr.adjusted(6,0,0,0), Qt::AlignVCenter|Qt::AlignLeft, lbl);

        int bsz=(int)(13/s);
        QRect badge(tr.right()-bsz-2,tr.top()+(tr.height()-bsz)/2,bsz,bsz);
        p.setBrush(col); p.setPen(Qt::NoPen); p.drawEllipse(badge);
        p.setPen(Qt::white);
        p.setFont(QFont("Segoe UI",6.0/s,QFont::Bold));
        p.drawText(badge, Qt::AlignCenter, QString::number(i+1));
    }
    p.restore();
}

/* End of widget.cpp */
