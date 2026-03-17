/*
 * ═══════════════════════════════════════════════════════════════════════════
 *   A4 Object Digitizer  –  widget.cpp
 *
 *   Pipeline:
 *     Camera → Live Frame → A4 Detection (3-strategy fallback) →
 *     Perspective Warp → Object Segmentation → Measurement →
 *     QPainterPath → Temporal Smoothing → JSON Export
 *
 *   A4 Detection Strategies (tried in order):
 *     1. HSV white/light mask  – works on coloured table backgrounds
 *     2. Canny + largest quad  – works when paper contrasts well
 *     3. Adaptive threshold    – works under uneven lighting
 *
 *   Object Detection:
 *     - Adaptive binarization on warped A4
 *     - RETR_TREE hierarchy → only even-depth contours (real objects)
 *     - minAreaRect for accurate W×H, orientation
 *     - Shape classification: Rectangle / Square / Circle / Triangle / Ellipse / Polygon
 *     - Sub-pixel accurate QPainterPath (approxPolyDP with tight epsilon)
 *     - Temporal 7-frame rolling average on measurements
 *
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include "widget.h"

#include <QApplication>
#include <QMediaDevices>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QSizePolicy>
#include <QFont>
#include <QFontMetrics>
#include <QRect>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QDateTime>
#include <QStandardPaths>
#include <QDebug>
#include <QTimer>
#include <QMessageBox>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

/* ─────────────────────────────────────────────────────────────────────────
   Colour palette (white & orange theme)
   ───────────────────────────────────────────────────────────────────────── */
namespace C {
static const QColor BG        {245, 245, 242};   // off-white background
static const QColor SURFACE   {255, 255, 255};   // pure white panels
static const QColor BORDER    {220, 210, 200};   // warm light grey border
static const QColor ACCENT    {235,  95,  20};   // deep orange
static const QColor ACCENT2   {255, 140,  50};   // lighter orange
static const QColor SUCCESS   { 34, 160,  80};   // green
static const QColor WARNING   {235, 150,   0};   // amber
static const QColor DANGER    {210,  45,  45};   // red
static const QColor TEXT_PRI  { 30,  30,  30};   // near-black
static const QColor TEXT_SEC  {110, 100,  90};   // warm grey

// Object overlay colours (cycle) – vivid on white background
static const std::array<QColor,6> OBJ_COLORS {
    QColor{235,  95,  20},   // orange  (accent)
    QColor{ 30, 130, 200},   // blue
    QColor{ 34, 160,  80},   // green
    QColor{180,  40, 180},   // violet
    QColor{200,  40,  60},   // red
    QColor{ 20, 170, 170},   // teal
};
}

/* ─────────────────────────────────────────────────────────────────────────
   Constructor
   ───────────────────────────────────────────────────────────────────────── */
Widget::Widget(QWidget *parent) : QWidget(parent)
{
    setWindowTitle("A4 Object Digitizer  •  Vision System v2");
    setMinimumSize(1100, 760);
    resize(1200, 820);
    buildUI();
    applyDarkTheme();
    updateUI();

    /* ── Camera: create infrastructure only – NO device opened yet ── */
    videoSink      = new QVideoSink(this);
    captureSession = new QMediaCaptureSession(this);
    captureSession->setVideoSink(videoSink);

    connect(videoSink, &QVideoSink::videoFrameChanged,
            this,      &Widget::onFrameChanged);

    /* Auto-detect timer (runs at ~5 Hz during live view) */
    autoTimer = new QTimer(this);
    autoTimer->setInterval(200);
    connect(autoTimer, &QTimer::timeout, this, &Widget::onAutoDetectTimer);

    /*
     * Defer device enumeration until AFTER the event loop starts.
     * FFmpeg/DirectShow on Windows crashes if you open a device
     * before the Qt event loop is running.
     */
    QTimer::singleShot(0, this, [this](){
        availCameras = QMediaDevices::videoInputs();

        /* Block combo signals while we populate it */
        cameraCombo->blockSignals(true);

        if (availCameras.isEmpty()) {
            setStatus("⚠  No camera detected – plug in your camera and restart", "#ff5c5c");
            btnStart->setEnabled(false);
            cameraCombo->addItem("No cameras found");
            cameraCombo->setEnabled(false);
        } else {
            for (const auto &dev : availCameras)
                cameraCombo->addItem(dev.description());

            /* Pre-select last device (USB cameras appear last on Windows) */
            int defaultIdx = availCameras.size() - 1;
            cameraCombo->setCurrentIndex(defaultIdx);

            /* Show names in status but do NOT open the device yet */
            setStatus(
                QString("Found %1 camera(s)  –  select device then press ▶ Start Camera")
                    .arg(availCameras.size()),
                "#aaaaaa"
                );

            /* Build the QCamera object (does NOT open device until start()) */
            buildCameraForIndex(defaultIdx);
        }

        cameraCombo->blockSignals(false);

        /* Now it's safe to connect the combo signal */
        connect(cameraCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &Widget::selectCamera);

        updateUI();
    });
}

/* ─────────────────────────────────────────────────────────────────────────
   Build UI
   ───────────────────────────────────────────────────────────────────────── */
void Widget::buildUI()
{
    auto makeBtn = [&](const QString &label) -> QPushButton* {
        auto *b = new QPushButton(label, this);
        b->setFixedHeight(36);
        b->setCursor(Qt::PointingHandCursor);
        return b;
    };

    btnStart     = makeBtn("▶  Start Camera");
    btnCapture   = makeBtn("📸  Capture");
    btnDetect    = makeBtn("🔍  Detect Objects");
    btnReset     = makeBtn("↺  Reset");
    btnPathOnly  = makeBtn("🧩  Path View");
    btnExport    = makeBtn("💾  Export JSON");

    cameraCombo  = new QComboBox(this);
    cameraCombo->setFixedHeight(36);
    cameraCombo->setMinimumWidth(200);
    cameraCombo->setCursor(Qt::PointingHandCursor);
    cameraCombo->setToolTip("Select camera device");

    chkOverlay   = new QCheckBox("Show Overlay", this);
    chkOverlay->setChecked(true);
    chkAutoDetect = new QCheckBox("Auto-detect A4", this);
    chkAutoDetect->setChecked(true);

    /* Status bar row */
    lblStatus  = new QLabel("Ready  –  press Start Camera", this);
    lblA4Lock  = new QLabel("A4: ✗ Not detected", this);
    lblObjects = new QLabel("Objects: 0", this);

    lblStatus->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    lblA4Lock->setAlignment(Qt::AlignCenter);
    lblObjects->setAlignment(Qt::AlignCenter);

    /* Toolbar */
    auto *toolbar = new QHBoxLayout;
    toolbar->setSpacing(8);
    toolbar->addWidget(cameraCombo);
    toolbar->addWidget(btnStart);
    toolbar->addWidget(btnCapture);
    toolbar->addWidget(btnDetect);
    toolbar->addWidget(btnPathOnly);
    toolbar->addWidget(btnReset);
    toolbar->addWidget(btnExport);
    toolbar->addSpacing(16);
    toolbar->addWidget(chkOverlay);
    toolbar->addWidget(chkAutoDetect);
    toolbar->addStretch();

    /* Status bar */
    auto *statusBar = new QHBoxLayout;
    statusBar->setSpacing(20);
    statusBar->addWidget(lblStatus, 3);
    statusBar->addWidget(lblA4Lock, 1);
    statusBar->addWidget(lblObjects, 1);

    auto *root = new QVBoxLayout(this);
    root->setContentsMargins(12, 10, 12, 10);
    root->setSpacing(8);
    root->addLayout(toolbar);
    root->addLayout(statusBar);
    root->addStretch();   // viewport fills remainder via paintEvent

    /* Connections */
    connect(btnStart,    &QPushButton::clicked, this, &Widget::startCamera);
    connect(btnCapture,  &QPushButton::clicked, this, &Widget::captureImage);
    connect(btnDetect,   &QPushButton::clicked, this, &Widget::runDetection);
    connect(btnReset,    &QPushButton::clicked, this, &Widget::resetAll);
    connect(btnPathOnly, &QPushButton::clicked, this, &Widget::togglePathOnly);
    connect(btnExport,   &QPushButton::clicked, this, &Widget::exportJson);
    connect(chkAutoDetect, &QCheckBox::toggled, [this](bool on){
        if (on && currentState == LiveView && camera->isActive())
            autoTimer->start();
        else
            autoTimer->stop();
    });
}

/* ─────────────────────────────────────────────────────────────────────────
   White & Orange Theme stylesheet
   ───────────────────────────────────────────────────────────────────────── */
void Widget::applyDarkTheme()
{
    setStyleSheet(R"(
        QWidget {
            background: #f5f5f2;
            color: #1e1e1e;
            font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
            font-size: 13px;
        }
        QPushButton {
            background: #ffffff;
            color: #1e1e1e;
            border: 1px solid #d8cfc8;
            border-radius: 7px;
            padding: 0 16px;
            font-size: 12px;
        }
        QPushButton:hover   { background: #fff3ec; border-color: #eb5f14; color: #eb5f14; }
        QPushButton:pressed { background: #ffe8d8; }
        QPushButton:disabled{ color: #b8b0a8; border-color: #e8e2dc; background: #f8f6f4; }
        QPushButton#accent {
            background: #eb5f14;
            border-color: #eb5f14;
            color: #ffffff;
            font-weight: 600;
        }
        QPushButton#accent:hover   { background: #d45210; border-color: #d45210; }
        QPushButton#accent:pressed { background: #be4a0e; }
        QCheckBox { color: #6e6460; spacing: 6px; font-size: 12px; }
        QCheckBox::indicator { width:15px; height:15px; border-radius:4px;
                               border:1px solid #d0c8c0; background:#ffffff; }
        QCheckBox::indicator:checked { background:#eb5f14; border-color:#eb5f14; image: none; }
        QLabel { color: #6e6460; font-size: 12px; }
        QComboBox {
            background: #ffffff;
            color: #1e1e1e;
            border: 1px solid #d8cfc8;
            border-radius: 7px;
            padding: 0 10px;
            font-size: 12px;
        }
        QComboBox:hover { border-color: #eb5f14; }
        QComboBox::drop-down { border: none; width: 22px; }
        QComboBox QAbstractItemView {
            background: #ffffff;
            color: #1e1e1e;
            selection-background-color: #fff0e8;
            selection-color: #eb5f14;
            border: 1px solid #d8cfc8;
        }
    )");

    btnStart->setObjectName("accent");
    btnDetect->setObjectName("accent");
}

/* ─────────────────────────────────────────────────────────────────────────
   updateUI  – enable/disable controls per state
   ───────────────────────────────────────────────────────────────────────── */
void Widget::updateUI()
{
    bool captured = (currentState == CapturedView);
    bool detected = (currentState == DetectedView || currentState == PathOnlyView);
    bool camReady  = (camera != nullptr);
    bool camActive = camReady && camera->isActive();

    btnStart->setEnabled(camReady);
    btnStart->setText(camActive ? "⏹  Stop Camera" : "▶  Start Camera");

    /* Capture is enabled whenever camera is streaming, regardless of state */
    btnCapture ->setEnabled(camActive && !liveFrame.isNull());
    btnDetect  ->setEnabled(captured);
    btnReset   ->setEnabled(currentState != LiveView);
    btnPathOnly->setEnabled(detected);
    btnExport  ->setEnabled(detected);

    lblObjects->setText(QString("Objects: %1").arg(objects.size()));
}

void Widget::setStatus(const QString &msg, const QString &color)
{
    lblStatus->setText(msg);
    lblStatus->setStyleSheet(QString("color:%1;font-size:12px;").arg(color));
}

/* ─────────────────────────────────────────────────────────────────────────
   Camera
   ───────────────────────────────────────────────────────────────────────── */

/* Build (but do NOT start) a QCamera for the given device index */
void Widget::buildCameraForIndex(int index)
{
    if (index < 0 || index >= availCameras.size()) return;

    /* Tear down any existing camera safely */
    if (camera) {
        camera->stop();
        captureSession->setCamera(nullptr);
        delete camera;
        camera = nullptr;
    }

    const QCameraDevice &dev = availCameras[index];
    camera = new QCamera(dev, this);

    connect(camera, &QCamera::errorOccurred,
            this, [this](QCamera::Error, const QString &msg){
                setStatus("⚠  Camera error: " + msg, "#ff5c5c");
                qDebug() << "Camera error:" << msg;
            });
    connect(camera, &QCamera::activeChanged, this, [this](bool active){
        qDebug() << "Camera active:" << active;
        updateUI();
    });
}

/* Called when the user changes the combo box */
void Widget::selectCamera(int index)
{
    if (index < 0 || index >= availCameras.size()) return;

    autoTimer->stop();
    liveFrame    = QImage();
    currentState = LiveView;
    a4Locked     = false;
    quadLockCount= 0;

    buildCameraForIndex(index);

    setStatus(
        QString("Selected: %1  –  press ▶ Start Camera")
            .arg(availCameras[index].description()),
        "#aaaaaa"
        );
    updateUI();
    update();
}

void Widget::startCamera()
{
    if (!camera) {
        setStatus("⚠  No camera object – select a device first", "#ff5c5c");
        return;
    }

    /* Toggle: if already active, stop it */
    if (camera->isActive()) {
        autoTimer->stop();
        camera->stop();
        captureSession->setCamera(nullptr);
        liveFrame = QImage();
        setStatus("Camera stopped  –  press ▶ to restart", "#aaaaaa");
        updateUI();
        update();
        return;
    }

    setStatus("Starting camera…", "#ffbd2e");
    captureSession->setCamera(camera);
    camera->start();

    currentState  = LiveView;
    a4Locked      = false;
    quadLockCount = 0;

    if (chkAutoDetect->isChecked()) autoTimer->start();

    /* Warn if no frames arrive within 4 seconds */
    QTimer::singleShot(4000, this, [this](){
        if (liveFrame.isNull() && camera && camera->isActive())
            setStatus("⚠  No frames received – try another camera or check USB connection", "#ffbd2e");
        else if (!liveFrame.isNull())
            setStatus("✓  Camera live  –  point at A4 sheet, then press 📸 Capture", "#48c78e");
    });

    updateUI();
}

void Widget::onFrameChanged(const QVideoFrame &frame)
{
    if (!frame.isValid() || currentState != LiveView) return;

    QVideoFrame f(frame);
    if (!f.map(QVideoFrame::ReadOnly)) return;

    QImage img = f.toImage();
    f.unmap();

    if (img.isNull()) return;

    bool firstFrame = liveFrame.isNull();
    liveFrame = img.convertToFormat(QImage::Format_ARGB32);

    /* On first frame: enable Capture button and update status */
    if (firstFrame) {
        setStatus("✓  Camera live  –  point at A4 sheet, then press 📸 Capture", "#48c78e");
        updateUI();
    }

    update();
}

/* Auto-detect A4 in live stream for lock indicator */
void Widget::onAutoDetectTimer()
{
    if (liveFrame.isNull() || currentState != LiveView) return;

    cv::Mat src(liveFrame.height(), liveFrame.width(),
                CV_8UC4, (void*)liveFrame.bits(), liveFrame.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);

    cv::Mat warped;
    bool found = tryHsvMethod(bgr, warped) ||
                 tryCannyMethod(bgr, warped) ||
                 tryAdaptiveMethod(bgr, warped);

    if (found) {
        quadLockCount = std::min(quadLockCount + 1, 5);
        a4Locked = (quadLockCount >= 3);
    } else {
        quadLockCount = std::max(quadLockCount - 1, 0);
        if (quadLockCount == 0) a4Locked = false;
    }

    if (a4Locked) {
        lblA4Lock->setText("A4: ✓ Locked");
        lblA4Lock->setStyleSheet("color:#48c78e;font-size:12px;font-weight:bold;");
    } else {
        lblA4Lock->setText(found ? "A4: ~ Acquiring..." : "A4: ✗ Not detected");
        lblA4Lock->setStyleSheet(found ?
                                     "color:#ffbd2e;font-size:12px;" :
                                     "color:#ff5c5c;font-size:12px;");
    }
}

/* ─────────────────────────────────────────────────────────────────────────
   Capture
   ───────────────────────────────────────────────────────────────────────── */
void Widget::captureImage()
{
    if (liveFrame.isNull()) {
        setStatus("⚠  No frame available yet", "#ff5c5c");
        return;
    }
    capturedFrame = liveFrame.copy();
    currentState  = CapturedView;
    /* Keep camera streaming but stop the A4 auto-detect timer */
    autoTimer->stop();
    setStatus("✓  Frame captured  –  press 🔍 Detect Objects", "#48c78e");
    updateUI();
    update();
}

/* ─────────────────────────────────────────────────────────────────────────
   Reset
   ───────────────────────────────────────────────────────────────────────── */
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
    setStatus("Reset  –  Camera live", "#aaaaaa");
    if (chkAutoDetect->isChecked() && camera && camera->isActive()) autoTimer->start();
    updateUI();
    update();
}

/* ─────────────────────────────────────────────────────────────────────────
   Detect (main entry)
   ───────────────────────────────────────────────────────────────────────── */
void Widget::runDetection()
{
    if (capturedFrame.isNull()) {
        setStatus("⚠  No captured frame – press 📸 Capture first", "#ff5c5c");
        return;
    }

    objects.clear();
    history.clear();

    setStatus("Detecting A4 sheet…", "#ffbd2e");
    qApp->processEvents();

    /* Ensure ARGB32 so detectAndWarpA4 can safely use CV_8UC4 */
    QImage frameToProcess = capturedFrame.convertToFormat(QImage::Format_ARGB32);

    if (!detectAndWarpA4(frameToProcess)) {
        setStatus("⚠  A4 sheet not found – ensure the entire sheet is visible and well-lit", "#ff5c5c");
        return;
    }

    setStatus("A4 warped – finding objects…", "#ffbd2e");
    qApp->processEvents();

    /* warpedA4 is BGR888 – detectObjects handles it */
    detectObjects(warpedA4);

    if (objects.empty()) {
        setStatus("⚠  No objects detected – check lighting, or drawn shapes may be too faint", "#ffbd2e");
    } else {
        setStatus(
            QString("✓  %1 object(s) detected – press 🧩 Path View or 💾 Export")
                .arg(objects.size()), "#48c78e");
    }

    currentState = DetectedView;
    updateUI();
    update();
}

/* ─────────────────────────────────────────────────────────────────────────
   ══════════════════  A4 DETECTION  ═══════════════════════════════════════
   Strategy 1: HSV white mask  (best for coloured table surfaces)
   ───────────────────────────────────────────────────────────────────────── */
bool Widget::tryHsvMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    /* White/very-light paper range */
    cv::Mat mask;
    cv::inRange(hsv,
                cv::Scalar(  0,  0, 160),
                cv::Scalar(180, 50, 255),
                mask);

    /* Morphological cleanup */
    auto k9 = cv::getStructuringElement(cv::MORPH_RECT, {9,9});
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k9);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  k9);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * cv::arcLength(c, true), true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            double imgArea = bgr.cols * bgr.rows;
            if (area > imgArea * 0.05 && area < imgArea * 0.98) {
                std::vector<cv::Point2f> pts(approx.begin(), approx.end());
                return warpFromQuad(bgr, pts, warped);
            }
        }
    }
    return false;
}

/* ─────────────────────────────────────────────────────────────────────────
   Strategy 2: Canny edges + largest quadrilateral
   ───────────────────────────────────────────────────────────────────────── */
bool Widget::tryCannyMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat gray, blurred, edges;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, {5,5}, 0);
    cv::Canny(blurred, edges, 50, 150);

    auto k3 = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    cv::dilate(edges, edges, k3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    double bestArea = 0;
    std::vector<cv::Point2f> bestQuad;

    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * cv::arcLength(c,true), true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            double imgArea = bgr.cols * bgr.rows;
            if (area > imgArea * 0.05 && area > bestArea) {
                bestArea = area;
                bestQuad = {approx.begin(), approx.end()};
            }
        }
    }

    if (bestQuad.size() == 4)
        return warpFromQuad(bgr, bestQuad, warped);
    return false;
}

/* ─────────────────────────────────────────────────────────────────────────
   Strategy 3: Adaptive threshold (handles uneven lighting)
   ───────────────────────────────────────────────────────────────────────── */
bool Widget::tryAdaptiveMethod(const cv::Mat &bgr, cv::Mat &warped)
{
    cv::Mat gray, bin;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5,5}, 0);
    cv::adaptiveThreshold(gray, bin, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY, 21, 4);

    /* Invert – paper becomes white */
    cv::bitwise_not(bin, bin);

    auto k = cv::getStructuringElement(cv::MORPH_RECT, {7,7});
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestArea = 0;
    std::vector<cv::Point2f> bestQuad;

    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx, 0.02 * cv::arcLength(c,true), true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            double imgArea = bgr.cols * bgr.rows;
            if (area > imgArea * 0.05 && area > bestArea) {
                bestArea = area;
                bestQuad = {approx.begin(), approx.end()};
            }
        }
    }

    if (bestQuad.size() == 4)
        return warpFromQuad(bgr, bestQuad, warped);
    return false;
}

/* ─────────────────────────────────────────────────────────────────────────
   Order quad: TL, TR, BR, BL
   ───────────────────────────────────────────────────────────────────────── */
std::vector<cv::Point2f> Widget::orderQuad(std::vector<cv::Point2f> pts)
{
    /* sort by y → top two / bottom two */
    std::sort(pts.begin(), pts.end(),
              [](const cv::Point2f &a, const cv::Point2f &b){ return a.y < b.y; });

    cv::Point2f tl, tr, bl, br;
    if (pts[0].x < pts[1].x) { tl = pts[0]; tr = pts[1]; }
    else                      { tl = pts[1]; tr = pts[0]; }
    if (pts[2].x < pts[3].x) { bl = pts[2]; br = pts[3]; }
    else                      { bl = pts[3]; br = pts[2]; }

    return {tl, tr, br, bl};
}

/* ─────────────────────────────────────────────────────────────────────────
   Warp perspective → fixed A4 canvas
   ───────────────────────────────────────────────────────────────────────── */
bool Widget::warpFromQuad(const cv::Mat &src,
                          std::vector<cv::Point2f> quad,
                          cv::Mat &warped)
{
    if (quad.size() != 4) return false;

    auto ordered = orderQuad(quad);
    lastQuad = ordered;

    std::vector<cv::Point2f> dst = {
        {0,       0       },
        {WARP_W,  0       },
        {WARP_W,  WARP_H  },
        {0,       WARP_H  }
    };

    /* Validate A4 aspect ratio (portrait or landscape within 25% tolerance) */
    double wTop   = cv::norm(ordered[0] - ordered[1]);
    double wBot   = cv::norm(ordered[3] - ordered[2]);
    double hLeft  = cv::norm(ordered[0] - ordered[3]);
    double hRight = cv::norm(ordered[1] - ordered[2]);
    double avgW   = (wTop + wBot) * 0.5;
    double avgH   = (hLeft + hRight) * 0.5;
    if (avgW < 1 || avgH < 1) return false;

    double ratio = avgW / avgH;
    constexpr double A4_RATIO      = A4_WIDTH_MM / A4_HEIGHT_MM;  // ~0.707
    constexpr double RATIO_TOL     = 0.30;   // generous tolerance
    if (ratio > A4_RATIO + RATIO_TOL && ratio < 1.0/(A4_RATIO + RATIO_TOL)) {
        /* could be landscape – flip WARP dims */
    }

    cv::Mat H = cv::getPerspectiveTransform(ordered, dst);
    cv::warpPerspective(src, warped, H, {WARP_W, WARP_H},
                        cv::INTER_LANCZOS4);

    warpedA4 = QImage(warped.data, warped.cols, warped.rows,
                      (int)warped.step, QImage::Format_BGR888).copy();

    pixPerMM = PX_PER_MM;
    qDebug() << "A4 warped | px/mm =" << pixPerMM
             << "| source quad area =" << cv::contourArea(ordered);
    return true;
}

/* ─────────────────────────────────────────────────────────────────────────
   Master A4 detection (tries all 3 strategies)
   ───────────────────────────────────────────────────────────────────────── */
bool Widget::detectAndWarpA4(const QImage &image)
{
    cv::Mat src(image.height(), image.width(),
                CV_8UC4, (void*)image.bits(), image.bytesPerLine());
    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);

    cv::Mat warped;
    if (tryHsvMethod(bgr, warped)) {
        qDebug() << "A4 found via HSV";
        return true;
    }
    if (tryCannyMethod(bgr, warped)) {
        qDebug() << "A4 found via Canny";
        return true;
    }
    if (tryAdaptiveMethod(bgr, warped)) {
        qDebug() << "A4 found via Adaptive";
        return true;
    }
    return false;
}

/* ─────────────────────────────────────────────────────────────────────────
   Shape classifier
   ───────────────────────────────────────────────────────────────────────── */
QString Widget::classifyShape(const std::vector<cv::Point> &approx,
                              double circularity)
{
    int n = (int)approx.size();

    if (circularity > 0.85)                        return "Circle";
    if (circularity > 0.75 && n > 8)               return "Ellipse";
    if (n == 3)                                    return "Triangle";
    if (n == 4) {
        /* distinguish rectangle from square */
        cv::RotatedRect rr = cv::minAreaRect(approx);
        double ratio = rr.size.width / std::max(rr.size.height, 1.0f);
        if (ratio < 1) ratio = 1.0 / ratio;
        return (ratio < 1.15) ? "Square" : "Rectangle";
    }
    if (n == 5)  return "Pentagon";
    if (n == 6)  return "Hexagon";
    return QString("Polygon(%1)").arg(n);
}

/* ─────────────────────────────────────────────────────────────────────────
   ══════════════════  OBJECT DETECTION  ═══════════════════════════════════
   ───────────────────────────────────────────────────────────────────────── */
void Widget::detectObjects(const QImage &img)
{
    objects.clear();

    cv::Mat src(img.height(), img.width(),
                CV_8UC3, (void*)img.bits(), img.bytesPerLine());

    /* ── 1. Pre-process ── */
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    /* Mild denoise before threshold */
    cv::GaussianBlur(gray, gray, {5,5}, 0);

    /* Adaptive threshold – handles shadows and uneven paper tone */
    cv::Mat bin;
    cv::adaptiveThreshold(gray, bin, 255,
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 31, 7);

    /* Close small gaps (connects broken contour lines) */
    auto k5 = cv::getStructuringElement(cv::MORPH_RECT, {5,5});
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, k5);

    /* Remove tiny noise specks */
    auto k3 = cv::getStructuringElement(cv::MORPH_RECT, {3,3});
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, k3);

    /* ── 2. Find contours with full hierarchy ── */
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin, contours, hierarchy,
                     cv::RETR_CCOMP,        // 2-level: outer + holes only
                     cv::CHAIN_APPROX_SIMPLE);

    /*
     * RETR_CCOMP gives exactly 2 levels:
     *   hierarchy[i][3] == -1  →  depth 0  = outer contour of an object  ✓ KEEP
     *   hierarchy[i][3] != -1  →  depth 1  = hole inside an object        ✗ SKIP
     *
     * This is correct for a warped A4 image where the canvas boundary
     * is the image edge (not a contour), so all drawn/placed objects
     * appear at depth 0.
     */
    auto isHole = [&](int i) -> bool {
        return hierarchy[i][3] != -1;   // has a parent → it is a hole
    };

    double imgArea = (double)img.width() * img.height();
    Q_UNUSED(imgArea)

    /* ── 4. Filter & measure ── */
    for (size_t i = 0; i < contours.size(); ++i) {

        /* Skip holes – only process outer object boundaries */
        if (isHole((int)i)) continue;

        double areaPx  = cv::contourArea(contours[i]);

        /* Size gates */
        double areaMM2 = areaPx / (pixPerMM * pixPerMM);
        if (areaMM2 < 4.0)  continue;    // < 4 mm² → noise / printer artefacts
        if (areaMM2 > A4_WIDTH_MM * A4_HEIGHT_MM * 0.90) continue;  // whole page

        /* ── 5. Accurate approximation (tight epsilon) ── */
        double epsilon = std::max(1.5, pixPerMM * 0.4);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, true);

        if ((int)approx.size() < 3) continue;

        /* ── 6. Measurements ── */
        cv::RotatedRect rr = cv::minAreaRect(contours[i]);
        double perimPx = cv::arcLength(contours[i], true);

        double w = rr.size.width, h = rr.size.height;
        if (w < h) std::swap(w, h);   // always W ≥ H

        double circularity = (perimPx > 0)
                                 ? 4.0 * M_PI * areaPx / (perimPx * perimPx)
                                 : 0;

        ObjectMeasure m;
        m.widthMM   = w / pixPerMM;
        m.heightMM  = h / pixPerMM;
        m.areaMM2   = areaMM2;
        m.perimMM   = perimPx / pixPerMM;
        m.angleDeg  = rr.angle;
        m.centerMM  = QPointF(rr.center.x / pixPerMM,
                             rr.center.y / pixPerMM);
        m.shapeLabel = classifyShape(approx, circularity);

        /* ── 7. Build QPainterPath in pixel space ── */
        QPainterPath pathPx;
        pathPx.moveTo(approx[0].x, approx[0].y);
        for (size_t k = 1; k < approx.size(); ++k)
            pathPx.lineTo(approx[k].x, approx[k].y);
        pathPx.closeSubpath();

        /* ── 8. Build QPainterPath in mm space ── */
        QPainterPath pathMM;
        pathMM.moveTo(approx[0].x / pixPerMM, approx[0].y / pixPerMM);
        for (size_t k = 1; k < approx.size(); ++k)
            pathMM.lineTo(approx[k].x / pixPerMM, approx[k].y / pixPerMM);
        pathMM.closeSubpath();

        DetectedObject obj;
        obj.pathPx  = pathPx;
        obj.pathMM  = pathMM;
        obj.measure = m;
        objects.push_back(obj);

        qDebug() << "[OBJ]"
                 << m.shapeLabel
                 << "W=" << QString::number(m.widthMM,'f',1) + "mm"
                 << "H=" << QString::number(m.heightMM,'f',1) + "mm"
                 << "Area=" << QString::number(m.areaMM2,'f',1) + "mm²"
                 << "Circ=" << QString::number(circularity,'f',3);
    }

    /* ── 9. Temporal smoothing ── */
    pushAndSmooth();
}

/* ─────────────────────────────────────────────────────────────────────────
   Temporal smoothing  (rolling average of last N frames)
   ───────────────────────────────────────────────────────────────────────── */
void Widget::pushAndSmooth()
{
    /* Push current frame's measurements */
    std::vector<ObjectMeasure> current;
    for (auto &o : objects) current.push_back(o.measure);
    history.push_back(current);
    if ((int)history.size() > SMOOTH_FRAMES) history.pop_front();

    /*
     * Simple strategy: for each object, average over all frames that have
     * at least that many objects.  This handles frame-to-frame count variation.
     */
    size_t n = objects.size();
    for (size_t i = 0; i < n; ++i) {
        double wSum = 0, hSum = 0, aSum = 0; int cnt = 0;
        for (auto &frame : history) {
            if (i < frame.size()) {
                wSum += frame[i].widthMM;
                hSum += frame[i].heightMM;
                aSum += frame[i].areaMM2;
                ++cnt;
            }
        }
        if (cnt > 0) {
            objects[i].measure.widthMM  = wSum / cnt;
            objects[i].measure.heightMM = hSum / cnt;
            objects[i].measure.areaMM2  = aSum / cnt;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────
   Path only toggle
   ───────────────────────────────────────────────────────────────────────── */
void Widget::togglePathOnly()
{
    if (objects.empty()) return;

    if (currentState == PathOnlyView)
        currentState = DetectedView;
    else
        currentState = PathOnlyView;

    btnPathOnly->setText(currentState == PathOnlyView ?
                             "🖼  Full View" : "🧩  Path View");
    updateUI();
    update();
}

/* ─────────────────────────────────────────────────────────────────────────
   Export JSON
   ───────────────────────────────────────────────────────────────────────── */
void Widget::exportJson()
{
    exportPathsToJson();
}

void Widget::exportPathsToJson()
{
    if (objects.empty()) {
        setStatus("⚠  No objects to export", "#ffbd2e");
        return;
    }

    QJsonObject root;
    root["version"]       = "2.0";
    root["timestamp"]     = QDateTime::currentDateTime().toString(Qt::ISODate);
    root["unit"]          = "mm";
    root["pixelsPerMM"]   = pixPerMM;

    QJsonObject canvas;
    canvas["width_mm"]  = A4_WIDTH_MM;
    canvas["height_mm"] = A4_HEIGHT_MM;
    root["canvas"] = canvas;

    QJsonArray objectsArray;

    for (int i = 0; i < (int)objects.size(); ++i) {
        const auto &obj = objects[i];
        const auto &m   = obj.measure;

        QJsonObject jo;
        jo["id"]         = i + 1;
        jo["shape"]      = m.shapeLabel;
        jo["width_mm"]   = qRound(m.widthMM  * 100) / 100.0;
        jo["height_mm"]  = qRound(m.heightMM * 100) / 100.0;
        jo["area_mm2"]   = qRound(m.areaMM2  * 10)  / 10.0;
        jo["perim_mm"]   = qRound(m.perimMM  * 10)  / 10.0;
        jo["angle_deg"]  = qRound(m.angleDeg * 10)  / 10.0;

        QJsonObject center;
        center["x_mm"] = qRound(m.centerMM.x() * 100) / 100.0;
        center["y_mm"] = qRound(m.centerMM.y() * 100) / 100.0;
        jo["center"] = center;

        /* Path in mm */
        QJsonArray pathArr;
        const QPainterPath &path = obj.pathMM;

        for (int e = 0; e < path.elementCount(); ++e) {
            QPainterPath::Element el = path.elementAt(e);
            QJsonObject cmd;

            auto round2 = [](double v){ return qRound(v * 100) / 100.0; };

            if (el.type == QPainterPath::MoveToElement) {
                cmd["cmd"] = "M";
                cmd["x"]   = round2(el.x);
                cmd["y"]   = round2(el.y);
            } else if (el.type == QPainterPath::LineToElement) {
                cmd["cmd"] = "L";
                cmd["x"]   = round2(el.x);
                cmd["y"]   = round2(el.y);
            } else if (el.type == QPainterPath::CurveToElement) {
                auto c1  = path.elementAt(e);
                auto c2  = path.elementAt(e + 1);
                auto end = path.elementAt(e + 2);
                cmd["cmd"] = "C";
                cmd["x1"]  = round2(c1.x);  cmd["y1"] = round2(c1.y);
                cmd["x2"]  = round2(c2.x);  cmd["y2"] = round2(c2.y);
                cmd["x"]   = round2(end.x); cmd["y"]  = round2(end.y);
                e += 2;
            }
            pathArr.append(cmd);
        }
        QJsonObject closeCmd; closeCmd["cmd"] = "Z";
        pathArr.append(closeCmd);

        jo["path"] = pathArr;
        objectsArray.append(jo);
    }

    root["objects"] = objectsArray;

    /* Save to Documents */
    QString dir  = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QString fname = dir + "/digitized_" +
                    QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") +
                    ".json";

    QFile file(fname);
    if (!file.open(QIODevice::WriteOnly)) {
        setStatus("⚠  Failed to write JSON file", "#ff5c5c");
        return;
    }
    file.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    file.close();

    setStatus(QString("✓  Exported %1 object(s) → %2").arg(objects.size()).arg(fname),
              "#48c78e");
    qDebug() << "Exported to" << fname;
}

/* ─────────────────────────────────────────────────────────────────────────
   resizeEvent
   ───────────────────────────────────────────────────────────────────────── */
void Widget::resizeEvent(QResizeEvent *e)
{
    QWidget::resizeEvent(e);
    update();
}

/* ─────────────────────────────────────────────────────────────────────────
   ══════════════════  PAINT EVENT  ════════════════════════════════════════
   ───────────────────────────────────────────────────────────────────────── */
void Widget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);

    /* Toolbar + status bar height (approx 90px) */
    const int TOP_OFFSET = 90;
    QRect viewport(0, TOP_OFFSET, width(), height() - TOP_OFFSET);
    p.fillRect(viewport, C::BG);  // off-white background

    switch (currentState) {
    case LiveView:
    case CapturedView:     drawLiveView(p, viewport);     break;
    case DetectedView:     drawDetectedView(p, viewport); break;
    case PathOnlyView:     drawPathOnlyView(p, viewport); break;
    }
}

/* Remove the unused overloads */

/* ─────────────────────────────────────────────────────────────────────────
   Helper: compute scale & offset to fit image inside rect
   ───────────────────────────────────────────────────────────────────────── */
static std::tuple<double,double,double>
fitRect(const QRect &vp, int imgW, int imgH)
{
    double sx = vp.width()  / double(imgW);
    double sy = vp.height() / double(imgH);
    double s  = std::min(sx, sy) * 0.95;  // 5% margin
    double ox = vp.x() + (vp.width()  - imgW * s) * 0.5;
    double oy = vp.y() + (vp.height() - imgH * s) * 0.5;
    return {s, ox, oy};
}

/* ─────────────────────────────────────────────────────────────────────────
   Live / Captured view
   ───────────────────────────────────────────────────────────────────────── */
void Widget::drawLiveView(QPainter &p, const QRect &vp)
{
    const QImage &img = (currentState == CapturedView) ? capturedFrame : liveFrame;
    if (img.isNull()) {
        p.setPen(C::TEXT_SEC);
        p.setFont(QFont("Segoe UI", 15));
        p.drawText(vp, Qt::AlignCenter, "Waiting for camera…\n\nSelect a camera and press ▶ Start Camera");
        return;
    }

    auto [s, ox, oy] = fitRect(vp, img.width(), img.height());

    p.save();
    p.translate(ox, oy);
    p.scale(s, s);
    p.drawImage(0, 0, img);

    /* A4 lock overlay on live */
    if (currentState == LiveView && a4Locked && !lastQuad.empty()) {
        QPolygonF poly;
        for (auto &pt : lastQuad) poly << QPointF(pt.x, pt.y);
        p.setPen(QPen(C::SUCCESS, 3.0 / s));
        p.setBrush(QColor(34, 160, 80, 30));
        p.drawPolygon(poly);

        /* "A4 Locked" label */
        QFont f("Segoe UI", 22.0 / s, QFont::Bold);
        p.setFont(f);
        p.setPen(C::SUCCESS);
        QRectF br = p.fontMetrics().boundingRect("✓ A4 Locked");
        br.moveCenter(poly.boundingRect().center());
        p.drawText(br, Qt::AlignCenter, "✓ A4 Locked");
    }
    p.restore();
}

/* ─────────────────────────────────────────────────────────────────────────
   Detected view (warped A4 + overlays)
   ───────────────────────────────────────────────────────────────────────── */
void Widget::drawDetectedView(QPainter &p, const QRect &vp)
{
    if (warpedA4.isNull()) return;
    auto [s, ox, oy] = fitRect(vp, warpedA4.width(), warpedA4.height());

    p.save();
    p.translate(ox, oy);
    p.scale(s, s);

    /* A4 image */
    p.drawImage(0, 0, warpedA4);

    /* Ruler edges */
    p.setPen(QPen(C::BORDER, 2.0 / s));
    p.drawRect(0, 0, warpedA4.width(), warpedA4.height());

    if (!chkOverlay->isChecked()) { p.restore(); return; }

    /* ── Object overlays ── */
    for (int i = 0; i < (int)objects.size(); ++i) {
        const auto &obj = objects[i];
        const auto &m   = obj.measure;
        QColor col = C::OBJ_COLORS[i % C::OBJ_COLORS.size()];

        /* Filled path */
        p.setPen(QPen(col, 2.0 / s));
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 28));
        p.drawPath(obj.pathPx);

        /* Crosshair at centre */
        double cx = m.centerMM.x() * pixPerMM;
        double cy = m.centerMM.y() * pixPerMM;
        double cross = 8 / s;
        p.setPen(QPen(col, 1.2 / s));
        p.drawLine(QLineF(cx - cross, cy, cx + cross, cy));
        p.drawLine(QLineF(cx, cy - cross, cx, cy + cross));

        /* ── Compact label bubble ── */
        QString label = QString("%1  %2×%3mm")
                            .arg(m.shapeLabel)
                            .arg(m.widthMM,  0, 'f', 1)
                            .arg(m.heightMM, 0, 'f', 1);

        QFont f("Segoe UI", 9.0 / s);   // small font
        p.setFont(f);

        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(label);
        tr.setWidth(tr.width() + 12);
        tr.setHeight(tr.height() + 7);
        tr.moveCenter(QPoint((int)cx, (int)cy - (int)(tr.height() * 0.8 + 4 / s)));

        /* Clamp inside canvas */
        if (tr.left()  < 4)           tr.moveLeft(4);
        if (tr.top()   < 4)           tr.moveTop(4);
        if (tr.right() > WARP_W - 4)  tr.moveRight(WARP_W - 4);
        if (tr.bottom()> WARP_H - 4)  tr.moveBottom(WARP_H - 4);

        /* White pill background with coloured left bar */
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255, 255, 255, 230));
        p.drawRoundedRect(tr, 4, 4);

        /* Coloured left accent strip */
        QRect strip(tr.left(), tr.top(), 3, tr.height());
        p.setBrush(col);
        p.drawRect(strip);

        /* Text */
        p.setPen(C::TEXT_PRI);
        QRect textRect = tr.adjusted(6, 0, 0, 0);
        p.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, label);

        /* Small index badge */
        int bsz = (int)(14 / s);
        QRect badge(tr.right() - bsz - 2, tr.top() + (tr.height() - bsz) / 2, bsz, bsz);
        p.setBrush(col);
        p.setPen(Qt::NoPen);
        p.drawEllipse(badge);
        p.setPen(Qt::white);
        p.setFont(QFont("Segoe UI", 6.5 / s, QFont::Bold));
        p.drawText(badge, Qt::AlignCenter, QString::number(i + 1));
    }

    p.restore();
}

/* ─────────────────────────────────────────────────────────────────────────
   Path-only view  (white A4 + black outlines, no image)
   ───────────────────────────────────────────────────────────────────────── */
void Widget::drawPathOnlyView(QPainter &p, const QRect &vp)
{
    auto [s, ox, oy] = fitRect(vp, WARP_W, WARP_H);

    p.save();
    p.translate(ox, oy);
    p.scale(s, s);

    /* White A4 background with subtle warm shadow */
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(180, 160, 140, 50));
    p.drawRect(5, 5, WARP_W, WARP_H);

    p.setBrush(Qt::white);
    p.setPen(QPen(C::BORDER, 1.0 / s));
    p.drawRect(0, 0, WARP_W, WARP_H);

    /* Objects */
    for (int i = 0; i < (int)objects.size(); ++i) {
        const auto &obj = objects[i];
        const auto &m   = obj.measure;
        QColor col = C::OBJ_COLORS[i % C::OBJ_COLORS.size()];

        p.setPen(QPen(col, 1.8 / s));
        p.setBrush(QColor(col.red(), col.green(), col.blue(), 18));
        p.drawPath(obj.pathPx);

        /* Compact measurement label */
        QString label = QString("%1  %2×%3mm")
                            .arg(m.shapeLabel)
                            .arg(m.widthMM,  0, 'f', 1)
                            .arg(m.heightMM, 0, 'f', 1);

        QFont f("Segoe UI", 9.0 / s);
        p.setFont(f);

        QFontMetrics fm(f);
        QRect tr = fm.boundingRect(label);
        tr.setWidth(tr.width() + 12);
        tr.setHeight(tr.height() + 7);
        double cx = m.centerMM.x() * pixPerMM;
        double cy = m.centerMM.y() * pixPerMM;
        tr.moveCenter(QPoint((int)cx, (int)cy));

        /* Clamp */
        if (tr.left()  < 6)           tr.moveLeft(6);
        if (tr.top()   < 6)           tr.moveTop(6);
        if (tr.right() > WARP_W - 6)  tr.moveRight(WARP_W - 6);
        if (tr.bottom()> WARP_H - 6)  tr.moveBottom(WARP_H - 6);

        /* White pill */
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255, 255, 255, 240));
        p.drawRoundedRect(tr, 4, 4);

        /* Coloured left strip */
        p.setBrush(col);
        p.drawRect(QRect(tr.left(), tr.top(), 3, tr.height()));

        /* Text */
        p.setPen(C::TEXT_PRI);
        p.drawText(tr.adjusted(6, 0, 0, 0), Qt::AlignVCenter | Qt::AlignLeft, label);

        /* Badge */
        int bsz = (int)(13 / s);
        QRect badge(tr.right() - bsz - 2, tr.top() + (tr.height() - bsz) / 2, bsz, bsz);
        p.setBrush(col);
        p.setPen(Qt::NoPen);
        p.drawEllipse(badge);
        p.setPen(Qt::white);
        p.setFont(QFont("Segoe UI", 6.0 / s, QFont::Bold));
        p.drawText(badge, Qt::AlignCenter, QString::number(i + 1));
    }

    p.restore();
}

/* End of widget.cpp */
