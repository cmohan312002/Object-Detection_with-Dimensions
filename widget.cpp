#include "widget.h"

#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>

/* ===================== CONSTANTS ===================== */
static const int A4_W = 2480;   // pixels
static const int A4_H = 3508;   // pixels

/* ===================== HELPER FUNCTIONS ===================== */

// Build QPainterPath by sampling contour every ~1mm
static QPainterPath buildPathFromContour(
    const std::vector<cv::Point> &contour,
    double stepPx
    ) {
    QPainterPath path;
    if (contour.empty()) return path;

    QPointF prev(contour[0].x, contour[0].y);
    path.moveTo(prev);

    double acc = 0.0;

    for (size_t i = 1; i < contour.size(); ++i) {
        QPointF curr(contour[i].x, contour[i].y);
        double d = QLineF(prev, curr).length();
        acc += d;

        if (acc >= stepPx) {
            path.lineTo(curr);
            acc = 0.0;
        }
        prev = curr;
    }

    path.closeSubpath();
    return path;
}

// Dump painter path commands
void Widget::dumpPainterPath(const QPainterPath &path, int index)
{
    qDebug() << "========== OBJECT" << index << "==========";
    for (int i = 0; i < path.elementCount(); ++i) {
        auto e = path.elementAt(i);
        if (e.isMoveTo())
            qDebug() << "moveTo(" << e.x << "," << e.y << ")";
        else if (e.isLineTo())
            qDebug() << "lineTo(" << e.x << "," << e.y << ")";
        else if (e.isCurveTo())
            qDebug() << "curveTo(" << e.x << "," << e.y << ")";
    }
    qDebug() << "====================================";
}


/* ===================== CONSTRUCTOR ===================== */
Widget::Widget(QWidget *parent) : QWidget(parent)
{
    setWindowTitle("A4 Object Digitizer");
    resize(1000, 750);

    startButton   = new QPushButton("▶ Start");
    captureButton = new QPushButton("📸 Capture");
    detectButton  = new QPushButton("🔍 Detect");
    resetButton   = new QPushButton("🔄 Reset");
    showOverlayCheck = new QCheckBox("Show Overlay");
    showOverlayCheck->setChecked(true);

    QHBoxLayout *controls = new QHBoxLayout;
    controls->addWidget(startButton);
    controls->addWidget(captureButton);
    controls->addWidget(detectButton);
    controls->addWidget(resetButton);
    controls->addWidget(showOverlayCheck);

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addLayout(controls);
    layout->addStretch();
    setLayout(layout);

    camera = new QCamera(this);
    videoSink = new QVideoSink(this);
    captureSession = new QMediaCaptureSession(this);
    captureSession->setCamera(camera);
    captureSession->setVideoSink(videoSink);

    connect(videoSink, &QVideoSink::videoFrameChanged,
            this, &Widget::onFrameChanged);

    connect(startButton, &QPushButton::clicked, this, &Widget::startCamera);
    connect(captureButton, &QPushButton::clicked, this, &Widget::captureImage);
    connect(detectButton, &QPushButton::clicked, this, &Widget::runDetection);
    connect(resetButton, &QPushButton::clicked, this, &Widget::resetView);

    updateUI();
}

/* ===================== UI ===================== */
void Widget::updateUI()
{
    startButton->setEnabled(currentState == LiveView);
    captureButton->setEnabled(currentState == LiveView);
    detectButton->setEnabled(currentState == CapturedView);
    resetButton->setEnabled(currentState != LiveView);
}

void Widget::startCamera()
{
    camera->start();
    currentState = LiveView;
    updateUI();
}

/* ===================== LIVE ===================== */
void Widget::onFrameChanged(const QVideoFrame &frame)
{
    if (!frame.isValid()) return;
    QVideoFrame f(frame);
    f.map(QVideoFrame::ReadOnly);
    liveFrame = f.toImage();
    f.unmap();
    update();
}

/* ===================== CAPTURE ===================== */
void Widget::captureImage()
{
    if (liveFrame.isNull()) return;
    capturedFrame = liveFrame.copy();
    currentState = CapturedView;
    updateUI();
    update();
}

/* ===================== RESET ===================== */
void Widget::resetView()
{
    objectPaths.clear();
    capturedFrame = QImage();
    warpedA4 = QImage();
    currentState = LiveView;
    updateUI();
    update();
}

/* ===================== DETECT ===================== */
void Widget::runDetection()
{
    objectPaths.clear();

    if (!detectAndWarpA4(capturedFrame)) {
        qDebug() << "❌ A4 not detected";
        return;
    }

    detectObjectsInsideA4(warpedA4);

    currentState = DetectedView;
    updateUI();
    update();
}

/* ===================== A4 DETECTION ===================== */
bool Widget::detectAndWarpA4(const QImage &image)
{
    cv::Mat src(image.height(), image.width(),
                CV_8UC4, (void*)image.bits(), image.bytesPerLine());

    cv::Mat bgr;
    cv::cvtColor(src, bgr, cv::COLOR_BGRA2BGR);

    cv::Mat hsv, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(
        hsv,
        cv::Scalar(0, 0, 150),
        cv::Scalar(180, 80, 255),
        mask
        );

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(11,11));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours,
                     cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Point> best;
    double maxArea = 0;

    for (auto &c : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(c, approx,
                         0.02 * cv::arcLength(c,true), true);
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            double area = cv::contourArea(approx);
            if (area > maxArea) {
                maxArea = area;
                best = approx;
            }
        }
    }

    if (best.size() != 4) return false;

    std::sort(best.begin(), best.end(),
              [](auto &a, auto &b){ return a.y < b.y; });

    cv::Point2f tl,tr,bl,br;
    if (best[0].x < best[1].x) { tl=best[0]; tr=best[1]; }
    else { tl=best[1]; tr=best[0]; }
    if (best[2].x < best[3].x) { bl=best[2]; br=best[3]; }
    else { bl=best[3]; br=best[2]; }

    cv::Mat H = cv::getPerspectiveTransform(
        std::vector<cv::Point2f>{tl,tr,br,bl},
        std::vector<cv::Point2f>{
            {0,0},{A4_W,0},{A4_W,A4_H},{0,A4_H}
        });

    cv::Mat warped;
    cv::warpPerspective(bgr, warped, H, cv::Size(A4_W, A4_H));

    warpedA4 = QImage(
                   warped.data, warped.cols, warped.rows,
                   warped.step, QImage::Format_BGR888).copy();

    pixelsPerMM = A4_W / 210.0;
    qDebug() << "✅ A4 locked | pixelsPerMM =" << pixelsPerMM;
    return true;
}

/* ===================== OBJECT DETECTION ===================== */
void Widget::detectObjectsInsideA4(const QImage &img)
{
    cv::Mat src(img.height(), img.width(),
                CV_8UC3, (void*)img.bits(), img.bytesPerLine());

    cv::Mat gray, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::adaptiveThreshold(
        gray, bin, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        31, 7
        );

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(5,5));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(
        bin, contours, hierarchy,
        cv::RETR_TREE,
        cv::CHAIN_APPROX_NONE
        );

    double stepPx = pixelsPerMM * 1.0; // 1mm resolution
    int index = 1;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Skip if this contour has a parent (it's a hole/inner contour)
        // hierarchy[i][3] contains parent index (-1 means no parent)
        if (hierarchy[i][3] != -1) continue;

        double area = cv::contourArea(contours[i]);
        if (area < 2000) continue;
        if (area > img.width()*img.height()*0.9) continue;

        QPainterPath path =
            buildPathFromContour(contours[i], stepPx);

        objectPaths.push_back(path);
        dumpPainterPath(path, index++);
    }
}

/* ===================== PAINT ===================== */
void Widget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    QImage img = (currentState==LiveView)?liveFrame:
                     (currentState==CapturedView)?capturedFrame:
                     warpedA4;

    if (img.isNull()) return;


    p.scale(width()/double(img.width()),
            height()/double(img.height()));
    p.drawImage(0,0,img);

    if (showOverlayCheck->isChecked()) {
        p.setPen(QPen(Qt::red,3));
        for (auto &path:objectPaths)
            p.drawPath(path);
    }
}
