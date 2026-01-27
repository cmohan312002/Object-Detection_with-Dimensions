
#include "widget.h"

#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>



/* ===================== CONSTANTS ===================== */
static const int A4_W = 2480;
static const int A4_H = 3508;
static const int AVERAGE_FRAMES = 5;


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
    pathOnlyButton = new QPushButton("🧩 Path Only");
    pathOnlyButton->setEnabled(currentState == DetectedView);





    QHBoxLayout *controls = new QHBoxLayout;
    controls->addWidget(startButton);
    controls->addWidget(captureButton);
    controls->addWidget(detectButton);
    controls->addWidget(pathOnlyButton);
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
    connect(pathOnlyButton, &QPushButton::clicked,
            this, &Widget::showPathOnlyView);


    updateUI();
}

/* ===================== UI ===================== */
void Widget::updateUI()
{
    startButton->setEnabled(currentState == LiveView);
    captureButton->setEnabled(currentState == LiveView);
    detectButton->setEnabled(currentState == CapturedView);
    resetButton->setEnabled(currentState != LiveView);
    pathOnlyButton->setEnabled(
        currentState == DetectedView || currentState == PathOnlyView
        );
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
    measurementHistory.clear();
    frameMeasures.clear();
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
    frameMeasures.clear();

    if (!detectAndWarpA4(capturedFrame)) {
        qDebug() << " A4 not detected";
        return;
    }

    detectObjectsInsideA4(warpedA4);

    measurementHistory.push_back(frameMeasures);
    if (measurementHistory.size() > AVERAGE_FRAMES)
        measurementHistory.erase(measurementHistory.begin());

    computeAveragedResults();

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
        cv::Scalar(0, 0, 140),
        cv::Scalar(180, 80, 255),
        mask
        );

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(9,9));
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
              [](const cv::Point&a,const cv::Point&b){return a.y<b.y;});

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

/* ===================== OBJECTS ===================== */
void Widget::detectObjectsInsideA4(const QImage &img)
{
    objectPaths.clear();
    frameMeasures.clear();

    cv::Mat src(img.height(), img.width(),
                CV_8UC3, (void*)img.bits(), img.bytesPerLine());

    /* ---------- 1. BINARIZE ---------- */
    cv::Mat gray, bin;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);

    cv::adaptiveThreshold(
        gray, bin, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        31, 7
        );

    /* ---------- 2. CLOSE SMALL GAPS ---------- */
    cv::morphologyEx(
        bin, bin, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, {3,3})
        );

    /* ---------- 3. FIND CONTOURS + HIERARCHY ---------- */
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(
        bin,
        contours,
        hierarchy,
        cv::RETR_TREE,
        cv::CHAIN_APPROX_NONE
        );

    int index = 1;

    /* ---------- 4. PROCESS ONLY REAL OBJECTS ---------- */
    for (size_t i = 0; i < contours.size(); ++i)
    {
        // --- hierarchy depth ---
        int depth = 0;
        int parent = hierarchy[i][3];
        while (parent != -1) {
            depth++;
            parent = hierarchy[parent][3];
        }

        // 🔥 Even depth = real object (including inner objects)
        if (depth % 2 != 0)
            continue;

        double areaPx = cv::contourArea(contours[i]);
        if (areaPx < 1500)
            continue;

        if (areaPx > img.width() * img.height() * 0.9)
            continue;

        /* ---------- 5. MEASURE OBJECT ---------- */
        cv::RotatedRect rr = cv::minAreaRect(contours[i]);

        ObjectMeasure m;
        m.widthMM  = rr.size.width  / pixelsPerMM;
        m.heightMM = rr.size.height / pixelsPerMM;
        m.areaMM2  = areaPx / (pixelsPerMM * pixelsPerMM);

        // normalize W >= H
        if (m.widthMM < m.heightMM)
            std::swap(m.widthMM, m.heightMM);

        frameMeasures.push_back(m);

        /* ---------- 6. CLEAN PATH FOR DRAWING ---------- */
        std::vector<cv::Point> approx;
        cv::approxPolyDP(
            contours[i],
            approx,
            pixelsPerMM * 0.6,   // smooth but accurate
            true
            );

        if (approx.size() < 3)
            continue;

        QPainterPath path;
        path.moveTo(approx[0].x, approx[0].y);
        for (size_t k = 1; k < approx.size(); ++k)
            path.lineTo(approx[k].x, approx[k].y);
        path.closeSubpath();

        objectPaths.push_back(path);
        dumpPainterPath(path, index++);

        qDebug()
            << "OBJECT"
            << "W(mm)=" << m.widthMM
            << "H(mm)=" << m.heightMM
            << "Area(mm²)=" << m.areaMM2;
    }
}
void Widget::dumpPainterPath(const QPainterPath &path, int objectIndex)
{
    qDebug() << "================ OBJECT" << objectIndex << "================";

    for (int i = 0; i < path.elementCount(); ++i)
    {
        QPainterPath::Element e = path.elementAt(i);

        if (e.type == QPainterPath::MoveToElement)
        {
            qDebug().nospace()
            << "moveTo("
            << e.x << ", "
            << e.y << ")";
        }
        else if (e.type == QPainterPath::LineToElement)
        {
            qDebug().nospace()
            << "lineTo("
            << e.x << ", "
            << e.y << ")";
        }
        else if (e.type == QPainterPath::CurveToElement)
        {
            QPainterPath::Element c1 = path.elementAt(i);
            QPainterPath::Element c2 = path.elementAt(i + 1);
            QPainterPath::Element end = path.elementAt(i + 2);

            qDebug().nospace()
                << "cubicTo("
                << c1.x << ", " << c1.y << ", "
                << c2.x << ", " << c2.y << ", "
                << end.x << ", " << end.y << ")";

            i += 2;
        }
    }

    qDebug() << "============================================";
}


/* ===================== AVERAGE ===================== */
void Widget::computeAveragedResults()
{
    if (measurementHistory.empty()) return;

    for (size_t i=0;i<measurementHistory[0].size();++i) {
        double w=0,h=0; int c=0;
        for (auto &f:measurementHistory) {
            if (i>=f.size()) continue;
            w+=f[i].widthMM;
            h+=f[i].heightMM;
            c++;
        }
        qDebug() << "AVG OBJECT"
                 << "W(mm)=" << (w/c)
                 << "H(mm)=" << (h/c);
    }
}

/* ===================== PAINT ===================== */
void Widget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // ===================== PATH ONLY VIEW =====================
    if (currentState == PathOnlyView)
    {
        // Use A4 logical size
        QSize logicalSize(A4_W, A4_H);

        double sx = width()  / double(logicalSize.width());
        double sy = height() / double(logicalSize.height());
        double scale = std::min(sx, sy);

        double ox = (width()  - logicalSize.width()  * scale) * 0.5;
        double oy = (height() - logicalSize.height() * scale) * 0.5;

        p.translate(ox, oy);
        p.scale(scale, scale);

        // Optional white background
        p.fillRect(0, 0, A4_W, A4_H, Qt::white);

        // Draw ONLY paths
        p.setPen(QPen(Qt::black, 2));
        p.setBrush(Qt::NoBrush);

        for (const auto &path : objectPaths)
            p.drawPath(path);

        return; // 🔥 IMPORTANT
    }

    // ===================== NORMAL VIEWS =====================
    QImage img;

    if (currentState == LiveView)
        img = liveFrame;
    else if (currentState == CapturedView)
        img = capturedFrame;
    else
        img = warpedA4;

    if (img.isNull()) return;

    double sx = width()  / double(img.width());
    double sy = height() / double(img.height());
    double scale = std::min(sx, sy);

    double ox = (width()  - img.width()  * scale) * 0.5;
    double oy = (height() - img.height() * scale) * 0.5;

    p.translate(ox, oy);
    p.scale(scale, scale);

    p.drawImage(0, 0, img);

    if (showOverlayCheck->isChecked()) {
        p.setPen(QPen(Qt::red, 2));
        for (const auto &path : objectPaths)
            p.drawPath(path);
    }
}

void Widget::showPathOnlyView()
{
    if (objectPaths.empty()) {
        qDebug() << "No paths to display";
        return;
    }

    currentState = PathOnlyView;
    updateUI();
    update();
}


