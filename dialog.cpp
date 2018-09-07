#include "dialog.h"
#include "ui_dialog.h"
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <detector.h>

Dialog::Dialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::Dialog)
{
    ui->setupUi(this);
}

Dialog::~Dialog()
{
    delete ui;
}

void Dialog::on_browsePushButton_pressed()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Video", QDir::currentPath(), "Video (*.mp4 *.avi)");
    if(QFile::exists(fileName))
    {
        ui->VideoURLLineEdit->setText(fileName);
    }
}

void Dialog::on_playPushButton_pressed()
{

    // Get path to Video
    QString videoPath = ui->VideoURLLineEdit->text();

    // Throw a message box when video path is blank
    if(videoPath == "")
    {
        QMessageBox::about(this, "Warning", "Please choose a video");
    }
    else
    {
        detector(videoPath);

    }

}

void Dialog::closeEvent(QCloseEvent *event)
{
    int result = QMessageBox::warning(this, "Exit", "Are you sure you want to close this program?", QMessageBox::Yes, QMessageBox::No);

    if(result == QMessageBox::Yes)
    {
        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void Dialog::detector(QString videoPath)
{
    detect(videoPath);
}
