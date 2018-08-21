#include "dialog.h"
#include "ui_dialog.h"
#include <QFileDialog>
#include <QInputDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <detectors.h>

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

    //std::string videoPath = ui->VideoURLLineEdit->text().toStdString();
    QString videoPath = ui->VideoURLLineEdit->text();

    // Throw a message box when video path is blank
    if(videoPath == "")
    {
        QMessageBox::about(this, "Warning", "Please choose a video");
    }
    else
    {
        detect(videoPath);

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

void Dialog::detect(QString videoPath)
{
    detect(videoPath);
}
