#ifndef DIALOG_H
#define DIALOG_H

#include <QDialog>
#include <qstring.h>

namespace Ui {
class Dialog;
}

class Dialog : public QDialog
{
    Q_OBJECT

public:
    explicit Dialog(QWidget *parent = 0);
    ~Dialog();

    void detector(QString videoPath);

private slots:
    void on_browsePushButton_pressed();

    void on_playPushButton_pressed();

private:
    Ui::Dialog *ui;

protected:
void closeEvent(QCloseEvent *event);

};

#endif // DIALOG_H
