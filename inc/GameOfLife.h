#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_GameOfLife.h"
#include "QSFML_EditorWidget.h"
#include "WorldPlane_CPU.h"
#include "WorldPlane_CPU_colored.h"
#include "WorldPlane_GPU.h"

class GameOfLife : public QMainWindow
{
    Q_OBJECT

public:
    GameOfLife(QWidget *parent = nullptr);
    ~GameOfLife();

private slots:
    void on_loadImage_pushButton_clicked();
    void on_speed_slider_valueChanged(int value);
private:
    void setupCanvas();
    void setupGameObjects();
    void loadImage(const std::string& imagePath, WorldPlaneLoader::Origin insertPos = WorldPlaneLoader::Origin::center);
    Ui::GameOfLifeClass ui;
    QSFML::Canvas* m_canvas;
    sf::Vector2u m_planeSize;

    struct GameObjects
    {
        WorldPlane_CPU* m_worldPlaneCPU = nullptr;
        WorldPlane_CPU_colored* m_worldPlaneCPU_colored = nullptr;
        WorldPlane_GPU* m_worldPlaneGPU = nullptr;
    };

    GameObjects m_gameObjects;

};
