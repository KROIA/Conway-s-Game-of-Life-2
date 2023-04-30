#include "GameOfLife.h"
#include <QFileDialog>

GameOfLife::GameOfLife(QWidget *parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    m_planeSize = sf::Vector2u(1000, 1000);
    setupCanvas();
    setupGameObjects();


    
}

GameOfLife::~GameOfLife()
{}


void GameOfLife::setupCanvas()
{
    QSFML::CanvasSettings settings;
    settings.timing.frameTime = 16;
    m_canvas = new QSFML::Canvas(ui.view_widget, settings);
    m_canvas->addObject(new QSFML::Objects::DefaultEditor("GameOfLife",sf::Vector2f(m_planeSize.x, m_planeSize.y)));
}
void GameOfLife::setupGameObjects()
{
    
    std::string image = "E:\\Dokumente\\Visual Studio 2022\\Projects\\GameOfLife\\images\\Glider_gun.png";
    WorldPlaneLoader::Origin insertPos = WorldPlaneLoader::Origin::center;
    //m_gameObjects.m_worldPlaneCPU = new WorldPlane_CPU(m_planeSize);
    //m_canvas->addObject(m_gameObjects.m_worldPlaneCPU);

    //m_gameObjects.m_worldPlaneCPU_colored = new WorldPlane_CPU_colored(m_planeSize);
    //m_canvas->addObject(m_gameObjects.m_worldPlaneCPU_colored);

    m_gameObjects.m_worldPlaneGPU = new WorldPlane_GPU(m_planeSize);
    m_canvas->addObject(m_gameObjects.m_worldPlaneGPU);
    
    //loadImage(image, insertPos);

}
void GameOfLife::loadImage(const std::string& imagePath, WorldPlaneLoader::Origin insertPos)
{
    if(m_gameObjects.m_worldPlaneCPU)
        m_gameObjects.m_worldPlaneCPU->loadFromImage(imagePath, insertPos);
    if(m_gameObjects.m_worldPlaneCPU_colored)
        m_gameObjects.m_worldPlaneCPU_colored->loadFromImage(imagePath, insertPos, true);
    if(m_gameObjects.m_worldPlaneGPU)
         m_gameObjects.m_worldPlaneGPU->loadFromImage(imagePath, insertPos, true);
}

void GameOfLife::on_loadImage_pushButton_clicked()
{
    QString path = QFileDialog::getOpenFileName(this,
        tr("Open Image"), QDir::currentPath(), tr("Image Files (*.png *.jpg)"));

    if (path.size() > 0)
    {
        loadImage(path.toStdString());
    }
}
void GameOfLife::on_speed_slider_valueChanged(int value)
{
    QSFML::CanvasSettings settings = m_canvas->getSettings();
    settings.timing.frameTime = value;
    m_canvas->setSettings(settings);
}