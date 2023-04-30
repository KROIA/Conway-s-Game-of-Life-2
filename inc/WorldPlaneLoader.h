#pragma once
#include "QSFML_EditorWidget.h"

class WorldPlaneLoader
{
public:
    enum Origin
    {
        center,
        topLeft,
        topCenter,
        topRight,
        rightCenter,
        bottomRight,
        bottomCenter,
        bottomLeft,
        leftCenter
    };

	void loadFromImage(const std::string& path, Origin insertOrigin = Origin::center, bool coloredImport = false);

    virtual void clearCells() = 0;
    virtual void setCell(unsigned int x, unsigned int y, bool alive) = 0;
    virtual void setCell(unsigned int x, unsigned int y, bool alive, const sf::Color &color) = 0;
    virtual unsigned int getWidth() const = 0;
    virtual unsigned int getHeight() const = 0;

};