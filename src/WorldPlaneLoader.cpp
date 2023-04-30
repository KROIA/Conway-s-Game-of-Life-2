#include "WorldPlaneLoader.h"


void WorldPlaneLoader::loadFromImage(const std::string& path, Origin insertOrigin, bool coloredImport)
{
    sf::Image sfImage;
    sfImage.loadFromFile(path);
    sf::Vector2u imageSize = sfImage.getSize();

    unsigned int worldWidth = getWidth();
    unsigned int worldHeight = getHeight();

    size_t xSize = std::min(imageSize.x, worldWidth);
    size_t ySize = std::min(imageSize.y, worldHeight);

    unsigned int worldOffsetX = 0;
    unsigned int worldOffsetY = 0;

    switch (insertOrigin)
    {


    case Origin::bottomRight:
        worldOffsetY = worldHeight - ySize;
    case Origin::topRight:
        worldOffsetX = worldWidth - xSize;
        break;
    case Origin::bottomLeft:
        worldOffsetY = worldHeight - ySize;
        break;
    case Origin::center:
        worldOffsetX = (worldWidth - xSize) / 2;
    case Origin::leftCenter:
        worldOffsetY = (worldHeight - ySize) / 2;
        break;
    case Origin::topCenter:
        worldOffsetX = (worldWidth - xSize) / 2;
        break;
    case Origin::rightCenter:
        worldOffsetX = worldWidth - xSize;
        worldOffsetY = (worldHeight - ySize) / 2;
        break;
    case Origin::bottomCenter:
        worldOffsetX = (worldWidth - xSize) / 2;
        worldOffsetY = worldHeight - ySize;
        break;
    }

    clearCells();

    for (size_t x = 0; x < xSize; x++)
    {
        for (size_t y = 0; y < ySize; y++)
        {
            sf::Color col = sfImage.getPixel(x, y);
            int sum = col.r + col.g + col.b;
            if (sum > 10)
            {
                if(coloredImport)
                    setCell(worldOffsetX + x, worldOffsetY + y, true, col);
                else
                    setCell(worldOffsetX + x, worldOffsetY + y, true);
            }
        }
    }
}