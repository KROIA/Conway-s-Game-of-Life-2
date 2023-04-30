#include "WorldPlane_CPU_colored.h"

WorldPlane_CPU_colored::WorldPlane_CPU_colored(const sf::Vector2u& worldSize,
                                               const std::string& name,
                                               CanvasObject* parent)
    : WorldPlane_CPU(worldSize, name, parent)
{
    m_colorWorld1 = new sf::Color* [m_worldSize.x];
    m_colorWorld2 = new sf::Color* [m_worldSize.x];
    for (size_t i = 0; i < m_worldSize.x; ++i)
    {
        m_colorWorld1[i] = new sf::Color[m_worldSize.y];
        m_colorWorld2[i] = new sf::Color[m_worldSize.y];
    }
    m_currentColorWorld = m_colorWorld1;
    clearCells();
}
WorldPlane_CPU_colored::~WorldPlane_CPU_colored()
{
    for (size_t i = 0; i < m_worldSize.x; ++i)
    {
        delete[] m_colorWorld1[i];
        delete[] m_colorWorld2[i];
    }
    delete[] m_colorWorld1;
    delete[] m_colorWorld2;
}

void WorldPlane_CPU_colored::clearCells()
{
    for (size_t x = 0; x < m_worldSize.x; x++)
    {
        for (size_t y = 0; y < m_worldSize.y; y++)
        {
            m_world1[x][y] = 0;
            m_world2[x][y] = 0;
            m_colorWorld1[x][y] = sf::Color::Black;
            m_colorWorld2[x][y] = sf::Color::Black;
        }
    }
}
void WorldPlane_CPU_colored::setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color)
{
    m_world1[x][y] = alive;
    m_world2[x][y] = alive;
    m_colorWorld1[x][y] = color;
    m_colorWorld2[x][y] = color;
}
std::vector<sf::Color*> WorldPlane_CPU_colored::getAliveNeigbourColor(unsigned int x, unsigned int y)
{
    std::vector<sf::Color*> colors;
    colors.reserve(8);
    int aliveCount = 0;
    unsigned int width = m_worldSize.x;
    unsigned int height = m_worldSize.y;

    unsigned int x_sub1 = (width + x - 1) % width;
    unsigned int x_add1 = (x + 1) % width;

    unsigned int y_sub1 = (height + y - 1) % height;
    unsigned int y_add1 = (y + 1) % height;

    getColorIfAlive(colors, x_sub1, y_sub1);
    getColorIfAlive(colors, x, y_sub1);
    getColorIfAlive(colors, x_add1, y_sub1);
    getColorIfAlive(colors, x_add1, y);
    getColorIfAlive(colors, x_add1, y_add1);
    getColorIfAlive(colors, x, y_add1);
    getColorIfAlive(colors, x_sub1, y_add1);
    getColorIfAlive(colors, x_sub1, y);

    return colors;
}
void WorldPlane_CPU_colored::getColorIfAlive(std::vector<sf::Color*>& list, unsigned int x, unsigned int y)
{
    if (m_currentWorld[x][y])
        list.push_back(&m_currentColorWorld[x][y]);
}
sf::Color WorldPlane_CPU_colored::mixColor(const std::vector<sf::Color*>& colors) const
{
    size_t size = colors.size();
    if (size == 0)
        return sf::Color::Black;
    unsigned int r = 0;
    unsigned int g = 0;
    unsigned int b = 0;
    //unsigned int a = 0;
    for (size_t i = 0; i < colors.size(); ++i)
    {
        sf::Color& col = *colors[i];
        r += col.r;
        g += col.g;
        b += col.b;
       // a += col.a;
    }
    
    r /= size;
    g /= size;
    b /= size;
    //a /= size;
    int toAdd = 100 - (r + g + b) / 3;
    if (toAdd > 0)
    {
        r += toAdd;
        g += toAdd;
        b += toAdd;
    }
    return sf::Color(r, g, b);
}

void WorldPlane_CPU_colored::update()
{
    unsigned char** nextWorld = m_world1;
    sf::Color** nextColoredWorld = m_colorWorld1;
    if (nextWorld == m_currentWorld)
        nextWorld = m_world2;
    if (nextColoredWorld == m_currentColorWorld)
        nextColoredWorld = m_colorWorld2;
    for (unsigned int x = 0; x < m_worldSize.x; ++x)
    {
        for (unsigned int y = 0; y < m_worldSize.y; ++y)
        {
            std::vector<sf::Color*> colors = getAliveNeigbourColor(x, y);
            unsigned char cell = getNewCellState(m_currentWorld[x][y], colors.size());
            nextWorld[x][y] = cell;
            nextColoredWorld[x][y] = mixColor(colors);
            if (cell)
                m_painter->setPixelColor(x, y, nextColoredWorld[x][y]);
            else
                m_painter->setPixelColor(x, y, m_deadColor);

        }
    }
    m_currentWorld = nextWorld;
    m_currentColorWorld = nextColoredWorld;
}