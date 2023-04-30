#include "WorldPlane_CPU.h"

WorldPlane_CPU::WorldPlane_CPU(const sf::Vector2u& worldSize,
                               const std::string& name,
                               CanvasObject* parent)
    : CanvasObject(name, parent)
    , m_worldSize(worldSize)
{

    m_painter = new Painter(worldSize);
    addComponent(m_painter);

    m_world1 = new unsigned char* [m_worldSize.x];
    m_world2 = new unsigned char* [m_worldSize.x];
    for (size_t i = 0; i < m_worldSize.x; ++i)
    {
        m_world1[i] = new unsigned char[m_worldSize.y];
        m_world2[i] = new unsigned char[m_worldSize.y];
        memset(m_world1[i], 0, m_worldSize.y);
        memset(m_world2[i], 0, m_worldSize.y);
    }
    m_currentWorld = m_world1;

    m_aliveColor = sf::Color::White;
    m_deadColor = sf::Color::Black;
}
WorldPlane_CPU::~WorldPlane_CPU()
{
    for (size_t i = 0; i < m_worldSize.x; ++i)
    {
        delete[] m_world1[i];
        delete[] m_world2[i];
    }
    delete[] m_world1;
    delete[] m_world2;   
}


void WorldPlane_CPU::clearCells()
{
    for (size_t x = 0; x < m_worldSize.x; x++)
    {
        for (size_t y = 0; y < m_worldSize.y; y++)
        {
            m_world1[x][y] = 0;
            m_world2[x][y] = 0;
        }
    }
}
void WorldPlane_CPU::setCell(unsigned int x, unsigned int y, bool alive)
{
    m_world1[x][y] = alive;
    m_world2[x][y] = alive;
}
void WorldPlane_CPU::setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color)
{
    m_world1[x][y] = alive;
    m_world2[x][y] = alive;
    // Color not implemented
}
unsigned int WorldPlane_CPU::getWidth() const
{
    return m_worldSize.x;
}
unsigned int WorldPlane_CPU::getHeight() const
{
    return m_worldSize.y;
}

void WorldPlane_CPU::update()
{
    unsigned char** nextWorld = m_world1;
    if (nextWorld == m_currentWorld)
        nextWorld = m_world2;
    for (unsigned int x = 0; x < m_worldSize.x; ++x)
    {
        for (unsigned int y = 0; y < m_worldSize.y; ++y)
        {
            unsigned char cell = getNewCellState(m_currentWorld[x][y], getAliveNeighbourCount(x, y));
            nextWorld[x][y] = cell;
            if (cell)
                m_painter->setPixelColor(x, y, m_aliveColor);
            else
                m_painter->setPixelColor(x, y, m_deadColor);

        }
    }
    m_currentWorld = nextWorld;
}
int WorldPlane_CPU::getAliveNeighbourCount(unsigned int x, unsigned int y)
{
    int aliveCount = 0;
    unsigned int width = m_worldSize.x;
    unsigned int height = m_worldSize.y;
    
    unsigned int x_sub1 = (width + x - 1) % width;
    unsigned int x_add1 = (x + 1) % width;

    unsigned int y_sub1 = (height + y - 1) % height;
    unsigned int y_add1 = (y + 1) % height;


    aliveCount += m_currentWorld[x_sub1][y_sub1];
    aliveCount += m_currentWorld[x][y_sub1];
    aliveCount += m_currentWorld[x_add1][y_sub1];
    aliveCount += m_currentWorld[x_add1][y];
    aliveCount += m_currentWorld[x_add1][y_add1];
    aliveCount += m_currentWorld[x][y_add1];
    aliveCount += m_currentWorld[x_sub1][y_add1];
    aliveCount += m_currentWorld[x_sub1][y];
    return aliveCount;
}
unsigned char WorldPlane_CPU::getNewCellState(unsigned char oldState, int aliveNeighbourCount)
{
    switch (aliveNeighbourCount)
    {
    case 3:
        oldState = 1;
    case 2:
        break;
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:

    case 0:
    case 1:
    deafault:
        oldState = 0;

    }
    return oldState;
}



WorldPlane_CPU::Painter::Painter(const sf::Vector2u& worldSize, const std::string& name)
    : Drawable(name)
    , m_width(worldSize.x)
    , m_height(worldSize.y)
    , m_pixels(new sf::Uint8[4 * worldSize.x * worldSize.y])
{
    m_texture.create(m_width, m_height);
    m_sprite.setTexture(m_texture);
    fill(sf::Color::Black);
}
WorldPlane_CPU::Painter::~Painter()
{
    delete m_pixels;
}
void WorldPlane_CPU::Painter::setScale(const sf::Vector2f& scale)
{
    m_sprite.setScale(scale);
}
void WorldPlane_CPU::Painter::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    m_texture.update(m_pixels);
    target.draw(m_sprite);
}

void WorldPlane_CPU::Painter::fill(const sf::Color& color)
{
    sf::Uint8* pixel = m_pixels-1;
    for (unsigned int i = 0; i < m_width * m_height; ++i)
    {
        *(++pixel) = color.r;
        *(++pixel) = color.g;
        *(++pixel) = color.b;
        *(++pixel) = color.a;
    }
}
void WorldPlane_CPU::Painter::setPixelColor(unsigned int x, unsigned int y, const sf::Color& color)
{
    // Calculate the index of the pixel in the pixel array
    unsigned int index = (y * m_width + x) * 4;

    // Set the color of the pixel
    sf::Uint8* pixel = m_pixels + index;
    *pixel = color.r;
    *(++pixel) = color.g;
    *(++pixel) = color.b;
    *(++pixel) = color.a;
}
sf::Color WorldPlane_CPU::Painter::getPixelColor(unsigned int x, unsigned int y) const
{
    // Calculate the index of the pixel in the pixel array
    unsigned int index = (y * m_width + x) * 4;

    sf::Color color;
    sf::Uint8* pixel = m_pixels + index;
    color.r = *(pixel++);
    color.g = *(pixel++);
    color.b = *(pixel++);
    color.a = *(pixel++);
    return color;
}