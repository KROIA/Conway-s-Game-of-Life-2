#include "WorldPlane_GPU.h"
#include <iostream>

WorldPlane_GPU::WorldPlane_GPU(const sf::Vector2u& worldSize,
                               const std::string& name,
                               CanvasObject* parent)
    : CanvasObject(name, parent)
    , m_worldSize(worldSize)
{
    m_painter = new Painter(worldSize);
    addComponent(m_painter);

    size_t cellCount = m_worldSize.x * m_worldSize.y;
    cudaCheck(cudaMalloc(&m_d_world1, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&m_d_world2, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&m_d_pixels, cellCount * 4 * sizeof(sf::Uint8)));
    m_d_currentWorld = m_d_world1;
}
WorldPlane_GPU::~WorldPlane_GPU()
{
    cudaCheck(cudaFree(m_d_world1));
    cudaCheck(cudaFree(m_d_world2));
    cudaCheck(cudaFree(m_d_pixels));
}

void WorldPlane_GPU::clearCells()
{
    size_t cellCount = m_worldSize.x * m_worldSize.y;
    cudaCheck(cudaMemset(m_d_world1, 0, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMemset(m_d_world2, 0, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMemset(m_d_pixels, 0, cellCount * 4 * sizeof(sf::Uint8)));
}
void WorldPlane_GPU::setCell(unsigned int x, unsigned int y, bool alive)
{
    unsigned char value = 0;
    if (alive)
        value = 1;
    cudaCheck(cudaMemcpyAsync(m_d_world1 + x+y*m_worldSize.x, &value, sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_world2 + x+y*m_worldSize.x, &value, sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
}
void WorldPlane_GPU::setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color)
{
    unsigned char value = 0;
    if (alive)
        value = 1;
    cudaCheck(cudaMemcpyAsync(m_d_world1 + x + y * m_worldSize.x, &value, sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_world2 + x + y * m_worldSize.x, &value, sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_pixels + (y * m_worldSize.x + x) * 4    , &color.r, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_pixels + (y * m_worldSize.x + x) * 4 + 1, &color.g, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_pixels + (y * m_worldSize.x + x) * 4 + 2, &color.b, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(m_d_pixels + (y * m_worldSize.x + x) * 4 + 3, &color.a, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaDeviceSynchronize());
}
unsigned int WorldPlane_GPU::getWidth() const
{
    return m_worldSize.x;
}
unsigned int WorldPlane_GPU::getHeight() const
{
    return m_worldSize.y;
}

void WorldPlane_GPU::cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << cudaGetErrorString(err) << std::endl;
    }
}



WorldPlane_GPU::Painter::Painter(const sf::Vector2u& worldSize, const std::string& name)
    : Drawable(name)
    , m_width(worldSize.x)
    , m_height(worldSize.y)
    , m_pixels(new sf::Uint8[4 * worldSize.x * worldSize.y])
{
    m_texture.create(m_width, m_height);
    m_sprite.setTexture(m_texture);
    fill(sf::Color::Black);
}
WorldPlane_GPU::Painter::~Painter()
{
    delete m_pixels;
}
void WorldPlane_GPU::Painter::setScale(const sf::Vector2f& scale)
{
    m_sprite.setScale(scale);
}
void WorldPlane_GPU::Painter::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    m_texture.update(m_pixels);
    target.draw(m_sprite);
}

void WorldPlane_GPU::Painter::fill(const sf::Color& color)
{
    sf::Uint8* pixel = m_pixels - 1;
    for (unsigned int i = 0; i < m_width * m_height; ++i)
    {
        *(++pixel) = color.r;
        *(++pixel) = color.g;
        *(++pixel) = color.b;
        *(++pixel) = color.a;
    }
}
void WorldPlane_GPU::Painter::setPixelColor(unsigned int x, unsigned int y, const sf::Color& color)
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
sf::Color WorldPlane_GPU::Painter::getPixelColor(unsigned int x, unsigned int y) const
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
