#include "WorldPlane_GPU.h"
#include <iostream>

WorldPlane_GPU::WorldPlane_GPU(const sf::Vector2u& worldSize,
                               const std::string& name,
                               CanvasObject* parent)
    : CanvasObject(name, parent)
    , m_worldSize(worldSize)
{
    size_t cellCount = m_worldSize.x * m_worldSize.y;
    cudaCheck(cudaMalloc(&m_d_world1, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&m_d_world2, cellCount * sizeof(unsigned char)));
    cudaCheck(cudaMalloc(&m_d_pixels, cellCount * 4 * sizeof(sf::Uint8)));
    m_d_currentWorld = m_d_world1;


    m_painter = new Painter(this, worldSize);

    addComponent(m_painter);

    
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
void WorldPlane_GPU::update()
{
    unsigned char* d_nextWorld = m_d_world1;
    if (d_nextWorld == m_d_currentWorld)
        d_nextWorld = m_d_world2;

    dim3 block_size(32, 32);
    dim3 num_blocks((m_worldSize.x + block_size.x - 1) / block_size.x, (m_worldSize.y + block_size.y - 1) / block_size.y);
    {
        QSFML_PROFILE_CANVASOBJECT(EASY_BLOCK("CUDA kernel: updateMap", profiler::colors::Orange200));
        kernelCallUpdateMap();
        QSFML_PROFILE_CANVASOBJECT(EASY_END_BLOCK);
    }
    {
        QSFML_PROFILE_CANVASOBJECT(EASY_BLOCK("CUDA memcpy async to host", profiler::colors::Orange200));
        cudaCheck(cudaMemcpyAsync((void*)m_painter->m_pixels, m_d_pixels,
            m_worldSize.x * m_worldSize.y * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost));
        QSFML_PROFILE_CANVASOBJECT(EASY_END_BLOCK);
    }
    m_d_currentWorld = d_nextWorld;
}
void WorldPlane_GPU::cudaCheck(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << cudaGetErrorString(err) << std::endl;
    }
}



WorldPlane_GPU::Painter::Painter(WorldPlane_GPU* parent, 
                                 const sf::Vector2u& worldSize, 
                                 const std::string& name)
    : Drawable(name)
    //, m_pixels(new sf::Uint8[4 * worldSize.x * worldSize.y])
    , m_width(worldSize.x)
    , m_height(worldSize.y)
    , m_pixels(nullptr)
    , m_parent(parent)

{
    WorldPlane_GPU::cudaCheck(cudaHostAlloc(&m_pixels, 4 * worldSize.x * worldSize.y * sizeof(sf::Uint8), cudaHostAllocDefault));
    m_texture.create(m_width, m_height);
    m_sprite.setTexture(m_texture);
    fill(sf::Color::Black);
}
WorldPlane_GPU::Painter::~Painter()
{
    cudaFreeHost(m_pixels);
    //delete[] m_pixels;
}
void WorldPlane_GPU::Painter::setScale(const sf::Vector2f& scale)
{
    m_sprite.setScale(scale);
}
void WorldPlane_GPU::Painter::draw(sf::RenderTarget& target, sf::RenderStates states) const
{
    //cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    m_texture.update(m_pixels);
    target.draw(m_sprite);
}
void WorldPlane_GPU::Painter::setPixelColor(unsigned int x, unsigned int y, const sf::Color& color)
{
    // Calculate the index of the pixel in the pixel array
    unsigned int index = (y * m_width + x) * 4;

    // Set the color of the pixel
    sf::Uint8* pixel = m_parent->m_d_pixels + index;
    /**pixel = color.r;
    *(++pixel) = color.g;
    *(++pixel) = color.b;
    *(++pixel) = color.a;*/

    cudaCheck(cudaMemcpyAsync(pixel, &color.r, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(pixel+1, &color.g, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(pixel+2, &color.b, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpyAsync(pixel+3, &color.a, sizeof(sf::Uint8), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
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
