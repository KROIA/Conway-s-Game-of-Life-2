#pragma once
#include "QSFML_EditorWidget.h"
#include "WorldPlaneLoader.h"
#include <cuda_runtime.h>

using namespace QSFML;
class WorldPlane_GPU: public Objects::CanvasObject, public WorldPlaneLoader
{
    class Painter;
public:
    WorldPlane_GPU(const sf::Vector2u& worldSize,
        const std::string& name = "WorldPlane_GPU",
        CanvasObject* parent = nullptr);
    ~WorldPlane_GPU();

    void clearCells() override;
    void setCell(unsigned int x, unsigned int y, bool alive) override;
    void setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color) override;
    unsigned int getWidth() const override;
    unsigned int getHeight() const override;

protected:
    void update() override;
    void cudaCheck(cudaError_t err);

    const sf::Vector2u m_worldSize;
    Painter* m_painter;

    unsigned char* m_d_world1;
    unsigned char* m_d_world2;
    unsigned char* m_d_currentWorld;
    sf::Uint8* m_d_pixels;

    
    class Painter : public Components::Drawable
    {
    public:
        Painter(const sf::Vector2u& worldSize, const std::string& name = "Painter");
        ~Painter();

        void setScale(const sf::Vector2f& scale);
        void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

        void setPixelColor(unsigned int x, unsigned int y, const sf::Color& color);
        sf::Color getPixelColor(unsigned int x, unsigned int y) const;
        void fill(const sf::Color& color);

        sf::Uint8* m_pixels;
    private:
        unsigned int m_width;
        unsigned int m_height;
        
        mutable sf::Texture m_texture;
        sf::Sprite m_sprite;
    };


};
namespace WorldPlane_GPU_kernel
{
    __global__ void updateMap(unsigned char* d_currentWorld, 
                              unsigned char* d_nextWorld,
                              sf::Uint8* d_pixels,
                              unsigned int worldSizeX,
                              unsigned int worldSizeY);
    __device__ int getAliveNeighbourCount(unsigned int x, unsigned int y,
                                          unsigned char *d_currentWorld,
                                          unsigned int worldSizeX, 
                                          unsigned int worldSizeY);
    __device__ int getAliveNeighbourCount_andMixColor(unsigned int x, unsigned int y,
                                                      unsigned char* d_currentWorld,
                                                      unsigned int worldSizeX,
                                                      unsigned int worldSizeY,
                                                      sf::Uint8* d_pixels);
    __device__ unsigned char getNewCellState(unsigned char oldState,
                                             int aliveNeighbourCount);

    __device__ void mixColor(sf::Uint8* colors, unsigned int colorCount, sf::Uint8* output);
}

