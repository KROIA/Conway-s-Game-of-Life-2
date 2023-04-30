#pragma once
#include "QSFML_EditorWidget.h"
#include "WorldPlaneLoader.h"

using namespace QSFML;
class WorldPlane_CPU: public Objects::CanvasObject, public WorldPlaneLoader
{
    class Painter;
public:
    
    WorldPlane_CPU(const sf::Vector2u &worldSize, 
                   const std::string& name = "WorldPlane_CPU",
                   CanvasObject* parent = nullptr);
    ~WorldPlane_CPU();

    void clearCells() override;
    void setCell(unsigned int x, unsigned int y, bool alive) override;
    void setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color) override;
    unsigned int getWidth() const override;
    unsigned int getHeight() const override;
    
protected:
    void update() override;

    int getAliveNeighbourCount(unsigned int x, unsigned int y);
    unsigned char getNewCellState(unsigned char oldState, int aliveNeighbourCount);

    const sf::Vector2u m_worldSize;
    Painter* m_painter;

    unsigned char** m_world1;
    unsigned char** m_world2;
    unsigned char** m_currentWorld;

    sf::Color m_aliveColor;
    sf::Color m_deadColor;

    class Painter: public Components::Drawable
    {
    public:
        Painter(const sf::Vector2u& worldSize, const std::string& name = "Painter");
        ~Painter();

        void setScale(const sf::Vector2f& scale);
        void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

        void setPixelColor(unsigned int x, unsigned int y, const sf::Color& color);
        sf::Color getPixelColor(unsigned int x, unsigned int y) const;
        void fill(const sf::Color& color);
    private:
        unsigned int m_width;
        unsigned int m_height;
        sf::Uint8* m_pixels;
        mutable sf::Texture m_texture;
        sf::Sprite m_sprite;
    };
};