#pragma once

#include "WorldPlane_CPU.h"

class WorldPlane_CPU_colored : public WorldPlane_CPU
{
public:
    WorldPlane_CPU_colored(const sf::Vector2u& worldSize,
                           const std::string& name = "WorldPlane_CPU_colored",
                           CanvasObject* parent = nullptr);
    ~WorldPlane_CPU_colored();

    void clearCells() override;
    void setCell(unsigned int x, unsigned int y, bool alive, const sf::Color& color) override;
private:
    std::vector<sf::Color*> getAliveNeigbourColor(unsigned int x, unsigned int y);
    inline void getColorIfAlive(std::vector<sf::Color*>& list, unsigned int x, unsigned int y);
    sf::Color mixColor(const std::vector<sf::Color*>& colors) const;

    void update() override;


    sf::Color** m_colorWorld1;
    sf::Color** m_colorWorld2;
    sf::Color** m_currentColorWorld;

};