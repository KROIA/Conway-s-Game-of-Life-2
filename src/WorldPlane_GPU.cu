#include "WorldPlane_GPU.h"



void WorldPlane_GPU::update()
{
    unsigned char* d_nextWorld = m_d_world1;
    if (d_nextWorld == m_d_currentWorld)
        d_nextWorld = m_d_world2;

    dim3 block_size(32, 32);
    dim3 num_blocks((m_worldSize.x + block_size.x - 1) / block_size.x, (m_worldSize.y + block_size.y - 1) / block_size.y);
    WorldPlane_GPU_kernel::updateMap << <num_blocks, block_size >> > (m_d_currentWorld, d_nextWorld,
                                                    m_d_pixels, m_worldSize.x, m_worldSize.y);

    cudaCheck(cudaMemcpy(m_painter->m_pixels, m_d_pixels, 
                         m_worldSize.x * m_worldSize.y * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost));
    m_d_currentWorld = d_nextWorld;
}

namespace WorldPlane_GPU_kernel
{
    __global__ void updateMap(unsigned char* d_currentWorld, 
                              unsigned char* d_nextWorld,
                              sf::Uint8* d_pixels,
                              unsigned int worldSizeX,
                              unsigned int worldSizeY)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= worldSizeX || y >= worldSizeY)
            return;
        unsigned int cellIndex = x + y * worldSizeX;

        int aliveNeighbour = getAliveNeighbourCount_andMixColor(x, y, d_currentWorld, worldSizeX, worldSizeY, d_pixels);
        //int aliveNeighbour = getAliveNeighbourCount(x, y, d_currentWorld, worldSizeX, worldSizeY);
        unsigned char cell = getNewCellState(d_currentWorld[cellIndex], aliveNeighbour);
        d_nextWorld[cellIndex] = cell;


        sf::Uint8* pixel = d_pixels + (y * worldSizeX + x) * 4;
       /* if (cell)
        {
            *pixel     = 255;
            *(++pixel) = 255;
            *(++pixel) = 255;
            *(++pixel) = 255;
        }
        else
        {
            *pixel     = 0;
            *(++pixel) = 0;
            *(++pixel) = 0;
            *(++pixel) = 255;
        }*/
        if (!cell)
        {
            *pixel = 0;
            *(++pixel) = 0;
            *(++pixel) = 0;
            *(++pixel) = 255;
        }
    }
    __device__ int getAliveNeighbourCount(unsigned int x, unsigned int y,
                                          unsigned char* d_currentWorld,
                                          unsigned int worldSizeX, 
                                          unsigned int worldSizeY)
    {
        int aliveCount = 0;
        unsigned int width = worldSizeX;
        unsigned int height = worldSizeY;

        unsigned int x_sub1 = (width + x - 1) % width;
        unsigned int x_add1 = (x + 1) % width;

        unsigned int y_sub1 = ((height + y - 1) % height) * worldSizeX;
        unsigned int y_add1 = ((y + 1) % height) * worldSizeX;


        aliveCount += d_currentWorld[x_sub1 + y_sub1];
        aliveCount += d_currentWorld[x + y_sub1];
        aliveCount += d_currentWorld[x_add1 + y_sub1];
        aliveCount += d_currentWorld[x_add1 + y * worldSizeX];
        aliveCount += d_currentWorld[x_add1 + y_add1];
        aliveCount += d_currentWorld[x + y_add1];
        aliveCount += d_currentWorld[x_sub1 + y_add1];
        aliveCount += d_currentWorld[x_sub1 + y * worldSizeX];
        return aliveCount;
    }
    __device__ int getAliveNeighbourCount_andMixColor(unsigned int x, unsigned int y,
                                                      unsigned char* d_currentWorld,
                                                      unsigned int worldSizeX,
                                                      unsigned int worldSizeY,
                                                      sf::Uint8* d_pixels)
    {
        int aliveCount = 0;
        //unsigned int width = worldSizeX;
        //unsigned int height = worldSizeY;

        //d_pixels += x + y * worldSizeX;

        unsigned int x_sub1 = (worldSizeX + x - 1) % worldSizeX;
        unsigned int x_add1 = (x + 1) % worldSizeX;

        unsigned int y_sub1 = ((worldSizeY + y - 1) % worldSizeY);
        unsigned int y_add1 = ((y + 1) % worldSizeY);

        unsigned int r = 0;
        unsigned int g = 0;
        unsigned int b = 0;
        //unsigned int a = 0;

      /*  unsigned char c1 = d_currentWorld[x_sub1 + y_sub1];
        unsigned char c2 = d_currentWorld[x + y_sub1];
        unsigned char c3 = d_currentWorld[x_add1 + y_sub1];
        unsigned char c4 = d_currentWorld[x_add1 + y * worldSizeX];
        unsigned char c5 = d_currentWorld[x_add1 + y_add1];
        unsigned char c6 = d_currentWorld[x + y_add1];
        unsigned char c7 = d_currentWorld[x_sub1 + y_add1];
        unsigned char c8 = d_currentWorld[x_sub1 + y * worldSizeX];

        {
            unsigned char cellValue = d_currentWorld[x_sub1 + y_sub1];
            aliveCount += cellValue;
            r += cellValue * d_pixels[(y * worldSizeX + x) * 4];
            g += cellValue * d_pixels[(y * worldSizeX + x) * 4 + 1];
            b += cellValue * d_pixels[(y * worldSizeX + x) * 4 + 2];
            a += cellValue * d_pixels[(y * worldSizeX + x) * 4 + 3];
        }*/

#define _KERNEL_ADD_COLOR(xPos, yPos) \
        { \
            unsigned char cellValue = d_currentWorld[xPos + yPos * worldSizeX]; \
            unsigned long colorPosVal = ((yPos * worldSizeX + xPos) * 4); \
            aliveCount += cellValue; \
            r += cellValue * d_pixels[colorPosVal]; \
            g += cellValue * d_pixels[colorPosVal + 1]; \
            b += cellValue * d_pixels[colorPosVal + 2]; \
            /*a += cellValue * d_pixels[colorPosVal + 3];*/ \
        }
        _KERNEL_ADD_COLOR(x_sub1, y_sub1);
        _KERNEL_ADD_COLOR(x, y_sub1);
        _KERNEL_ADD_COLOR(x_add1, y_sub1);
        _KERNEL_ADD_COLOR(x_add1, y);
        _KERNEL_ADD_COLOR(x_add1, y_add1);
        _KERNEL_ADD_COLOR(x, y_add1);
        _KERNEL_ADD_COLOR(x_sub1, y_add1);
        _KERNEL_ADD_COLOR(x_sub1, y);

        if (aliveCount > 0)
        {
            r /= aliveCount;
            g /= aliveCount;
            b /= aliveCount;
            //a /= aliveCount;

            //if (r < 50) r = 50;
            //if (g < 50) g = 50;
            //if (b < 50) b = 50;
            int toAdd = 100-(r + g + b)/3;
            if (toAdd > 0)
            {
                r += toAdd;
                g += toAdd;
                b += toAdd;
            }



            d_pixels[(x + y * worldSizeX) * 4] = r;
            d_pixels[(x + y * worldSizeX) * 4 + 1] = g;
            d_pixels[(x + y * worldSizeX) * 4 + 2] = b;
            //d_pixels[(x + y * worldSizeX) * 4 + 3] = a;
        }
        return aliveCount;
    }


    __device__ unsigned char getNewCellState(unsigned char oldState, 
                                             int aliveNeighbourCount)
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

    __device__ void mixColor(sf::Uint8* colors, unsigned int colorCount, sf::Uint8* output)
    {
        if (colorCount == 0)
            return;
        unsigned int r = 0;
        unsigned int g = 0;
        unsigned int b = 0;
        unsigned int a = 0;
        colors -= 1;
        for (size_t i = 0; i < colorCount; ++i)
        {
            r += *(++colors);
            g += *(++colors);
            b += *(++colors);
            a += *(++colors);
        }

        r /= colorCount;
        g /= colorCount;
        b /= colorCount;
        a /= colorCount;

        output[0] = r;
        output[1] = g;
        output[2] = b;
        output[3] = a;
    }
}

