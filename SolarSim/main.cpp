#include "imgui.h"
#include "imgui-SFML.h"

#include "absl/container/flat_hash_map.h"

#include <SFML/Graphics.hpp>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <mmintrin.h>
#include <xmmintrin.h>
//#include "FastNoiseSIMD.h"
namespace fs = std::experimental::filesystem;

sf::Texture circle;
const int groupSize = 4;
double pi = 2 * acos(0.0);
const float G = 10;
const float zero = 0.f;
const float m_r = 0.5f;
__m128 gravity = _mm_broadcast_ss(&G);
__m128 merge_r = _mm_broadcast_ss(&m_r);

struct Planet {
    int* id;
    float* x;
    float* y;
    float* dx;
    float* dy;
    float* mass;
    float* r;
};

struct PlanetOption {
    bool valid = false;
    Planet planet;
    operator bool() { return valid; }
    operator Planet() { return planet; }
};


struct PlanetGroup {
    union {
        int   id[groupSize] = {};
        __m128i m_id;
    };
    union {
        float x[groupSize] = {};
        __m128 m_x;
    };
    union {
        float y[groupSize] = {};
        __m128 m_y;
    };
    union {
        float dx[groupSize] = {};
        __m128 m_dx;
    };
    union {
        float dy[groupSize] = {};
        __m128 m_dy;
    };
    union {
        float mass[groupSize] = {};
        __m128 m_mass;
    };
    union {
        float r[groupSize] = {};
        __m128 m_r;
    };
};

class SolarSystem {
    std::vector<PlanetGroup> planets;
public:
    int planetsLength = 0;
    void AddPlanet(float x, float y, float dx, float dy, float mass)
    {
        if (planetsLength % groupSize == 0)
            planets.emplace_back(PlanetGroup());
        PlanetGroup* group = &planets[planetsLength / 4];
        int index = planetsLength % groupSize;
        // Reserver id 0 for invalid planets
        group->id  [index] = planetsLength + 1;
        group->x   [index] = x;
        group->y   [index] = y;
        group->dx  [index] = dx;
        group->dy  [index] = dy;
        group->mass[index] = mass;
        group->r   [index] = sqrt(mass / pi) * 128.f;
        planetsLength++;
    }
    PlanetOption GetPlanet(int id)
    {
        id--;
        if (id / groupSize < planets.size())
        {
            PlanetGroup* group = &planets[id / groupSize];
            int index = id % groupSize;
            Planet planet = { 
                &group->id[index], 
                &group->x[index], 
                &group->y[index],
                &group->dx[index],
                &group->dy[index],
                &group->mass[index],
                &group->r[index]
            };
            // Check for uninitialized planet
            if (group->id[index] == 0)
                return PlanetOption();
            return PlanetOption{ true,planet };
        }
        return PlanetOption();
    }
    void RemovePlanet(int id)
    {
        id--;
        if (id / groupSize < planets.size())
        {
            PlanetGroup* group = &planets[id / groupSize];
            PlanetGroup* lastGroup = &planets[(planetsLength-1) / groupSize];
            int index = id % groupSize;
            if (PlanetOption temp = GetPlanet(planetsLength))
            {
                Planet lastPlanet = temp;
                group->x[index]     = *lastPlanet.x;
                group->y[index]     = *lastPlanet.y;
                group->dx[index]    = *lastPlanet.dx;
                group->dy[index]    = *lastPlanet.dy;
                group->mass[index]  = *lastPlanet.mass;
                group->r[index]     = *lastPlanet.r;
                // Don't forget to clear because simd will still do ops on these values
                int lastIndex = planetsLength-1 % groupSize;
                lastGroup->id[lastIndex]   = 0;
                lastGroup->x[lastIndex]    = 0.f;
                lastGroup->y[lastIndex]    = 0.f;
                lastGroup->dx[lastIndex]   = 0.f;
                lastGroup->dy[lastIndex]   = 0.f;
                lastGroup->mass[lastIndex] = 0.f;
                lastGroup->r[lastIndex]    = 0.f;
                planetsLength--;
            }
        }
    }
    void AddRandomSatellite(int id) {
        if (PlanetOption temp = GetPlanet(id))
        {
            Planet parent = temp;
        }
    }
    SolarSystem()
    {
        AddPlanet(-200, 0, 0, -1, 0.5);
        AddPlanet(200, 0, 0, 1, 0.5);
        AddPlanet(600, 0, 0, 1, 0.05);
        AddPlanet(-600, 0, 0, -1, 0.05);
        //AddPlanet(-72, 0, 0, 1, 1);
        //AddPlanet(72, 0, 0, -2, 0.5);
        //AddPlanet(-216, 0, 0, 2, 0.5);
    }
    std::vector<sf::Sprite> DrawSolarSystem()
    {
        std::vector<sf::Sprite> sprites;
        for (int i = 1; i <= planetsLength;i++)
        {
            if (PlanetOption temp = GetPlanet(i))
            {
                Planet planet = temp;
                sf::Sprite sprite;
                sprite.setTexture(circle);
                sprite.setOrigin(128, 128);
                sprite.setPosition(*planet.x, *planet.y);
                sprite.setScale(*planet.r / 128.f, *planet.r / 128.f);
                sprites.emplace_back(sprite);
            }
        }
        return sprites;
    }
    void MergePlanets(int idA, int idB)
    {
        assert(idA != idB);
        if (PlanetOption tempA = GetPlanet(idA))
        {
            if (PlanetOption tempB = GetPlanet(idB))
            {
                Planet planetA = tempA;
                Planet planetB = tempB;
                float total_mass = *planetA.mass + *planetB.mass;
                float relative_mass = (*planetB.mass / *planetA.mass) * 0.5f;
                *planetA.x += *planetB.x * relative_mass;
                *planetA.y += *planetB.y * relative_mass;
                *planetA.dx = ((*planetA.dx) * (*planetA.mass) + *planetB.dx * *planetB.mass) / total_mass;
                *planetA.dy = ((*planetA.dy) * (*planetA.mass) + *planetB.dy * *planetB.mass) / total_mass;
                *planetA.mass = total_mass;
                *planetA.r = sqrt(total_mass / pi) * 128.f;
                // Remove planet B
                // Recalculate planet As forces since its mass changed
                RemovePlanet(idB);
            }
        }
    }
    void UpdatePlanets()
    {
        // F = (G * m1 * m2) / r^2
        // F = ma
        // a = m / F;
        // a = m / ((G * m * m2) / r^2)
        
        for (int i = 1; i <= planetsLength; i++)
        {
            if (PlanetOption tempA = GetPlanet(i))
            {
                Planet planetA = tempA;
                float Fx = 0;
                float Fy = 0;
                for (int j = 1; j <= planetsLength; j++)
                {
                    if (PlanetOption tempB = GetPlanet(j))
                    {
                        Planet planetB = tempB;
                        if (*planetA.id != *planetB.id)
                        {
                            float rx = ((*planetB.x) - (*planetA.x));
                            float ry = ((*planetB.y) - (*planetA.y));
                            float r2 = pow(rx,2) + pow(ry,2);

                            // Check if planets merge
                            float mass_r2 = pow(((*planetA.r + *planetB.r) * m_r),2);
                            if (r2 < mass_r2)
                            {
                                MergePlanets(i, j);
                                i--;
                                goto skip_calc;
                            }
                            float F = (G * (*planetA.mass) * (*planetB.mass)) / r2;
                            Fx += F * rx;
                            Fy += F * ry;
                        }
                    }
                }
                *planetA.dx += Fx / (*planetA.mass);
                *planetA.dy += Fy / (*planetA.mass);
                *planetA.x += *planetA.dx;
                *planetA.y += *planetA.dy;

            skip_calc: continue;
            }
        }
    }
    void UpdatePlanetsGrouped()
    {
        for (int i = 1; i <= planetsLength; i++)
        {
            int j = 0;
            if (PlanetOption tempA = GetPlanet(i))
            {
                Planet planetA = tempA;
                __m128i planetA_id = _mm_set1_epi32(*planetA.id);
                __m128 planetA_x    = _mm_broadcast_ss(planetA.x);
                __m128 planetA_y    = _mm_broadcast_ss(planetA.y);
                union {
                    float  planetA_Fx[groupSize];
                    __m128 mplanetA_Fx = _mm_broadcast_ss(&zero);
                };
                union {
                    float  planetA_Fy[groupSize];
                    __m128 mplanetA_Fy = _mm_broadcast_ss(&zero);
                };
                __m128 planetA_mass = _mm_broadcast_ss(planetA.mass);
                __m128 planetA_r = _mm_broadcast_ss(planetA.r);
                float planet_id = *planetA.id;

                for (PlanetGroup group : planets)
                {
                    // Subtract planet As position from groups positions to find relative distance
                    __m128 rx = _mm_sub_ps(group.m_x, planetA_x);
                    __m128 ry = _mm_sub_ps(group.m_y, planetA_y);
                    // Find the square of each distance
                    __m128 rx2 = _mm_mul_ps(rx, rx);
                    __m128 ry2 = _mm_mul_ps(ry, ry);
                    // Find the radius squared
                    union {
                        float  i_r2[groupSize];
                        __m128 r2;
                    };
                    r2 = _mm_add_ps(rx2, ry2);
                    // Check if planets merge
                    
                    __m128 total_mass_r = _mm_add_ps(group.m_r, planetA_r);
                    __m128 total_mass_r_check = _mm_mul_ps(total_mass_r, merge_r);
                    union {
                        float total_mass_r2[groupSize];
                        __m128 m_total_mass_r2;
                    };
                    m_total_mass_r2 = _mm_mul_ps(total_mass_r_check, total_mass_r_check);
                    
                    for (int k = 0; k < groupSize; k++)
                    {
                        if (group.id[k] != 0 && planet_id != group.id[k] && i_r2[k] < total_mass_r2[k])
                        {
                            MergePlanets(i, j * groupSize + k + 1);
                            i--;
                            goto skip_group_calc;
                        }
                    }
                    
                    // Calculate gravity
                    __m128 mass = _mm_mul_ps(group.m_mass, planetA_mass);
                    __m128 gm = _mm_mul_ps(mass, gravity); 
                    // Find the forces for each dimension
                    __m128 F = _mm_div_ps(gm, r2);
                    union {
                        float F_x[groupSize];
                        __m128 Fx;
                    };
                    Fx = _mm_mul_ps(F, rx);
                    union {
                        float F_y[groupSize];
                        __m128 Fy;
                    };
                    Fy = _mm_mul_ps(F, ry);
                    // Remove nan values such as planets affecting themselves
                    for (int k = 0; k < groupSize; k++)
                    {
                        if (isnan(F_x[k]))
                            F_x[k] = 0;
                        if (isnan(F_y[k]))
                            F_y[k] = 0;
                    }
                    // Apply the forces 
                    mplanetA_Fx = _mm_add_ps(mplanetA_Fx, Fx);
                    mplanetA_Fy = _mm_add_ps(mplanetA_Fy, Fy);
                }
                for (int k = 0; k < groupSize; k++)
                {
                    *planetA.dx += planetA_Fx[k] / (*planetA.mass);
                    *planetA.dy += planetA_Fy[k] / (*planetA.mass);
                }
                *planetA.x += *planetA.dx;
                *planetA.y += *planetA.dy;
                j++;
            }
        skip_group_calc: continue;
        }
    }
};

int main()
{
    sf::Clock deltaClock;
    sf::Clock frameClock;

    sf::RenderWindow window(sf::VideoMode(1000, 1000), "ImGui + SFML = <3");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    // Just for debugging
    /*for (const auto& entry : fs::directory_iterator(fs::current_path()))
    {
        std::cout << entry.path() << '\n';
    }*/

    sf::View centreView;
    sf::Vector2u size = window.getSize();
    centreView.setSize(sf::Vector2f(size.x, size.y));
    centreView.setCenter(0, 0);
    float prevZoom = 1.f;
    float zoom = 1.f;

    circle.loadFromFile("circle.png");
    int selectedPlanet = 1;

    srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));
    SolarSystem system;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);

            if (event.type == sf::Event::Resized)
            {
                sf::Vector2u size = window.getSize();
                centreView.setSize(sf::Vector2f(size.x, size.y));
                centreView.setCenter(0, 0);
            }
            else if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta != 0)
                {
                    const float minZoom = 0.5;
                    const float maxZoom = 16.f;
                    zoom -= event.mouseWheelScroll.delta * 0.5;
                    if (zoom < minZoom)
                        zoom = minZoom;
                    if (zoom > maxZoom)
                        zoom = maxZoom;
                }
            }
            else if (event.type == sf::Event::Closed) {
                window.close();
            }
        }
        if (prevZoom != zoom)
        {
            centreView.zoom(zoom / prevZoom);
            prevZoom = zoom;
        }
        window.setView(centreView);
        window.clear();
        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Update Rate");
        int frameRate = 1000000 / frameClock.getElapsedTime().asMicroseconds();
        static int updatesPerFrame = 1;
        ImGui::Text(std::string("FPS: "+std::to_string(frameRate)).c_str());
        ImGui::Text(std::string("UPS: " + std::to_string(frameRate * updatesPerFrame)).c_str());
        ImGui::SliderInt(" :Updates Per Frame", &updatesPerFrame, 1, 10);
        ImGui::End();
        frameClock.restart();

        ImGui::Begin("Modify Planet");
        ImGui::SliderInt(": ID", &selectedPlanet, 1, system.planetsLength);
        if (PlanetOption temp = system.GetPlanet(selectedPlanet))
        {
            float smin = -1000;
            float smax = 1000;
            float dmin = -1.0;
            float dmax = 1.0;
            Planet planet = temp;
            ImGui::SliderFloat(": X",  planet.x,    smin, smax);
            ImGui::SliderFloat(": Y",  planet.y,    smin, smax);
            ImGui::SliderFloat(": DX", planet.dx,   dmin, dmax);
            ImGui::SliderFloat(": DY", planet.dy,   dmin, dmax);
            ImGui::SliderFloat(": M",  planet.mass, 0.0001, 10,"%.3f",2.f);
            if (ImGui::Button("Add Planet"))
            {
                system.AddPlanet(smin, smin, 0, 0, 0.5);
                selectedPlanet = system.planetsLength;
            }
        }
        if (!ImGui::IsWindowFocused())
        {
            for (int i = 0; i < updatesPerFrame; i++)
            {
                //system.UpdatePlanets();
                system.UpdatePlanetsGrouped();
            }
        }

        ImGui::End();

        //ImGui::ShowTestWindow();

        std::vector<sf::Sprite> sprites = system.DrawSolarSystem();
        for (int i=0;i<sprites.size();i++)
        {
            sf::Sprite sprite = sprites[i];
            if (i + 1 == selectedPlanet)
                sprite.setColor(sf::Color(255, 0, 0, 255));
            window.draw(sprite);
        }
       
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
