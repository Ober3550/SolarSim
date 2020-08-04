#include "imgui.h"
#include "imgui-SFML.h"

#include "absl/container/flat_hash_map.h"

#include <SFML/Graphics.hpp>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <emmintrin.h>
namespace fs = std::experimental::filesystem;

sf::Texture circle;

struct Planet {
    int* id;
    float* x;
    float* y;
    float* dx;
    float* dy;
    float* mass;
};

struct PlanetOption {
    bool valid = false;
    Planet planet;
    operator bool() { return valid; }
    operator Planet() { return planet; }
};

const int groupSize = 4;
const float G = 10;
struct PlanetGroup {
    int   id[groupSize] = {};
    float x [groupSize] = {};
    float y [groupSize] = {};
    float dx[groupSize] = {};
    float dy[groupSize] = {};
    float mass[groupSize] = {};
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
        planetsLength++;
    }
    SolarSystem()
    {
        AddPlanet(-200, 0, 0, -1, 0.5);
        AddPlanet(200, 0,  0, 1, 0.5);
        AddPlanet(600, 0,  0, 1, 0.05);
        AddPlanet(-600, 0,  0, -1, 0.05);
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
                &group->mass[index] };
            // Check for uninitialized planet
            if (group->id[index] == 0)
                return PlanetOption();
            return PlanetOption{ true,planet };
        }
        return PlanetOption();
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
                sprite.setScale(*planet.mass, *planet.mass);
                sprites.emplace_back(sprite);
            }
        }
        return sprites;
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
            }
        }
    }
    void UpdatePlanetsGrouped()
    {
        for (int i = 1; i <= planetsLength; i++)
        {
            if (PlanetOption tempA = GetPlanet(i))
            {
                Planet planetA = tempA;
                for (PlanetGroup group : planets)
                {
                    
                }
            }
        }
    }
};

int main()
{
    sf::RenderWindow window(sf::VideoMode(1000, 1000), "ImGui + SFML = <3");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    // Just for debugging
    /*for (const auto& entry : fs::directory_iterator(fs::current_path()))
    {
        std::cout << entry.path() << '\n';
    }*/

    circle.loadFromFile("circle.png");

    SolarSystem system;

    sf::View centreView;
    sf::Vector2u size = window.getSize();
    centreView.setSize(sf::Vector2f(size.x, size.y));
    centreView.setCenter(0, 0);

    int selectedPlanet = 1;

    float prevZoom  = 1.f;
    float zoom      = 1.f;

    sf::Clock deltaClock;
    sf::Clock frameClock;
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
        ImGui::Text(std::string("FPS: "+std::to_string(1000000/frameClock.getElapsedTime().asMicroseconds())).c_str());
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
            system.UpdatePlanets();

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
