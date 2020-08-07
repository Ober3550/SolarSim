#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

#include <string>
#include <iostream>
#include <experimental/filesystem>

#include <immintrin.h>

#include <sstream>
#include <thread>

namespace fs = std::experimental::filesystem;
namespace std {
    template <typename T>
    string to_string_with_precision(const T a_value, const int n = 2)
    {
        ostringstream out;
        out.precision(n);
        out << fixed << a_value;
        return out.str();
    }
}

sf::Texture circle;
const int VECWIDTH = 8;
const double pi = 2 * acos(0.0);
float G = 4.f;
const float zero = 0.f;
const float m_r = 0.5f;
__m256 gravity   = _mm256_set1_ps(G);
__m256 merge_r   = _mm256_set1_ps(m_r);
__m256i m_zeroi  = _mm256_set1_epi32(0);
__m256i m_onesi  = _mm256_set1_epi32(uint32_t(-1));
int selectedPlanet = 1;
bool gotoSelected = false;
bool merging = false;
bool multi_threaded = true;
bool simd = true;
sf::Vector2f camPos = { 0.f,0.f };

const float minZoom = 0.5;
const float maxZoom = 256.f;
int tiers = 3;
int children = 20;
bool static_framerate = true;

struct Planet {
    int* id;
    float* x;
    float* y;
    float* dx;
    float* dy;
    float* Fx;
    float* Fy;
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
        int   id[VECWIDTH] = {};
        __m256i m_id;
    };
    union {
        float x[VECWIDTH] = {};
        __m256 m_x;
    };
    union {
        float y[VECWIDTH] = {};
        __m256 m_y;
    };
    union {
        float dx[VECWIDTH] = {};
        __m256 m_dx;
    };
    union {
        float dy[VECWIDTH] = {};
        __m256 m_dy;
    };
    union {
        float Fx[VECWIDTH] = {};
        __m256 m_Fx;
    };
    union {
        float Fy[VECWIDTH] = {};
        __m256 m_Fy;
    };
    union {
        float mass[VECWIDTH] = {};
        __m256 m_mass;
    };
    union {
        float r[VECWIDTH] = {};
        __m256 m_r;
    };
};

class SolarSystem {
    std::vector<PlanetGroup> planets;
public:
    int planetsLength = 0;
    void AddPlanet(float x, float y, float dx, float dy, float mass)
    {
        if (planetsLength % VECWIDTH == 0)
            planets.emplace_back(PlanetGroup());
        PlanetGroup* group = &planets[planetsLength / VECWIDTH];
        int index = planetsLength % VECWIDTH;
        // Reserver id 0 for invalid planets
        group->id  [index] = planetsLength + 1;
        group->x   [index] = x;
        group->y   [index] = y;
        group->dx  [index] = dx;
        group->dy  [index] = dy;
        group->Fx  [index] = 0.f;
        group->Fy  [index] = 0.f;
        group->mass[index] = mass;
        group->r   [index] = sqrt(mass / pi) * 128.f;
        planetsLength++;
    }
    PlanetOption GetPlanet(int id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* group = &planets[id / VECWIDTH];
            int index = id % VECWIDTH;
            Planet planet = { 
                &group->id[index], 
                &group->x[index], 
                &group->y[index],
                &group->dx[index],
                &group->dy[index],
                &group->Fx[index],
                &group->Fy[index],
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
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* group = &planets[id / VECWIDTH];
            PlanetGroup* lastGroup = &planets[(planetsLength-1) / VECWIDTH];
            int index = id % VECWIDTH;
            if (id == planetsLength)
                goto clear_planet;
            if (PlanetOption temp = GetPlanet(planetsLength))
            {
                Planet lastPlanet = temp;
                group->x[index]     = *lastPlanet.x;
                group->y[index]     = *lastPlanet.y;
                group->dx[index]    = *lastPlanet.dx;
                group->dy[index]    = *lastPlanet.dy;
                group->Fx[index]    = *lastPlanet.Fx;
                group->Fy[index]    = *lastPlanet.Fy;
                group->mass[index]  = *lastPlanet.mass;
                group->r[index]     = *lastPlanet.r;
            }
            // Don't forget to clear because simd will still do ops on these values
            clear_planet:;
            int lastIndex = (planetsLength - 1) % VECWIDTH;
            lastGroup->id[lastIndex] = 0;
            lastGroup->x[lastIndex] = 0.f;
            lastGroup->y[lastIndex] = 0.f;
            lastGroup->dx[lastIndex] = 0.f;
            lastGroup->dy[lastIndex] = 0.f;
            lastGroup->Fx[lastIndex] = 0.f;
            lastGroup->Fy[lastIndex] = 0.f;
            lastGroup->mass[lastIndex] = 0.f;
            lastGroup->r[lastIndex] = 0.f;
            planetsLength--;
        }
    }
    void AddRandomSatellite(int id) {
        if (PlanetOption temp = GetPlanet(id))
        {
            Planet parent = temp;
            const int angles = (360 * 8);
            const int steps = 1000;
            const float min_dis = 500.f;
            const float max_dis = 10000.f;
            const float dif_dis = max_dis - min_dis;
            const float min_prop = 0.05;
            const float max_prop = 0.5f;
            const float dif_prop = max_prop - min_prop;
            const int   mas_step = 1000;
            float new_mass = min_prop + float(rand() % mas_step) / mas_step * dif_prop * (*parent.mass);
            float angle = float(rand() % angles) / float(angles);
            float magni = (min_dis + float(rand() % steps) / float(steps) * dif_dis) * (*parent.mass);
            float new_x = (*parent.x) + cos(angle * 2.f * pi) * magni;
            float new_y = (*parent.y) + sin(angle * 2.f * pi) * magni;
            float new_dx;
            float new_dy;
            if (rand() & 1)
            {
                new_dx = (*parent.dx) + cos(angle * 2.f * pi + pi * 0.5f) * 2.f;
                new_dy = (*parent.dy) + sin(angle * 2.f * pi + pi * 0.5f) * 2.f;
            }
            else
            {
                new_dx = (*parent.dx) + cos(angle * 2.f * pi + pi * 0.5f) * 2.f;
                new_dy = (*parent.dy) + sin(angle * 2.f * pi + pi * 0.5f) * 2.f;
            }
            
            AddPlanet(new_x, new_y, new_dx, new_dy, new_mass);
        }
    }
    // Parent, children per call, recursion depth
    void RecursivelyAddPlanets(int parent, int n, int m)
    {
        if (m == 0)
            return;
        for (int i = 0; i < n; i++)
        {
            AddRandomSatellite(parent);
            RecursivelyAddPlanets(planetsLength, n, m-1);
        }
    }
    SolarSystem()
    {
        //AddPlanet(-200, 0, 0, -1, 0.5);
        //AddPlanet(200, 0, 0, 1, 0.5);
        //AddPlanet(600, 0, 0, 1, 0.05);
        //AddPlanet(-600, 0, 0, -1, 0.05);
        AddPlanet(0, 0, 0, 0, 1);
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
                if (i == selectedPlanet)
                {
                    if (gotoSelected)
                        camPos = { *planet.x, *planet.y };
                }
                sprite.setScale(*planet.r / 128.f, *planet.r / 128.f);
                sprites.emplace_back(sprite);
            }
        }
        return sprites;
    }
    void MergePlanets(int idA, int idB)
    {
        if (idA > idB)
        {
            int temp = idA;
            idA = idB;
            idB = temp;
        }
        assert(idA != idB);
        if (PlanetOption tempA = GetPlanet(idA))
        {
            if (PlanetOption tempB = GetPlanet(idB))
            {
                Planet planetA = tempA;
                Planet planetB = tempB;
                float total_mass = *planetA.mass + *planetB.mass;
                float relative_mass = (*planetB.mass / *planetA.mass) * 0.5f;
                *planetA.x += (*planetB.x - *planetA.x) * relative_mass;
                *planetA.y += (*planetB.y - *planetA.y) * relative_mass;
                *planetA.dx = ((*planetA.dx) * (*planetA.mass) + *planetB.dx * *planetB.mass) / total_mass;
                *planetA.dy = ((*planetA.dy) * (*planetA.mass) + *planetB.dy * *planetB.mass) / total_mass;
                *planetA.mass = total_mass;
                *planetA.r = sqrt(total_mass / pi) * 128.f;
                if (idB == selectedPlanet)
                    selectedPlanet = idA;
                // Remove planet B
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
                for (int j = 1; j <= planetsLength; j++)
                {
                    if (PlanetOption tempB = GetPlanet(j))
                    {
                        Planet planetB = tempB;
                        if (*planetA.id != *planetB.id)
                        {
                            float rx = ((*planetB.x) - (*planetA.x));
                            float ry = ((*planetB.y) - (*planetA.y));
                            float r2 = pow(rx, 2) + pow(ry, 2);

                            // Check if planets merge
                            float mass_r2 = pow(((*planetA.r + *planetB.r) * m_r), 2);
                            if (merging && r2 < mass_r2)
                            {
                                MergePlanets(i, j);
                                // Recalculate planet As forces since its mass changed
                                i--;
                                goto skip_calc;
                            }
                            float F = (G * (*planetA.mass) * (*planetB.mass)) / r2;
                            *planetA.Fx += F * rx;
                            *planetA.Fy += F * ry;
                        }
                    }
                }
            skip_calc:;
            }
        }
    }
    void MergeAllPlanets(const int start, const int end)
    {
        int start2 = (start - 1) / VECWIDTH;
        int end2 = end / VECWIDTH;
        for (int i = start2; i < end2; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    for (int k = 0; k < planetsLength / VECWIDTH; k++)
                    {
                        PlanetGroup* groupB = &planets[k];
                        for (int l = 0; l < VECWIDTH; l++)
                        {
                            if (groupB->id[l] != 0 && groupA->id[j] != groupB->id[l])
                            {
                                float rx = ((groupB->x[l]) - groupA->x[j]);
                                float ry = ((groupB->y[l]) - groupA->y[j]);
                                float r2 = (rx * rx) + (ry * ry);
                                // Check if planets merge
                                float mass_r2 = ((groupA->r[j] + groupB->r[l]) * m_r) * ((groupA->r[j] + groupB->r[l]) * m_r);
                                if (r2 < mass_r2)
                                {
                                    MergePlanets(groupA->id[j], groupB->id[l]);
                                    j--;
                                    goto skip_grouped;
                                }
                            }
                        }
                    }
                }
            skip_grouped:;
            }
        }
    }
    void Threaded(const int start, const int end)
    {
        // F = (G * m1 * m2) / r^2
        // F = ma
        // a = m / F;
        // a = m / ((G * m * m2) / r^2)
        for (int i = start; i < end; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    for (int k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup* groupB = &planets[k];
                        for (int l = 0; l < VECWIDTH; l++)
                        {
                            if (groupB->id[l] != 0 && groupA->id[j] != groupB->id[l])
                            {
                                float rx = groupB->x[l] - groupA->x[j];
                                float ry = groupB->y[l] - groupA->y[j];
                                float r2 = (rx * rx) + (ry * ry);
                                float F = (G * (groupA->mass[j]) * groupB->mass[j]) / r2;
                                groupA->Fx[j] += F * rx;
                                groupA->Fy[j] += F * ry;
                            }
                        }
                    }
                }
            }
        }
    }
    void SimdThreaded(const int start, const int end)
    {
        for (int i = start; i < end; i++)
        {
            if (start == 0)
                bool hello;
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                __m256 planetA_x    = _mm256_set1_ps(groupA->x[j]);
                __m256 planetA_y    = _mm256_set1_ps(groupA->y[j]);
                __m256 planetA_mass = _mm256_set1_ps(groupA->mass[j]);
                __m256 planetA_r    = _mm256_set1_ps(groupA->r[j]);
                __m256i planetA_id    = _mm256_set1_epi32(groupA->id[j]);
                union {
                    float  planetA_Fx[VECWIDTH];
                    __m256 mplanetA_Fx = _mm256_set1_ps(zero);
                };
                union {
                    float  planetA_Fy[VECWIDTH];
                    __m256 mplanetA_Fy = _mm256_set1_ps(zero);
                };

                for (int k = 0; k < planets.size(); k++)
                {
                    PlanetGroup* groupB = &planets[k];
                    // Subtract planet As position from groups positions to find relative distance
                    // Find the square of each distance
                    // Code readibility may suffer due to functions not being optimized such that
                    // Simd vectors aren't being stored in registers properly and may be passed to cache or stack preemtively
                    __m256 rx = _mm256_sub_ps(groupB->m_x, planetA_x);
                    __m256 rx2 = _mm256_mul_ps(rx, rx);
                    __m256 ry = _mm256_sub_ps(groupB->m_y, planetA_y);
                    __m256 ry2 = _mm256_mul_ps(ry, ry);
                    // Find the radius squared
                    __m256 r2 = _mm256_add_ps(rx2, ry2);
                    // Calculate gravity
                    __m256 mass = _mm256_mul_ps(groupB->m_mass, planetA_mass);
                    __m256 gm = _mm256_mul_ps(mass, gravity);
                    // Find the forces for each dimension
                    __m256 F = _mm256_div_ps(gm, r2);
                    union {
                        float F_x[VECWIDTH];
                        __m256 Fx;
                    };
                    Fx = _mm256_mul_ps(F, rx);
                    union {
                        float F_y[VECWIDTH];
                        __m256 Fy;
                    };
                    Fy = _mm256_mul_ps(F, ry);

                    // Remove nan values such as planets affecting themselves
                    // If id == 0
                    __m256i zeromask = _mm256_cmpeq_epi32(groupB->m_id, m_zeroi);
                    __m256i idmask = _mm256_cmpeq_epi32(groupB->m_id, planetA_id);
                    // If groupA.id == groupB.id
                    __m256i bothmask = _mm256_or_si256(zeromask, idmask);
                    bothmask = _mm256_xor_si256(bothmask, m_onesi);
                    Fx = _mm256_and_ps(Fx, _mm256_castsi256_ps(bothmask));
                    Fy = _mm256_and_ps(Fy, _mm256_castsi256_ps(bothmask));

                    mplanetA_Fx = _mm256_add_ps(mplanetA_Fx, Fx);
                    mplanetA_Fy = _mm256_add_ps(mplanetA_Fy, Fy);
                }
                for (int l = 0; l < VECWIDTH; l++)
                {
                    groupA->Fx[j] += planetA_Fx[l];
                    groupA->Fy[j] += planetA_Fy[l];
                }
            }
        }
    }
    void ThreadedUpdatePlanets()
    {
        // I don't think this can be multithreaded since it relies on removing elements being thread safe... 
        // which is obvious why it wouldn't be
        if(merging)
            MergeAllPlanets(1, planetsLength);

        // Seems like threads don't like to be moved or recreated
        std::vector<std::thread> threads;
        const int NUM_THREADS = 4;
        int block_size = (planetsLength / VECWIDTH) / NUM_THREADS + 1;
        for (int i = 0; i < NUM_THREADS; i++)
        {
            if (simd)
            {
                threads.emplace_back(std::thread([&](SolarSystem* system, const int start, const int end) {
                    system->SimdThreaded(start, end);
                    }, this, i * block_size, (i + 1) * block_size));
            }
            else
            {
                threads.emplace_back(std::thread([&](SolarSystem* system, const int start, const int end) {
                    system->Threaded(start, end);
                    }, this, i * block_size, (i + 1) * block_size));
            }
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threads[i].join();
        }

        // Separate applying forces and velocity since it's O(n)
        for (int i = 0; i < planetsLength / VECWIDTH + 1; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    groupA->dx[j] += groupA->Fx[j] / groupA->mass[j];
                    groupA->dy[j] += groupA->Fy[j] / groupA->mass[j];
                    groupA->x[j] += groupA->dx[j];
                    groupA->y[j] += groupA->dy[j];
                    groupA->Fx[j] = 0.f;
                    groupA->Fy[j] = 0.f;
                }
            }
        }
    }
    void UpdatePlanetsSimd()
    {
        if(merging)
            MergeAllPlanets(1, planetsLength);
        for (int i = 0; i < planets.size(); i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                __m256 planetA_x = _mm256_set1_ps(groupA->x[j]);
                __m256 planetA_y = _mm256_set1_ps(groupA->y[j]);
                __m256 planetA_mass = _mm256_set1_ps(groupA->mass[j]);
                __m256 planetA_r = _mm256_set1_ps(groupA->r[j]);
                __m256i planetA_id = _mm256_set1_epi32(groupA->id[j]);
                union {
                    float  planetA_Fx[VECWIDTH];
                    __m256 mplanetA_Fx = _mm256_set1_ps(zero);
                };
                union {
                    float  planetA_Fy[VECWIDTH];
                    __m256 mplanetA_Fy = _mm256_set1_ps(zero);
                };
                int planet_id = groupA->id[j];

                for (int k = 0; k < planets.size(); k++)
                {
                    const PlanetGroup* groupB = &planets[k];
                    // Subtract planet As position from groups positions to find relative distance
                    // Find the square of each distance
                    // Code readibility may suffer due to functions not being optimized such that
                    // Simd vectors aren't being stored in registers properly and may be passed to cache or stack preemtively
                    __m256 rx = _mm256_sub_ps(groupB->m_x, planetA_x);
                    __m256 rx2 = _mm256_mul_ps(rx, rx);
                    __m256 ry = _mm256_sub_ps(groupB->m_y, planetA_y);
                    __m256 ry2 = _mm256_mul_ps(ry, ry);
                    // Find the radius squared
                    union {
                        float  i_r2[VECWIDTH];
                        __m256 r2;
                    };
                    r2 = _mm256_add_ps(rx2, ry2);                    
                    // Calculate gravity
                    __m256 mass = _mm256_mul_ps(groupB->m_mass, planetA_mass);
                    __m256 gm = _mm256_mul_ps(mass, gravity); 
                    // Find the forces for each dimension
                    __m256 F = _mm256_div_ps(gm, r2);
                    union {
                        float F_x[VECWIDTH];
                        __m256 Fx;
                    };
                    Fx = _mm256_mul_ps(F, rx);
                    union {
                        float F_y[VECWIDTH];
                        __m256 Fy;
                    };
                    Fy = _mm256_mul_ps(F, ry);

                    // Remove nan values such as planets affecting themselves
                    // If id == 0
                    __m256i zeromask = _mm256_cmpeq_epi32(groupB->m_id, m_zeroi);
                    __m256i idmask = _mm256_cmpeq_epi32(groupB->m_id, planetA_id);
                    // If groupA.id == groupB.id
                    __m256i bothmask = _mm256_or_si256(zeromask, idmask);
                    bothmask = _mm256_xor_si256(bothmask, m_onesi);
                    Fx = _mm256_and_ps(Fx, _mm256_castsi256_ps(bothmask));
                    Fy = _mm256_and_ps(Fy, _mm256_castsi256_ps(bothmask));

                    // Apply the forces 
                    mplanetA_Fx = _mm256_add_ps(mplanetA_Fx, Fx);
                    mplanetA_Fy = _mm256_add_ps(mplanetA_Fy, Fy);
                }
                for (int l = 0; l < VECWIDTH; l++)
                {
                    groupA->Fx[j] += planetA_Fx[l];
                    groupA->Fy[j] += planetA_Fy[l];
                }
            }
        }
        // Separate applying forces and velocity since it's O(n)
        for (int i = 0; i < planetsLength / VECWIDTH + 1; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    groupA->dx[j] += groupA->Fx[j] / groupA->mass[j];
                    groupA->dy[j] += groupA->Fy[j] / groupA->mass[j];
                    groupA->x[j] += groupA->dx[j];
                    groupA->y[j] += groupA->dy[j];
                    groupA->Fx[j] = 0.f;
                    groupA->Fy[j] = 0.f;
                }
            }
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
    sf::Vector2f prevCamPos = { 0.f,0.f };

    circle.loadFromFile("circle.png");
    uint8_t KEYW = 1;
    uint8_t KEYS = 2;
    uint8_t KEYA = 4;
    uint8_t KEYD = 8;
    uint8_t pressed = 0;
    
    srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));
    SolarSystem system;
    system.RecursivelyAddPlanets(selectedPlanet, children, tiers);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::W)
                    pressed |= KEYW;
                else if (event.key.code == sf::Keyboard::S)
                    pressed |= KEYS;
                else if (event.key.code == sf::Keyboard::A)
                    pressed |= KEYA;
                else if (event.key.code == sf::Keyboard::D)
                    pressed |= KEYD;
            }
            else if (event.type == sf::Event::KeyReleased)
            {
                if (event.key.code == sf::Keyboard::W)
                    pressed &= (0xff ^ KEYW);
                else if (event.key.code == sf::Keyboard::S)
                    pressed &= (0xff ^ KEYS);
                else if (event.key.code == sf::Keyboard::A)
                    pressed &= (0xff ^ KEYA);
                else if (event.key.code == sf::Keyboard::D)
                    pressed &= (0xff ^ KEYD);
            }
            else if (event.type == sf::Event::Resized)
            {
                sf::Vector2u size = window.getSize();
                centreView.setSize(sf::Vector2f(size.x, size.y));
                centreView.setCenter(0, 0);
            }
            else if (event.type == sf::Event::MouseWheelScrolled)
            {
                if (event.mouseWheelScroll.delta != 0)
                {
                    zoom -= event.mouseWheelScroll.delta * 1.f;
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
        const float move_speed = 5.f;
        for (int i = 0; i < 4; i++)
        {
            if ((pressed >> i) & 1)
            {
                if (i < 2)
                    camPos.y += move_speed * zoom * ((i & 1) ? 1.f : -1.f);
                else
                    camPos.x += move_speed * zoom * ((i & 1) ? 1.f : -1.f);
            }
        }
        if (prevZoom != zoom)
        {
            centreView.zoom(pow(zoom / prevZoom,2));
            prevZoom = zoom;
        }
        if (prevCamPos != camPos)
        {
            centreView.move(camPos - prevCamPos);
            prevCamPos = camPos;
        }
        window.setView(centreView);
        window.clear();
        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Update Rate");
        float frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
        static int updatesPerFrame = 1;
        ImGui::Text(std::string("FPS:     "+std::to_string_with_precision(frameRate)).c_str());
        ImGui::Text(std::string("UPS:     " + std::to_string_with_precision(frameRate * float(updatesPerFrame))).c_str());
        std::string operations = std::to_string_with_precision(frameRate * float(updatesPerFrame * system.planetsLength * system.planetsLength));
        int outer = 0;
        bool decimal = false;
        for (auto iter = operations.end(); iter > operations.begin();iter--)
        {
            if (decimal)
            {
                if (outer % 3 == 0 && outer != 0)
                    iter = operations.insert(iter, ',');
            }
            else if (operations[operations.length() - outer - 1] == '.')
            {
                decimal = true;
                outer = -1;
            }
            outer++;
        }
        ImGui::Text(std::string("OPS:     " + operations).c_str());
        ImGui::Text(std::string("PLANETS: " + std::to_string(system.planetsLength)).c_str());
        ImGui::SliderInt(" :UPF", &updatesPerFrame, 1, 50);
        ImGui::Checkbox(" :Threaded", &multi_threaded);
        ImGui::Checkbox(" :Simd", &simd);
        ImGui::Checkbox(" :Lock Framerate", &static_framerate);
        ImGui::Checkbox(" :Planet Merging", &merging);
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
            if (ImGui::SliderFloat(": G", &G, 0.1, 10))
            {
                gravity = _mm256_set1_ps(G);
            }
            ImGui::SliderFloat(": ZOOM", &zoom, minZoom, maxZoom);
            ImGui::SliderInt(": Tiers",    &tiers,    1, 4);
            ImGui::SliderInt(": Children", &children, 1, 10);
            ImGui::Text(std::string("Settings will add " + std::to_string(int(std::pow(children, tiers))) + " planets.").c_str());
            if (ImGui::Button("Remove Planet"))
            {
                system.RemovePlanet(selectedPlanet);
            }
            if (ImGui::Button("Add Planet"))
            {
                system.AddPlanet(smin, smin, 0, 0, 0.5);
                selectedPlanet = system.planetsLength;
            }
            if (ImGui::Button("Add Random"))
            {
                system.AddRandomSatellite(selectedPlanet);
            }
            if (ImGui::Button("Add Universe"))
            {
                system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
            }
            ImGui::Checkbox(": Follow Selected", &gotoSelected);
        }
        if (!ImGui::IsAnyWindowFocused())
        {
            if (static_framerate)
            {
                if (updatesPerFrame > 1 && frameRate < 50.f)
                    updatesPerFrame--;
                else if (updatesPerFrame < 50 && frameRate > 55.f)
                    updatesPerFrame++;
            }
            if (!multi_threaded)
                for (int i = 0; i < updatesPerFrame; i++)
                {
                    if (simd)
                        system.UpdatePlanetsSimd();
                    else
                        system.UpdatePlanets();
                }
            else
                for (int i = 0; i < updatesPerFrame; i++)
                    system.ThreadedUpdatePlanets();
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
