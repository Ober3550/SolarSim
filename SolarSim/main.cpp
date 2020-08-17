#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <immintrin.h>
#include <sstream>
#include <thread>
#include <array>

#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include "imgui.h"
#include "imgui-SFML.h"

#include <glm/glm.hpp>
#include <glm/vec3.hpp> // glm::vec3
#include <glm/vec4.hpp> // glm::vec4
#include <glm/mat4x4.hpp> // glm::mat4
#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
#include <glm/ext/scalar_constants.hpp> // glm::pi
#include <glm/gtc/type_ptr.hpp>

namespace fs = std::experimental::filesystem;
namespace std {
    template <typename T>
    string to_string_with_precision(const T a_value, const int64_t n = 2)
    {
        ostringstream out;
        out.precision(n);
        out << fixed << a_value;
        return out.str();
    }
}

sf::Texture circle;
const int64_t VECWIDTH = 4;
const double pi = 2 * acos(0.0);
float G = 4.f;
const float zero = 0.f;
const float merge_r = 0.5f;

__m256i m_zeroi  = _mm256_set1_epi64x(0);
__m256i m_onesi  = _mm256_set1_epi64x(uint64_t(-1));
__m256d gravity  = _mm256_set1_pd(G);

int64_t selectedPlanet = 1;
bool gotoSelected = false;
bool merging = false;
bool multi_threaded = true;
bool simd = true;
sf::Vector2f camPos = { 0.f,0.f };

const float minZoom = 0.5;
const float maxZoom = 256.f;
int64_t tiers = 2;
int64_t children = 10;
bool static_framerate = false;
bool runSimulation = true;

void drawSphere(double r, int lats, int longs) {
    int i, j;
    for (i = 0; i <= lats; i++) {
        double lat0 = pi * (-0.5 + (double)(i - 1) / lats);
        double z0 = sin(lat0);
        double zr0 = cos(lat0);

        double lat1 = pi * (-0.5 + (double)i / lats);
        double z1 = sin(lat1);
        double zr1 = cos(lat1);

        glBegin(GL_QUAD_STRIP);
        for (j = 0; j <= longs; j++) {
            double lng = 2 * pi * (double)(j - 1) / longs;
            double x = cos(lng);
            double y = sin(lng);

            glNormal3f(x * zr0, y * zr0, z0);
            glVertex3f(r * x * zr0, r * y * zr0, r * z0);
            glNormal3f(x * zr1, y * zr1, z1);
            glVertex3f(r * x * zr1, r * y * zr1, r * z1);
        }
        glEnd();
    }
}

struct Planet {
    int64_t* id;
    double* x;
    double* y;
    double* z;
    double* mass;
    double* dx;
    double* dy;
    double* dz;
    double* Fx;
    double* Fy;
    double* Fz;
    double* r;
};

struct PlanetOption {
    bool valid = false;
    Planet planet;
    operator bool() { return valid; }
    operator Planet() { return planet; }
};

struct PlanetGroup {
    union {
        int64_t   id[VECWIDTH] = {};
        __m256i m_id;
    };
    union {
        double x[VECWIDTH] = {};
        __m256d m_x;
    };
    union {
        double y[VECWIDTH] = {};
        __m256d m_y;
    };
    union {
        double z[VECWIDTH] = {};
        __m256d m_z;
    };
    union {
        double mass[VECWIDTH] = {};
        __m256d m_mass;
    };
};

struct PlanetGroupExtra {
    union {
        double dx[VECWIDTH] = {};
        __m256d m_dx;
    };
    union {
        double dy[VECWIDTH] = {};
        __m256d m_dy;
    };
    union {
        double dz[VECWIDTH] = {};
        __m256d m_dz;
    };
    union {
        double Fx[VECWIDTH] = {};
        __m256d m_Fx;
    };
    union {
        double Fy[VECWIDTH] = {};
        __m256d m_Fy;
    };
    union {
        double Fz[VECWIDTH] = {};
        __m256d m_Fz;
    };
    union {
        double r[VECWIDTH] = {};
        __m256d m_r;
    };
};

sf::Color HSV2RGB(sf::Color input)
{
    float h = input.r * 360.f / 256.f, s = input.g / 256.f, v = input.b / 256.f;
    float c = v * s;
    float val = h / 60.f;
    while (val > 2.0f)
        val -= 2.f;
    float x = c * (1.f - abs(val - 1.f));
    float m = v - c;
    std::array<std::array<float, 3>, 6> colorTable;
    colorTable[0] = { c,x,0 };
    colorTable[1] = { x,c,0 };
    colorTable[2] = { 0,c,x };
    colorTable[3] = { 0,x,c };
    colorTable[4] = { x,0,c };
    colorTable[5] = { c,0,x };
    std::array<float, 3> color = colorTable[int(h / 60)];
    return sf::Color(uint8_t((color[0] + m) * 256), uint8_t((color[1] + m) * 256), uint8_t((color[2] + m) * 256), input.a);
}

class SolarSystem {
    std::vector<PlanetGroup>        planets;
    std::vector<PlanetGroupExtra>   planetsExtra;
    double minz = -1;
    double maxz =  1;
public:
    int64_t planetsLength = 0;
    void AddPlanet(double x, double y, double z, double dx, double dy, double dz, double mass)
    {
        if (planetsLength % VECWIDTH == 0)
        {
            planets.emplace_back(PlanetGroup());
            planetsExtra.emplace_back(PlanetGroupExtra());
        }
        PlanetGroup*      group         = &planets[planetsLength / VECWIDTH];
        PlanetGroupExtra* groupExtra    = &planetsExtra[planetsLength / VECWIDTH];
        int64_t index = planetsLength % VECWIDTH;
        // Reserver id 0 for invalid planets
        group->id  [index] = int64_t(planetsLength + 1);
        group->x   [index] = x;
        group->y   [index] = y;
        group->z   [index] = z;
        group->mass[index] = mass;
        groupExtra->dx[index] = dx;
        groupExtra->dy[index] = dy;
        groupExtra->dz[index] = dz;
        groupExtra->Fx[index] = 0.f;
        groupExtra->Fy[index] = 0.f;
        groupExtra->Fz[index] = 0.f;
        groupExtra->r[index]  = sqrt(mass/pi) * 128.f;
        planetsLength++;
    }
    PlanetOption GetPlanet(int64_t id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup*      group      = &planets[id / VECWIDTH];
            PlanetGroupExtra* groupExtra = &planetsExtra[id / VECWIDTH];
            int64_t index = id % VECWIDTH;
            Planet planet = { 
                & group->id[index], 
                & group->x[index], 
                & group->y[index],
                & group->z[index],
                & group->mass[index],
                & groupExtra->dx[index],
                & groupExtra->dy[index],
                & groupExtra->dz[index],
                & groupExtra->Fx[index],
                & groupExtra->Fy[index],
                & groupExtra->Fz[index],
                & groupExtra->r[index]
            };
            // Check for uninitialized planet
            if (group->id[index] == 0)
                return PlanetOption();
            return PlanetOption{ true,planet };
        }
        return PlanetOption();
    }
    void RemovePlanet(int64_t id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* lastGroup = &planets[(planetsLength - 1) / VECWIDTH];
            PlanetGroupExtra* lastGroupExtra = &planetsExtra[(planetsLength - 1) / VECWIDTH];
            if (id == planetsLength)
                goto clear_planet;
            {
                PlanetGroup* group = &planets[id / VECWIDTH];
                PlanetGroupExtra* groupExtra = &planetsExtra[id / VECWIDTH];
                int64_t index = id % VECWIDTH;
                if (PlanetOption temp = GetPlanet(planetsLength))
                {
                    Planet lastPlanet = temp;
                    group->x[index] = *lastPlanet.x;
                    group->y[index] = *lastPlanet.y;
                    group->z[index] = *lastPlanet.z;
                    group->mass[index] = *lastPlanet.mass;
                    groupExtra->dx[index] = *lastPlanet.dx;
                    groupExtra->dy[index] = *lastPlanet.dy;
                    groupExtra->dz[index] = *lastPlanet.dz;
                    groupExtra->Fx[index] = *lastPlanet.Fx;
                    groupExtra->Fy[index] = *lastPlanet.Fy;
                    groupExtra->Fz[index] = *lastPlanet.Fz;
                    groupExtra->r[index] = *lastPlanet.r;
                }
            }
            // Don't forget to clear because simd will still do ops on these values
            clear_planet:;
            int64_t lastIndex = (planetsLength - 1) % VECWIDTH;
            lastGroup->id[lastIndex] = 0;
            lastGroup->x[lastIndex] = 0.f;
            lastGroup->y[lastIndex] = 0.f;
            lastGroup->z[lastIndex] = 0.f;
            lastGroup->mass[lastIndex] = 0.f;
            lastGroupExtra->dx[lastIndex] = 0.f;
            lastGroupExtra->dy[lastIndex] = 0.f;
            lastGroupExtra->dz[lastIndex] = 0.f;
            lastGroupExtra->Fx[lastIndex] = 0.f;
            lastGroupExtra->Fy[lastIndex] = 0.f;
            lastGroupExtra->Fz[lastIndex] = 0.f;
            lastGroupExtra->r[lastIndex] = 0.f;
            planetsLength--;
        }
    }
    void AddRandomSatellite(int64_t id) {
        if (PlanetOption temp = GetPlanet(id))
        {
            Planet parent = temp;
            const int       angles = (360 * 32);
            const int       steps = 1000;
            const double    min_dis = 5;
            const double    max_dis = 100;
            const double    dif_dis = max_dis - min_dis;
            const double    min_prop = 0.005;
            const double    max_prop = 0.05;
            const double    dif_prop = max_prop - min_prop;
            const int       mas_step = 1000;
            double new_mass = min_prop + double(rand() % mas_step) / mas_step * dif_prop * (*parent.mass);
            double angleA = double(rand() % angles) / double(angles);
            double angleB = double(rand() % angles) / double(angles);
            angleA *= 2.f * pi;
            angleB *= 2.f * pi;
            double magni = (*parent.r) + (*parent.mass) * (min_dis + double(rand() % steps) / double(steps) * dif_dis);
            double new_x = (*parent.x) + sin(angleA) * cos(angleB) * magni;
            double new_y = (*parent.y) + sin(angleB)               * magni;
            double new_z = (*parent.z) + cos(angleA) * cos(angleB) * magni;
            double new_dx;
            double new_dy;
            double new_dz;
            double scalar = sqrt(G*((*parent.mass) + 0.5 * children * new_mass));
            new_dx = (*parent.dx) + cos(angleA + pi * 0.5) * scalar;
            new_dy = (*parent.dy) + sin(angleA + pi * 0.5) * scalar;
            new_dz = (*parent.dz) + sin(angleB + pi * 0.5) * scalar;

            
            /*if (rand() & 1)
            {
                
            }
            else
            {
                new_dx = (*parent.dx) + cos(angle + pi * 0.5f) * scalar;
                new_dy = (*parent.dy) + sin(angle + pi * 0.5f) * scalar;
            }*/
            
            AddPlanet(new_x, new_y, new_z, new_dx, new_dy, new_dz, new_mass);
        }
    }
    // Parent, children per call, recursion depth
    void RecursivelyAddPlanets(int64_t parent, int64_t n, int64_t m)
    {
        std::vector<int64_t> parents;
        if (m == 0)
            return;
        for (int64_t i = 0; i < n; i++)
        {
            AddRandomSatellite(parent);
            parents.emplace_back(planetsLength);
        }
        for (int64_t i = 0; i < parents.size(); i++)
        {
            RecursivelyAddPlanets(parents[i], n, m - 1);
        }
    }
    SolarSystem()
    {
        AddPlanet(0, 0, 0, 0, 50, 0, 1000);
        AddPlanet(5000, 0, 0, 0, -50, 0, 1000);
    }
    void DrawSolarSystem()
    {
        for (int64_t i = 1; i <= planetsLength;i++)
        {
            if (PlanetOption temp = GetPlanet(i))
            {
                Planet planet = temp;
                const float scale = 1.f/10000.f;
                glPushMatrix();
                glTranslatef(*planet.x * scale, *planet.y * scale, *planet.z * scale);
                drawSphere(*planet.r * scale, 10, 10);
                glPopMatrix();
            }
        }
    }
    void MergePlanets(int64_t idA, int64_t idB)
    {
        if (idA > idB)
        {
            int64_t temp = idA;
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
                *planetA.z += (*planetB.z - *planetA.z) * relative_mass;
                *planetA.dx = ((*planetA.dx) * (*planetA.mass) + *planetB.dx * *planetB.mass) / total_mass;
                *planetA.dy = ((*planetA.dy) * (*planetA.mass) + *planetB.dy * *planetB.mass) / total_mass;
                *planetA.dz = ((*planetA.dz) * (*planetA.mass) + *planetB.dz * *planetB.mass) / total_mass;
                *planetA.mass = total_mass;
                *planetA.r = sqrt(total_mass / pi) * 128.f;
                if (idB == selectedPlanet)
                    selectedPlanet = idA;
                // Remove planet B
                RemovePlanet(idB);
            }
        }
    }
    void MergeAllPlanets()
    {
        for (int64_t i = 0; i < planets.size(); i++)
        {
            PlanetGroup*        groupA      = &planets[i];
            PlanetGroupExtra*   groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                if (i * VECWIDTH + j < planetsLength && groupA->id[j] != 0)
                {
                    for (int64_t k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup*        groupB      = &planets[k];
                        PlanetGroupExtra*   groupBExtra = &planetsExtra[k];
                        for (int64_t l = 0; l < VECWIDTH; l++)
                        {
                            if (k * VECWIDTH + l < planetsLength && groupB->id[l] != 0)
                            {
                                if (groupB->id[l] != 0 && groupA->id[j] != groupB->id[l])
                                {
                                    double rx = ((groupB->x[l]) - groupA->x[j]);
                                    double ry = ((groupB->y[l]) - groupA->y[j]);
                                    double rz = ((groupB->z[l]) - groupA->z[j]);
                                    double r2 = (rx * rx) + (ry * ry) + (rz * rz);
                                    // Check if planets merge
                                    double mass_r2 = ((groupAExtra->r[j] + groupBExtra->r[l]) * merge_r) * ((groupAExtra->r[j] + groupBExtra->r[l]) * merge_r);
                                    if (r2 < mass_r2)
                                    {
                                        MergePlanets(groupA->id[j], groupB->id[l]);
                                        j--;
                                        goto check_again;
                                    }
                                }
                            }
                        }
                    }
                }
            check_again:;
            }
        }
    }
    void ApplyForces()
    {
        // Separate applying forces and velocity since it's O(n)
        for (int64_t i = 0; i < planetsLength / VECWIDTH + 1; i++)
        {
            PlanetGroup* groupA = &planets[i];
            PlanetGroupExtra* groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    groupAExtra->dx[j] += groupAExtra->Fx[j] / groupA->mass[j];
                    groupAExtra->dy[j] += groupAExtra->Fy[j] / groupA->mass[j];
                    groupAExtra->dz[j] += groupAExtra->Fz[j] / groupA->mass[j];
                    groupA->x[j] += groupAExtra->dx[j];
                    groupA->y[j] += groupAExtra->dy[j];
                    groupA->z[j] += groupAExtra->dz[j];
                    if (groupA->z[j] < minz)
                        minz = groupA->z[j];
                    if (groupA->z[j] > maxz)
                        maxz = groupA->z[j];
                    groupAExtra->Fx[j] = 0.f;
                    groupAExtra->Fy[j] = 0.f;
                    groupAExtra->Fz[j] = 0.f;
                }
            }
        }
    }
    void Threaded(const int64_t start, const int64_t end)
    {
        // F = (G * m1 * m2) / r^2
        // F = ma
        // a = m / F;
        // a = m / ((G * m * m2) / r^2)
        for (int64_t i = start; i < end; i++)
        {
            PlanetGroup* groupA = &planets[i];
            PlanetGroupExtra* groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    for (int64_t k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup* groupB = &planets[k];
                        for (int64_t l = 0; l < VECWIDTH; l++)
                        {
                            if (groupB->id[l] != 0 && groupA->id[j] != groupB->id[l])
                            {
                                double rx = groupB->x[l] - groupA->x[j];
                                double ry = groupB->y[l] - groupA->y[j];
                                double rz = groupB->z[l] - groupA->z[j];
                                double r2 = (rx * rx) + (ry * ry) + (rz * rz);
                                double F = (G * (groupA->mass[j]) * groupB->mass[j]) / r2;
                                groupAExtra->Fx[j] += F * rx;
                                groupAExtra->Fy[j] += F * ry;
                                groupAExtra->Fz[j] += F * rz;
                            }
                        }
                    }
                }
            }
        }
    }
    void UpdatePlanets()
    {
        // F = (G * m1 * m2) / r^2
        // F = ma
        // a = m / F;
        // a = m / ((G * m * m2) / r^2)
        if(merging)
            MergeAllPlanets();
        Threaded(0, planets.size());
        ApplyForces();
    }
    
    void SimdThreaded(const int64_t start, const int64_t end)
    {
        for (int64_t i = start; i < end; i++)
        {
            PlanetGroup*        groupA      = &planets[i];
            PlanetGroupExtra*   groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                __m256d planetA_x       = _mm256_set1_pd(groupA->x[j]);
                __m256d planetA_y       = _mm256_set1_pd(groupA->y[j]);
                __m256d planetA_z       = _mm256_set1_pd(groupA->z[j]);
                __m256d planetA_mass    = _mm256_set1_pd(groupA->mass[j]);
                __m256i planetA_id      = _mm256_set1_epi64x(groupA->id[j]);
                union {
                    double  planetA_Fx[VECWIDTH];
                    __m256d mplanetA_Fx = _mm256_set1_pd(0);
                };
                union {
                    double  planetA_Fy[VECWIDTH];
                    __m256d mplanetA_Fy = _mm256_set1_pd(0);
                };
                union {
                    double  planetA_Fz[VECWIDTH];
                    __m256d mplanetA_Fz = _mm256_set1_pd(0);
                };
                for (int64_t k = 0; k < planets.size(); k++)
                {
                    PlanetGroup* groupB = &planets[k];
                    // Subtract planet As position from groups positions to find relative distance
                    // Find the square of each distance
                    // Code readibility may suffer due to functions not being optimized such that
                    // Simd vectors aren't being stored in registers properly and may be passed to cache or stack preemtively
                    __m256d rx   = _mm256_sub_pd(groupB->m_x, planetA_x);
                    __m256d rx2  = _mm256_mul_pd(rx, rx);
                    __m256d ry   = _mm256_sub_pd(groupB->m_y, planetA_y);
                    __m256d ry2  = _mm256_mul_pd(ry, ry);
                    __m256d rz   = _mm256_sub_pd(groupB->m_z, planetA_z);
                    __m256d rz2  = _mm256_mul_pd(rz, rz);
                    // Find the radius squared
                    __m256d r2   = _mm256_add_pd(_mm256_add_pd(rx2, ry2), rz2);
                    // Calculate gravity
                    __m256d mass  = _mm256_mul_pd(groupB->m_mass, planetA_mass);
                    __m256d gm    = _mm256_mul_pd(mass, gravity);
                    // Find the forces for each dimension
                    __m256d F = _mm256_div_pd(gm, r2);
                    union {
                        double F_x[VECWIDTH];
                        __m256d Fx;
                    };
                    Fx = _mm256_mul_pd(F, rx);
                    union {
                        double F_y[VECWIDTH];
                        __m256d Fy;
                    };
                    Fy = _mm256_mul_pd(F, ry);
                    union {
                        double F_z[VECWIDTH];
                        __m256d Fz;
                    };
                    Fz = _mm256_mul_pd(F, rz);

                    // Remove nan values such as planets affecting themselves
                    // If id == 0
                    __m256i zeromask    = _mm256_cmpeq_epi64(groupB->m_id, m_zeroi);
                    __m256i idmask      = _mm256_cmpeq_epi64(groupB->m_id, planetA_id);
                    // If groupA.id == groupB.id
                    __m256i bothmask    = _mm256_or_si256(zeromask, idmask);
                    bothmask            = _mm256_xor_si256(bothmask, m_onesi);
                    Fx = _mm256_and_pd(Fx, _mm256_castsi256_pd(bothmask));
                    Fy = _mm256_and_pd(Fy, _mm256_castsi256_pd(bothmask));
                    Fz = _mm256_and_pd(Fz, _mm256_castsi256_pd(bothmask));

                    mplanetA_Fx = _mm256_add_pd(mplanetA_Fx, Fx);
                    mplanetA_Fy = _mm256_add_pd(mplanetA_Fy, Fy);
                    mplanetA_Fz = _mm256_add_pd(mplanetA_Fz, Fz);
                }
                for (int64_t l = 0; l < VECWIDTH; l++)
                {
                    groupAExtra->Fx[j] += planetA_Fx[l];
                    groupAExtra->Fy[j] += planetA_Fy[l];
                    groupAExtra->Fz[j] += planetA_Fz[l];
                }
            }
        }
    }
    void ThreadedUpdatePlanets()
    {
        // I don't think this can be multithreaded since it relies on removing elements being thread safe... 
        // which is obvious why it wouldn't be
        if(merging)
            MergeAllPlanets();

        // Seems like threads don't like to be moved or recreated
        std::vector<std::thread> threads;
        const int64_t NUM_THREADS = 4;
        if (planetsLength < 32)
            UpdatePlanetsSimd();
        else
        {
            int64_t block_size = (planetsLength / VECWIDTH) / NUM_THREADS + 1;
            for (int64_t i = 0; i < NUM_THREADS; i++)
            {
                if (simd)
                {
                    threads.emplace_back(std::thread([&](SolarSystem* system, const int64_t start, const int64_t end) {
                        system->SimdThreaded(start, end);
                        }, this, i * block_size, (i + 1) * block_size));
                }
                else
                {
                    threads.emplace_back(std::thread([&](SolarSystem* system, const int64_t start, const int64_t end) {
                        system->Threaded(start, end);
                        }, this, i * block_size, (i + 1) * block_size));
                }
            }
            for (int64_t i = 0; i < NUM_THREADS; i++)
            {
                threads[i].join();
            }
            ApplyForces();
        }
    }
    void UpdatePlanetsSimd()
    {
        if(merging)
            MergeAllPlanets();
        SimdThreaded(0, planets.size());
        ApplyForces();
    }
    void CheckPlanets()
    {
        // Separate applying forces and velocity since it's O(n)
        for (int64_t i = 0; i < planetsLength / VECWIDTH + 1; i++)
        {
            PlanetGroup* groupA = &planets[i];
            PlanetGroupExtra* groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id[j] != 0)
                {
                    if (groupA->z[j] != 0)
                        bool hello = true;
                }
            }
        }
    }
};

void glLoadMatrixf(glm::mat4 matrix)
{
    float* fM;
    fM = glm::value_ptr(matrix);
    glLoadMatrixf(fM);
}

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

    circle.loadFromFile("circle.png");

    sf::View centreView;
    sf::Vector2u windowSize = window.getSize();
    sf::Vector2i windowMiddle = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);
    bool cursorGrabbed = true;
    centreView.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    centreView.setCenter(0, 0);
    float prevZoom = 1.f;
    float zoom = 1.f;
    sf::Vector2f prevCamPos = { 0.f,0.f };
    sf::Vector2f mousePos = { 0.f,0.f };
    uint8_t KEYW = 1;
    uint8_t KEYS = 2;
    uint8_t KEYA = 4;
    uint8_t KEYD = 8;
    uint8_t pressed = 0;
    
    // Opengl stuff -------------------------------
    
    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraPitch = 0.f;
    float cameraYaw = 0.f;

    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraDirection = glm::normalize(cameraPos - cameraTarget);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraRight = glm::normalize(glm::cross(up, cameraDirection));
    cameraUp = glm::cross(cameraDirection, cameraRight);

    double frustRight = 1;
    double frustUp = 1;
    double nearClip = 0.1f;
    double farClip = 3.f;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-frustRight, frustRight, -frustUp, frustUp, nearClip, farClip);
    glMatrixMode(GL_MODELVIEW);
    window.pushGLStates();

    // --------------------------------------------
    
    srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));
    SolarSystem system;
    system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
    system.MergeAllPlanets();

    bool running = true;
    while (running) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::KeyPressed)
            {
                const float cameraSpeed = 0.05f; // adjust accordingly
                switch (event.key.code)
                {
                case sf::Keyboard::E: {
                    cursorGrabbed = !cursorGrabbed;
                }break;
                case sf::Keyboard::W: {
                    cameraPos += cameraSpeed * cameraFront;
                    pressed |= KEYW;
                }break;
                case sf::Keyboard::S: {
                    cameraPos -= cameraSpeed * cameraFront;
                    pressed |= KEYS;
                }break;
                case sf::Keyboard::A: {
                    cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
                    pressed |= KEYA;
                }break;
                case sf::Keyboard::D: {
                    cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
                    pressed |= KEYD;
                }break;
                }
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
            else if (event.type == sf::Event::MouseMoved)
            {
                if (cursorGrabbed)
                {
                    sf::Mouse::setPosition(sf::Vector2i(windowMiddle.x, windowMiddle.y), window);
                    if (event.mouseMove.x != windowMiddle.x || event.mouseMove.y != windowMiddle.y)
                    {
                        const float sensitivity = 0.1f;

                        cameraYaw += (event.mouseMove.x - windowMiddle.x) * sensitivity;
                        cameraPitch += (event.mouseMove.y - windowMiddle.y) * -sensitivity;

                        if (cameraYaw > 180.f)
                            cameraYaw = -180.f;
                        if (cameraYaw < -180.f)
                            cameraYaw = 180.f;
                        if (cameraPitch > 89.0f)
                            cameraPitch = 89.0f;
                        if (cameraPitch < -89.0f)
                            cameraPitch = -89.0f;

                        glm::vec3 direction;
                        direction.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
                        direction.y = sin(glm::radians(cameraPitch));
                        direction.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
                        cameraFront = glm::normalize(direction);
                    }
                }
                mousePos = sf::Vector2f(event.mouseMove.x - float(windowMiddle.x), event.mouseMove.y - float(windowMiddle.y));
            }
            else if (event.type == sf::Event::Resized)
            {
                windowSize = window.getSize();
                windowMiddle = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);
                centreView.setSize(sf::Vector2f(windowSize.x, windowSize.y));
                centreView.setCenter(0, 0);
                double ratio = double(windowSize.x) / double(windowSize.y);
                window.popGLStates();
                glViewport(0, 0, windowSize.x, windowSize.y);
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                glFrustum(-frustRight * ratio, frustRight * ratio, -frustUp, frustUp, nearClip, farClip);
                glMatrixMode(GL_MODELVIEW);
                window.pushGLStates();
                prevZoom = 1.f;
                zoom = 1.f;
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
                running = false;
            }
        }
        float frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
        const float move_speed = 50.f;
        for (int64_t i = 0; i < 4; i++)
        {
            if ((pressed >> i) & 1)
            {
                if (i < 2)
                    camPos.y += move_speed * zoom * ((i & 1) ? 1.f : -1.f) / (frameRate / 60);
                else
                    camPos.x += move_speed * zoom * ((i & 1) ? 1.f : -1.f) / (frameRate / 60);
            }
        }
        if (prevZoom != zoom)
        {
            if (zoom < prevZoom)
                camPos += (mousePos * prevZoom * 2.f * (prevZoom/zoom));
            centreView.zoom(pow(zoom / prevZoom,2));
            prevZoom = zoom;
        }
        if (prevCamPos != camPos)
        {
            centreView.move(camPos - prevCamPos);
            prevCamPos = camPos;
        }
        window.clear();
        ImGui::SFML::Update(window, deltaClock.restart());
        window.setView(centreView);

        ImGui::Begin("Update Rate");
        
        static int updatesPerFrame = 1;
        ImGui::Text(std::string("FPS:     "+std::to_string_with_precision(frameRate)).c_str());
        ImGui::Text(std::string("UPS:     " + std::to_string_with_precision(frameRate * float(updatesPerFrame))).c_str());
        std::string operations = std::to_string_with_precision(frameRate * float(updatesPerFrame * system.planetsLength * system.planetsLength));
        int64_t outer = 0;
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
        ImGui::Checkbox(" :Run Simulation", &runSimulation);
        ImGui::Text(std::string("POS : " + std::to_string_with_precision(camPos.x) + ", " + std::to_string_with_precision(camPos.y)).c_str());
        ImGui::Text(std::string("MPOS: " + std::to_string_with_precision(mousePos.x) + ", " + std::to_string_with_precision(mousePos.y)).c_str());
        ImGui::Text(std::string("ZOOM: " + std::to_string_with_precision(zoom)).c_str());
        ImGui::End();
        frameClock.restart();

        ImGui::Begin("Modify Planet");
        ImGui::SliderInt(": ID", (int*)&selectedPlanet, 1, system.planetsLength);
        if (PlanetOption temp = system.GetPlanet(selectedPlanet))
        {
            float smin = -1000;
            float smax = 1000;
            float dmin = -1.0;
            float dmax = 1.0;
            Planet planet = temp;
            float px    = float(*planet.x);
            float py    = float(*planet.y);
            float pz    = float(*planet.z);
            float pdx   = float(*planet.dx);
            float pdy   = float(*planet.dy);
            float pdz   = float(*planet.dz);
            float pm    = float(*planet.mass);
            ImGui::SliderFloat(": X",  &px,    smin, smax);
            ImGui::SliderFloat(": Y",  &py,    smin, smax);
            ImGui::SliderFloat(": Z",  &pz,    smin, smax);
            ImGui::SliderFloat(": DX", &pdx,   dmin, dmax);
            ImGui::SliderFloat(": DY", &pdy,   dmin, dmax);
            ImGui::SliderFloat(": DZ", &pdz,   dmin, dmax);
            ImGui::SliderFloat(": M",  &pm, 0.0001, 10,"%.3f",2.f);
            if (ImGui::SliderFloat(": G", &G, 0.1, 10))
            {
                gravity = _mm256_set1_pd(G);
            }
            ImGui::SliderInt(": Tiers",    (int*)&tiers,    1, 4);
            ImGui::SliderInt(": Children", (int*)&children, 1, 10);
            ImGui::Text(std::string("Settings will add " + std::to_string(int(std::pow(children, tiers))) + " planets.").c_str());
            if (ImGui::Button("Remove Planet"))
            {
                system.RemovePlanet(selectedPlanet);
            }
            if (ImGui::Button("Add Planet"))
            {
                system.AddPlanet(smin, smin, 0, 0, 0, 0, 0.5);
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
        ImGui::End();

        if (!ImGui::IsAnyWindowFocused())
        {
            if (runSimulation)
            {
                if (static_framerate)
                {
                    if (updatesPerFrame > 1 && frameRate < 50.f)
                        updatesPerFrame--;
                    else if (updatesPerFrame < 50 && frameRate > 55.f)
                        updatesPerFrame++;
                }
                if (!multi_threaded)
                    for (int64_t i = 0; i < updatesPerFrame; i++)
                    {
                        if (simd)
                            system.UpdatePlanetsSimd();
                        else
                            system.UpdatePlanets();
                    }
                else
                    for (int64_t i = 0; i < updatesPerFrame; i++)
                        system.ThreadedUpdatePlanets();
            }
        }

        //ImGui::ShowTestWindow();

        window.popGLStates();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadMatrixf(glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp));

        // Draw a white grid "floor" for the tetrahedron to sit on.
        glColor3f(1.0, 1.0, 1.0);
        glBegin(GL_LINES);
        const float lines = 50;
        const float gap   = 0.5;
        float dist = lines * gap * 0.5;
        for (GLfloat i = -dist; i <= dist; i += gap) {
            glVertex3f(i, 0, dist); glVertex3f(i, 0, -dist);
            glVertex3f(dist, 0, i); glVertex3f(-dist, 0, i);
        }
        glEnd();

        system.DrawSolarSystem();

        glFlush();

        window.pushGLStates();
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
