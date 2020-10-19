#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <immintrin.h>
#include <sstream>
#include <thread>
#include <array>
#include <queue>

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
    template <typename T>
    string to_string_with_formatting(const T a_value, const int64_t n = 2)
    {
        std::string operations = std::to_string_with_precision(a_value);
        int64_t outer = 0;
        bool decimal = false;
        for (auto iter = operations.end(); iter > operations.begin(); iter--)
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
        return operations;
    }
}

int NUM_THREADS = std::thread::hardware_concurrency();
const int64_t VECWIDTH = 4;
const double pi = 2 * acos(0.0);
float G = 4.f;
const float zero = 0.f;
const float merge_r = 0.5f;

__m256i m_zeroi  = _mm256_set1_epi64x(0);
__m256i m_onesi  = _mm256_set1_epi64x(uint64_t(-1));
__m256d gravity  = _mm256_set1_pd(G);

int64_t selectedPlanet = 1;
bool merging = false;
bool multi_threaded = true;
bool simd = true;
int forward_calculation = 2000;
int tick_spacing = 10;
sf::Vector2f camPos = { 0.f,0.f };

int64_t tiers = 1;
int64_t children = 40;
bool static_framerate = false;
bool runSimulation = true;
bool drawPlane = true;
int global_tick = 0;

// Basic cube primitive for drawing planet paths
const std::vector<std::array<float, 3>> box_vert = { {
    {-1.f,-1.f,-1.f},
    {-1.f,1.f,-1.f},
    {-1.f,1.f,1.f},
    {-1.f,-1.f,1.f},
    {1.f,-1.f,1.f},
    {1.f,1.f,1.f},
    {1.f,1.f,-1.f},
    {1.f,-1.f,-1.f}
} };

const std::vector<std::array<int, 3>> box_face = { {
    {0,1,2},
    {2,3,0},
    {5,4,3},
    {3,2,5},
    {4,5,6},
    {6,7,4},
    {1,0,7},
    {7,6,1},
    {0,3,4},
    {4,7,0},
    {6,5,2},
    {2,1,6}
} };

const std::vector<std::array<float, 3>> box_array = { {
    {-1.f,-1.f,-1.f},   // 0
    {-1.f,1.f,-1.f},    // 1
    {-1.f,1.f,1.f},     // 2

    {-1.f,1.f,1.f},     // 2
    {-1.f,-1.f,1.f},    // 3
    {-1.f,-1.f,-1.f},   // 0

    {1.f,1.f,1.f},      // 5
    {1.f,-1.f,1.f},     // 4
    {-1.f,-1.f,1.f},    // 3

    {-1.f,-1.f,1.f},    // 3
    {-1.f,1.f,1.f},     // 2
    {1.f,1.f,1.f},      // 5

    {1.f,-1.f,1.f},     // 4
    {1.f,1.f,1.f},      // 5
    {1.f,1.f,-1.f},     // 6

    {1.f,1.f,-1.f},     // 6
    {1.f,-1.f,-1.f},    // 7
    {1.f,-1.f,1.f},     // 4

    {-1.f,1.f,-1.f},    // 1
    {-1.f,-1.f,-1.f},   // 0
    {1.f,-1.f,-1.f},    // 7

    {1.f,-1.f,-1.f},    // 7
    {1.f,1.f,-1.f},     // 6
    {-1.f,1.f,-1.f},    // 1

    {-1.f,-1.f,-1.f},   // 0
    {-1.f,-1.f,1.f},    // 3
    {1.f,-1.f,1.f},     // 4
    
    {1.f,-1.f,1.f},     // 4
    {1.f,-1.f,-1.f},    // 7
    {-1.f,-1.f,-1.f},   // 0

    {1.f,1.f,-1.f},     // 6
    {1.f,1.f,1.f},      // 5
    {-1.f,1.f,1.f},     // 2

    {-1.f,1.f,1.f},     // 2
    {-1.f,1.f,-1.f},    // 1
    {1.f,1.f,-1.f},     // 6
} };

void drawWireFrame(const std::vector<std::array<float, 3>> vert, const std::vector<std::array<int, 3>>& tri, const float scale = 1.f)
{
    glBegin(GL_LINES);
    for (int i = 0; i < tri.size(); i++)
    {
        glVertex3f(vert[tri[i][0]][0] * scale, vert[tri[i][0]][1] * scale, vert[tri[i][0]][2] * scale);
        glVertex3f(vert[tri[i][1]][0] * scale, vert[tri[i][1]][1] * scale, vert[tri[i][1]][2] * scale);
        glVertex3f(vert[tri[i][1]][0] * scale, vert[tri[i][1]][1] * scale, vert[tri[i][1]][2] * scale);
        glVertex3f(vert[tri[i][2]][0] * scale, vert[tri[i][2]][1] * scale, vert[tri[i][2]][2] * scale);
        glVertex3f(vert[tri[i][2]][0] * scale, vert[tri[i][2]][1] * scale, vert[tri[i][2]][2] * scale);
        glVertex3f(vert[tri[i][0]][0] * scale, vert[tri[i][0]][1] * scale, vert[tri[i][0]][2] * scale);
    }
    glEnd();
}

void drawFilled(const std::vector<std::array<float, 3>>& vert, const std::vector<std::array<int, 3>>& tri, const float scale = 1.f)
{
    glBegin(GL_TRIANGLES);
    for (int i = 0; i < tri.size(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            glNormal3f(vert[tri[i][j]][0], vert[tri[i][j]][1], vert[tri[i][j]][2]);
            glVertex3f(vert[tri[i][j]][0] * scale, vert[tri[i][j]][1] * scale, vert[tri[i][j]][2] * scale);
        }
    }
    glEnd();
}

void drawSphere() {
    int lats = 10, longs = 10;
    int i = 10, j = 0;
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
            glVertex3f(x * zr0, y * zr0, z0);
            glNormal3f(x * zr1, y * zr1, z1);
            glVertex3f(x * zr1, y * zr1, z1);
        }
        glEnd();
    }
}

struct PlanetObject {
    int64_t id;
    double x;
    double y;
    double z;
    double mass;
    double r;
    double dx;
    double dy;
    double dz;
    double Fx;
    double Fy;
    double Fz;
};

struct PlanetReference {
    int64_t* id;
    double* x;
    double* y;
    double* z;
    double* mass;
    double* r;
    double* dx;
    double* dy;
    double* dz;
    double* Fx;
    double* Fy;
    double* Fz;
};

struct PlanetOption {
    bool valid = false;
    PlanetReference planet;
    operator bool() { return valid; }
    operator PlanetReference() { return planet; }
};

struct PlanetGroup {
    __m256i id;
    __m256d x;
    __m256d y;
    __m256d z;
    __m256d mass;
    __m256d r;
};

struct PlanetGroupExtra {
    __m256d dx;
    __m256d dy;
    __m256d dz;
    __m256d Fx;
    __m256d Fy;
    __m256d Fz;
};

class SolarSystem {
public:
    std::vector<PlanetGroup>        planets;
    std::vector<PlanetGroupExtra>   planetsExtra;
    int tick;
    int64_t planetsLength = 0;
    void AddPlanet(double x, double y, double z, double dx, double dy, double dz, double mass, double Fx=0.f, double Fy=0.f, double Fz=0.f)
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
        group->id.m256i_i64[index] = int64_t(planetsLength + 1);
        group->x.m256d_f64[index] = x;
        group->y.m256d_f64[index] = y;
        group->z.m256d_f64[index] = z;
        group->mass.m256d_f64[index] = mass;
        group->r.m256d_f64[index] = sqrt(mass / pi) * 128.f;
        groupExtra->dx.m256d_f64[index] = dx;
        groupExtra->dy.m256d_f64[index] = dy;
        groupExtra->dz.m256d_f64[index] = dz;
        groupExtra->Fx.m256d_f64[index] = Fx;
        groupExtra->Fy.m256d_f64[index] = Fy;
        groupExtra->Fz.m256d_f64[index] = Fz;
        planetsLength++;
    }
    void AddPlanet(PlanetObject planet)
    {
        AddPlanet(planet.x, planet.y, planet.z, planet.dx, planet.dy, planet.dz, planet.mass, planet.Fx, planet.Fy, planet.Fz);
    }
    SolarSystem(int tick)
    {
        this->tick = tick;
        if (!tick)
        {
            AddPlanet(0, 0, 0, 0, 50, 0, 1000);
            AddPlanet(5000, 0, 0, 0, -50, 0, 1000);
        }
    }
    PlanetObject GetPlanetObject(int64_t id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* group = &planets[id / VECWIDTH];
            PlanetGroupExtra* groupExtra = &planetsExtra[id / VECWIDTH];
            int64_t index = id % VECWIDTH;
            PlanetObject new_planet = {
                group->id.m256i_i64[index],
                group->x.m256d_f64[index],
                group->y.m256d_f64[index],
                group->z.m256d_f64[index],
                group->mass.m256d_f64[index],
                group->r.m256d_f64[index],
                groupExtra->dx.m256d_f64[index],
                groupExtra->dy.m256d_f64[index],
                groupExtra->dz.m256d_f64[index],
                groupExtra->Fx.m256d_f64[index],
                groupExtra->Fy.m256d_f64[index],
                groupExtra->Fz.m256d_f64[index]
            };
            return new_planet;
        }
    }
    PlanetOption GetPlanetReference(int64_t id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetReference planet;
            if (planetsExtra.size() > 0)
            {
                PlanetGroup* group = &planets[id / VECWIDTH];
                PlanetGroupExtra* groupExtra = &planetsExtra[id / VECWIDTH];
                int64_t index = id % VECWIDTH;
                planet = PlanetReference{
                    &group->id.m256i_i64[index],
                    &group->x.m256d_f64[index],
                    &group->y.m256d_f64[index],
                    &group->z.m256d_f64[index],
                    &group->mass.m256d_f64[index],
                    &group->r.m256d_f64[index],
                    &groupExtra->dx.m256d_f64[index],
                    &groupExtra->dy.m256d_f64[index],
                    &groupExtra->dz.m256d_f64[index],
                    &groupExtra->Fx.m256d_f64[index],
                    &groupExtra->Fy.m256d_f64[index],
                    &groupExtra->Fz.m256d_f64[index],
                };
                // Check for uninitialized planet
                if (group->id.m256i_i64[index] == 0)
                    return PlanetOption();
            }
            else
            {
                PlanetGroup* group = &planets[id / VECWIDTH];
                int64_t index = id % VECWIDTH;
                planet = PlanetReference{
                    &group->id.m256i_i64[index],
                    &group->x.m256d_f64[index],
                    &group->y.m256d_f64[index],
                    &group->z.m256d_f64[index],
                    &group->mass.m256d_f64[index],
                    &group->r.m256d_f64[index],
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                };
                // Check for uninitialized planet
                if (group->id.m256i_i64[index] == 0)
                    return PlanetOption();
            }
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
                if (PlanetOption temp = GetPlanetReference(planetsLength))
                {
                    PlanetReference lastPlanet = temp;
                    group->x.m256d_f64[index] = *lastPlanet.x;
                    group->y.m256d_f64[index] = *lastPlanet.y;
                    group->z.m256d_f64[index] = *lastPlanet.z;
                    group->mass.m256d_f64[index] = *lastPlanet.mass;
                    group->r.m256d_f64[index] = *lastPlanet.r;
                    groupExtra->dx.m256d_f64[index] = *lastPlanet.dx;
                    groupExtra->dy.m256d_f64[index] = *lastPlanet.dy;
                    groupExtra->dz.m256d_f64[index] = *lastPlanet.dz;
                    groupExtra->Fx.m256d_f64[index] = *lastPlanet.Fx;
                    groupExtra->Fy.m256d_f64[index] = *lastPlanet.Fy;
                    groupExtra->Fz.m256d_f64[index] = *lastPlanet.Fz;
                }
            }
            // Don't forget to clear because simd will still do ops on these values
            clear_planet:;
            int64_t lastIndex = (planetsLength - 1) % VECWIDTH;
            lastGroup->id.m256i_i64[lastIndex] = 0;
            lastGroup->x.m256d_f64[lastIndex] = 0.f;
            lastGroup->y.m256d_f64[lastIndex] = 0.f;
            lastGroup->z.m256d_f64[lastIndex] = 0.f;
            lastGroup->mass.m256d_f64[lastIndex] = 0.f;
            lastGroup->r.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->dx.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->dy.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->dz.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->Fx.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->Fy.m256d_f64[lastIndex] = 0.f;
            lastGroupExtra->Fz.m256d_f64[lastIndex] = 0.f;
            planetsLength--;
        }
    }
    void AddRandomSatellite(int64_t id) {
        if (PlanetOption temp = GetPlanetReference(id))
        {
            PlanetReference parent = temp;
            // Find two angles to determine the relative polar vector from the parent
            const int       angles = (360 * 32);
            double angleA = (double(rand() % angles) / double(angles)) * 2.f * pi;
            double angleB = (double(rand() % angles) / double(angles)) * 2.f * pi;

            // Vector Magnitude
            const int       steps = 1000;
            const double    min_dis = 5;
            const double    max_dis = 100;
            const double    dif_dis = max_dis - min_dis;
            double magni = (*parent.r) + (*parent.mass) * (min_dis + double(rand() % steps) / double(steps) * dif_dis);
            double new_x = (*parent.x) + sin(angleA) * cos(angleB) * magni;
            double new_y = (*parent.y) + sin(angleB)               * magni;
            double new_z = (*parent.z) + cos(angleA) * cos(angleB) * magni;

            // Calculate Mass
            const double    min_prop = 0.005;
            const double    max_prop = 0.05;
            const double    dif_prop = max_prop - min_prop;
            const int       mas_step = 1000;
            double new_mass = min_prop + double(rand() % mas_step) / mas_step * dif_prop * (*parent.mass);

            // Calculate Velocity
            double scalar = sqrt(G*((*parent.mass) + 0.5 * children * new_mass));
            double new_dx = (*parent.dx) + cos(angleA + pi * 0.5) * scalar;
            double new_dy = (*parent.dy) + sin(angleA + pi * 0.5) * scalar;
            double new_dz = (*parent.dz) + sin(angleB + pi * 0.5) * scalar;
            
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
    void MergePlanets(int64_t idA, int64_t idB)
    {
        if (idA > idB)
        {
            int64_t temp = idA;
            idA = idB;
            idB = temp;
        }
        assert(idA != idB);
        if (PlanetOption tempA = GetPlanetReference(idA))
        {
            if (PlanetOption tempB = GetPlanetReference(idB))
            {
                PlanetReference planetA = tempA;
                PlanetReference planetB = tempB;
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
                if (i * VECWIDTH + j < planetsLength && groupA->id.m256i_i64[j] != 0)
                {
                    for (int64_t k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup*        groupB      = &planets[k];
                        PlanetGroupExtra*   groupBExtra = &planetsExtra[k];
                        for (int64_t l = 0; l < VECWIDTH; l++)
                        {
                            if (k * VECWIDTH + l < planetsLength && groupB->id.m256i_i64[l] != 0)
                            {
                                if (groupB->id.m256i_i64[l] != 0 && groupA->id.m256i_i64[j] != groupB->id.m256i_i64[l])
                                {
                                    double rx = ((groupB->x.m256d_f64[l]) - groupA->x.m256d_f64[j]);
                                    double ry = ((groupB->y.m256d_f64[l]) - groupA->y.m256d_f64[j]);
                                    double rz = ((groupB->z.m256d_f64[l]) - groupA->z.m256d_f64[j]);
                                    double r2 = (rx * rx) + (ry * ry) + (rz * rz);
                                    // Check if planets merge
                                    double mass_r2 = ((groupA->r.m256d_f64[j] + groupB->r.m256d_f64[l]) * merge_r) * ((groupA->r.m256d_f64[j] + groupB->r.m256d_f64[l]) * merge_r);
                                    if (r2 < mass_r2)
                                    {
                                        MergePlanets(groupA->id.m256i_i64[j], groupB->id.m256i_i64[l]);
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
    void SimdThreaded(const int64_t start, int64_t end)
    {
        if (end > planets.size())
            end = planets.size();
        for (int64_t i = start; i < end; i++)
        {
            PlanetGroup*        groupA      = &planets[i];
            PlanetGroupExtra*   groupAExtra = &planetsExtra[i];
            for (int64_t j = 0; j < VECWIDTH; j++)
            {
                // Skip unintialized planets
                if (groupA->id.m256i_i64[j] != 0)
                {
                    // Get position, mass and id of current planet
                    __m256d planetA_x = _mm256_set1_pd(groupA->x.m256d_f64[j]);
                    __m256d planetA_y = _mm256_set1_pd(groupA->y.m256d_f64[j]);
                    __m256d planetA_z = _mm256_set1_pd(groupA->z.m256d_f64[j]);
                    __m256d planetA_mass = _mm256_set1_pd(groupA->mass.m256d_f64[j]);
                    __m256i planetA_id = _mm256_set1_epi64x(groupA->id.m256i_i64[j]);
                    // Create variables to accumulate forces
                    __m256d mplanetA_Fx = _mm256_set1_pd(0);
                    __m256d mplanetA_Fy = _mm256_set1_pd(0);
                    __m256d mplanetA_Fz = _mm256_set1_pd(0);
                    for (int64_t k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup* groupB = &planets[k];
                        // What does simd do?
                        //   xxxx
                        // + yyyy
                        // = zzzz
                        // Subtract planet As position from groups positions to find relative distance
                        __m256d rx = _mm256_sub_pd(groupB->x, planetA_x);
                        __m256d ry = _mm256_sub_pd(groupB->y, planetA_y);
                        __m256d rz = _mm256_sub_pd(groupB->z, planetA_z);
                        // Find the square of each distance
                        __m256d rx2 = _mm256_mul_pd(rx, rx);
                        __m256d ry2 = _mm256_mul_pd(ry, ry);
                        __m256d rz2 = _mm256_mul_pd(rz, rz);
                        // Find the radius squared
                        __m256d r2 = _mm256_add_pd(_mm256_add_pd(rx2, ry2), rz2);
                        // Calculate gravitational force
                        __m256d mass = _mm256_mul_pd(groupB->mass, planetA_mass);
                        __m256d gm = _mm256_mul_pd(mass, gravity);
                        __m256d F = _mm256_div_pd(gm, r2);
                        // Find the forces for each dimension
                        __m256d Fx = _mm256_mul_pd(F, rx);
                        __m256d Fy = _mm256_mul_pd(F, ry);
                        __m256d Fz = _mm256_mul_pd(F, rz);

                        // Remove nan values such as planets affecting themselves
                        // If id == 0
                        __m256i zeromask = _mm256_cmpeq_epi64(groupB->id, m_zeroi);
                        __m256i idmask = _mm256_cmpeq_epi64(groupB->id, planetA_id);
                        // If groupA.id == groupB.id
                        __m256i bothmask = _mm256_or_si256(zeromask, idmask);
                        bothmask = _mm256_xor_si256(bothmask, m_onesi);
                        Fx = _mm256_and_pd(Fx, _mm256_castsi256_pd(bothmask));
                        Fy = _mm256_and_pd(Fy, _mm256_castsi256_pd(bothmask));
                        Fz = _mm256_and_pd(Fz, _mm256_castsi256_pd(bothmask));
                        // Accumulate forces
                        mplanetA_Fx = _mm256_add_pd(mplanetA_Fx, Fx);
                        mplanetA_Fy = _mm256_add_pd(mplanetA_Fy, Fy);
                        mplanetA_Fz = _mm256_add_pd(mplanetA_Fz, Fz);
                    }
                    // Flatten 4 wide force accumulator onto single planet
                    for (int64_t l = 0; l < VECWIDTH; l++)
                    {
                        groupAExtra->Fx.m256d_f64[j] += mplanetA_Fx.m256d_f64[l];
                        groupAExtra->Fy.m256d_f64[j] += mplanetA_Fy.m256d_f64[l];
                        groupAExtra->Fz.m256d_f64[j] += mplanetA_Fz.m256d_f64[l];
                    }
                }
            }
        }
    }
    void ApplyForces()
    {
        if (planetsLength)
        {
            // Separate applying forces and velocity since it's O(n)
            for (int64_t i = 0; i < planetsLength / VECWIDTH + 1; i++)
            {
                PlanetGroup* groupA = &planets[i];
                PlanetGroupExtra* groupAExtra = &planetsExtra[i];
                for (int64_t j = 0; j < VECWIDTH; j++)
                {
                    if (groupA->id.m256i_i64[j] != 0)
                    {
                        // Add force to velocity (acceleration)
                        groupAExtra->dx.m256d_f64[j] += groupAExtra->Fx.m256d_f64[j] / groupA->mass.m256d_f64[j];
                        groupAExtra->dy.m256d_f64[j] += groupAExtra->Fy.m256d_f64[j] / groupA->mass.m256d_f64[j];
                        groupAExtra->dz.m256d_f64[j] += groupAExtra->Fz.m256d_f64[j] / groupA->mass.m256d_f64[j];
                        // Add velocity to position
                        groupA->x.m256d_f64[j] += groupAExtra->dx.m256d_f64[j];
                        groupA->y.m256d_f64[j] += groupAExtra->dy.m256d_f64[j];
                        groupA->z.m256d_f64[j] += groupAExtra->dz.m256d_f64[j];
                        // Set force to 0 for recalculation
                        groupAExtra->Fx.m256d_f64[j] = 0.f;
                        groupAExtra->Fy.m256d_f64[j] = 0.f;
                        groupAExtra->Fz.m256d_f64[j] = 0.f;
                    }
                }
            }
        }
    }
    void ThreadedUpdatePlanets(bool apply_forces)
    {
        // I don't think this can be multithreaded since it relies on removing elements being thread safe... 
        // which is obvious why it wouldn't be
        if(merging)
            MergeAllPlanets();        
        std::vector<std::thread> threads;
        int64_t block_size = (planetsLength / VECWIDTH) / NUM_THREADS + 1;
        for (int64_t i = 0; i < NUM_THREADS; i++)
        {
            if (simd)
            {
                threads.emplace_back(std::thread([&](SolarSystem* system, const int64_t start, const int64_t end) {
                    system->SimdThreaded(start, end);
                    }, this, i * block_size, (i + 1) * block_size));
            }
            //else
            //{
            //    threads.emplace_back(std::thread([&](SolarSystem* system, const int64_t start, const int64_t end) {
            //        system->Threaded(start, end);
            //        }, this, i * block_size, (i + 1) * block_size));
            //}
        }
        for (int64_t i = 0; i < NUM_THREADS; i++)
        {
            threads[i].join();
        }
        if(apply_forces)
            ApplyForces();
    }
    void DrawSolarSystem(bool orbit_path)
    {
        for (int64_t i = 1; i <= planetsLength; i++)
        {
            if (PlanetOption temp = GetPlanetReference(i))
            {
                PlanetReference planet = temp;

                // Change the scale so that the renderer looks better
                const float scale = 1.f / 100.f;
                glPushMatrix();
                glScalef(scale, scale, scale);

                // Draw planet at this position
                glTranslatef(*planet.x * scale, *planet.y * scale, *planet.z * scale);
                if (orbit_path)
                {
                    // Super fast drawcall for a cube (cube vertices are already loaded onto graphics card)
                    glDrawArrays(GL_TRIANGLES, 0, 36);
                }
                else
                {   
                    // Relatively slow draw call
                    glScalef(*planet.r * scale, *planet.r * scale, *planet.r * scale);
                    drawSphere();
                }
                glPopMatrix();
            }
        }
    }
};

int main()
{
    sf::Clock deltaClock;
    sf::Clock frameClock;

    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Solar Simulator", sf::Style::Default, sf::ContextSettings(32));
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    sf::View centreView;
    sf::Vector2u windowSize = window.getSize();
    sf::Vector2i windowMiddle = sf::Vector2i(windowSize.x / 2, windowSize.y / 2);
    bool cursorGrabbed = true;
    bool recalculate_frames = true;
    bool time_recalculation = false;
    std::string recalc_time = "Recalculated in ";
    centreView.setSize(sf::Vector2f(windowSize.x, windowSize.y));
    centreView.setCenter(0, 0);
    
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
    double frustUp = frustRight * double(windowMiddle.y) / double(windowMiddle.x);
    double nearClip = 1.f;
    double fov = 70;
    double farClip = double(windowMiddle.x) / double(tan(fov * pi / 360));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-frustRight, frustRight, -frustUp, frustUp, nearClip, farClip);
    glMatrixMode(GL_MODELVIEW);

    GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
    GLfloat direction[] = { 0.0, 0.0, 1.0, 1.0 };

    glMaterialfv(GL_FRONT,  GL_AMBIENT_AND_DIFFUSE, white);
    glMaterialfv(GL_FRONT,  GL_SPECULAR,            white);
    glMaterialf(GL_FRONT,   GL_SHININESS,           30);

    glLightfv(GL_LIGHT0, GL_AMBIENT,  black);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  white);
    glLightfv(GL_LIGHT0, GL_SPECULAR, white);
    glLightfv(GL_LIGHT0, GL_POSITION, direction);

    glEnable(GL_LIGHTING);                // so the renderer considers light
    glEnable(GL_LIGHT0);                  // turn LIGHT0 on
    glEnable(GL_DEPTH_TEST);              // so the renderer considers depth
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);

    window.pushGLStates();

    // --------------------------------------------
    
    srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));
    std::list<SolarSystem> simulation;
    SolarSystem system = SolarSystem(0);
    //system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
    //system.MergeAllPlanets();
    simulation.emplace_back(system);

    bool running = true;
    while (running) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::KeyPressed)
            {
                switch (event.key.code)
                {
                case sf::Keyboard::E: {
                    cursorGrabbed = !cursorGrabbed;
                    sf::Mouse::setPosition(sf::Vector2i(windowMiddle.x, windowMiddle.y), window);
                }break;
                case sf::Keyboard::Escape: {
                    cursorGrabbed = !cursorGrabbed;
                    sf::Mouse::setPosition(sf::Vector2i(windowMiddle.x, windowMiddle.y), window);
                }break;
                }
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
                //mousePos = sf::Vector2f(event.mouseMove.x - float(windowMiddle.x), event.mouseMove.y - float(windowMiddle.y));
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
            }
            else if (event.type == sf::Event::Closed) {
                running = false;
            }
        }
        // Movement code
        const float cameraSpeed = 0.05f; // adjust accordingly
        float accelerate = (1.f + float(sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)));
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
        {
            cameraPos += cameraSpeed * cameraFront * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
        {
            cameraPos -= cameraSpeed * cameraFront * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
        {
            cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * accelerate;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
        {
            cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed * accelerate;
        }


        if (running)
        {
            float frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
            window.clear();
            ImGui::SFML::Update(window, deltaClock.restart());
            window.setView(centreView);

            ImGui::Begin("Update Rate");

            static int updatesPerFrame = 1;
            ImGui::PushItemWidth(150);
            //ImGui::Text(std::string("POS : " + std::to_string_with_precision(camPos.x) + ", " + std::to_string_with_precision(camPos.y)).c_str());
            //ImGui::Text(std::string("MPOS: " + std::to_string_with_precision(mousePos.x) + ", " + std::to_string_with_precision(mousePos.y)).c_str());
            ImGui::Text(std::string("FPS:     " + std::to_string_with_precision(frameRate)).c_str());
            ImGui::Text(std::string("UPS:     " + std::to_string_with_precision(frameRate * float(updatesPerFrame))).c_str());
            ImGui::Text(std::string("OPS:     " + std::to_string_with_formatting(frameRate * float(updatesPerFrame * simulation.front().planetsLength * simulation.front().planetsLength))).c_str());
            ImGui::Text(std::string("Tick: " + std::to_string(simulation.front().tick)).c_str());
            
            ImGui::SliderInt(":UPF", &updatesPerFrame, 1, 50);
            ImGui::SliderInt(":THREADS", &NUM_THREADS, 1, 20);
            //ImGui::Checkbox(" :Threaded", &multi_threaded);
            //ImGui::Checkbox(" :Simd", &simd);
            //ImGui::Checkbox(" :Lock Framerate", &static_framerate);
            if (ImGui::SliderInt(":Forward Frames", &forward_calculation, 1, 2000))
            {
                recalculate_frames = true;
            }
            ImGui::SliderInt(":Orbit spacing", &tick_spacing, 1, 50);
            if (ImGui::SliderFloat(":Gravity", &G, 0.1, 10))
            {
                gravity = _mm256_set1_pd(G);
            }
            ImGui::Checkbox(":Planet Merging", &merging);
            ImGui::Checkbox(":Run Simulation", &runSimulation);
            ImGui::Checkbox(":Draw Plane", &drawPlane);
            if (time_recalculation) 
            {
                if (frameClock.getElapsedTime().asMilliseconds() > 1000)
                {
                    recalc_time = std::string("Recalculated in " + std::to_string_with_formatting(frameClock.getElapsedTime().asSeconds()) + "s");
                }
                else
                {
                    recalc_time = std::string("Recalculated in " + std::to_string_with_formatting(frameClock.getElapsedTime().asMilliseconds()) + "ms");
                }
                time_recalculation = false;
            }
            ImGui::Text(recalc_time.c_str());
            ImGui::End();
            frameClock.restart();

            ImGui::Begin("Modify Planet");
            ImGui::SliderInt(": ID", (int*)&selectedPlanet, 1, simulation.front().planetsLength);
            if (PlanetOption temp = simulation.front().GetPlanetReference(selectedPlanet))
            {
                bool modified = false;
                float smin = -100000;
                float smax = 100000;
                float dmin = -50.0;
                float dmax = 50.0;
                PlanetReference planet = temp;
                float px = float(*planet.x);
                float py = float(*planet.y);
                float pz = float(*planet.z);
                float pdx = float(*planet.dx);
                float pdy = float(*planet.dy);
                float pdz = float(*planet.dz);
                float pm  = float(*planet.mass);
                modified |= ImGui::SliderFloat(": Mass", &pm, 0.0001, 1000, "%.3f", 2.f);
                ImGui::Text("Position:");
                modified |= ImGui::SliderFloat(": X", &px, smin, smax, "%.3f", 2.f);
                modified |= ImGui::SliderFloat(": Y", &py, smin, smax, "%.3f", 2.f);
                modified |= ImGui::SliderFloat(": Z", &pz, smin, smax, "%.3f", 2.f);
                ImGui::Text("Velocity:");
                modified |= ImGui::SliderFloat(": DX", &pdx, dmin, dmax, "%.3f", 2.f);
                modified |= ImGui::SliderFloat(": DY", &pdy, dmin, dmax, "%.3f", 2.f);
                modified |= ImGui::SliderFloat(": DZ", &pdz, dmin, dmax, "%.3f", 2.f);
                if (modified) {
                    *planet.x = double(px);
                    *planet.y = double(py);
                    *planet.z = double(pz);
                    *planet.dx = double(pdx);
                    *planet.dy = double(pdy);
                    *planet.dz = double(pdz);
                    *planet.mass = double(pm);
                }
                if (ImGui::Button("Recalculate Frames"))
                {
                    while (simulation.size() > 1)
                        simulation.pop_back();
                    recalculate_frames = true;
                }
            }
            ImGui::End();

            ImGui::Begin("Manipulate Universe");
            ImGui::PushItemWidth(150);
            ImGui::Text(std::string("Planets: " + std::to_string(simulation.front().planetsLength)).c_str());
            ImGui::Text(std::string("Planet "+std::to_string(selectedPlanet)+" is the parent").c_str());
            ImGui::SliderInt(": Tiers", (int*)&tiers, 1, 4);
            ImGui::SliderInt(": Children", (int*)&children, 1, 10);
            ImGui::Text(std::string("Settings will add " + std::to_string(int(std::pow(children, tiers))) + " planets.").c_str());
            if (ImGui::Button("Remove Planet"))
            {
                simulation.front().RemovePlanet(selectedPlanet);
            }
            if (ImGui::Button("Add Universe"))
            {
                simulation.front().RecursivelyAddPlanets(selectedPlanet, children, tiers);
                simulation.front().MergeAllPlanets();
                while (simulation.size() > 1)
                    simulation.pop_back();
            }
            if (ImGui::Button("Remove All But One"))
            {
                PlanetObject saved = simulation.front().GetPlanetObject(selectedPlanet);
                simulation.front().planets.clear();
                simulation.front().planetsExtra.clear();
                simulation.front().planetsLength = 0;
                simulation.front().AddPlanet(saved);
                while (simulation.size() > 1)
                    simulation.pop_back();
                selectedPlanet = 1;
            }
            ImGui::End();
            
            if (recalculate_frames || !ImGui::IsAnyWindowFocused())
            {
                if (recalculate_frames)
                {
                    time_recalculation = true;
                }
                while (simulation.size() > forward_calculation)
                {
                    simulation.pop_back();
                }

                while (simulation.size() <= forward_calculation + updatesPerFrame)
                {
                    simulation.back().ThreadedUpdatePlanets(false);
                    SolarSystem new_system = SolarSystem(simulation.back());
                    new_system.tick++;
                    new_system.ApplyForces();
                    simulation.emplace_back(new_system);
                }
                recalculate_frames = false;
            }
            if (!ImGui::IsAnyWindowFocused())
            {
                if (runSimulation)
                {
                    for (int i = 0; i < updatesPerFrame; i++)
                    {
                        if (simulation.front().tick < global_tick)
                            simulation.pop_front();
                        global_tick++;
                    }
                }
            }
            //ImGui::ShowDemoWindow();

            {
                window.popGLStates();

                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                glLoadMatrixf(glm::value_ptr(glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp)));

                // Draw a white grid "floor" for the tetrahedron to sit on.
                glColor3f(1.0, 1.0, 1.0);

                if (drawPlane)
                {
                    glBegin(GL_LINES);
                    const float lines = 50;
                    const float gap = 0.5;
                    float dist = lines * gap * 0.5;
                    for (GLfloat i = -dist; i <= dist; i += gap) {
                        glNormal3f(0.f, 1.f, 0.f);
                        glVertex3f(i, 0, dist); glVertex3f(i, 0, -dist);
                        glNormal3f(0.f, 1.f, 0.f);
                        glVertex3f(dist, 0, i); glVertex3f(-dist, 0, i);
                    }
                    glEnd();
                }
                for (auto it = simulation.begin(); it != simulation.end(); it++)
                {
                    if (it == simulation.begin())
                    {
                        simulation.front().DrawSolarSystem(false);
                        // Load cube into graphics card for next draw calls
                        glEnableClientState(GL_VERTEX_ARRAY);
                        glEnableClientState(GL_NORMAL_ARRAY);
                        glVertexPointer(3, GL_FLOAT, 3 * sizeof(float), &box_array[0]);
                        glNormalPointer(GL_FLOAT, 3 * sizeof(float), &box_array[0]);
                    }
                    else if (it->tick % tick_spacing == 0)
                    {
                        it->DrawSolarSystem(true);
                    }
                }
                glDisableClientState(GL_VERTEX_ARRAY);
                glDisableClientState(GL_NORMAL_ARRAY);
                
                glCullFace(GL_FRONT);
                glFlush();
                window.pushGLStates();
            }
            ImGui::SFML::Render(window);
            window.display();
        }
    }
    ImGui::SFML::Shutdown();
    return 0;
}
