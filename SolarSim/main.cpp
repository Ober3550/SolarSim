#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"
#include <fstream>
#include <iostream>
#include <string>
#include <ostream>
#include <experimental/filesystem>
#include <immintrin.h>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <mutex>

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

std::string add_commas(std::string str)
{
    int outer = 0;
    bool decimal = false;
    for (auto iter = str.end(); iter > str.begin(); iter--)
    {
        if (decimal)
        {
            if (outer % 3 == 0 && outer != 0)
                iter = str.insert(iter, ',');
        }
        else if (str[str.length() - outer - 1] == '.')
        {
            decimal = true;
            outer = -1;
        }
        outer++;
    }
    return str;
}

const int VECWIDTH = 8;
const double pi = 2 * acos(0.0);
float G = 4.f;
const float zero = 0.f;
const float m_r = 0.9f;
const float base_zoom = 20.f;

__m256 merge_r   = _mm256_set1_ps(m_r);
__m256i m_zeroi  = _mm256_set1_epi32(0);
__m256i m_onesi  = _mm256_set1_epi32(uint32_t(-1));
__m256 gravity = _mm256_set1_ps(G);

int selectedPlanet = 1;
bool gotoSelected = false;
bool merging = false;
bool simd = true;
bool script_speed = false;
bool script_compare = false;
int NUM_THREADS = 4;
sf::Vector2f camPos = { 0.f,0.f };

const float minZoom = 0.5;
const float maxZoom = 256.f;
int tiers = 2;
int children = 120;
bool static_framerate = false;
int global_tick = 0;

std::string results = "Test,SIMD,Threads,Time\n";

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
    __m256i id;
    __m256 x;
    __m256 y;
    __m256 dx;
    __m256 dy;
    __m256 Fx;
    __m256 Fy;
    __m256 mass;
    __m256 r;
};
struct PlanetMerge {
    int A;
    int B;
    operator int64_t() { 
        return ((int64_t(this->A) << 32) | int32_t(this->B)); 
    }
    static PlanetMerge Decode(int64_t input) {
        return PlanetMerge{ int32_t(input >> 32), int32_t(input & 0xFFFFFFFF) };
    }
};

class SolarSystem {
public:
    std::vector<PlanetGroup> planets;
    std::unordered_set<int64_t> merger;
    std::mutex merger_lock;
    int planetsLength = 0;
    void AddPlanet(float x, float y, float dx, float dy, float mass)
    {
        if (planetsLength % VECWIDTH == 0)
            planets.emplace_back(PlanetGroup());
        PlanetGroup* group = &planets[planetsLength / VECWIDTH];
        int index = planetsLength % VECWIDTH;
        // Reserver id 0 for invalid planets
        group->id.m256i_i32 [index] = planetsLength + 1;
        group->x.m256_f32   [index] = x;
        group->y.m256_f32   [index] = y;
        group->dx.m256_f32  [index] = dx;
        group->dy.m256_f32  [index] = dy;
        group->Fx.m256_f32  [index] = 0.f;
        group->Fy.m256_f32  [index] = 0.f;
        group->mass.m256_f32[index] = mass;
        group->r.m256_f32   [index] = sqrt(mass / pi) * 128.f;
        planetsLength++;
    }
    SolarSystem(const SolarSystem& other) {
        this->planets = std::vector<PlanetGroup>(other.planets);
        this->merger = std::unordered_set<int64_t>();
        this->planetsLength = other.planetsLength;
    }
    bool operator==(const SolarSystem& other)
    {
        for (int i=0;i<planets.size();i++)
        {
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (planets[i].id.m256i_i32[j] != 0 || other.planets[i].id.m256i_i32[j] != 0)
                {
                    if (planets[i].x.m256_f32[j] != other.planets[i].x.m256_f32[j])
                        return false;
                    if (planets[i].y.m256_f32[j] != other.planets[i].y.m256_f32[j])
                        return false;
                    if (planets[i].dx.m256_f32[j] != other.planets[i].dx.m256_f32[j])
                        return false;
                    if (planets[i].dy.m256_f32[j] != other.planets[i].dy.m256_f32[j])
                        return false;
                    if (planets[i].Fx.m256_f32[j] != other.planets[i].Fx.m256_f32[j])
                        return false;
                    if (planets[i].Fy.m256_f32[j] != other.planets[i].Fy.m256_f32[j])
                        return false;
                }
            }
        }
        return true;
    }
    PlanetOption GetPlanet(int id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* group = &planets[id / VECWIDTH];
            int index = id % VECWIDTH;
            Planet planet = { 
                &group->id.m256i_i32[index],
                &group->x.m256_f32[index],
                &group->y.m256_f32[index],
                &group->dx.m256_f32[index],
                &group->dy.m256_f32[index],
                &group->Fx.m256_f32[index],
                &group->Fy.m256_f32[index],
                &group->mass.m256_f32[index],
                &group->r.m256_f32[index]
            };
            // Check for uninitialized planet
            if (group->id.m256i_i32[index] == 0)
                return PlanetOption();
            return PlanetOption{ true,planet };
        }
        return PlanetOption();
    }
    // Removes the planets contents
    void EmptyPlanet(int id)
    {
        id--;
        if (id / VECWIDTH < planets.size())
        {
            PlanetGroup* group = &planets[id / VECWIDTH];
            int index = id % VECWIDTH;
            group->id.m256i_i32[index]  = 0;
            group->x.m256_f32[index]    = 0.f;
            group->y.m256_f32[index]    = 0.f;
            group->dx.m256_f32[index]   = 0.f;
            group->dy.m256_f32[index]   = 0.f;
            group->Fx.m256_f32[index]   = 0.f;
            group->Fy.m256_f32[index]   = 0.f;
            group->mass.m256_f32[index] = 0.f;
            group->r.m256_f32[index]    = 0.f;
        }
    }
    void RemoveAllButSelected(int id) {
        for (int i = 0; i < planets.size(); i++) {
            PlanetGroup* group = &planets[i];
            for (int j = 0; j < VECWIDTH; j++) {
                if (group->id.m256i_i32[j] != id) {
                    group->id.m256i_i32[j] = 0;
                }
            }
        }
        RemoveHoles();
    }
    // Removes the empty planets and packs everything 
    // closer together to use less memory and processing
    void RemoveHoles()
    {
        planetsLength = 0;
        int prev_id = 0;
        for (int i = 0; i < planets.size(); i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
            next_hole:;
                if (groupA->id.m256i_i32[j] == 0)
                {
                    for (int k = planets.size() - 1; k >= 0; k--)
                    {
                        PlanetGroup* groupB = &planets[k];
                        for (int l = VECWIDTH - 1; l >= 0; l--)
                        {
                            if (i == k && j == l)
                                goto reduce_vector;
                            if (groupB->id.m256i_i32[l] != 0)
                            {
                                prev_id++;
                                // Move planet into hole
                                groupA->id.m256i_i32[j] = prev_id;
                                groupA->x.m256_f32[j] = groupB->x.m256_f32[l];
                                groupA->y.m256_f32[j] = groupB->y.m256_f32[l];
                                groupA->dx.m256_f32[j] = groupB->dx.m256_f32[l];
                                groupA->dy.m256_f32[j] = groupB->dy.m256_f32[l];
                                groupA->Fx.m256_f32[j] = groupB->Fx.m256_f32[l];
                                groupA->Fy.m256_f32[j] = groupB->Fy.m256_f32[l];
                                groupA->mass.m256_f32[j] = groupB->mass.m256_f32[l];
                                groupA->r.m256_f32[j] = groupB->r.m256_f32[l];
                                // Empty the just moved planet
                                groupB->id.m256i_i32[l] = 0;
                                groupB->x.m256_f32[l] = 0.f;
                                groupB->y.m256_f32[l] = 0.f;
                                groupB->dx.m256_f32[l] = 0.f;
                                groupB->dy.m256_f32[l] = 0.f;
                                groupB->Fx.m256_f32[l] = 0.f;
                                groupB->Fy.m256_f32[l] = 0.f;
                                groupB->mass.m256_f32[l] = 0.f;
                                groupB->r.m256_f32[l] = 0.f;
                                planetsLength = ((planets.size()-1) * VECWIDTH) + l;
                                goto next_hole;
                            }
                        }
                    }
                }
                else {
                    planetsLength++;
                    prev_id = groupA->id.m256i_i32[j];
                }
            }
        }
    reduce_vector:;
        for (int k = planets.size() - 1; k >= 0; k--)
        {
            PlanetGroup* groupB = &planets[k];
            for (int l = VECWIDTH - 1; l >= 0; l--)
            {
                if (groupB->id.m256i_i32[l] == 0)
                {
                    if (l == 0)
                        planets.pop_back();
                }
                else
                    return;
            }
        }
    }
    void AddRandomSatellite(int id) {
        if (PlanetOption temp = GetPlanet(id))
        {
            Planet parent = temp;
            const int angles = (360 * 32);
            const int steps = 1000;
            const float min_dis = 5.f;
            const float max_dis = 100.f;
            const float dif_dis = max_dis - min_dis;
            const float min_prop = 0.005;
            const float max_prop = 0.05f;
            const float dif_prop = max_prop - min_prop;
            const int   mas_step = 1000;
            float new_mass = min_prop + float(rand() % mas_step) / mas_step * dif_prop * (*parent.mass);
            float angle = float(rand() % angles) / float(angles);
            angle *= 2.f * pi;
            float magni = (*parent.r) + (*parent.mass) * (min_dis + float(rand() % steps) / float(steps) * dif_dis);
            float new_x = (*parent.x) + cos(angle) * magni;
            float new_y = (*parent.y) + sin(angle) * magni;
            float new_dx;
            float new_dy;
            float scalar = sqrt(0.5f*G*((*parent.mass) + 0.5f * children * new_mass));
            new_dx = (*parent.dx) + cos(angle + pi * 0.5f) * scalar;
            new_dy = (*parent.dy) + sin(angle + pi * 0.5f) * scalar;
            AddPlanet(new_x, new_y, new_dx, new_dy, new_mass);
        }
    }
    // Parent, children per call, recursion depth
    void RecursivelyAddPlanets(int parent, int n, int m)
    {
        std::vector<int> parents;
        if (m == 0)
            return;
        for (int i = 0; i < n; i++)
        {
            AddRandomSatellite(parent);
            parents.emplace_back(planetsLength);
        }
        for (int i = 0; i < parents.size(); i++)
        {
            RecursivelyAddPlanets(parents[i], n, m - 1);
        }
    }
    SolarSystem()
    {
        //AddPlanet(-200, 0, 0, -1, 0.5);
        //AddPlanet(200, 0, 0, 1, 0.5);
        //AddPlanet(600, 0, 0, 1, 0.05);
        //AddPlanet(-600, 0, 0, -1, 0.05);
        AddPlanet(0, 0, 0, 0, 1000);
        //AddPlanet(72, 0, 0, -2, 0.5);
        //AddPlanet(-216, 0, 0, 2, 0.5);
    }
    std::vector<sf::Sprite> DrawSolarSystem(sf::Texture& circle)
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
                if (*planetA.mass < * planetB.mass)
                {
                    *planetA.x = *planetB.x;
                    *planetA.y = *planetB.y;
                }
                *planetA.dx = ((*planetA.dx) * (*planetA.mass) + *planetB.dx * *planetB.mass) / total_mass;
                *planetA.dy = ((*planetA.dy) * (*planetA.mass) + *planetB.dy * *planetB.mass) / total_mass;
                *planetA.mass = total_mass;
                *planetA.r = sqrt(total_mass / pi) * 128.f;
                if (idB == selectedPlanet)
                    selectedPlanet = idA;
                // Remove planet B
                EmptyPlanet(idB);
            }
        }
    }
    void ThreadedCheckMerge(const int start, int end)
    {
        if (end > planets.size()) end = planets.size();
        for (int i = start; i < end; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA->id.m256i_i32[j] != 0)
                {
                    __m256 planetA_x = _mm256_set1_ps(groupA->x.m256_f32[j]);
                    __m256 planetA_y = _mm256_set1_ps(groupA->y.m256_f32[j]);
                    __m256 planetA_r = _mm256_set1_ps(groupA->r.m256_f32[j]);
                    for (PlanetGroup& groupB : planets)
                    {
                        __m256 rx = _mm256_sub_ps(planetA_x, groupB.x);
                        __m256 ry = _mm256_sub_ps(planetA_y, groupB.y);
                        rx = _mm256_mul_ps(rx, rx);
                        ry = _mm256_mul_ps(ry, ry);
                        __m256 r2 = _mm256_add_ps(rx, ry);
                        __m256 r = _mm256_sqrt_ps(r2);
                        __m256 prox = _mm256_mul_ps(merge_r, _mm256_add_ps(planetA_r, groupB.r));
                        for (int l = 0; l < VECWIDTH; l++)
                        {
                            if (r.m256_f32[l] < prox.m256_f32[l])
                            {
                                if (groupB.id.m256i_i32[l] != 0)
                                {
                                    int ida = groupA->id.m256i_i32[j];
                                    int idb = groupB.id.m256i_i32[l];
                                    if (ida > idb)
                                    {
                                        int temp = ida;
                                        ida = idb;
                                        idb = temp;
                                    }
                                    if (ida != idb)
                                    {
                                        merger_lock.lock();
                                        merger.insert(PlanetMerge{ ida,idb });
                                        merger_lock.unlock();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    void ThreadedMergePlanets()
    {
        std::vector<std::thread> threads;
        int block_size = (planetsLength / VECWIDTH) / NUM_THREADS + 1;
        if (NUM_THREADS == 1)
            block_size = planets.size();
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threads.emplace_back(std::thread([&](SolarSystem* system, const int start, const int end) {
                system->ThreadedCheckMerge(start, end);
                }, this, i * block_size, (i + 1) * block_size));
        }
        for (int i = 0; i < NUM_THREADS; i++)
        {
            threads[i].join();
        }
        for (int64_t temp : merger)
        {
            PlanetMerge merge = PlanetMerge::Decode(temp);
            MergePlanets(merge.A, merge.B);
        }
        merger.clear();
        RemoveHoles();
    }
    void ApplyForces()
    {
        // Separate applying forces and velocity since it's O(n)
        for (PlanetGroup& groupA : planets)
        {
            for (int j = 0; j < VECWIDTH; j++)
            {
                if (groupA.id.m256i_i32[j] != 0)
                {
                    groupA.dx.m256_f32[j]+= groupA.Fx.m256_f32[j] / groupA.mass.m256_f32[j];
                    groupA.dy.m256_f32[j]+= groupA.Fy.m256_f32[j] / groupA.mass.m256_f32[j];
                    groupA.x.m256_f32[j] += groupA.dx.m256_f32[j];
                    groupA.y.m256_f32[j] += groupA.dy.m256_f32[j];
                    groupA.Fx.m256_f32[j] = 0.f;
                    groupA.Fy.m256_f32[j] = 0.f;
                }
            }
        }
    }
    void Threaded(const int start, int end)
    {
        // F = (G * m1 * m2) / r^2
        // F = ma
        // a = m / F;
        // a = m / ((G * m * m2) / r^2)
        if (end > planets.size()) end = planets.size();
        for (int i = start; i < end; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                // Adding accumulator arrays to test theory
                float Fx[VECWIDTH] = {};
                float Fy[VECWIDTH] = {};
                if (groupA->id.m256i_i32[j] != 0)
                {
                    for (int k = 0; k < planets.size(); k++)
                    {
                        PlanetGroup* groupB = &planets[k];
                        for (int l = 0; l < VECWIDTH; l++)
                        {
                            if (groupB->id.m256i_i32[l] != 0 && groupA->id.m256i_i32[j] != groupB->id.m256i_i32[l])
                            {
                                float rx = groupB->x.m256_f32[l] - groupA->x.m256_f32[j];
                                float ry = groupB->y.m256_f32[l] - groupA->y.m256_f32[j];
                                float r2 = (rx * rx) + (ry * ry);
                                float F = (G * (groupA->mass.m256_f32[j]) * groupB->mass.m256_f32[l]) / r2;
                                Fx[l] += F * rx;
                                Fy[l] += F * ry;
                                // Previous way of accumulating values
                                //groupA->Fx.m256_f32[j] += F * rx;
                                //groupA->Fy.m256_f32[j] += F * ry;
                            }
                        }
                    }
                    for (int l = 0; l < VECWIDTH; l++)
                    {
                        groupA->Fx.m256_f32[j] += Fx[l];
                        groupA->Fy.m256_f32[j] += Fy[l];
                    }
                }
            }
        }
    }
    void SimdApplyForces()
    {
        // Separate applying forces and velocity since it's O(n)
        for (PlanetGroup& groupA : planets)
        {
            groupA.dx = _mm256_add_ps(groupA.dx, _mm256_div_ps(groupA.Fx, groupA.mass));
            groupA.dy = _mm256_add_ps(groupA.dy, _mm256_div_ps(groupA.Fy, groupA.mass));
            groupA.x =  _mm256_add_ps(groupA.x, groupA.dx);
            groupA.y =  _mm256_add_ps(groupA.y, groupA.dy);
            groupA.Fx = _mm256_set1_ps(0.f);
            groupA.Fy = _mm256_set1_ps(0.f);
        }
    }
    void SimdThreaded(const int start, int end)
    {
        if (end > planets.size()) end = planets.size();
        for (int i = start; i < end; i++)
        {
            PlanetGroup* groupA = &planets[i];
            for (int j = 0; j < VECWIDTH; j++)
            {
                // Create accumulators for forces
                __m256 planetA_Fx = _mm256_set1_ps(zero);
                __m256 planetA_Fy = _mm256_set1_ps(zero);

                // Create common registers
                __m256  planetA_x    = _mm256_set1_ps(groupA->x.m256_f32[j]);
                __m256  planetA_y    = _mm256_set1_ps(groupA->y.m256_f32[j]);
                __m256  planetA_mass = _mm256_set1_ps(groupA->mass.m256_f32[j]);
                __m256i planetA_id = _mm256_set1_epi32(groupA->id.m256i_i32[j]);

                for (int k = 0; k < planets.size(); k++)
                {
                    // START CORE LOOP
                    PlanetGroup* groupB = &planets[k];
                    // Subtract planet As position from groups positions to find relative distance
/*1*/               __m256 rx = _mm256_sub_ps(groupB->x, planetA_x);
/*2*/               __m256 ry = _mm256_sub_ps(groupB->y, planetA_y);
                    // Find the square of each distance
/*3*/               __m256 rx2 = _mm256_mul_ps(rx, rx);
/*4*/               __m256 ry2 = _mm256_mul_ps(ry, ry);
                    // Find the euclidean distance squared
/*5*/               __m256 r2 = _mm256_add_ps(rx2, ry2);
                    // Calculate gravity
/*6*/               __m256 mass = _mm256_mul_ps(groupB->mass, planetA_mass);
/*7*/               __m256 gm   = _mm256_mul_ps(mass, gravity);
/*8*/               __m256 F    = _mm256_div_ps(gm, r2);
                    // Find the forces for each dimension
                    __m256 Fx; 
                    __m256 Fy;
                    Fx = _mm256_mul_ps(F, rx);
                    Fy = _mm256_mul_ps(F, ry);
                    // Remove nan values such as planets affecting themselves
                    // If id == 0
/*9*/               __m256i zeromask = _mm256_cmpeq_epi32(groupB->id, m_zeroi);
/*10*/              __m256i idmask   = _mm256_cmpeq_epi32(groupB->id, planetA_id);
                    // If groupA.id == groupB.id
/*11*/              __m256i bothmask = _mm256_or_si256(zeromask, idmask);
/*12*/              bothmask = _mm256_xor_si256(bothmask, m_onesi);
/*13*/              Fx = _mm256_and_ps(Fx, _mm256_castsi256_ps(bothmask));
/*14*/              Fy = _mm256_and_ps(Fy, _mm256_castsi256_ps(bothmask));
                    // Accumulate forces in 8 wide simd vectors
/*15*/              planetA_Fx = _mm256_add_ps(planetA_Fx, Fx);
/*16*/              planetA_Fy = _mm256_add_ps(planetA_Fy, Fy);
                    // END CORE LOOP
                }
                // Accumulate 8 wide force vector onto single variable within planet A
                for (int l = 0; l < VECWIDTH; l++)
                {
                    groupA->Fx.m256_f32[j] += planetA_Fx.m256_f32[l];
                    groupA->Fy.m256_f32[j] += planetA_Fy.m256_f32[l];
                }
            }
        }
    }
    void ThreadedUpdatePlanets()
    {
        // I don't think this can be multithreaded since it relies on removing elements being thread safe... 
        // which is obvious why it wouldn't be
        if(merging)
            ThreadedMergePlanets();

        if (NUM_THREADS == 1)
        {
            if (simd)
                SimdThreaded(0, this->planets.size());
            else
                Threaded(0, this->planets.size());
        }
        else
        {
            // Create thread container
            std::vector<std::thread> threads;
            // Calculate work division size
            int block_size = (planetsLength / VECWIDTH) / NUM_THREADS + 1;
            // Construct threads with given work
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
            // Join threads
            for (int i = 0; i < NUM_THREADS; i++)
            {
                threads[i].join();
            }
        }
        if (simd)
            SimdApplyForces();
        else
            ApplyForces();
    }
};

int main()
{
    sf::Clock deltaClock;
    sf::Clock frameClock;

    sf::RenderWindow window(sf::VideoMode(1000, 1000), "Gravity Simulation");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    // Just for debugging
    /*for (const auto& entry : fs::directory_iterator(fs::current_path()))
    {
        std::cout << entry.path() << '\n';
    }*/

    sf::Texture circle;
    circle.loadFromFile("circle.png");

    sf::View centreView;
    sf::Vector2u size = window.getSize();
    centreView.setSize(sf::Vector2f(size.x, size.y));
    centreView.setCenter(0, 0);
    float prevZoom = 1.f;
    float zoom = base_zoom;
    sf::Vector2f prevCamPos = { 0.f,0.f };
    sf::Vector2f mousePos = { 0.f,0.f };

    bool simulating = true;
    float max_ops = 0.f;
    int test_num = 0;
    int result_num = 0;
    float frameRate = 0.f;
    
    //srand(std::hash<int>{}(frameClock.getElapsedTime().asMicroseconds()));
    srand(200);
    SolarSystem system;
    system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
    system.ThreadedMergePlanets();
    bool running = true;

    while (running && window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);
            if (event.type == sf::Event::MouseMoved)
            {
                mousePos = sf::Vector2f(event.mouseMove.x - float(size.x)/2, event.mouseMove.y - float(size.y)/2);
            }
            else if (event.type == sf::Event::Resized)
            {
                size = window.getSize();
                centreView.setSize(sf::Vector2f(size.x, size.y));
                centreView.setCenter(0, 0);
                prevZoom = 1.f;
                zoom = base_zoom;
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
        window.setView(centreView);
        window.clear();
        ImGui::SFML::Update(window, deltaClock.restart());

        ImGui::Begin("Update Rate");
        static int updatesPerFrame = 1;
        ImGui::Text(std::string("FPS:     "+std::to_string_with_precision(frameRate)).c_str());
        ImGui::Text(std::string("UPS:     " + std::to_string_with_precision(frameRate * float(updatesPerFrame))).c_str());
        double num_ops = 16 * frameRate * double(int64_t(updatesPerFrame) * int64_t(system.planetsLength) * int64_t(system.planetsLength));
        std::string operations = add_commas(std::to_string_with_precision(num_ops));
        if (simulating)
        {
            if (num_ops > max_ops)
            {
                max_ops = num_ops;
            }
        }
        std::string max_operat = add_commas(std::to_string_with_precision(max_ops));
        ImGui::Text(std::string("OPS:     " + operations).c_str());
        ImGui::Text(std::string("MAX OPS: " + max_operat).c_str());
        ImGui::Text("OPS = n^2 * UPS");
        ImGui::Text(std::string("PLANETS: " + std::to_string(system.planetsLength)).c_str());
        ImGui::SliderInt(" :UPF", &updatesPerFrame, 1, 50);
        ImGui::SliderInt(" :Threads", &NUM_THREADS, 1, 20);
        ImGui::Checkbox(" :SIMD", &simd);
        ImGui::Checkbox(" :Lock Framerate", &static_framerate);
        //ImGui::Text(std::string("POS : " + std::to_string_with_precision(camPos.x) + ", " + std::to_string_with_precision(camPos.y)).c_str());
        //ImGui::Text(std::string("MPOS: " + std::to_string_with_precision(mousePos.x) + ", " + std::to_string_with_precision(mousePos.y)).c_str());
        ImGui::Text(std::string("ZOOM: " + std::to_string_with_precision(zoom)).c_str());
        ImGui::Checkbox(": Run Simulation", &simulating);
        /*if (ImGui::Button("Run Tests"))
        {
            NUM_THREADS = std::thread::hardware_concurrency();
            script_speed = true;
        }
        if (script_speed) ImGui::Text(std::string("Running").c_str());*/
        ImGui::End();

        float smin = -1000000;
        float smax = 1000000;
        ImGui::Begin("Modify Planet");
        ImGui::SliderInt(": ID", &selectedPlanet, 1, system.planetsLength);
        if (PlanetOption temp = system.GetPlanet(selectedPlanet))
        {
            float dmin = -1000.0;
            float dmax = 1000.0;
            Planet planet = temp;
            ImGui::SliderFloat(": X",  planet.x,    smin, smax);
            ImGui::SliderFloat(": Y",  planet.y,    smin, smax);
            ImGui::SliderFloat(": DX", planet.dx,   dmin, dmax);
            ImGui::SliderFloat(": DY", planet.dy,   dmin, dmax);
            if (ImGui::SliderFloat(": M", planet.mass, 0.001, 10000, "%.3f", 2.f))
            {
                *planet.r = sqrt(*planet.mass / pi) * 128.f;
            }
            if (ImGui::SliderFloat(": G", &G, 0.1, 10))
            {
                gravity = _mm256_set1_ps(G);
            }
            
            ImGui::Checkbox(": Follow Selected", &gotoSelected);
            
        }
        ImGui::End();

        ImGui::Begin("Modify Universe");
        
        ImGui::SliderInt(": Tiers", &tiers, 1, 4);
        ImGui::SliderInt(": Children", &children, 1, 10);
        ImGui::Text(std::string("Settings will add " + std::to_string(int(std::pow(children, tiers))) + " planets.").c_str());
        if (ImGui::Button("Add Universe", {200,20}))
        {
            system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
            system.ThreadedMergePlanets();
        }
        if (ImGui::Button("Remove Planet", { 200,20 }))
        {
            system.EmptyPlanet(selectedPlanet);
        }
        if (ImGui::Button("Remove All But Selected", { 200,20 })) {
            system.RemoveAllButSelected(selectedPlanet);
        }
        if (ImGui::Button("Add Planet", { 200,20 }))
        {
            system.AddPlanet(smin, smin, 0, 0, 0.5);
            selectedPlanet = system.planetsLength;
        }
        if (ImGui::Button("Add Random", { 200,20 }))
        {
            system.AddRandomSatellite(selectedPlanet);
        }
        ImGui::Checkbox(" :Planet Merging", &merging);
        ImGui::End();

        ImGui::Begin("Instructions");
        ImGui::TextWrapped(
            "Welcome to the 2D Solar Simulator!\n"
            "The best way to move around is to scroll to wherever you point the mouse. \n"
            "You can ctrl+click on a slider to enter a specific value. \n"
            "You can select follow selected for the camera to follow the planet that's currently selected. \n"
            "If you have any questions contact me at Ober3550@gmail.com! "
        );
        ImGui::End();
        
        if (simulating)
        {
            if (static_framerate)
            {
                if (updatesPerFrame > 1 && frameRate < 50.f)
                    updatesPerFrame--;
                else if (updatesPerFrame < 50 && frameRate > 55.f)
                    updatesPerFrame++;
            }
            for (int i = 0; i < updatesPerFrame; i++)
            {
                if (script_compare)
                {
                    SolarSystem systemSerial = SolarSystem(system);
                    SolarSystem systemNoSimd = SolarSystem(system);
                    SolarSystem systemSimd = SolarSystem(system);
                    int temp = std::thread::hardware_concurrency();
                    simd = false;
                    
                    NUM_THREADS = 1;
                    systemSerial.ThreadedUpdatePlanets();
                    NUM_THREADS = temp;
                    systemNoSimd.ThreadedUpdatePlanets();
                    assert(systemSerial == systemNoSimd);
                    simd = true;
                    NUM_THREADS = 1;              
                    systemSimd.ThreadedUpdatePlanets();
                    NUM_THREADS = 4;
                    system.ThreadedUpdatePlanets();
                    assert(systemSimd == system);
                    assert(systemNoSimd == systemSimd);
                }
                system.ThreadedUpdatePlanets();
            }
        }
        //ImGui::ShowTestWindow();

        std::vector<sf::Sprite> sprites = system.DrawSolarSystem(circle);
        for (int i=0;i<sprites.size();i++)
        {
            sf::Sprite sprite = sprites[i];
            if (i + 1 == selectedPlanet)
                sprite.setColor(sf::Color(255, 0, 0, 255));
            window.draw(sprite);
        }
       
        ImGui::SFML::Render(window);
        window.display();
        frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
        if (script_speed)
        {
            // Give the program a few frames to settle the memory layout in to remove any discrepancies due to startup processes
            if (global_tick > 0)
            {
                result_num++;
                test_num++;
                // Create a new entry to the table
                results += std::to_string(result_num) + ","
                    + (simd ? "true," : "false,")
                    + std::to_string(NUM_THREADS) + ","
                    + std::to_string(frameClock.getElapsedTime().asMicroseconds()) + "\n";
                // Run 4 tests for each configuration
                if (test_num == 1)
                {
                    // Decrement the number of threads untill there's only 1
                    if (NUM_THREADS != 1)
                        NUM_THREADS--;
                    else
                    {
                        // If there's only 1 thread and simd is active that means the first 16 tests have been completed and switch to the non-simd tests.
                        if (simd)
                        {
                            NUM_THREADS = std::thread::hardware_concurrency();
                            simd = false;
                        }
                        else
                        {
                            // If there's only 1 thread and simd was not active this means all tests have been completed
                            // Write the results to a file and exit the program.
                            std::ofstream myfile;
                            myfile.open("results.csv", std::ios::out | std::ios::trunc | std::ios::binary);
                            if (myfile.is_open())
                            {
                                myfile << results;
                            }
                            running = false;
                        }
                    }
                    test_num = 0;
                }
            }
            global_tick++;
        }
        frameClock.restart();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
