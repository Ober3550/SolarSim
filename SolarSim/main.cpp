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
#include <SolarSim/clBuilder.h>

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

int selectedPlanet = 0;
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
int children = 200;
bool static_framerate = false;
int global_tick = 0;

#define SIZE 100000
#define MASS 0
#define X 1
#define Y 2
#define DX 3
#define DY 4
#define FX 5
#define FY 6
#define R 7
#define PLANET_SIZE 8
#define VEC_WIDTH 8
#define SIMD_IDX(a,b) ((a / VEC_WIDTH)*(VEC_WIDTH*PLANET_SIZE)+(a % VEC_WIDTH)+VEC_WIDTH*b)
#define G 4.0
__m256 gravity  = _mm256_set1_ps(G);
__m256 smallest = _mm256_set1_ps(0.00000000000001f);

float min_dis = 10.f;
float max_dis = 150.f;
float max_prop = 50.f;
float min_prop = 200.f;
float start_vel = 1.5f;
float planets[SIZE*PLANET_SIZE] = {};
float cpu_fx[SIZE];
float cpu_fy[SIZE];
int planetsLength = 0;

struct Planet {
    int id      = 0;
    float x     = 0;
    float y     = 0;
    float dx    = 0;
    float dy    = 0;
    float Fx    = 0;
    float Fy    = 0;
    float mass  = 0;
    float r     = 0;
};

void SimdUpdatePlanets(int start, int end) {
    for (int i = start; i < std::min(end,planetsLength); i++)
    {
        // Create accumulators for forces
        __m256 planetA_Fx = _mm256_set1_ps(zero);
        __m256 planetA_Fy = _mm256_set1_ps(zero);

        // Create common registers
        __m256  planetA_x = _mm256_set1_ps(planets[SIMD_IDX(i, X)]);
        __m256  planetA_y = _mm256_set1_ps(planets[SIMD_IDX(i, Y)]);
        __m256  planetA_mass = _mm256_set1_ps(planets[SIMD_IDX(i, MASS)]);

        for (int k = 0; k < (planetsLength / VEC_WIDTH); k++)
        {
            __m256 group_x = _mm256_load_ps(&planets[SIMD_IDX((k * VEC_WIDTH), X)]);
            __m256 group_y = _mm256_load_ps(&planets[SIMD_IDX((k * VEC_WIDTH), Y)]);
            __m256 group_mass = _mm256_load_ps(&planets[SIMD_IDX((k * VEC_WIDTH), MASS)]);
            // START CORE LOOP
            // Subtract planet As position from groups positions to find relative distance
            __m256 rx = _mm256_sub_ps(group_x, planetA_x);
            __m256 ry = _mm256_sub_ps(group_y, planetA_y);
            // Find the square of each distance
            __m256 rx2 = _mm256_mul_ps(rx, rx);
            __m256 ry2 = _mm256_mul_ps(ry, ry);
            // Find the euclidean distance squared
            __m256 r2 = _mm256_add_ps(rx2, ry2);
            r2 = _mm256_add_ps(r2, smallest);
            // Calculate gravity
            __m256 mass = _mm256_mul_ps(group_mass, planetA_mass);
            __m256 gm = _mm256_mul_ps(mass, gravity);
            __m256 F = _mm256_div_ps(gm, r2);
            // Find the forces for each dimension
            __m256 Fx = _mm256_mul_ps(F, rx);
            __m256 Fy = _mm256_mul_ps(F, ry);
            // Accumulate forces in 8 wide simd vectors
            planetA_Fx = _mm256_add_ps(planetA_Fx, Fx);
            planetA_Fy = _mm256_add_ps(planetA_Fy, Fy);
            // END CORE LOOP
        }
        // Accumulate 8 wide force vector onto single variable within planet A
        for (int l = 0; l < VEC_WIDTH; l++)
        {
            cpu_fx[i] += planetA_Fx.m256_f32[l];
            cpu_fy[i] += planetA_Fy.m256_f32[l];
        }
    }
}

class SolarSystem {
public:
    clProgram prog;
    cl::CommandQueue queue;
    SolarSystem()
    {
        prog.context = cl::Context({ cl::Device::getDefault() });
        cl::Program::Sources sources;
        const std::string kernelString = getFileData("kernels.cl");
        sources.push_back({ kernelString.c_str(),kernelString.length() });
        prog.program = cl::Program(prog.context, sources);
        std::cout << "Device name: " << cl::Device::getDefault().getInfo<CL_DEVICE_NAME>() << std::endl;
        if(prog.program.build({cl::Device::getDefault()}) != CL_SUCCESS)
            std::cout << "error: " << prog.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
        prog.buffer = cl::Buffer(prog.context, CL_MEM_READ_WRITE, (sizeof(float) * SIZE * PLANET_SIZE));
        prog.kernel = cl::Kernel(prog.program, "planetForce");
        prog.kernel.setArg(0, prog.buffer);
        AddPlanet(0, 0, 0, 0, 1000);
        //RecursivelyAddPlanets(selectedPlanet, children, tiers);
        queue = cl::CommandQueue(prog.context, cl::Device::getDefault());
    }
    void UpdatePlanets() {
        if (planetsLength > 20000) {
            // buffer initialisation
            queue.enqueueWriteBuffer(prog.buffer, CL_TRUE, 0, (sizeof(float) * SIZE * PLANET_SIZE), planets);
            // Executing the kernel object
            queue.enqueueNDRangeKernel(prog.kernel, cl::NullRange, cl::NDRange(SIZE), cl::NullRange);
        }
        // Create thread container
        std::vector<std::thread> threads;
        // Calculate work division size
        int block_size = 20000 / 4;
        // Construct threads with given work
        for (int i = 0; i < NUM_THREADS; i++) {
            threads.emplace_back(std::thread([&](const int start, const int end) {
                SimdUpdatePlanets(start, end);
                }, i * block_size, (i + 1) * block_size));
        }
        if (planetsLength > 20000) {
            queue.finish();
            // reading results from bufferC into the C array
            queue.enqueueReadBuffer(prog.buffer, CL_TRUE, 0, (sizeof(float) * SIZE * PLANET_SIZE), planets);
        }
        // Join threads
        for (int i = 0; i < NUM_THREADS; i++){
            threads[i].join();
        }
        ApplyForces();
    }
    void ApplyForces() {
        for (int i = 0; i < SIZE; i++) {
            if (cpu_fx[i] != 0.f) {
                planets[SIMD_IDX(i, DX)] += cpu_fx[i] / planets[SIMD_IDX(i, MASS)];
                planets[SIMD_IDX(i, DY)] += cpu_fy[i] / planets[SIMD_IDX(i, MASS)];
                cpu_fx[i] = 0.f;
                cpu_fy[i] = 0.f;
            }
            else {
                planets[SIMD_IDX(i, DX)] += planets[SIMD_IDX(i, FX)] / planets[SIMD_IDX(i, MASS)];
                planets[SIMD_IDX(i, DY)] += planets[SIMD_IDX(i, FY)] / planets[SIMD_IDX(i, MASS)];
            }
            planets[SIMD_IDX(i, X)]  += planets[SIMD_IDX(i, DX)];
            planets[SIMD_IDX(i, Y)]  += planets[SIMD_IDX(i, DY)];
            planets[SIMD_IDX(i, FX)]  = 0;
            planets[SIMD_IDX(i, FY)]  = 0;
        }
    }
    void AddPlanet(float x, float y, float dx, float dy, float mass)
    {
        if (planetsLength < SIZE) {
            int i = planetsLength;
            planets[SIMD_IDX(i,X)]      = x;
            planets[SIMD_IDX(i,Y)]      = y;
            planets[SIMD_IDX(i,DX)]     = dx;
            planets[SIMD_IDX(i,DY)]     = dy;
            planets[SIMD_IDX(i,FX)]     = 0.f;
            planets[SIMD_IDX(i,FY)]     = 0.f;
            planets[SIMD_IDX(i,MASS)]   = mass;
            planets[SIMD_IDX(i,R)]      = sqrt(mass / pi) * 128.f;
            planetsLength++;
        }
    }
    Planet GetPlanet(int id)
    {
        Planet planet;
        planet.id   = id;
        planet.x    = planets[SIMD_IDX(id, X)];
        planet.y    = planets[SIMD_IDX(id, Y)];
        planet.dx   = planets[SIMD_IDX(id, DX)];
        planet.dy   = planets[SIMD_IDX(id, DY)];
        planet.Fx   = planets[SIMD_IDX(id, FX)];
        planet.Fy   = planets[SIMD_IDX(id, FY)];
        planet.mass = planets[SIMD_IDX(id, MASS)];
        planet.r    = planets[SIMD_IDX(id, R)];
        return planet;
    }
    void SetPlanet(Planet planet) {
        int id = planet.id;
        planets[SIMD_IDX(id, X)]    = planet.x;
        planets[SIMD_IDX(id, Y)]    = planet.y;
        planets[SIMD_IDX(id, DX)]   = planet.dx;
        planets[SIMD_IDX(id, DY)]   = planet.dy;
        planets[SIMD_IDX(id, FX)]   = 0.f;
        planets[SIMD_IDX(id, FY)]   = 0.f;
        planets[SIMD_IDX(id, MASS)] = planet.mass;
        planets[SIMD_IDX(id, R)]    = sqrt(planet.mass / pi) * 128.f;
    }
    void RemovePlanet(int id) {
        planets[SIMD_IDX(id, MASS)] = 0;
    }
    void RemoveAllButSelected(int id) {
        for (int i = 0; i < SIZE; i++) {
            if(i!=id)
                planets[SIMD_IDX(i, MASS)] = 0;
        }
    }
    void AddRandomSatellite(int id) {
        Planet parent = GetPlanet(id);
        int angles = (360 * 32);
        int steps = 1000;
        int   mas_step = 1000;
        float dif_dis = max_dis - min_dis;
        float dif_prop = (1 / max_prop) - (1 / min_prop);
        float new_mass = (1 / min_prop) + float(rand() % mas_step) / mas_step * dif_prop * (parent.mass);
        float angle = (float(rand() % angles) / float(angles)) * 2.f * pi;
        float magni = (parent.r) + (parent.mass) * (min_dis + float(rand() % steps) / float(steps) * dif_dis);
        float new_x = (parent.x) + cos(angle) * magni;
        float new_y = (parent.y) + sin(angle) * magni;
        float scalar = start_vel * sqrt(0.5f * G * ((parent.mass) + 0.5f * children * new_mass));
        float new_dx = (parent.dx) + cos(angle + pi * 0.5f) * scalar;
        float new_dy = (parent.dy) + sin(angle + pi * 0.5f) * scalar;
        AddPlanet(new_x, new_y, new_dx, new_dy, new_mass);
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
    std::vector<sf::Sprite> DrawSolarSystem(sf::Texture& circle)
    {
        std::vector<sf::Sprite> sprites;
        for (int i = 0; i < SIZE;i++)
        {
            if (planets[SIMD_IDX(i,MASS)] != 0.f) {
                sf::Sprite sprite;
                sprite.setTexture(circle);
                sprite.setOrigin(128, 128);
                sprite.setPosition(planets[SIMD_IDX(i, X)], planets[SIMD_IDX(i, Y)]);
                if (i == selectedPlanet)
                {
                    if (gotoSelected)
                        camPos = { planets[SIMD_IDX(i,X)], planets[SIMD_IDX(i,Y)] };
                }
                sprite.setScale(planets[SIMD_IDX(i, R)] / 128.f, planets[SIMD_IDX(i, R)] / 128.f);
                sprites.emplace_back(sprite);
            }
        }
        return sprites;
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
        double num_ops = 16 * frameRate * double(int64_t(updatesPerFrame) * int64_t(planetsLength) * int64_t(planetsLength));
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
        ImGui::Text(std::string("PLANETS: " + std::to_string(planetsLength)).c_str());
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
        ImGui::SliderInt(": ID", &selectedPlanet, 0, planetsLength-1);
        Planet planet = system.GetPlanet(selectedPlanet);
        float dmin = -1000.0;
        float dmax = 1000.0;
        bool modified = false;
        modified |= ImGui::SliderFloat(": X",  &planet.x,    smin, smax);
        modified |= ImGui::SliderFloat(": Y",  &planet.y,    smin, smax);
        modified |= ImGui::SliderFloat(": DX", &planet.dx,   dmin, dmax);
        modified |= ImGui::SliderFloat(": DY", &planet.dy,   dmin, dmax);
        modified |= ImGui::SliderFloat(": M", &planet.mass, 0.001, 10000, "%.3f", 2.f);
        if(modified)system.SetPlanet(planet);
        ImGui::Checkbox(": Follow Selected", &gotoSelected);
        ImGui::End();

        ImGui::Begin("Modify Universe");
        ImGui::Text("Children Properties:");
        ImGui::DragFloatRange2(": 1/Size", &min_prop, &max_prop);
        ImGui::DragFloatRange2(": Distance", &min_dis, &max_dis);
        ImGui::SliderFloat(": Velocity", &start_vel,0.5f,3.f);
        ImGui::SliderInt(": Tiers", &tiers, 1, 4);
        ImGui::SliderInt(": Children", &children, 1, 300);
        ImGui::Text(std::string("Settings will add " + std::to_string(int(std::pow(children, tiers))) + " planets.").c_str());
        if (ImGui::Button("Add Universe", { 200,20 })) {
            system.RecursivelyAddPlanets(selectedPlanet, children, tiers);
        }
        if (ImGui::Button("Remove Planet", { 200,20 })) {
            system.RemovePlanet(selectedPlanet);
        }
        if (ImGui::Button("Remove All But Selected", { 200,20 })) {
            system.RemoveAllButSelected(selectedPlanet);
        }
        if (ImGui::Button("Add Planet", { 200,20 })){
            system.AddPlanet(smin, smin, 0, 0, 0.5);
            selectedPlanet = planetsLength;
        }
        if (ImGui::Button("Add Random", { 200,20 })){
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
        
        for (int i = 0; i < updatesPerFrame; i++)
        {
            system.UpdatePlanets();
        }
        //ImGui::ShowTestWindow();

        std::vector<sf::Sprite> sprites = system.DrawSolarSystem(circle);
        for (int i=0;i<sprites.size();i++)
        {
            sf::Sprite sprite = sprites[i];
            if (i == selectedPlanet)
                sprite.setColor(sf::Color(255, 0, 0, 255));
            window.draw(sprite);
        }
       
        ImGui::SFML::Render(window);
        window.display();
        frameRate = 1000000.f / float(frameClock.getElapsedTime().asMicroseconds());
        frameClock.restart();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
