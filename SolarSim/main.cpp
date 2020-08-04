#include "imgui.h"
#include "imgui-SFML.h"
#include "zlib.h"

#include "absl/container/flat_hash_map.h"

#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    sf::RenderWindow window(sf::VideoMode(640, 480), "ImGui + SFML = <3");
    window.setFramerateLimit(60);
    ImGui::SFML::Init(window);

    absl::flat_hash_map<std::string, std::string> ducks = {{"a", "huey"}, {"b", "dewey"}, {"c", "louie"}};
    ducks.emplace("d", "brewey");
    std::string search_key = "d";
    auto result = ducks.find(search_key);

    if (result != ducks.end()) {
        std::cout << "Result: " << result->second << std::endl;
    }

    sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    sf::Clock deltaClock;
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);

            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        //ImGui::ShowTestWindow();
        
        ImGui::Begin("Hello, world!");
        if (ImGui::Button("Look at this pretty button"))
            std::cout << "Hey its rudimentary but it works \n";
        ImGui::End();

        window.clear();
        window.draw(shape);
        ImGui::SFML::Render(window);
        window.display();
    }

    ImGui::SFML::Shutdown();
    
    return 0;
}
