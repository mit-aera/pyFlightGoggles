#include "pyFlightGogglesClient.hpp"

PyFlightGogglesClient::PyFlightGogglesClient(
    std::string yaml_path,
    std::string input_port,
    std::string output_port){
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    // flightGoggles.initializeConnections();
    flightGoggles = new FlightGogglesClient(input_port, output_port);

    flightGoggles->state.sceneFilename = cfg["state"]["sceneFilename"].as<std::string>();
    // std::cout << flightGoggles->state.sceneFilename << std::endl;
    flightGoggles->state.camWidth = cfg["state"]["camWidth"].as<int>();
    flightGoggles->state.camHeight = cfg["state"]["camHeight"].as<int>();
    flightGoggles->state.camFOV = cfg["state"]["camFOV"].as<float>();
    flightGoggles->state.camDepthScale = cfg["state"]["camDepthScale"].as<double>();

    _f = (flightGoggles->state.camHeight / 2.0) / tan((M_PI * (flightGoggles->state.camFOV / 180.0)) / 2.0);
    _cx = flightGoggles->state.camWidth / 2.0;
    _cy = flightGoggles->state.camHeight / 2.0;
    _tx = 0.0;
    _ty = 0.0;
}
