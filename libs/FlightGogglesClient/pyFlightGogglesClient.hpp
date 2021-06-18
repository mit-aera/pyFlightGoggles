#ifndef PYFLIGHTGOGGLESCLIENT_HPP
#define PYFLIGHTGOGGLESCLIENT_HPP

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

#include "yaml-cpp/yaml.h"
#include "FlightGogglesClient.hpp"

typedef Eigen::Matrix<uchar, Eigen::Dynamic, 1> VectorXuc;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1> VectorXi;

namespace py = pybind11;

class PyFlightGogglesClient{
public:
    PyFlightGogglesClient() = default;
    PyFlightGogglesClient(
        std::string yaml_path,
        std::string input_port,
        std::string output_port);

    void setCameraPose(const Eigen::Vector3d & pos, const Eigen::Vector4d & quat, int cam_index) {
        Eigen::Affine3d camera_pose;
        camera_pose.translation() = pos;
        Eigen::Quaterniond q = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
        camera_pose.linear() = q.normalized().toRotationMatrix();

        flightGoggles->setCameraPoseUsingNEDCoordinates(camera_pose, cam_index);
    }
    
    void setObjectPose(const Eigen::Vector3d & pos, const Eigen::Vector4d & quat, int object_index) {
        Eigen::Affine3d object_pose;
        object_pose.translation() = pos;
        Eigen::Quaterniond q = Eigen::Quaterniond(quat[0], quat[1], quat[2], quat[3]);
        object_pose.linear() = q.normalized().toRotationMatrix();

        flightGoggles->setObjectPoseUsingNEDCoordinates(object_pose, object_index);
    }
    
    void setCameraPoseUsingNEDCoordinates(const Eigen::Affine3d & ros_pose, int cam_index) {
        flightGoggles->setCameraPoseUsingNEDCoordinates(ros_pose, cam_index);
    }
    
    void setObjectPoseUsingNEDCoordinates(const Eigen::Affine3d & ros_pose, int object_index) {
        flightGoggles->setObjectPoseUsingNEDCoordinates(ros_pose, object_index);
    }
    
    void addCamera(std::string ID, int outputIndex, int shaderType=-1, bool hasCollisionCheck=true) {
        unity_outgoing::Camera_t cam;
        cam.ID = ID;
        cam.outputShaderType = shaderType;
    cam.hasCollisionCheck = hasCollisionCheck;
        flightGoggles->state.cameras.push_back(cam);}

    void addObject(std::string ID, std::string prefabID, double size_x, double size_y, double size_z) {
        unity_outgoing::Object_t object;
        object.ID = ID;
        object.prefabID = prefabID;
        object.size.resize(3);
        object.size.at(0) = size_x;
        object.size.at(1) = size_y;
        object.size.at(2) = size_z;
        flightGoggles->state.objects.push_back(object);
    }

    
    void setStateTime(int64_t timestamp) {
        flightGoggles->state.ntime = timestamp;}

    int64_t getTimestamp() {return flightGoggles->getTimestamp();}
    
    bool requestRender() {return flightGoggles->requestRender();}
    // std::map<std::string,VectorXuc> getImage();
    std::tuple<std::map<std::string,VectorXi>, bool, std::map<std::string,VectorXd>, double> getImage();

    void terminate() {flightGoggles->terminateConnections();}

private:
    // FlightGogglesClient flightGoggles;
    FlightGogglesClient* flightGoggles;

    // Camera Info
    float _f, _cx, _cy, _tx, _ty;
};


std::tuple<std::map<std::string,VectorXi>, bool, std::map<std::string,VectorXd>, double> 
    PyFlightGogglesClient::getImage(){
    std::map<std::string, VectorXi> vec;

    // const cv::Mat & img = cv::Mat::zeros(1024,768,CV_8UC3);
    unity_incoming::RenderOutput_t rendered_output = flightGoggles->handleImageResponse();
    for (auto i=0; i<rendered_output.renderMetadata.cameraIDs.size(); i++) {
        //int * arr = rendered_output.images.at(i).isContinuous() ? 
        //    rendered_output.images.at(i).data : rendered_output.images.at(i).clone().data;
        int length = rendered_output.images.at(i).total()*rendered_output.images.at(i).channels();
        int* arr;
        arr = (int*) malloc(sizeof(int)*length);
    if (rendered_output.renderMetadata.channels[i] == 2)
            for (auto ii=0; ii<length;ii++) arr[ii] = rendered_output.images.at(i).at<uint16_t>(ii);
        else
            for (auto ii=0; ii<length;ii++) arr[ii] = rendered_output.images.at(i).at<uint8_t>(ii);

        //memcpy(rendered_output.images.at(i).data, arr, sizeof(int)*length);

        Eigen::Map<VectorXi> img_eigen(arr, length);

        vec.insert(std::pair<std::string, VectorXi>
            (rendered_output.renderMetadata.cameraIDs[i], img_eigen));
    delete arr;
    }

    // Collision
    bool hasCameraCollision = rendered_output.renderMetadata.hasCameraCollision;
    
    // Landmark
    std::map<std::string, VectorXd> landmark_t;
    for (auto i=0; i<rendered_output.renderMetadata.landmarksInView.size(); i++) {
        Eigen::VectorXd ld_pos_t = Eigen::Map<Eigen::VectorXd>(
            rendered_output.renderMetadata.landmarksInView[i].position.data(), 
            rendered_output.renderMetadata.landmarksInView[i].position.size());
        landmark_t.insert(std::pair<std::string, VectorXd>
            (rendered_output.renderMetadata.landmarksInView[i].ID, ld_pos_t));
    }
    
    // Lidar
    double lidarReturn = rendered_output.renderMetadata.lidarReturn;
    
    // return vec;
    return std::tuple<std::map<std::string,VectorXi>, bool, std::map<std::string,VectorXd>, double>
        (vec, hasCameraCollision, landmark_t, lidarReturn);
}

void bind_FlightGogglesClient(py::module &m) {
    py::class_<PyFlightGogglesClient>(m, "FlightGogglesClient")
        .def(py::init<>())
        .def(py::init<
            std::string, std::string, std::string>(),
            py::arg("yaml_path"),
            py::arg("input_port")="10253", 
            py::arg("output_port")="10254")
        .def("setCameraPose", &PyFlightGogglesClient::setCameraPose, "set camera pose")
        .def("setCameraPoseUsingNEDCoordinates", 
            &PyFlightGogglesClient::setCameraPoseUsingNEDCoordinates, 
            "set camera pose using NED coordinates")
        .def("setObjectPose", &PyFlightGogglesClient::setObjectPose, "set object pose")
        .def("setObjectPoseUsingNEDCoordinates", 
            &PyFlightGogglesClient::setObjectPoseUsingNEDCoordinates, 
            "set object pose using NED coordinates")
        .def("addCamera", &PyFlightGogglesClient::addCamera, "add camera")
        .def("addObject", &PyFlightGogglesClient::addObject, "add object")
        .def("setStateTime", &PyFlightGogglesClient::setStateTime, "set state timestamp")
        .def("getTimestamp", &PyFlightGogglesClient::getTimestamp, "get state timestamp")
        .def("requestRender", &PyFlightGogglesClient::requestRender, "request rendering")
        .def("getImage", &PyFlightGogglesClient::getImage, "get image rendered")
        .def("terminate", &PyFlightGogglesClient::terminate, "terminate connections");
}

#endif
