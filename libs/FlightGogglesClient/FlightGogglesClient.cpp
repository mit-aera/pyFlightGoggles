/**
 * @file   FlightGogglesClient.cpp
 * @author Winter Guerra
 * @brief  Node that outputs the expected pose of cameras based on motion capture
 * data to Unity for FlightGoggles operation.
 */

#include "FlightGogglesClient.hpp"

/**
 * Constructor
 */
FlightGogglesClient::FlightGogglesClient()
{
    initializeConnections();
}

/**
 * Constructor
 */
FlightGogglesClient::FlightGogglesClient(
    std::string _upload_port, 
    std::string _download_port)
{
    upload_port = _upload_port;
    download_port = _download_port;

    initializeConnections();
}

/**
 * initializeConnections initializeConnections with Unity
 */
void FlightGogglesClient::initializeConnections()
{
    std::cout << "Initializing ZMQ connections..." << std::endl;
    // create and bind a upload_socket
    upload_socket.set(zmqpp::socket_option::send_high_water_mark, 6);
    upload_socket.bind(client_address + ":" + upload_port);
    // create and bind a download_socket
    download_socket.set(zmqpp::socket_option::receive_high_water_mark, 6);
    download_socket.bind(client_address + ":" + download_port);
    download_socket.subscribe("");
    std::cout << "Done!" << std::endl;
}

/**
 * terminateConnections terminateConnections with Unity
 */
void FlightGogglesClient::terminateConnections()
{
    std::cout << "Terminating ZMQ connections..." << std::endl;
    // upload_socket.unbind(client_address + ":" + upload_port);
    // download_socket.unbind(client_address + ":" + download_port);
    upload_socket.close();
    download_socket.close();
    delete upload_socket;
    delete download_socket;
    std::cout << "Done!" << std::endl;
}

/**
 * setCameraPoseUsingROSCoordinates accepts camera pose in ros frame
 * @param ros_pose Pose of the camera
 * @param cam_index Index of the camera
 */
void FlightGogglesClient::setCameraPoseUsingNEDCoordinates(Transform3 NED_pose, int cam_index) {
    // To transforms
    Transform3 unity_pose = convertNEDGlobalPoseToGlobalUnityCoordinates(NED_pose);
    // Transform3 unity_pose = convertEDNGlobalPoseToGlobalUnityCoordinates(ros_pose);

    // Extract position and rotation
    std::vector<double> position = {
        unity_pose.translation()[0],
        unity_pose.translation()[1],
        unity_pose.translation()[2],
    };

    Eigen::Matrix3d rotationMatrix = unity_pose.rotation();
    Quaternionx quat(rotationMatrix);

    std::vector<double> rotation = {
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w(),
    };

    // Set camera position and rotation
    state.cameras[cam_index].position = position;
    state.cameras[cam_index].rotation = rotation;
}

/**
 * setCameraPoseUsingROSCoordinates accepts camera pose in ros frame
 * @param ros_pose Pose of the camera
 * @param cam_index Index of the camera
 */
void FlightGogglesClient::setObjectPoseUsingNEDCoordinates(Transform3 NED_pose, int object_index) {
    // To transforms
    Transform3 unity_pose = convertNEDGlobalPoseToGlobalUnityCoordinates(NED_pose);
    // Transform3 unity_pose = convertEDNGlobalPoseToGlobalUnityCoordinates(ros_pose);

    // Extract position and rotation
    std::vector<double> position = {
        unity_pose.translation()[0],
        unity_pose.translation()[1],
        unity_pose.translation()[2],
    };

    Eigen::Matrix3d rotationMatrix = unity_pose.rotation();
    Quaternionx quat(rotationMatrix);

    std::vector<double> rotation = {
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w(),
    };

    std::cout << "Objects size: " << state.objects.size() << std::endl;
    // Set camera position and rotation
    state.objects[object_index].position = position;
    state.objects[object_index].rotation = rotation;
}

/**
 * setCameraPoseUsingROSCoordinates accepts camera pose in ros frame
 * @param ros_pose Pose of the camera
 * @param cam_index Index of the camera
 */
void FlightGogglesClient::setCameraPoseUsingROSCoordinates(Transform3 ros_pose, int cam_index) {
    // To transforms
    Transform3 NED_pose = convertROSToNEDCoordinates(ros_pose);
    Transform3 unity_pose = convertNEDGlobalPoseToGlobalUnityCoordinates(NED_pose);
    // Transform3 unity_pose = convertEDNGlobalPoseToGlobalUnityCoordinates(ros_pose);

    // Extract position and rotation
    std::vector<double> position = {
        unity_pose.translation()[0],
        unity_pose.translation()[1],
        unity_pose.translation()[2],
    };

    Eigen::Matrix3d rotationMatrix = unity_pose.rotation();
    Quaternionx quat(rotationMatrix);

    std::vector<double> rotation = {
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w(),
    };

    // Set camera position and rotation
    state.cameras[cam_index].position = position;
    state.cameras[cam_index].rotation = rotation;
}

/**
 * setCameraPoseUsingROSCoordinates accepts camera pose in ros frame
 * @param ros_pose Pose of the camera
 * @param cam_index Index of the camera
 */
void FlightGogglesClient::setObjectPoseUsingROSCoordinates(Transform3 ros_pose, int object_index) {
    // To transforms
    Transform3 NED_pose = convertROSToNEDCoordinates(ros_pose);
    Transform3 unity_pose = convertNEDGlobalPoseToGlobalUnityCoordinates(NED_pose);
    // Transform3 unity_pose = convertEDNGlobalPoseToGlobalUnityCoordinates(ros_pose);

    // Extract position and rotation
    std::vector<double> position = {
        unity_pose.translation()[0],
        unity_pose.translation()[1],
        unity_pose.translation()[2],
    };

    Eigen::Matrix3d rotationMatrix = unity_pose.rotation();
    Quaternionx quat(rotationMatrix);

    std::vector<double> rotation = {
        quat.x(),
        quat.y(),
        quat.z(),
        quat.w(),
    };

    std::cout << "Objects size: " << state.objects.size() << std::endl;
    // Set camera position and rotation
    state.objects[object_index].position = position;
    state.objects[object_index].rotation = rotation;
}

/**
 * This function is called when a new pose has been received.
 * If the pose is good, asks Unity to render another frame by sending a ZMQ
 * message.
*/
bool FlightGogglesClient::requestRender()
{
    // Create new message object
    zmqpp::message msg;
    // Add topic header
    msg << "Pose";

    // Update timestamp
    last_uploaded_utime = state.ntime;

    // Create JSON object for status update & append to message.
    json json_msg = state;
    msg << json_msg.dump();

    // Output debug messages at a low rate
    if (state.ntime > last_upload_debug_utime + 1e9)
    {
        last_upload_debug_utime = state.ntime;
    }
    // Send message without blocking.
    upload_socket.send(msg, true);
    return true;
}

/**
 * handleImageResponse handles the image response from Unity
 * Note: This is a blocking call.
 *
 * @return RenderOutput_t returns the render output with metadata
 */
unity_incoming::RenderOutput_t FlightGogglesClient::handleImageResponse()
{
    // Populate output
    unity_incoming::RenderOutput_t output;

    // Get data from client as fast as possible
    zmqpp::message msg;
    download_socket.receive(msg);

    // Unpack message metadata.
    std::string json_metadata_string = msg.get(0);
    
    // Parse metadata.
    unity_incoming::RenderMetadata_t renderMetadata = json::parse(json_metadata_string);

    // Log the latency in ms (1,000 microseconds)
    if (!u_packet_latency)
    {
        u_packet_latency = (getTimestamp() - renderMetadata.ntime);
    }
    else
    {
        // avg over last ~10 frames in ms.
        u_packet_latency =
            ((u_packet_latency * (9) + (getTimestamp() - renderMetadata.ntime*1e-3)) / 10.0f);
    }

    ensureBufferIsAllocated(renderMetadata);

    output.images.resize(renderMetadata.cameraIDs.size());

    // For each camera, save the received image.
    auto num_threads = renderMetadata.cameraIDs.size();
    const uint8_t stride = 3;
    
    for (int i = 0; i < num_threads; i++) {
        cv::Mat new_image;
        // Reshape the received image
        if (renderMetadata.channels[i] != 2) {
            // Get raw image bytes from ZMQ message.
            // WARNING: This is a zero-copy operation that also casts the input to an array of unit8_t.
            // when the message is deleted, this pointer is also dereferenced.
            const uint8_t* imageData;
            msg.get(imageData, i + 1);
            // // ALL images comes as 3-channel RGB images from Unity. Calculate the row length
            // uint32_t bufferRowLength = renderMetadata.camWidth * renderMetadata.channels[i];

            // Pack image into cv::Mat
            new_image = cv::Mat(renderMetadata.camHeight, renderMetadata.camWidth, CV_MAKETYPE(CV_8U, stride));
            memcpy(new_image.data, imageData, renderMetadata.camWidth * renderMetadata.camHeight * stride );
        } else {
            // This is a 16UC1 depth image
            // Get raw image bytes from ZMQ message.
            // WARNING: This is a zero-copy operation that also casts the input to an array of unit8_t.
            // when the message is deleted, this pointer is also dereferenced.
            const uint16_t* imageData;
            msg.get(imageData, i + 1);
            // // ALL images comes as 3-channel RGB images from Unity. Calculate the row length
            // uint32_t bufferRowLength = renderMetadata.camWidth * renderMetadata.channels[i];
            new_image = cv::Mat(renderMetadata.camHeight, renderMetadata.camWidth, CV_MAKETYPE(CV_16U, 1));
            memcpy(new_image.data, imageData, renderMetadata.camWidth * renderMetadata.camHeight * 2);
       }

        // Tell OpenCv that the input is RGB.
        if (renderMetadata.channels[i]==3){
            if (stride == 3)
                cv::cvtColor(new_image, new_image, CV_RGB2BGR);
            if (stride == 4)
                cv::cvtColor(new_image, new_image, CV_BGRA2BGR);
        }

        // Flip image since OpenCV origin is upper left, but Unity's is lower left.
        cv::flip(new_image, new_image, 0);

        // Add image to output vector
        output.images.at(i) = new_image;
    }

    // Add metadata to output
    output.renderMetadata = renderMetadata;

    // Output debug at 1hz
    if (getTimestamp() > last_download_debug_utime + 1e6)
    {
        last_download_debug_utime = getTimestamp();
        num_frames = 0;
    }
    num_frames++;

    return output;
}
