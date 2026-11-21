cmake_minimum_required(VERSION 3.20)
project(fftgyro_executor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FFTGYRO_BUILD_TESTS "Build unit tests" ON)
option(FFTGYRO_RPM_ONLY "Build standalone RPM receiver without MAVLink dependencies" OFF)
option(FFTGYRO_REQUIRE_GPIOD "Require GPIO support for aircraft deployment" OFF)
set(FFTGYRO_MAVLINK_ROOT "" CACHE PATH "Path to the PX4 MAVLink checkout")

find_package(Threads REQUIRED)
add_executable(fftgyro-rpm src/rpm_capture.cpp)
target_link_libraries(fftgyro-rpm PRIVATE Threads::Threads)
target_compile_options(fftgyro-rpm PRIVATE -Wall -Wextra -Wpedantic -Werror)
install(TARGETS fftgyro-rpm RUNTIME DESTINATION bin)
if(FFTGYRO_RPM_ONLY)
  return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(nlohmann_json 3.11 QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
  find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "Install nlohmann-json3-dev")
  endif()
  add_library(nlohmann_json_headers INTERFACE)
  target_include_directories(nlohmann_json_headers INTERFACE "${NLOHMANN_JSON_INCLUDE_DIR}")
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json_headers)
endif()
if(NOT FFTGYRO_MAVLINK_ROOT)
  message(FATAL_ERROR "Set FFTGYRO_MAVLINK_ROOT to PX4-Autopilot/src/modules/mavlink/mavlink")
endif()

set(PROTOCOL_XML "${CMAKE_CURRENT_LIST_DIR}/../protocol/fftgyro.xml")
set(MAVGEN "${FFTGYRO_MAVLINK_ROOT}/pymavlink/tools/mavgen.py")
set(MAVLINK_DEFINITIONS "${FFTGYRO_MAVLINK_ROOT}/message_definitions/v1.0")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GENERATED_HEADER "${GENERATED_DIR}/fftgyro/fftgyro.h")
file(GLOB_RECURSE MAVLINK_GENERATOR_INPUTS CONFIGURE_DEPENDS
  "${FFTGYRO_MAVLINK_ROOT}/pymavlink/generator/*.py"
  "${MAVLINK_DEFINITIONS}/*.xml")

add_custom_command(
  OUTPUT "${GENERATED_HEADER}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/cmake/generate_mavlink.py"
          --mavgen "${MAVGEN}"
          --definitions "${MAVLINK_DEFINITIONS}"
          --dialect "${PROTOCOL_XML}"
          --output "${GENERATED_DIR}"
  DEPENDS "${PROTOCOL_XML}" "${CMAKE_CURRENT_LIST_DIR}/cmake/generate_mavlink.py" "${MAVGEN}"
          ${MAVLINK_GENERATOR_INPUTS}
  COMMENT "Generating FFTGYRO MAVLink headers"
  VERBATIM
)
add_custom_target(fftgyro_mavlink_headers DEPENDS "${GENERATED_HEADER}")

add_library(fftgyro_experiment src/experiment.cpp src/session.cpp src/axis_controller.cpp src/collective_ramp.cpp src/coupled_controller.cpp)
add_dependencies(fftgyro_experiment fftgyro_mavlink_headers)
target_include_directories(fftgyro_experiment PUBLIC include)
target_include_directories(fftgyro_experiment SYSTEM PUBLIC "${GENERATED_DIR}")
target_link_libraries(fftgyro_experiment PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(fftgyro_experiment PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(fftgyro-executor src/main.cpp src/mavlink_uart.cpp src/session_runner.cpp)
add_dependencies(fftgyro-executor fftgyro_mavlink_headers)
target_link_libraries(fftgyro-executor PRIVATE fftgyro_experiment)
target_compile_options(fftgyro-executor PRIVATE -Wall -Wextra -Wpedantic -Werror)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(GPIOD QUIET IMPORTED_TARGET libgpiod)
endif()
if(TARGET PkgConfig::GPIOD)
  target_link_libraries(fftgyro-executor PRIVATE PkgConfig::GPIOD)
  target_compile_definitions(fftgyro-executor PRIVATE FFTGYRO_HAVE_GPIOD=1)
else()
  if(FFTGYRO_REQUIRE_GPIOD)
    message(FATAL_ERROR "Aircraft deployment requires libgpiod-dev and pkg-config")
  endif()
  message(STATUS "libgpiod not found: GPIO-enabled configs will be rejected")
endif()

install(TARGETS fftgyro-executor RUNTIME DESTINATION bin)

if(FFTGYRO_BUILD_TESTS)
  enable_testing()
  add_executable(fftgyro-upstep-tests tests/test_upstep.cpp)
  target_link_libraries(fftgyro-upstep-tests PRIVATE fftgyro_experiment)
  add_test(NAME fftgyro-upstep-tests COMMAND fftgyro-upstep-tests)
  add_executable(fftgyro-coupled-tests tests/test_coupled.cpp)
  target_link_libraries(fftgyro-coupled-tests PRIVATE fftgyro_experiment)
  target_compile_options(fftgyro-coupled-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME fftgyro-coupled-tests COMMAND fftgyro-coupled-tests
    "${CMAKE_CURRENT_LIST_DIR}/config/coupled-session.example.json")
  add_executable(fftgyro-cascade-tests tests/test_cascade.cpp)
  target_link_libraries(fftgyro-cascade-tests PRIVATE fftgyro_experiment)
  target_compile_options(fftgyro-cascade-tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
  add_test(NAME fftgyro-cascade-tests COMMAND fftgyro-cascade-tests
    "${CMAKE_CURRENT_LIST_DIR}/config/pitch-cascade-commissioning.example.json")
  add_executable(fftgyro-executor-tests tests/test_core.cpp)
  add_dependencies(fftgyro-executor-tests fftgyro_mavlink_headers)
  target_link_libraries(fftgyro-executor-tests PRIVATE fftgyro_experiment)
  target_compile_options(fftgyro-executor-tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  add_test(NAME fftgyro-executor-tests COMMAND fftgyro-executor-tests)
  add_executable(fftgyro-session-tests tests/test_session.cpp)
  target_link_libraries(fftgyro-session-tests PRIVATE fftgyro_experiment)
  target_compile_options(fftgyro-session-tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)
  add_test(NAME fftgyro-session-tests COMMAND fftgyro-session-tests
    "${CMAKE_CURRENT_LIST_DIR}/config/axis-session.example.json")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME fftgyro-coupled-transport COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_coupled_transport.py" "$<TARGET_FILE:fftgyro-executor>"
      "${CMAKE_CURRENT_LIST_DIR}/config/coupled-session.example.json")
    add_test(NAME fftgyro-absolute-pitch-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_absolute_pitch_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/pitch-cascade-commissioning.example.json")
    set_tests_properties(fftgyro-absolute-pitch-transport-tests PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-absolute-yaw-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_absolute_yaw_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/pitch-cascade-commissioning.example.json")
    set_tests_properties(fftgyro-absolute-yaw-transport-tests PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-collective-review-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_collective_review.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/pitch-collective-compensation.example.json")
    set_tests_properties(fftgyro-collective-review-tests PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-cascade-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_cascade_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/pitch-cascade-commissioning.example.json")
    set_tests_properties(fftgyro-cascade-transport-tests PROPERTIES TIMEOUT 200 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-upstep-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_upstep_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/axis-session.example.json")
    set_tests_properties(fftgyro-upstep-transport-tests PROPERTIES TIMEOUT 40 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-slow-fixed-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_slow_fixed.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/props-off-hold.example.json")
    set_tests_properties(fftgyro-slow-fixed-tests PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-pitch-hold-commissioning-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_pitch_hold_commissioning.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/pitch-hold-commissioning.example.json")
    set_tests_properties(fftgyro-pitch-hold-commissioning-tests PROPERTIES TIMEOUT 180 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-session-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_session_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/axis-session.example.json")
    set_tests_properties(fftgyro-session-transport-tests PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-commissioning-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_commissioning_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/props-off-hold.example.json")
    set_tests_properties(fftgyro-commissioning-transport-tests PROPERTIES TIMEOUT 420 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-fixed-commissioning-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_fixed_commissioning.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/props-off-hold.example.json")
    set_tests_properties(fftgyro-fixed-commissioning-tests PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
    add_test(NAME fftgyro-props-off-guard-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_props_off_guard.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/props-off-hold.example.json")
    add_test(NAME fftgyro-props-off-transport-tests
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/tests/test_session_transport.py"
        "$<TARGET_FILE:fftgyro-executor>" "${CMAKE_CURRENT_LIST_DIR}/config/props-off-hold.example.json" --props-off)
    set_tests_properties(fftgyro-props-off-guard-tests fftgyro-props-off-transport-tests
      PROPERTIES TIMEOUT 90 RUN_SERIAL TRUE)
  endif()
endif()
