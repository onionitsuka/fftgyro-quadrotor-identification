#include "fftgyro_executor/executor.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message)
{
	if (!condition) {
		throw std::runtime_error(message);
	}
}

bool near(double actual, double expected, double tolerance = 1e-6)
{
	return std::abs(actual - expected) <= tolerance;
}

fftgyro::Config base_config()
{
	fftgyro::Config config;
	config.run_name = "unit-test";
	config.uart_device = "/dev/null";
	config.collective = 0.4;
	config.motor_min = 0.1;
	config.motor_max = 0.8;
	config.axis_mix = {1.0, -1.0, -1.0, 1.0};
	config.motor_mapping_verified = true;
	config.collective_verified = true;
	config.startup_ramp_rate = 0.2;
	config.normal_ramp_rate = 0.1;
	config.abort_ramp_rate = 0.5;
	config.log_directory = "/tmp";
	config.gpio_enabled = true;
	config.waveform.type = fftgyro::WaveformType::Hold4;
	config.waveform.amplitude = 0.0;
	config.waveform.active_s = 0.1;
	return config;
}

void test_hold()
{
	const auto config = base_config();
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t startup = 200;
	require(samples.size() == startup + 10, "HOLD4 sample count");
	require(near(samples.front().motor[0], 0.002), "startup first step");
	require(samples.front().stage == fftgyro::SampleStage::Startup, "startup stage");

	for (auto i = startup; i < samples.size(); ++i) {
		const auto &sample = samples[i];
		for (const float motor : sample.motor) {
			require(near(motor, 0.4), "HOLD4 collective");
		}
	}
}

void test_pulse_coast()
{
	auto config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.05;
	config.waveform.preroll_s = 0.02;
	config.waveform.active_s = 0.03;
	config.waveform.posthold_s = 0.02;
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t startup = 200;
	require(samples.size() == startup + 7, "PULSE_COAST sample count");
	require(near(samples[startup].motor[0], 0.4), "pulse preroll");
	require(samples[startup + 2].sync_start, "pulse sync position");
	require(near(samples[startup + 2].motor[0], 0.45), "pulse positive pair");
	require(near(samples[startup + 2].motor[1], 0.35), "pulse negative pair");
	require(near(samples[startup + 5].motor[0], 0.4), "pulse coast");
}

void test_pulse_edge_and_negative_direction()
{
	auto config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = -0.05;
	config.waveform.active_s = 0.10;
	config.waveform.posthold_s = 0.02;
	config.waveform.edge_s = 0.02;
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t active = 200;
	require(near(samples[active].motor[0], 0.4), "pulse edge starts neutral");
	require(samples[active + 2].motor[0] < 0.36, "negative pulse reaches requested direction");
	require(near(samples[active + 9].motor[0], 0.4), "pulse edge ends neutral");
}

void test_axis_bias_preserves_collective_and_offsets_waveform()
{
	auto config = base_config();
	config.axis_bias = 0.01;
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.02;
	config.waveform.preroll_s = 0.02;
	config.waveform.active_s = 0.03;
	config.waveform.posthold_s = 0.02;
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t startup = 205;
	require(near(samples[startup].motor[0], 0.41), "biased preroll positive pair");
	require(near(samples[startup].motor[1], 0.39), "biased preroll negative pair");
	require(near(samples[startup + 2].motor[0], 0.43), "waveform adds to bias");
	require(near(samples[startup + 2].motor[1], 0.37), "waveform subtracts from bias");
	double mean = 0.0;
	for (float motor : samples[startup + 2].motor) mean += motor / 4.0;
	require(near(mean, 0.4), "axis bias and waveform preserve collective mean");
}

void test_sine_fade()
{
	auto config = base_config();
	config.gpio_enabled = false;
	config.waveform.type = fftgyro::WaveformType::SineAxis;
	config.waveform.amplitude = 0.05;
	config.waveform.active_s = 1.0;
	config.waveform.posthold_s = 0.02;
	config.waveform.fade_s = 0.2;
	config.waveform.frequency_hz = 1.0;
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t startup = 200;
	require(near(samples[startup].motor[0], 0.4), "sine fade-in starts at collective");
	require(near(samples.back().motor[0], 0.4), "sine fade-out ends at collective");
	require(std::abs(samples[startup + 25].motor[0] - 0.4) > 0.04,
		"sine reaches requested amplitude");
}

void test_chirp()
{
	auto config = base_config();
	config.gpio_enabled = false;
	config.waveform.type = fftgyro::WaveformType::LogChirpAxis;
	config.waveform.amplitude = 0.02;
	config.waveform.active_s = 2.0;
	config.waveform.posthold_s = 0.02;
	config.waveform.fade_s = 0.1;
	config.waveform.start_frequency_hz = 0.25;
	config.waveform.end_frequency_hz = 2.0;
	const auto samples = fftgyro::build_waveform(config);
	const std::size_t startup = 200;
	require(near(samples[startup].frequency_hz, 0.25), "chirp starts at 0.25 Hz");
	require(samples[startup + 100].frequency_hz > samples[startup + 50].frequency_hz,
		"chirp frequency increases");
	require(samples.back().frequency_hz < 2.0, "last sampled chirp frequency is below endpoint");
}

void test_bounds_rejection()
{
	auto config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.5;
	config.waveform.posthold_s = 0.02;

	bool rejected = false;

	try {
		(void)fftgyro::build_waveform(config);

	} catch (const std::runtime_error &) {
		rejected = true;
	}

	require(rejected, "out-of-bounds waveform must be rejected");
}

void test_slew_rejection()
{
	auto config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.10;
	config.waveform.active_s = 0.10;
	config.waveform.posthold_s = 0.02;

	bool rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "large single-sample decrease must be rejected");

	config.waveform.edge_s = 0.10;
	config.waveform.active_s = 0.30;
	require(!fftgyro::build_waveform(config).empty(), "ramped pulse should pass slew check");
}

void test_abort_neutralization_is_nonincreasing()
{
	const std::array<float, 4> motor{0.31F, 0.29F, 0.28F, 0.30F};
	const auto target = fftgyro::abort_neutral_target(motor);
	for (std::size_t i = 0; i < motor.size(); ++i) {
		require(target[i] <= motor[i], "abort neutralization must not increase a motor");
		require(near(target[i], 0.28), "abort target must be the current minimum");
	}
}

fftgyro::AngleGuardConfig angle_guard()
{
	fftgyro::AngleGuardConfig guard;
	guard.enabled = true;
	guard.limits_verified = true;
	guard.attitude_pitch_sign = 1.0;
	guard.attitude_center_deg = 0.0;
	guard.axis_response_sign = 1.0;
	guard.recovery_deg = 10.0;
	guard.warning_deg = 20.0;
	guard.soft_limit_deg = 25.0;
	guard.abort_limit_deg = 30.0;
	guard.brake_amplitude = 0.005;
	guard.max_abs_rate_dps = 100.0;
	guard.max_sample_age_ms = 100;
	return guard;
}

fftgyro::AttitudeSample attitude(double pitch_deg, double rate_dps)
{
	constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;
	fftgyro::AttitudeSample sample;
	sample.valid = true;
	sample.pitch_rad = pitch_deg * degrees_to_radians;
	sample.pitch_rate_rad_s = rate_dps * degrees_to_radians;
	return sample;
}

void test_pitch_angle_guard()
{
	const std::array<double, 4> mix{1.0, -1.0, -1.0, 1.0};
	const std::array<float, 4> outward{0.305F, 0.295F, 0.295F, 0.305F};

	auto decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, outward, attitude(0.0, 0.0), false);
	require(decision.state == fftgyro::AngleGuardState::Nominal, "guard nominal state");
	require(near(decision.applied_axis_delta, 0.005), "guard preserves nominal delta");

	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, outward, attitude(22.5, 1.0), false);
	require(decision.state == fftgyro::AngleGuardState::Fading, "guard warning fade");
	require(decision.applied_axis_delta > 0.0
		&& decision.applied_axis_delta < decision.nominal_axis_delta,
		"guard reduces outward input before soft limit");

	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, outward, attitude(25.0, 5.0), false);
	require(decision.state == fftgyro::AngleGuardState::Braking, "guard braking state");
	require(decision.latched, "guard latches at soft limit");
	require(near(decision.applied_axis_delta, -0.005), "guard reverses differential");

	const std::array<float, 4> inward{0.295F, 0.305F, 0.305F, 0.295F};
	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, inward, attitude(15.0, -5.0), true);
	require(decision.latched, "guard remains latched outside recovery band");
	require(near(decision.applied_axis_delta, -0.005), "guard permits inward input");

	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, inward, attitude(9.0, -2.0), true);
	require(!decision.latched, "guard releases inside recovery band while returning");
	require(decision.state == fftgyro::AngleGuardState::Nominal, "guard returns nominal");

	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, outward, attitude(30.0, 0.0), false);
	require(decision.abort_angle, "guard marks absolute angle trip");
	decision = fftgyro::apply_pitch_angle_guard(
		angle_guard(), mix, outward, attitude(0.0, 100.0), false);
	require(decision.abort_rate, "guard marks angular-rate trip");

	auto offset_guard = angle_guard();
	offset_guard.attitude_center_deg = 12.0;
	decision = fftgyro::apply_pitch_angle_guard(
		offset_guard, mix, outward, attitude(12.0, 0.0), false);
	require(near(decision.pitch_deg, 0.0), "guard subtracts measured center angle");
}

void test_persistent_boundary_reflection()
{
	const std::array<double, 4> mix{1.0, -1.0, 1.0, -1.0};
	const std::array<float, 4> outward{0.305F, 0.295F, 0.305F, 0.295F};
	auto guard = angle_guard();
	guard.boundary_reflection = true;

	auto decision = fftgyro::apply_pitch_angle_guard(
		guard, mix, outward, attitude(25.0, 2.0), false, 0);
	require(decision.state == fftgyro::AngleGuardState::ReflectingNegative,
		"positive boundary starts persistent negative reflection");
	require(decision.reflection_direction == -1, "negative reflection is latched");
	require(near(decision.applied_axis_delta, -0.005),
		"positive boundary reverses differential");

	decision = fftgyro::apply_pitch_angle_guard(
		guard, mix, outward, attitude(0.0, -3.0), true,
		decision.reflection_direction);
	require(decision.state == fftgyro::AngleGuardState::ReflectingNegative,
		"negative reflection persists through center");
	require(decision.reflection_direction == -1,
		"negative direction remains latched before opposite boundary");

	decision = fftgyro::apply_pitch_angle_guard(
		guard, mix, outward, attitude(-25.0, -2.0), true,
		decision.reflection_direction);
	require(decision.state == fftgyro::AngleGuardState::ReflectingPositive,
		"negative boundary flips reflection positive");
	require(decision.reflection_direction == 1, "positive reflection is latched");
	require(near(decision.applied_axis_delta, 0.005),
		"negative boundary applies positive differential");

	decision = fftgyro::apply_pitch_angle_guard(
		guard, mix, outward, attitude(0.0, 3.0), true,
		decision.reflection_direction);
	require(decision.state == fftgyro::AngleGuardState::ReflectingPositive,
		"positive reflection persists through center");

	decision = fftgyro::apply_pitch_angle_guard(
		guard, mix, outward, attitude(25.0, 2.0), true,
		decision.reflection_direction);
	require(decision.state == fftgyro::AngleGuardState::ReflectingNegative,
		"positive boundary flips reflection negative again");
	require(decision.reflection_direction == -1,
		"negative reflection relatches on the next cycle");
}

void test_preflight_gates()
{
	auto config = base_config();
	config.collective_verified = false;
	bool rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "unverified collective must be rejected");

	config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.01;
	config.waveform.posthold_s = 0.02;
	config.motor_mapping_verified = false;
	rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "unverified axis mapping must be rejected");

	config = base_config();
	config.waveform.type = fftgyro::WaveformType::PulseCoast;
	config.waveform.amplitude = 0.01;
	config.waveform.posthold_s = 0.02;
	config.axis_mix = {1.0, 0.0, 0.0, 0.0};
	rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "non-differential axis mix must be rejected");

	config = base_config();
	config.waveform.requires_mechanical_lock = true;
	rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "required mechanical lock must be confirmed");

	config = base_config();
	config.waveform.type = fftgyro::WaveformType::SineAxis;
	config.waveform.amplitude = 0.005;
	config.waveform.active_s = 4.0;
	config.waveform.posthold_s = 0.02;
	config.waveform.fade_s = 1.0;
	config.waveform.requires_angle_guard = true;
	rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "required angle guard must be enabled");

	config = base_config();
	config.startup_ramp_rate = 0.21;
	rejected = false;
	try {
		(void)fftgyro::build_waveform(config);
	} catch (const std::runtime_error &) {
		rejected = true;
	}
	require(rejected, "fast startup ramp must be rejected");
}

void test_mavlink_roundtrip()
{
	require(MAVLINK_MSG_ID_FFTGYRO_MOTOR_SET_LEN == 29, "MOTOR_SET payload length");
	require(MAVLINK_MSG_ID_FFTGYRO_MOTOR_SET_CRC == 50, "MOTOR_SET CRC");
	require(MAVLINK_MSG_ID_FFTGYRO_MOTOR_SET_ACK_LEN == 19, "MOTOR_SET_ACK payload length");
	require(MAVLINK_MSG_ID_FFTGYRO_MOTOR_SET_ACK_CRC == 221, "MOTOR_SET_ACK CRC");

	const std::array<float, 4> motor{0.1F, 0.2F, 0.3F, 0.4F};
	mavlink_message_t message{};
	mavlink_msg_fftgyro_motor_set_pack(
		42,
		191,
		&message,
		1234,
		55,
		motor.data(),
		150,
		1,
		1,
		FFTGYRO_MOTOR_ACTION_CONTROL);
	require(message.msgid == 60000, "MOTOR_SET message ID");

	mavlink_fftgyro_motor_set_t decoded{};
	mavlink_msg_fftgyro_motor_set_decode(&message, &decoded);
	require(decoded.run_id == 1234, "MOTOR_SET run ID");
	require(decoded.sequence == 55, "MOTOR_SET sequence");
	require(decoded.timeout_ms == 150, "MOTOR_SET timeout");

	for (std::size_t index = 0; index < motor.size(); ++index) {
		require(near(decoded.motor[index], motor[index]), "MOTOR_SET motor array");
	}

	mavlink_message_t ack_message{};
	mavlink_msg_fftgyro_motor_set_ack_pack(
		1,
		1,
		&ack_message,
		999,
		1234,
		55,
		42,
		191,
		FFTGYRO_MOTOR_SET_STATUS_ACCEPTED);
	require(ack_message.msgid == 60001, "MOTOR_SET_ACK message ID");

	mavlink_fftgyro_motor_set_ack_t ack{};
	mavlink_msg_fftgyro_motor_set_ack_decode(&ack_message, &ack);
	require(ack.rx_time_us == 999, "ACK PX4 receive time");
	require(ack.run_id == 1234 && ack.sequence == 55, "ACK strict identity");
	require(ack.status == FFTGYRO_MOTOR_SET_STATUS_ACCEPTED, "ACK status");
}

}  // namespace

int main()
{
	try {
		test_hold();
		test_pulse_coast();
		test_pulse_edge_and_negative_direction();
		test_axis_bias_preserves_collective_and_offsets_waveform();
		test_sine_fade();
		test_chirp();
		test_bounds_rejection();
		test_slew_rejection();
		test_abort_neutralization_is_nonincreasing();
		test_pitch_angle_guard();
		test_persistent_boundary_reflection();
		test_preflight_gates();
		test_mavlink_roundtrip();
		std::cout << "all executor core tests passed\n";
		return 0;

	} catch (const std::exception &error) {
		std::cerr << "test failure: " << error.what() << '\n';
		return 1;
	}
}
