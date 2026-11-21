#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <fftgyro/mavlink.h>

namespace fftgyro {

constexpr int kRateHz = 100;
constexpr int kBaud = 921600;
constexpr std::uint16_t kTimeoutMs = 150;
constexpr std::uint8_t kSourceSystem = 42;
constexpr std::uint8_t kSourceComponent = 191;

enum class WaveformType { Hold4, PulseCoast, SineAxis, LogChirpAxis };
enum class StopReason {
	Complete,
	Signal,
	AckTimeout,
	AckRejected,
	SchedulerLate,
	TransportFault,
	AngleUnavailable,
	AngleLimit,
	AngularRateLimit
};
enum class SampleStage { Startup, Preroll, Active, Posthold, Neutralize, RampDown, Release };
enum class AngleGuardState {
	Disabled,
	Nominal,
	Fading,
	Braking,
	ReflectingPositive,
	ReflectingNegative
};

struct AngleGuardConfig {
	bool enabled{false};
	bool limits_verified{false};
	double attitude_pitch_sign{1.0};
	double attitude_center_deg{0.0};
	double axis_response_sign{1.0};
	double recovery_deg{0.0};
	double warning_deg{0.0};
	double soft_limit_deg{0.0};
	double abort_limit_deg{0.0};
	double brake_amplitude{0.0};
	double max_abs_rate_dps{0.0};
	std::uint32_t max_sample_age_ms{100};
	bool boundary_reflection{false};
};

struct WaveformConfig {
	WaveformType type{WaveformType::Hold4};
	double amplitude{0.0};
	double preroll_s{0.0};
	double active_s{1.0};
	double posthold_s{0.0};
	double edge_s{0.0};
	double fade_s{0.0};
	double frequency_hz{1.0};
	double start_frequency_hz{0.25};
	double end_frequency_hz{2.0};
	bool requires_mechanical_lock{false};
	bool requires_angle_guard{false};
};

struct Config {
	std::string run_name;
	std::string uart_device{"/dev/serial0"};
	std::uint8_t target_system{1};
	std::uint8_t target_component{1};
	double collective{0.0};
	double motor_min{0.0};
	double motor_max{1.0};
	std::array<double, 4> axis_mix{};
	double axis_bias{0.0};
	bool motor_mapping_verified{false};
	bool collective_verified{false};
	bool mechanically_locked_verified{false};
	double startup_ramp_rate{0.2};
	double normal_ramp_rate{0.1};
	double abort_ramp_rate{0.5};
	std::filesystem::path log_directory;
	bool gpio_enabled{false};
	std::string gpio_chip{"/dev/gpiochip0"};
	unsigned int gpio_line{17};
	AngleGuardConfig angle_guard;
	WaveformConfig waveform;
};

struct Sample {
	double time_s{0.0};
	double phase_rad{0.0};
	double frequency_hz{0.0};
	bool sync_start{false};
	SampleStage stage{SampleStage::Startup};
	std::array<float, 4> motor{};
};

struct Record {
	std::uint32_t sequence{0};
	std::int64_t scheduled_ns{0};
	std::int64_t sent_ns{0};
	double time_s{0.0};
	double phase_rad{0.0};
	double frequency_hz{0.0};
	std::array<float, 4> nominal_motor{};
	std::array<float, 4> motor{};
	double pitch_deg{0.0};
	double pitch_rate_dps{0.0};
	std::uint32_t attitude_time_boot_ms{0};
	std::int64_t attitude_rx_ns{0};
	std::int64_t attitude_age_ns{0};
	double nominal_axis_delta{0.0};
	double applied_axis_delta{0.0};
	AngleGuardState angle_guard_state{AngleGuardState::Disabled};
	int reflection_direction{0};
	int ack_status{-1};
	std::uint64_t ack_rx_time_us{0};
	std::int64_t sync_edge_ns{0};
	bool sync_start{false};
	SampleStage stage{SampleStage::Startup};
	bool release{false};
};

struct PollResult {
	bool received{false};
	bool rejected{false};
};

struct AttitudeSample {
	bool valid{false};
	std::uint32_t time_boot_ms{0};
	std::int64_t rx_monotonic_ns{0};
	double pitch_rad{0.0};
	double pitch_rate_rad_s{0.0};
};

struct Feedback {
	std::array<double, 4> quaternion{1.0, 0.0, 0.0, 0.0};
	std::array<double, 3> body_rate_deg_s{};
	std::uint32_t boot_ms{0};
	std::int64_t received_ns{0};
	bool valid{false};
	bool fault{false};
};

struct AngleGuardDecision {
	std::array<float, 4> motor{};
	double pitch_deg{0.0};
	double pitch_rate_dps{0.0};
	double nominal_axis_delta{0.0};
	double applied_axis_delta{0.0};
	AngleGuardState state{AngleGuardState::Disabled};
	bool latched{false};
	int reflection_direction{0};
	bool abort_angle{false};
	bool abort_rate{false};
};

Config load_config(const std::filesystem::path &path);
std::vector<Sample> build_waveform(const Config &config);
std::array<float, 4> abort_neutral_target(const std::array<float, 4> &motor);
AngleGuardDecision apply_pitch_angle_guard(
	const AngleGuardConfig &guard,
	const std::array<double, 4> &axis_mix,
	const std::array<float, 4> &nominal_motor,
	const AttitudeSample &attitude,
	bool latched,
	int reflection_direction = 0);
void updateAttitudeFeedback(
	Feedback &feedback,
	const mavlink_message_t &message,
	std::uint8_t target_system,
	std::uint8_t target_component,
	std::int64_t received_ns) noexcept;
const char *waveform_name(WaveformType type);
const char *stop_reason_name(StopReason reason);
const char *stage_name(SampleStage stage);
const char *angle_guard_state_name(AngleGuardState state);

class MavlinkUart {
public:
	explicit MavlinkUart(const Config &config);
	~MavlinkUart();
	MavlinkUart(const MavlinkUart &) = delete;
	MavlinkUart &operator=(const MavlinkUart &) = delete;

	void send(std::uint32_t run_id, std::uint32_t sequence,
		  const std::array<float, 4> &motor, bool release);
	void request_attitude_stream();
	void requestAttitude();
	PollResult poll(std::uint32_t run_id, std::span<Record> records);
	const AttitudeSample &attitude() const { return attitude_; }
	const Feedback &feedback() const { return feedback_; }
	void consumeAttitude(const mavlink_message_t &message, std::int64_t received_ns);

private:
	int fd_{-1};
	std::uint8_t target_system_{1};
	std::uint8_t target_component_{1};
	mavlink_status_t parser_status_{};
	AttitudeSample attitude_{};
	Feedback feedback_{};
};

class SyncGpio {
public:
	explicit SyncGpio(const Config &config);
	~SyncGpio();
	SyncGpio(const SyncGpio &) = delete;
	SyncGpio &operator=(const SyncGpio &) = delete;
	void set(bool high);

private:
	bool enabled_{false};
	[[maybe_unused]] void *chip_{nullptr};
	[[maybe_unused]] void *line_{nullptr};
};

}  // namespace fftgyro
