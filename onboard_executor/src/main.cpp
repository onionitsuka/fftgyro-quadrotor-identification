#include "fftgyro_executor/executor.hpp"
#include "fftgyro_executor/session_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace fftgyro {
namespace {

constexpr std::int64_t kPeriodNs = 1'000'000'000LL / kRateHz;
constexpr std::int64_t kAckTimeoutNs = 1'000'000'000LL;
volatile std::sig_atomic_t stop_requested = 0;

void on_signal(int)
{
	stop_requested = 1;
}

std::int64_t now_ns()
{
	timespec now{};
	if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0)
		throw std::runtime_error("CLOCK_MONOTONIC failed");
	return static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec;
}

void sleep_until(std::int64_t target)
{
#if defined(__linux__)
	const timespec time{static_cast<time_t>(target / 1'000'000'000LL),
			    static_cast<long>(target % 1'000'000'000LL)};
	while (::clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &time, nullptr) == EINTR) {}
#else
	const auto remaining = target - now_ns();
	if (remaining > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(remaining));
#endif
}

std::uint32_t make_run_id()
{
	std::random_device random;
	std::uint32_t id = 0;
	while (id == 0) id = (static_cast<std::uint32_t>(random()) << 16U)
		^ static_cast<std::uint32_t>(random());
	return id;
}

std::string timestamp()
{
	const auto time = std::time(nullptr);
	std::tm utc{};
	(void)::gmtime_r(&time, &utc);
	std::ostringstream text;
	text << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
	return text.str();
}

bool active(const std::array<float, 4> &motor)
{
	return std::any_of(motor.begin(), motor.end(), [](float value) { return value > 0.0F; });
}

void ramp_step(std::array<float, 4> &motor, double rate)
{
	const float step = static_cast<float>(rate / kRateHz);
	for (float &value : motor) value = std::max(0.0F, value - step);
}

bool at_target(const std::array<float, 4> &motor, const std::array<float, 4> &target)
{
	for (std::size_t i = 0; i < motor.size(); ++i) {
		if (std::abs(motor[i] - target[i]) > 1e-6F) return false;
	}
	return true;
}

void ramp_toward(std::array<float, 4> &motor, const std::array<float, 4> &target, double rate)
{
	const float step = static_cast<float>(rate / kRateHz);
	for (std::size_t i = 0; i < motor.size(); ++i) {
		if (motor[i] < target[i]) motor[i] = std::min(target[i], motor[i] + step);
		else motor[i] = std::max(target[i], motor[i] - step);
	}
}

void append_record(std::vector<Record> &records, std::int64_t scheduled,
		   const Sample &sample, const std::array<float, 4> &applied_motor,
		   bool release, std::int64_t sync_edge_ns = 0,
		   const AngleGuardDecision *guard = nullptr,
		   const AttitudeSample *attitude = nullptr,
		   std::int64_t attitude_age_ns = 0)
{
	Record record;
	record.sequence = static_cast<std::uint32_t>(records.size());
	record.scheduled_ns = scheduled;
	record.sent_ns = now_ns();
	record.time_s = sample.time_s;
	record.phase_rad = sample.phase_rad;
	record.frequency_hz = sample.frequency_hz;
	record.nominal_motor = sample.motor;
	record.motor = applied_motor;
	if (guard != nullptr && attitude != nullptr) {
		record.pitch_deg = guard->pitch_deg;
		record.pitch_rate_dps = guard->pitch_rate_dps;
		record.attitude_time_boot_ms = attitude->time_boot_ms;
		record.attitude_rx_ns = attitude->rx_monotonic_ns;
		record.attitude_age_ns = attitude_age_ns;
		record.nominal_axis_delta = guard->nominal_axis_delta;
		record.applied_axis_delta = guard->applied_axis_delta;
		record.angle_guard_state = guard->state;
		record.reflection_direction = guard->reflection_direction;
	}
	record.sync_edge_ns = sync_edge_ns;
	record.sync_start = sample.sync_start;
	record.stage = sample.stage;
	record.release = release;
	records.push_back(record);
}

void write_log(const std::filesystem::path &path, std::uint32_t run_id,
	       StopReason reason, const std::string &detail, const std::vector<Record> &records)
{
	std::ofstream output(path);
	if (!output) throw std::runtime_error("cannot write log: " + path.string());
	output << "# run_id=" << run_id << ",result=" << stop_reason_name(reason)
	       << ",detail=" << detail << '\n';
	output << "sequence,scheduled_ns,sent_ns,jitter_ns,time_s,stage,sync_start,sync_edge_ns,"
		  "phase_rad,frequency_hz,nominal_motor1,nominal_motor2,nominal_motor3,"
		  "nominal_motor4,requested_motor1,requested_motor2,requested_motor3,"
		  "requested_motor4,pitch_deg,pitch_rate_dps,attitude_time_boot_ms,"
		  "attitude_rx_ns,attitude_age_ns,nominal_axis_delta,applied_axis_delta,"
		  "angle_guard_state,reflection_direction,ack_status,ack_rx_time_us,action\n";
	output << std::setprecision(17);
	for (const auto &r : records) {
		output << r.sequence << ',' << r.scheduled_ns << ',' << r.sent_ns << ','
		       << r.sent_ns - r.scheduled_ns << ',' << r.time_s << ',' << stage_name(r.stage)
		       << ',' << (r.sync_start ? 1 : 0) << ',' << r.sync_edge_ns << ','
		       << r.phase_rad << ',' << r.frequency_hz << ',' << r.nominal_motor[0]
		       << ',' << r.nominal_motor[1] << ',' << r.nominal_motor[2] << ','
		       << r.nominal_motor[3] << ',' << r.motor[0] << ',' << r.motor[1] << ','
		       << r.motor[2] << ',' << r.motor[3] << ',' << r.pitch_deg << ','
		       << r.pitch_rate_dps << ',' << r.attitude_time_boot_ms << ','
		       << r.attitude_rx_ns << ',' << r.attitude_age_ns << ','
		       << r.nominal_axis_delta << ',' << r.applied_axis_delta << ','
		       << angle_guard_state_name(r.angle_guard_state) << ','
		       << r.reflection_direction << ',' << r.ack_status << ','
		       << r.ack_rx_time_us << ',' << (r.release ? "RELEASE" : "CONTROL") << '\n';
	}
}

void write_dry_run(const std::filesystem::path &path, const std::vector<Sample> &waveform)
{
	std::ofstream output(path);
	if (!output) throw std::runtime_error("cannot write dry-run CSV: " + path.string());
	output << "sample,time_s,stage,sync_start,phase_rad,frequency_hz,"
		  "requested_motor1,requested_motor2,requested_motor3,requested_motor4\n";
	output << std::setprecision(17);
	for (std::size_t i = 0; i < waveform.size(); ++i) {
		const auto &sample = waveform[i];
		output << i << ',' << sample.time_s << ',' << stage_name(sample.stage) << ','
		       << (sample.sync_start ? 1 : 0) << ',' << sample.phase_rad << ','
		       << sample.frequency_hz << ',' << sample.motor[0] << ',' << sample.motor[1]
		       << ',' << sample.motor[2] << ',' << sample.motor[3] << '\n';
	}
}

StopReason execute(const Config &config, const std::vector<Sample> &waveform,
		   std::string &detail, std::filesystem::path &log_path)
{
	if (waveform.empty()) throw std::runtime_error("empty waveform");
	const auto run_id = make_run_id();
	const auto maximum_ramp = static_cast<std::size_t>(
		std::ceil(config.motor_max / std::min(config.normal_ramp_rate, config.abort_ramp_rate)
			  * kRateHz));
	std::vector<Record> records;
	records.reserve(waveform.size() + 2 * maximum_ramp + 1);
	std::filesystem::create_directories(config.log_directory);
	log_path = config.log_directory
		/ (config.run_name + "_" + timestamp() + "_" + std::to_string(run_id) + ".csv");
	{
		std::ofstream check(log_path);
		if (!check) throw std::runtime_error("log directory is not writable");
	}

	StopReason reason = StopReason::Complete;
	std::array<float, 4> motor{};

	try {
		MavlinkUart link(config);
		SyncGpio gpio(config);
		if (config.angle_guard.enabled) {
			link.request_attitude_stream();
			const auto attitude_deadline = now_ns() + 2'000'000'000LL;
			while (!link.attitude().valid && now_ns() < attitude_deadline) {
				(void)link.poll(run_id, records);
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
			}
			if (!link.attitude().valid) {
				reason = StopReason::AngleUnavailable;
				detail = "no PX4 ATTITUDE received during 2 s preflight";
			}
		}

		std::int64_t scheduled = now_ns() + kPeriodNs;
		const auto first_send = scheduled;
		auto last_ack = std::int64_t{0};
		bool sync_high = false;
		bool angle_guard_latched = false;
		int angle_guard_reflection_direction = 0;

		auto poll_ack = [&] {
			const PollResult poll = link.poll(run_id, records);
			if (poll.received) last_ack = now_ns();
			if (poll.rejected && reason == StopReason::Complete)
				reason = StopReason::AckRejected;
		};

		for (const auto &sample : waveform) {
			if (reason != StopReason::Complete) break;
			sleep_until(scheduled);
			if (stop_requested) {
				reason = StopReason::Signal;
				break;
			}
			if (now_ns() - scheduled > kPeriodNs) {
				reason = StopReason::SchedulerLate;
				break;
			}

			poll_ack();
			const auto ack_reference = last_ack == 0 ? first_send : last_ack;
			if (reason == StopReason::AckRejected) break;
			if (!records.empty() && now_ns() - ack_reference > kAckTimeoutNs) {
				reason = StopReason::AckTimeout;
				break;
			}

			if (sync_high) {
				gpio.set(false);
				sync_high = false;
			}
			std::int64_t sync_edge_ns = 0;
			if (sample.sync_start) {
				gpio.set(true);
				if (config.gpio_enabled) sync_edge_ns = now_ns();
				sync_high = true;
			}

			std::array<float, 4> applied_motor = sample.motor;
			AngleGuardDecision guard_decision;
			const AngleGuardDecision *guard_record = nullptr;
			const AttitudeSample *attitude_record = nullptr;
			std::int64_t attitude_age_ns = 0;
			if (config.angle_guard.enabled) {
				const auto &attitude = link.attitude();
				attitude_age_ns = now_ns() - attitude.rx_monotonic_ns;
				const auto maximum_age_ns =
					static_cast<std::int64_t>(config.angle_guard.max_sample_age_ms)
					* 1'000'000LL;
				if (!attitude.valid || attitude_age_ns < 0
				    || attitude_age_ns > maximum_age_ns) {
					reason = StopReason::AngleUnavailable;
					detail = "PX4 ATTITUDE stale or unavailable";
					break;
				}
				guard_decision = apply_pitch_angle_guard(
					config.angle_guard, config.axis_mix, sample.motor,
					attitude, angle_guard_latched,
					angle_guard_reflection_direction);
				if (guard_decision.abort_angle) {
					reason = StopReason::AngleLimit;
					detail = "absolute pitch angle limit reached";
					break;
				}
				if (guard_decision.abort_rate) {
					reason = StopReason::AngularRateLimit;
					detail = "absolute pitch angular-rate limit reached";
					break;
				}
				if (sample.stage == SampleStage::Startup
				    && std::abs(guard_decision.pitch_deg)
					    >= config.angle_guard.soft_limit_deg) {
					reason = StopReason::AngleLimit;
					detail = "pitch outside soft limit during startup";
					break;
				}
				angle_guard_latched = guard_decision.latched;
				angle_guard_reflection_direction =
					guard_decision.reflection_direction;
				applied_motor = guard_decision.motor;
				guard_record = &guard_decision;
				attitude_record = &attitude;
			}

			link.send(run_id, records.size(), applied_motor, false);
			append_record(records, scheduled, sample, applied_motor, false,
				      sync_edge_ns, guard_record, attitude_record, attitude_age_ns);
			motor = applied_motor;
			scheduled += kPeriodNs;
		}

		if (sync_high) gpio.set(false);

		scheduled = now_ns() + kPeriodNs;
		if (reason != StopReason::Complete && active(motor)) {
			const auto neutral = abort_neutral_target(motor);

			while (!at_target(motor, neutral)) {
				sleep_until(scheduled);
				if (now_ns() - scheduled > kPeriodNs) break;
				poll_ack();
				ramp_toward(motor, neutral, config.abort_ramp_rate);
				Sample sample;
				sample.time_s = static_cast<double>(records.size()) / kRateHz;
				sample.stage = SampleStage::Neutralize;
				sample.motor = motor;
				link.send(run_id, records.size(), motor, false);
				append_record(records, scheduled, sample, motor, false);
				scheduled += kPeriodNs;
			}
		}

		while (active(motor)) {
			sleep_until(scheduled);
			if (now_ns() - scheduled > kPeriodNs) {
				if (reason == StopReason::Complete) reason = StopReason::SchedulerLate;
				break;
			}
			poll_ack();
			if (reason == StopReason::Complete) {
				const auto reference = last_ack == 0 ? first_send : last_ack;
				if (now_ns() - reference > kAckTimeoutNs) reason = StopReason::AckTimeout;
			}

			const double rate = reason == StopReason::Complete
				? config.normal_ramp_rate : config.abort_ramp_rate;
			ramp_step(motor, rate);
			Sample sample;
			sample.time_s = static_cast<double>(records.size()) / kRateHz;
			sample.stage = SampleStage::RampDown;
			sample.motor = motor;
			link.send(run_id, records.size(), motor, false);
			append_record(records, scheduled, sample, motor, false);
			scheduled += kPeriodNs;
		}

		Sample release;
		release.time_s = static_cast<double>(records.size()) / kRateHz;
		release.stage = SampleStage::Release;
		const auto release_sequence = records.size();
		link.send(run_id, records.size(), release.motor, true);
		append_record(records, now_ns(), release, release.motor, true);
		const auto deadline = now_ns() + 200'000'000LL;
		while (now_ns() < deadline) {
			const PollResult poll = link.poll(run_id, records);
			if (poll.rejected && reason == StopReason::Complete)
				reason = StopReason::AckRejected;
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
		}
		if (reason == StopReason::Complete) {
			const int status = records[release_sequence].ack_status;
			if (status < 0) reason = StopReason::AckTimeout;
			else if (status != FFTGYRO_MOTOR_SET_STATUS_RELEASED)
				reason = StopReason::AckRejected;
		}
	} catch (const std::exception &error) {
		reason = StopReason::TransportFault;
		detail = error.what();
	}

	write_log(log_path, run_id, reason, detail, records);
	return reason;
}

}  // namespace

const char *stop_reason_name(StopReason reason)
{
	switch (reason) {
	case StopReason::Complete: return "COMPLETE";
	case StopReason::Signal: return "SIGNAL";
	case StopReason::AckTimeout: return "ACK_TIMEOUT";
	case StopReason::AckRejected: return "ACK_REJECTED";
	case StopReason::SchedulerLate: return "SCHEDULER_LATE";
	case StopReason::TransportFault: return "TRANSPORT_FAULT";
	case StopReason::AngleUnavailable: return "ANGLE_UNAVAILABLE";
	case StopReason::AngleLimit: return "ANGLE_LIMIT";
	case StopReason::AngularRateLimit: return "ANGULAR_RATE_LIMIT";
	}
	return "UNKNOWN";
}

}  // namespace fftgyro

int main(int argc, char **argv)
{
	const std::string_view command = argc > 1 ? argv[1] : "";
	const bool props_off_arguments = command == "session-props-off" && argc == 7 &&
		std::string_view(argv[3]) == "--props-removed" && std::string_view(argv[4]) == "--fixed" &&
		std::string_view(argv[5]) == "--operator-stop-ready" && std::string_view(argv[6]) == "--charged-4s";
	const bool fixed_arguments = command == "session-fixed-commissioning" && argc == 7 &&
		std::string_view(argv[3]) == "--props-installed" && std::string_view(argv[4]) == "--fixed" &&
		std::string_view(argv[5]) == "--operator-stop-ready" && std::string_view(argv[6]) == "--charged-4s";
	const bool pitch_arguments = command == "session-pitch-hold-commissioning" && argc == 7 &&
		std::string_view(argv[3]) == "--props-installed" && std::string_view(argv[4]) == "--pitch-only-free" &&
		std::string_view(argv[5]) == "--operator-stop-ready" && std::string_view(argv[6]) == "--charged-4s";
	const bool yaw_arguments = command == "session-yaw-hold-commissioning" && argc == 7 &&
		std::string_view(argv[3]) == "--props-installed" && std::string_view(argv[4]) == "--yaw-only-free" &&
		std::string_view(argv[5]) == "--operator-stop-ready" && std::string_view(argv[6]) == "--charged-4s";
	if (props_off_arguments || fixed_arguments || pitch_arguments || yaw_arguments ||
		(command == "session-pitch-hold-commissioning-validate" && argc == 3) ||
		(command == "session-yaw-hold-commissioning-validate" && argc == 3) ||
		(command == "session-fixed-commissioning-validate" && argc == 3) ||
		((command == "session-validate" || command == "session-run" ||
		  command == "session-props-off-validate") && argc == 3) ||
		(command == "session-preview" && argc == 4)) {
		try {
			return fftgyro::sessionCommand(std::string(command), argv[2], argc == 4 ? argv[3] : "");
		} catch (const std::exception &error) {
			std::cerr << error.what() << '\n';
			return 1;
		}
	}
	const bool valid_arguments = (command == "validate" && argc == 3)
		|| (command == "dry-run" && argc == 4)
		|| (command == "run" && argc == 3);
	if (!valid_arguments) {
		std::cerr << "Usage:\n"
			  "  fftgyro-executor validate CONFIG.json\n"
			  "  fftgyro-executor dry-run CONFIG.json OUTPUT.csv\n"
			  "  fftgyro-executor run CONFIG.json\n"
			  "  fftgyro-executor session-validate SESSION.json\n"
			  "  fftgyro-executor session-preview SESSION.json OUTPUT.csv\n"
			  "  fftgyro-executor session-run SESSION.json\n"
			  "  fftgyro-executor session-props-off-validate SESSION.json\n"
			  "  fftgyro-executor session-props-off SESSION.json --props-removed --fixed --operator-stop-ready --charged-4s\n"
			  "  fftgyro-executor session-fixed-commissioning-validate SESSION.json\n"
			  "  fftgyro-executor session-fixed-commissioning SESSION.json --props-installed --fixed --operator-stop-ready --charged-4s\n"
			  "  fftgyro-executor session-pitch-hold-commissioning-validate SESSION.json\n"
			  "  fftgyro-executor session-pitch-hold-commissioning SESSION.json --props-installed --pitch-only-free --operator-stop-ready --charged-4s\n"
			  "  fftgyro-executor session-yaw-hold-commissioning-validate SESSION.json\n"
			  "  fftgyro-executor session-yaw-hold-commissioning SESSION.json --props-installed --yaw-only-free --operator-stop-ready --charged-4s\n";
		return 2;
	}

	try {
		const auto config = fftgyro::load_config(argv[2]);
		const auto waveform = fftgyro::build_waveform(config);
		if (command == "validate") {
			std::cout << "VALID samples=" << waveform.size()
				  << " duration_s=" << static_cast<double>(waveform.size()) / fftgyro::kRateHz
				  << " waveform=" << fftgyro::waveform_name(config.waveform.type) << '\n';
			return 0;
		}
		if (command == "dry-run") {
			fftgyro::write_dry_run(argv[3], waveform);
			std::cout << "DRY_RUN samples=" << waveform.size() << " output=" << argv[3] << '\n';
			return 0;
		}
		std::signal(SIGINT, fftgyro::on_signal);
		std::signal(SIGTERM, fftgyro::on_signal);
		std::string detail;
		std::filesystem::path log;
		const auto result = fftgyro::execute(config, waveform, detail, log);
		auto config_log = log;
		config_log.replace_extension(".config.json");
		std::error_code copy_error;
		std::filesystem::copy_file(argv[2], config_log,
					   std::filesystem::copy_options::overwrite_existing, copy_error);
		if (copy_error) std::cerr << "config snapshot failed: " << copy_error.message() << '\n';
		std::cout << fftgyro::stop_reason_name(result) << " log=" << log << '\n';
		if (!detail.empty()) std::cerr << detail << '\n';
		return result == fftgyro::StopReason::Complete ? 0 : 1;
	} catch (const std::exception &error) {
		std::cerr << "fftgyro-executor: " << error.what() << '\n';
		return 1;
	}
}
