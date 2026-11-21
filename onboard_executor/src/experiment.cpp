#include "fftgyro_executor/executor.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace fftgyro {
namespace {

using Json = nlohmann::json;
constexpr double kMeasuredMotorMin = 0.10;
constexpr double kMeasuredMotorMax = 0.80;
constexpr double kMaxSampleIncrease = 0.10;
constexpr double kMaxSampleDecrease = 0.05;

WaveformType parse_type(const std::string &name)
{
	if (name == "HOLD4") return WaveformType::Hold4;
	if (name == "PULSE_COAST") return WaveformType::PulseCoast;
	if (name == "SINE_AXIS") return WaveformType::SineAxis;
	if (name == "LOG_CHIRP_AXIS") return WaveformType::LogChirpAxis;
	throw std::runtime_error("unknown waveform: " + name);
}

bool finite(double value)
{
	return std::isfinite(value);
}

void validate(const Config &c)
{
	const bool safe_name = !c.run_name.empty() && c.run_name.size() <= 64
		&& std::all_of(c.run_name.begin(), c.run_name.end(), [](unsigned char ch) {
			return std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-';
		});
	if (!safe_name) throw std::runtime_error("invalid run_name");
	if (c.uart_device.empty() || c.target_system == 0 || c.target_component == 0)
		throw std::runtime_error("invalid UART target");
	if (!finite(c.motor_min) || !finite(c.motor_max) || !finite(c.collective)
	    || !finite(c.axis_bias)
	    || c.motor_min < kMeasuredMotorMin || c.motor_max > kMeasuredMotorMax
	    || c.motor_min >= c.motor_max
	    || c.collective < c.motor_min || c.collective > c.motor_max)
		throw std::runtime_error("motor commands must stay inside measured envelope 0.10..0.80");
	if (!c.collective_verified)
		throw std::runtime_error("collective_verified must be true after preflight");
	if (!finite(c.startup_ramp_rate) || !finite(c.normal_ramp_rate)
	    || !finite(c.abort_ramp_rate) || c.startup_ramp_rate <= 0.0
	    || c.normal_ramp_rate <= 0.0 || c.abort_ramp_rate <= 0.0)
		throw std::runtime_error("ramp rates must be positive");
	if (c.startup_ramp_rate > 0.2)
		throw std::runtime_error("startup_ramp_rate exceeds tested limit 0.20 command/s");
	if (c.normal_ramp_rate > 0.1)
		throw std::runtime_error("normal_ramp_rate exceeds tested limit 0.10 command/s");
	if (c.abort_ramp_rate > 0.5)
		throw std::runtime_error("abort_ramp_rate exceeds provisional limit 0.50 command/s");
	if (c.log_directory.empty()) throw std::runtime_error("log_directory is empty");

	bool mixed = false;
	double mix_sum = 0.0;
	for (double value : c.axis_mix) {
		if (!finite(value) || std::abs(value) > 1.0)
			throw std::runtime_error("axis_mix entries must be finite and inside -1..1");
		mixed |= std::abs(value) > 1e-12;
		mix_sum += value;
	}
	if (std::abs(c.axis_bias) > 0.0 && !mixed)
		throw std::runtime_error("axis_bias needs nonzero axis_mix");
	if ((std::abs(c.axis_bias) > 0.0 || c.waveform.type != WaveformType::Hold4)
	    && std::abs(mix_sum) > 1e-9)
		throw std::runtime_error("axis_mix must sum to zero");
	for (double mix : c.axis_mix) {
		const double baseline = c.collective + mix * c.axis_bias;
		if (baseline < c.motor_min || baseline > c.motor_max)
			throw std::runtime_error("axis_bias exceeds motor bounds");
	}

	const auto &w = c.waveform;
	if (w.requires_mechanical_lock && !c.mechanically_locked_verified)
		throw std::runtime_error("mechanically_locked_verified is required by this waveform");
	if (w.requires_angle_guard && !c.angle_guard.enabled)
		throw std::runtime_error("enabled angle_guard is required by this waveform");
	if (c.angle_guard.enabled) {
		const auto &g = c.angle_guard;
		const std::array guard_values{
			g.attitude_pitch_sign, g.attitude_center_deg,
			g.axis_response_sign, g.recovery_deg, g.warning_deg,
			g.soft_limit_deg, g.abort_limit_deg,
			g.brake_amplitude, g.max_abs_rate_dps};
		if (!g.limits_verified)
			throw std::runtime_error("angle_guard limits_verified must be true after commissioning");
		if (!std::all_of(guard_values.begin(), guard_values.end(), finite)
		    || std::abs(g.attitude_pitch_sign) != 1.0
		    || std::abs(g.axis_response_sign) != 1.0
		    || g.recovery_deg < 0.0
		    || g.recovery_deg >= g.warning_deg
		    || g.warning_deg >= g.soft_limit_deg
		    || g.soft_limit_deg >= g.abort_limit_deg
		    || g.brake_amplitude <= 0.0
		    || g.brake_amplitude > 0.01
		    || g.max_abs_rate_dps <= 0.0
		    || g.max_sample_age_ms < 20 || g.max_sample_age_ms > 150)
			throw std::runtime_error("invalid angle_guard limits");
		for (double mix : c.axis_mix) {
			const double low_brake = c.collective - std::abs(mix) * g.brake_amplitude;
			const double high_brake = c.collective + std::abs(mix) * g.brake_amplitude;
			if (low_brake < c.motor_min || high_brake > c.motor_max)
				throw std::runtime_error("angle_guard brake exceeds motor bounds");
		}
	}
	const std::array values{w.amplitude, w.preroll_s, w.active_s, w.posthold_s,
				w.edge_s, w.fade_s, w.frequency_hz,
				w.start_frequency_hz, w.end_frequency_hz};
	if (!std::all_of(values.begin(), values.end(), finite)
	    || w.preroll_s < 0.0 || w.active_s <= 0.0 || w.posthold_s < 0.0
	    || w.edge_s < 0.0 || w.fade_s < 0.0)
		throw std::runtime_error("invalid waveform value");
	if (w.type == WaveformType::Hold4 && w.amplitude != 0.0)
		throw std::runtime_error("HOLD4 amplitude must be zero");
	if (w.type != WaveformType::Hold4 && !c.motor_mapping_verified)
		throw std::runtime_error("motor_mapping_verified must be true for axis waveform");
	if (w.type != WaveformType::Hold4 && (!mixed || std::abs(w.amplitude) <= 0.0))
		throw std::runtime_error("axis waveform needs mix and amplitude");
	if (w.type != WaveformType::Hold4 && w.posthold_s < 1.0 / kRateHz)
		throw std::runtime_error("axis waveform needs a neutral posthold sample");
	if (w.type == WaveformType::PulseCoast && 2.0 * w.edge_s > w.active_s)
		throw std::runtime_error("PULSE_COAST edges exceed active time");
	if (w.type == WaveformType::SineAxis
	    && (w.frequency_hz <= 0.0 || w.frequency_hz > 2.0
		|| w.fade_s <= 0.0 || 2.0 * w.fade_s > w.active_s))
		throw std::runtime_error("invalid SINE_AXIS frequency or fade");
	if (w.type == WaveformType::LogChirpAxis
	    && (w.start_frequency_hz < 0.25 || w.end_frequency_hz > 2.0
		|| w.start_frequency_hz >= w.end_frequency_hz
		|| w.fade_s <= 0.0 || 2.0 * w.fade_s > w.active_s))
		throw std::runtime_error("invalid LOG_CHIRP_AXIS frequency or fade");
}

std::size_t count(double seconds)
{
	return static_cast<std::size_t>(std::llround(seconds * kRateHz));
}

std::size_t ramp_count(double value, double rate)
{
	return static_cast<std::size_t>(std::ceil(value / rate * kRateHz));
}

double fade(std::size_t index, std::size_t active, std::size_t width)
{
	if (width == 0 || active <= 1) return 1.0;
	return std::min({1.0, static_cast<double>(index) / width,
			 static_cast<double>(active - 1 - index) / width});
}

}  // namespace

Config load_config(const std::filesystem::path &path)
{
	std::ifstream input(path);
	if (!input) throw std::runtime_error("cannot open config: " + path.string());

	Json j;
	input >> j;
	Config c;
	c.run_name = j.at("run_name").get<std::string>();
	c.collective = j.at("collective").get<double>();
	c.motor_min = j.at("motor_min").get<double>();
	c.motor_max = j.at("motor_max").get<double>();
	c.axis_mix = j.at("axis_mix").get<std::array<double, 4>>();
	c.axis_bias = j.value("axis_bias", 0.0);
	c.motor_mapping_verified = j.value("motor_mapping_verified", false);
	c.collective_verified = j.value("collective_verified", false);
	c.mechanically_locked_verified = j.value("mechanically_locked_verified", false);
	c.startup_ramp_rate = j.value("startup_ramp_rate", 0.2);
	c.normal_ramp_rate = j.value("normal_ramp_rate", 0.1);
	c.abort_ramp_rate = j.value("abort_ramp_rate", 0.5);
	c.log_directory = j.at("log_directory").get<std::string>();

	const auto &u = j.at("uart");
	c.uart_device = u.at("device").get<std::string>();
	c.target_system = u.value("target_system", 1);
	c.target_component = u.value("target_component", 1);
	if (u.value("baud", kBaud) != kBaud
	    || j.value("sample_rate_hz", kRateHz) != kRateHz
	    || j.value("timeout_ms", kTimeoutMs) != kTimeoutMs)
		throw std::runtime_error("UART, rate and timeout are fixed at 921600, 100 Hz and 150 ms");

	if (j.contains("gpio")) {
		const auto &g = j.at("gpio");
		c.gpio_enabled = g.value("enabled", false);
		c.gpio_chip = g.value("chip", std::string("/dev/gpiochip0"));
	c.gpio_line = g.value("line", 17U);
	}

	if (j.contains("angle_guard")) {
		const auto &g = j.at("angle_guard");
		c.angle_guard.enabled = g.value("enabled", false);
		c.angle_guard.limits_verified = g.value("limits_verified", false);
		c.angle_guard.attitude_pitch_sign = g.value("attitude_pitch_sign", 1.0);
		c.angle_guard.attitude_center_deg = g.value("attitude_center_deg", 0.0);
		c.angle_guard.axis_response_sign = g.value("axis_response_sign", 1.0);
		c.angle_guard.recovery_deg = g.value("recovery_deg", 0.0);
		c.angle_guard.warning_deg = g.value("warning_deg", 0.0);
		c.angle_guard.soft_limit_deg = g.value("soft_limit_deg", 0.0);
		c.angle_guard.abort_limit_deg = g.value("abort_limit_deg", 0.0);
		c.angle_guard.brake_amplitude = g.value("brake_amplitude", 0.0);
		c.angle_guard.max_abs_rate_dps = g.value("max_abs_rate_dps", 0.0);
		c.angle_guard.max_sample_age_ms = g.value("max_sample_age_ms", 100U);
		c.angle_guard.boundary_reflection = g.value("boundary_reflection", false);
	}

	const auto &w = j.at("waveform");
	c.waveform.type = parse_type(w.at("type").get<std::string>());
	c.waveform.amplitude = w.value("amplitude", 0.0);
	c.waveform.preroll_s = w.value("preroll_s", 0.0);
	c.waveform.active_s = w.at("active_s").get<double>();
	c.waveform.posthold_s = w.value("posthold_s", 0.0);
	c.waveform.edge_s = w.value("edge_s", 0.0);
	c.waveform.fade_s = w.value("fade_s", 0.0);
	c.waveform.frequency_hz = w.value("frequency_hz", 1.0);
	c.waveform.start_frequency_hz = w.value("start_frequency_hz", 0.25);
	c.waveform.end_frequency_hz = w.value("end_frequency_hz", 2.0);
	c.waveform.requires_mechanical_lock = w.value("requires_mechanical_lock", false);
	c.waveform.requires_angle_guard = w.value("requires_angle_guard", false);
	validate(c);
	return c;
}

std::vector<Sample> build_waveform(const Config &c)
{
	validate(c);
	const auto pre = count(c.waveform.preroll_s);
	const auto active = count(c.waveform.active_s);
	const auto post = count(c.waveform.posthold_s);
	double startup_peak = c.collective;
	for (double mix : c.axis_mix)
		startup_peak = std::max(startup_peak, c.collective + mix * c.axis_bias);
	const auto startup = ramp_count(startup_peak, c.startup_ramp_rate);
	const auto pulse_edge = count(c.waveform.edge_s);
	const auto fade_width = count(c.waveform.fade_s);
	if (active == 0) throw std::runtime_error("waveform must contain at least one active sample");
	std::vector<Sample> samples(startup + pre + active + post);

	for (std::size_t i = 0; i < samples.size(); ++i) {
		auto &sample = samples[i];
		sample.time_s = static_cast<double>(i) / kRateHz;
		double base = c.collective;
		double delta = c.axis_bias;

		if (i < startup) {
			sample.stage = SampleStage::Startup;
			const double scale = static_cast<double>(i + 1) / startup;
			base *= scale;
			delta *= scale;

		} else {
			const auto body = i - startup;
			sample.stage = body < pre ? SampleStage::Preroll
				: body < pre + active ? SampleStage::Active : SampleStage::Posthold;
		}

		if (sample.stage == SampleStage::Active) {
			const auto n = i - startup - pre;
			const double t = static_cast<double>(n) / kRateHz;
			sample.sync_start = n == 0;

			if (c.waveform.type == WaveformType::PulseCoast) {
				delta += c.waveform.amplitude * fade(n, active, pulse_edge);
			} else if (c.waveform.type == WaveformType::SineAxis) {
				sample.frequency_hz = c.waveform.frequency_hz;
				sample.phase_rad = 2.0 * std::numbers::pi * sample.frequency_hz * t;
				delta += c.waveform.amplitude * fade(n, active, fade_width)
					* std::sin(sample.phase_rad);
			} else if (c.waveform.type == WaveformType::LogChirpAxis) {
				const double f0 = c.waveform.start_frequency_hz;
				const double log_ratio = std::log(c.waveform.end_frequency_hz / f0);
				const double exponent = std::exp(log_ratio * t / c.waveform.active_s);
				sample.frequency_hz = f0 * exponent;
				sample.phase_rad = 2.0 * std::numbers::pi * f0
					* c.waveform.active_s / log_ratio * (exponent - 1.0);
				delta += c.waveform.amplitude * fade(n, active, fade_width)
					* std::sin(sample.phase_rad);
			}
		}

		for (std::size_t motor = 0; motor < 4; ++motor) {
			const double value = base + c.axis_mix[motor] * delta;
			const double lower = sample.stage == SampleStage::Startup ? 0.0 : c.motor_min;
			if (!finite(value) || value < lower - 1e-7 || value > c.motor_max + 1e-7)
				throw std::runtime_error("waveform exceeds motor bounds at sample "
							 + std::to_string(i));
			sample.motor[motor] = static_cast<float>(value);

			if (i > 0) {
				const double change = value - samples[i - 1].motor[motor];
				if (change > kMaxSampleIncrease + 1e-7
				    || change < -kMaxSampleDecrease - 1e-7)
					throw std::runtime_error("waveform exceeds measured sample slew at sample "
								 + std::to_string(i));
			}
		}
	}
	return samples;
}

std::array<float, 4> abort_neutral_target(const std::array<float, 4> &motor)
{
	const float minimum = *std::min_element(motor.begin(), motor.end());
	return {minimum, minimum, minimum, minimum};
}

AngleGuardDecision apply_pitch_angle_guard(
	const AngleGuardConfig &guard,
	const std::array<double, 4> &axis_mix,
	const std::array<float, 4> &nominal_motor,
	const AttitudeSample &attitude,
	bool latched,
	int reflection_direction)
{
	AngleGuardDecision decision;
	decision.motor = nominal_motor;
	decision.reflection_direction = reflection_direction;
	if (!guard.enabled) return decision;
	if (!attitude.valid) throw std::runtime_error("angle guard received no attitude");
	if (reflection_direction < -1 || reflection_direction > 1)
		throw std::runtime_error("invalid angle guard reflection direction");

	constexpr double radians_to_degrees = 180.0 / std::numbers::pi;
	decision.pitch_deg =
		guard.attitude_pitch_sign * attitude.pitch_rad * radians_to_degrees
		- guard.attitude_center_deg;
	decision.pitch_rate_dps =
		guard.attitude_pitch_sign * attitude.pitch_rate_rad_s * radians_to_degrees;

	double mix_energy = 0.0;
	double mean = 0.0;
	for (float value : nominal_motor) mean += value / nominal_motor.size();
	for (std::size_t i = 0; i < axis_mix.size(); ++i) {
		mix_energy += axis_mix[i] * axis_mix[i];
		decision.nominal_axis_delta += axis_mix[i] * (nominal_motor[i] - mean);
	}
	if (mix_energy <= 0.0) throw std::runtime_error("angle guard needs nonzero axis mix");
	decision.nominal_axis_delta /= mix_energy;
	decision.applied_axis_delta = decision.nominal_axis_delta;

	const double absolute_pitch = std::abs(decision.pitch_deg);
	const double absolute_rate = std::abs(decision.pitch_rate_dps);
	constexpr double comparison_epsilon = 1e-6;
	decision.abort_angle =
		absolute_pitch + comparison_epsilon >= guard.abort_limit_deg;
	decision.abort_rate =
		absolute_rate + comparison_epsilon >= guard.max_abs_rate_dps;
	decision.latched = latched;

	const auto sign = [](double value) { return value < 0.0 ? -1.0 : 1.0; };
	const double response = guard.axis_response_sign * decision.nominal_axis_delta;
	if (guard.boundary_reflection) {
		if (decision.reflection_direction == 0) {
			if (decision.pitch_deg >= guard.soft_limit_deg)
				decision.reflection_direction = -1;
			else if (decision.pitch_deg <= -guard.soft_limit_deg)
				decision.reflection_direction = 1;
		} else if (decision.reflection_direction < 0
			   && decision.pitch_deg <= -guard.soft_limit_deg) {
			decision.reflection_direction = 1;
		} else if (decision.reflection_direction > 0
			   && decision.pitch_deg >= guard.soft_limit_deg) {
			decision.reflection_direction = -1;
		}
		decision.latched = decision.reflection_direction != 0;
		if (decision.reflection_direction != 0) {
			decision.applied_axis_delta =
				static_cast<double>(decision.reflection_direction)
				* guard.brake_amplitude / guard.axis_response_sign;
			decision.state = decision.reflection_direction > 0
				? AngleGuardState::ReflectingPositive
				: AngleGuardState::ReflectingNegative;
		} else if (absolute_pitch >= guard.warning_deg
			   && decision.pitch_deg * response > 0.0) {
			const double scale = std::clamp(
				(guard.soft_limit_deg - absolute_pitch)
					/ (guard.soft_limit_deg - guard.warning_deg),
				0.0, 1.0);
			decision.applied_axis_delta *= scale;
			decision.state = AngleGuardState::Fading;
		} else {
			decision.state = AngleGuardState::Nominal;
		}
	} else {
		if (!decision.latched && absolute_pitch >= guard.soft_limit_deg)
			decision.latched = true;
		if (decision.latched && absolute_pitch <= guard.recovery_deg
		    && decision.pitch_deg * decision.pitch_rate_dps <= 0.0)
			decision.latched = false;
		if (decision.latched) {
		const bool moving_outward =
			decision.pitch_deg * decision.pitch_rate_dps > 0.0;
		const bool command_outward = decision.pitch_deg * response > 0.0;
		if (moving_outward || command_outward) {
			decision.applied_axis_delta =
				-sign(decision.pitch_deg) * guard.brake_amplitude
				/ guard.axis_response_sign;
			decision.state = AngleGuardState::Braking;
		} else {
			decision.state = AngleGuardState::Fading;
		}
		} else if (absolute_pitch >= guard.warning_deg
			   && decision.pitch_deg * response > 0.0) {
			const double scale = std::clamp(
				(guard.soft_limit_deg - absolute_pitch)
					/ (guard.soft_limit_deg - guard.warning_deg),
				0.0, 1.0);
			decision.applied_axis_delta *= scale;
			decision.state = AngleGuardState::Fading;
		} else {
			decision.state = AngleGuardState::Nominal;
		}
	}

	const double delta_change =
		decision.applied_axis_delta - decision.nominal_axis_delta;
	for (std::size_t i = 0; i < decision.motor.size(); ++i)
		decision.motor[i] += static_cast<float>(axis_mix[i] * delta_change);
	return decision;
}

const char *waveform_name(WaveformType type)
{
	switch (type) {
	case WaveformType::Hold4: return "HOLD4";
	case WaveformType::PulseCoast: return "PULSE_COAST";
	case WaveformType::SineAxis: return "SINE_AXIS";
	case WaveformType::LogChirpAxis: return "LOG_CHIRP_AXIS";
	}
	return "UNKNOWN";
}

const char *stage_name(SampleStage stage)
{
	switch (stage) {
	case SampleStage::Startup: return "STARTUP";
	case SampleStage::Preroll: return "PREROLL";
	case SampleStage::Active: return "ACTIVE";
	case SampleStage::Posthold: return "POSTHOLD";
	case SampleStage::Neutralize: return "NEUTRALIZE";
	case SampleStage::RampDown: return "RAMP_DOWN";
	case SampleStage::Release: return "RELEASE";
	}
	return "UNKNOWN";
}

const char *angle_guard_state_name(AngleGuardState state)
{
	switch (state) {
	case AngleGuardState::Disabled: return "DISABLED";
	case AngleGuardState::Nominal: return "NOMINAL";
	case AngleGuardState::Fading: return "FADING";
	case AngleGuardState::Braking: return "BRAKING";
	case AngleGuardState::ReflectingPositive: return "REFLECTING_POSITIVE";
	case AngleGuardState::ReflectingNegative: return "REFLECTING_NEGATIVE";
	}
	return "UNKNOWN";
}

}  // namespace fftgyro
