#include "fftgyro_executor/executor.hpp"

#include <array>
#include <cerrno>
#include <ctime>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <termios.h>
#include <unistd.h>

#if defined(FFTGYRO_HAVE_GPIOD)
#include <gpiod.h>
#endif

namespace fftgyro {
namespace {

// Do not let a busy PX4 telemetry stream monopolize the 100 Hz controller.
// poll() is called every 10 ms; it must return in time for the next motor
// command even when the UART never becomes momentarily empty.  Remaining
// bytes stay in the kernel buffer and are consumed on the following tick.
constexpr std::int64_t kPollBudgetNs = 2'000'000;
constexpr std::size_t kPollByteBudget = 4096;

std::runtime_error system_error(const char *operation)
{
	return std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

speed_t uart_speed()
{
#if defined(B921600)
	return B921600;
#else
	throw std::runtime_error("this host does not support 921600 baud");
#endif
}

bool accepted(std::uint8_t status)
{
	return status == FFTGYRO_MOTOR_SET_STATUS_ACCEPTED
		|| status == FFTGYRO_MOTOR_SET_STATUS_RELEASED;
}

std::int64_t monotonic_now_ns()
{
	timespec now{};
	if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0)
		throw system_error("CLOCK_MONOTONIC");
	return static_cast<std::int64_t>(now.tv_sec) * 1'000'000'000LL + now.tv_nsec;
}

}  // namespace

MavlinkUart::MavlinkUart(const Config &config)
	: target_system_(config.target_system), target_component_(config.target_component)
{
	fd_ = ::open(config.uart_device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
	if (fd_ < 0) throw system_error("UART open");

	termios options{};
	if (::tcgetattr(fd_, &options) != 0) {
		::close(fd_);
		fd_ = -1;
		throw system_error("UART tcgetattr");
	}
	::cfmakeraw(&options);
	::cfsetispeed(&options, uart_speed());
	::cfsetospeed(&options, uart_speed());
	options.c_cflag = (options.c_cflag | CLOCAL | CREAD) & ~CRTSCTS;
	options.c_cc[VMIN] = 0;
	options.c_cc[VTIME] = 0;
	if (::tcsetattr(fd_, TCSANOW, &options) != 0 || ::tcflush(fd_, TCIOFLUSH) != 0) {
		::close(fd_);
		fd_ = -1;
		throw system_error("UART configure");
	}
}

MavlinkUart::~MavlinkUart()
{
	if (fd_ >= 0) ::close(fd_);
}

void MavlinkUart::send(std::uint32_t run_id, std::uint32_t sequence,
		       const std::array<float, 4> &motor, bool release)
{
	mavlink_message_t message{};
	mavlink_msg_fftgyro_motor_set_pack(
		kSourceSystem, kSourceComponent, &message, run_id, sequence, motor.data(),
		kTimeoutMs, target_system_, target_component_,
		release ? FFTGYRO_MOTOR_ACTION_RELEASE : FFTGYRO_MOTOR_ACTION_CONTROL);

	std::array<std::uint8_t, MAVLINK_MAX_PACKET_LEN> frame{};
	const auto length = mavlink_msg_to_send_buffer(frame.data(), &message);
	const auto written = ::write(fd_, frame.data(), length);
	if (written < 0) throw system_error("UART write");
	if (written != static_cast<ssize_t>(length)) throw std::runtime_error("UART partial write");
}

void MavlinkUart::request_attitude_stream()
{
	mavlink_message_t message{};
	mavlink_msg_command_long_pack(
		kSourceSystem, kSourceComponent, &message,
		target_system_, target_component_, MAV_CMD_SET_MESSAGE_INTERVAL, 0,
		static_cast<float>(MAVLINK_MSG_ID_ATTITUDE),
		static_cast<float>(1'000'000 / kRateHz), 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);

	std::array<std::uint8_t, MAVLINK_MAX_PACKET_LEN> frame{};
	const auto length = mavlink_msg_to_send_buffer(frame.data(), &message);
	const auto written = ::write(fd_, frame.data(), length);
	if (written < 0) throw system_error("UART attitude request write");
	if (written != static_cast<ssize_t>(length))
		throw std::runtime_error("UART attitude request partial write");
}

void MavlinkUart::requestAttitude()
{
	mavlink_message_t message{};
	mavlink_msg_command_long_pack(
		kSourceSystem, kSourceComponent, &message,
		target_system_, target_component_, MAV_CMD_SET_MESSAGE_INTERVAL, 0,
		static_cast<float>(MAVLINK_MSG_ID_ATTITUDE_QUATERNION),
		static_cast<float>(1'000'000 / kRateHz), 0.0F, 0.0F, 0.0F, 0.0F, 0.0F);

	std::array<std::uint8_t, MAVLINK_MAX_PACKET_LEN> frame{};
	const auto length = mavlink_msg_to_send_buffer(frame.data(), &message);
	const auto written = ::write(fd_, frame.data(), length);
	if (written < 0) throw system_error("UART attitude quaternion request write");
	if (written != static_cast<ssize_t>(length))
		throw std::runtime_error("UART attitude quaternion request partial write");
}

void MavlinkUart::consumeAttitude(const mavlink_message_t &message, std::int64_t received_ns)
{
	updateAttitudeFeedback(feedback_, message, target_system_, target_component_, received_ns);
}

PollResult MavlinkUart::poll(std::uint32_t run_id, std::span<Record> records)
{
	PollResult result;
	std::array<std::uint8_t, 512> bytes{};
	const auto started_ns = monotonic_now_ns();
	std::size_t consumed = 0;

	for (;;) {
		const auto count = ::read(fd_, bytes.data(), bytes.size());
		if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
		if (count < 0) throw system_error("UART read");
		if (count == 0) break;
		consumed += static_cast<std::size_t>(count);

		for (ssize_t i = 0; i < count; ++i) {
			mavlink_message_t message{};
			if (!mavlink_parse_char(MAVLINK_COMM_0, bytes[i], &message, &parser_status_))
				continue;
			const auto received_ns = monotonic_now_ns();
			consumeAttitude(message, received_ns);

			if (message.msgid == MAVLINK_MSG_ID_ATTITUDE
			    && message.sysid == target_system_
			    && message.compid == target_component_) {
				mavlink_attitude_t attitude{};
				mavlink_msg_attitude_decode(&message, &attitude);
				attitude_.valid = true;
				attitude_.time_boot_ms = attitude.time_boot_ms;
				attitude_.rx_monotonic_ns = received_ns;
				attitude_.pitch_rad = attitude.pitch;
				attitude_.pitch_rate_rad_s = attitude.pitchspeed;
				continue;
			}

			if (message.msgid != MAVLINK_MSG_ID_FFTGYRO_MOTOR_SET_ACK) continue;
			mavlink_fftgyro_motor_set_ack_t ack{};
			mavlink_msg_fftgyro_motor_set_ack_decode(&message, &ack);
			if (message.sysid != target_system_ || message.compid != target_component_
			    || ack.target_system != kSourceSystem
			    || ack.target_component != kSourceComponent
			    || ack.run_id != run_id || ack.sequence >= records.size())
				continue;

			auto &record = records[ack.sequence];
			// A delayed duplicate must not refresh the controller ACK watchdog.
			// Only the first ACK for a still-pending sequence is new evidence that
			// PX4 is receiving the current command stream.
			if (record.ack_status >= 0) continue;
			record.ack_status = ack.status;
			record.ack_rx_time_us = ack.rx_time_us;
			result.received = true;
			result.rejected |= !accepted(ack.status);
		}
		if (consumed >= kPollByteBudget || monotonic_now_ns() - started_ns >= kPollBudgetNs)
			break;
	}
	return result;
}

SyncGpio::SyncGpio(const Config &config)
	: enabled_(config.gpio_enabled)
{
	if (!enabled_) return;
#if defined(FFTGYRO_HAVE_GPIOD)
	auto *chip = gpiod_chip_open(config.gpio_chip.c_str());
	auto *line = chip == nullptr ? nullptr : gpiod_chip_get_line(chip, config.gpio_line);
	if (line == nullptr || gpiod_line_request_output(line, "fftgyro-executor", 0) != 0) {
		if (chip != nullptr) gpiod_chip_close(chip);
		throw std::runtime_error("cannot open synchronization GPIO");
	}
	chip_ = chip;
	line_ = line;
#else
	throw std::runtime_error("GPIO requested but libgpiod is unavailable");
#endif
}

SyncGpio::~SyncGpio()
{
#if defined(FFTGYRO_HAVE_GPIOD)
	if (line_ != nullptr) {
		(void)gpiod_line_set_value(static_cast<gpiod_line *>(line_), 0);
		gpiod_line_release(static_cast<gpiod_line *>(line_));
	}
	if (chip_ != nullptr) gpiod_chip_close(static_cast<gpiod_chip *>(chip_));
#endif
}

void SyncGpio::set(bool high)
{
	if (!enabled_) return;
#if defined(FFTGYRO_HAVE_GPIOD)
	if (gpiod_line_set_value(static_cast<gpiod_line *>(line_), high ? 1 : 0) != 0)
		throw std::runtime_error("cannot set synchronization GPIO");
#else
	(void)high;
#endif
}

}  // namespace fftgyro
