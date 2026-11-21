// Standalone RPM receiver for Linux/Pi and macOS. No Pixhawk or motor access.
#include "../../firmware/rpm_capture_nano/rpm_wire.h"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#ifdef __APPLE__
#include <IOKit/serial/ioss.h>
#include <sys/ioctl.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
int64_t nowNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }

struct Sample {
  int64_t host_ns;
  uint32_t capture, sequence, nano_us, value;
  char kind;
};

class Logger {
 public:
  explicit Logger(const std::filesystem::path& path) : samples_(131072)
  {
    file_ = std::fopen(path.c_str(), "wx");
    if (!file_) throw std::runtime_error("cannot create RPM log");
    std::fprintf(file_, "host_ns,kind,capture_id,sequence,nano_us,value\n");
    worker_ = std::thread([this] { save(); });
  }
  ~Logger() { finish(); }
  void push(const Sample& sample)
  {
    if (failed_) throw std::runtime_error("RPM storage error");
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next = (head + 1) % samples_.size();
    if (next == tail_.load(std::memory_order_acquire)) throw std::runtime_error("RPM host buffer full");
    samples_[head] = sample;
    head_.store(next, std::memory_order_release);
  }
  bool finish()
  {
    done_ = true;
    if (worker_.joinable()) worker_.join();
    return !failed_;
  }
 private:
  void save()
  {
    auto synced = Clock::now();
    while (!done_ || tail_.load() != head_.load()) {
      const size_t tail = tail_.load(std::memory_order_relaxed);
      if (tail == head_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      } else {
        const auto s = samples_[tail];
        if (std::fprintf(file_, "%lld,%c,%u,%u,%u,%u\n", static_cast<long long>(s.host_ns), s.kind,
                         s.capture, s.sequence, s.nano_us, s.value) < 0) failed_ = true;
        tail_.store((tail + 1) % samples_.size(), std::memory_order_release);
      }
      if (Clock::now() - synced >= std::chrono::seconds(1)) {
        if (std::fflush(file_) != 0 || ::fsync(fileno(file_)) != 0) failed_ = true;
        synced = Clock::now();
      }
    }
    if (std::fflush(file_) != 0 || ::fsync(fileno(file_)) != 0) failed_ = true;
    if (std::fclose(file_) != 0) failed_ = true;
  }
  std::vector<Sample> samples_;
  std::atomic<size_t> head_{0}, tail_{0};
  std::atomic<bool> done_{false}, failed_{false};
  FILE* file_{};
  std::thread worker_;
};

class Serial {
 public:
  explicit Serial(const char* path)
  {
    fd = ::open(path, O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open Nano USB port");
    termios options{};
    if (tcgetattr(fd, &options) < 0) error();
    cfmakeraw(&options);
    options.c_cflag = (options.c_cflag | CLOCAL | CREAD) & ~CRTSCTS;
    options.c_cc[VMIN] = 0; options.c_cc[VTIME] = 0;
#ifdef __APPLE__
    cfsetispeed(&options, B9600); cfsetospeed(&options, B9600);
#else
    cfsetispeed(&options, B1000000); cfsetospeed(&options, B1000000);
#endif
    if (tcsetattr(fd, TCSANOW, &options) < 0) error();
#if defined(__APPLE__) && !defined(FFTGYRO_RPM_TEST_PTY)
    speed_t baud = 1000000;
    if (ioctl(fd, IOSSIOSPEED, &baud) < 0) error();
#endif
  }
  ~Serial() { ::close(fd); }
  void send(char kind, uint32_t value = 0)
  {
    char command[12];
    const int length = kind == 'Z' ? std::snprintf(command, sizeof(command), "Z\n") :
                                   std::snprintf(command, sizeof(command), "%c%08x\n", kind, value);
    if (::write(fd, command, length) != length) throw std::runtime_error("Nano command write failed");
  }
  int fd;
 private:
  void error() { ::close(fd); throw std::runtime_error("Nano serial configuration failed"); }
};
}  // namespace

int main(int argc, char** argv)
{
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: fftgyro-rpm DEVICE NEW_OUTPUT_DIRECTORY SECONDS [SYNC_INTERVAL_MS=1000]\n";
    return 2;
  }
  std::signal(SIGINT, interrupt); std::signal(SIGTERM, interrupt);
  try {
    size_t parsed = 0;
    const double seconds = std::stod(argv[3], &parsed);
    if (parsed != std::string(argv[3]).size() || !std::isfinite(seconds) || seconds <= 0 || seconds > 86400)
      throw std::runtime_error("duration must be in (0, 86400] seconds");
    int sync_ms = 1000;
    if (argc == 5) {
      sync_ms = std::stoi(argv[4], &parsed);
      if (parsed != std::string(argv[4]).size() || sync_ms < 50 || sync_ms > 1000)
        throw std::runtime_error("sync interval must be in [50, 1000] ms");
    }
    const std::filesystem::path directory(argv[2]);
    if (!std::filesystem::create_directory(directory)) throw std::runtime_error("output directory already exists");
    std::ofstream result(directory / "result.txt");
    result << "INCOMPLETE\n"; result.flush();
    Logger logger(directory / "rpm.csv");
    Serial serial(argv[1]);
    const uint32_t capture = std::random_device{}() | 1u;
    std::array<uint64_t, 4> counts{};
    bool began = false, ended = false, stopping = false;
    uint32_t expected = 0, token = 0;
    std::string failure;
    try {
      std::this_thread::sleep_for(std::chrono::seconds(2));  // USB open may reset Nano.
      serial.send('Z');
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      tcflush(serial.fd, TCIFLUSH);
      serial.send('B', capture);
      auto last = Clock::now(), start = last, next_sync = last, stop_time = last;
      std::array<uint8_t, rpm::kFrameSize> frame{};
      size_t fill = 0;
      while (!ended) {
        const auto now = Clock::now();
        if (!stopping && (interrupted || (began && std::chrono::duration<double>(now - start).count() >= seconds))) {
          serial.send('Z'); stopping = true; stop_time = now;
        }
        if (now - last > std::chrono::seconds(2) || (stopping && now - stop_time > std::chrono::seconds(2)))
          throw std::runtime_error("Nano timeout or missing END");
        if (began && !stopping && now >= next_sync) {
          const int64_t sent = nowNs();
          serial.send('T', ++token);
          logger.push({sent, capture, 0, 0, token, 'Q'});
          next_sync = now + std::chrono::milliseconds(sync_ms);
        }
        pollfd ready{serial.fd, POLLIN, 0};
        const int polled = ::poll(&ready, 1, 20);
        if (polled < 0) { if (errno == EINTR) continue; throw std::runtime_error("USB poll failed"); }
        if (ready.revents & (POLLHUP | POLLERR | POLLNVAL)) throw std::runtime_error("Nano disconnected");
        if (!(ready.revents & POLLIN)) continue;
        std::array<uint8_t, 4096> bytes{};
        const ssize_t size = ::read(serial.fd, bytes.data(), bytes.size());
        const int64_t received = nowNs();
        if (size < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (size <= 0) throw std::runtime_error("Nano read failed");
        for (ssize_t i = 0; i < size; ++i) {
          if (ended) throw std::runtime_error("data after END");
          frame[fill++] = bytes[i];
          if (fill != frame.size()) continue;
          fill = 0;
          if (!rpm::valid(frame.data())) throw std::runtime_error("RPM frame/CRC error");
          Sample s{received, rpm::get32(frame.data() + 4), rpm::get32(frame.data() + 8),
                   rpm::get32(frame.data() + 12), rpm::get32(frame.data() + 16), static_cast<char>(frame[3])};
          if (s.capture != capture || s.sequence != expected++) throw std::runtime_error("RPM capture/sequence mismatch");
          logger.push(s);
          last = Clock::now();
          if (!began) {
            if (s.kind != 'B') throw std::runtime_error("missing BEGIN");
            began = true; start = last;
            std::cout << "Recording D2-D5 (no motor commands)\n" << std::flush;
          } else if (s.kind == 'E' && s.value >= 1 && s.value <= 4) ++counts[s.value - 1];
          else if (s.kind == 'H' || s.kind == 'Z') {
            if (s.value) throw std::runtime_error("Nano edge buffer overflow");
            if (s.kind == 'Z') {
              if (!stopping) throw std::runtime_error("unexpected END");
              ended = true;
            }
          } else if (s.kind != 'T') throw std::runtime_error("unexpected RPM event");
        }
      }
    } catch (const std::exception& e) {
      failure = e.what();
      try { serial.send('Z'); } catch (...) {}
    }
    if (!logger.finish()) failure = "RPM storage error";
    result.close();
    result.open(directory / "result.txt", std::ios::trunc);
    result << (failure.empty() && ended ? "COMPLETE" : "INCOMPLETE") << '\n' << failure << '\n';
    result << "sync_interval_ms=" << sync_ms << '\n';
    for (size_t i = 0; i < counts.size(); ++i) result << "S" << i + 1 << "=" << counts[i] << '\n';
    result.flush();
    std::cout << "S1=" << counts[0] << " S2=" << counts[1] << " S3=" << counts[2] << " S4=" << counts[3] << '\n';
    if (!result || !failure.empty() || !ended) { std::cerr << "INCOMPLETE: " << failure << '\n'; return 1; }
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
