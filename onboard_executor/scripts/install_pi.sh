#!/bin/sh
set -eu

# Install only files whose definitions live in this repository.
# Run from the repository checkout:
#   sudo onboard_executor/scripts/install_pi.sh [CONFIG.json]

if [ "$(id -u)" -ne 0 ]; then
	echo "run this script with sudo" >&2
	exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
executor_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd)
repo_dir=$(CDPATH= cd -- "${executor_dir}/.." && pwd)

binary="${executor_dir}/build/fftgyro-executor"
offboard_binary="${executor_dir}/build/fftgyro-offboard-executor"
config="${1:-${executor_dir}/config/run.example.json}"
service="${executor_dir}/systemd/fftgyro-executor.service"
dispatcher="${executor_dir}/scripts/fftgyro-executor-dispatch"

for file in "$binary" "$offboard_binary" "$config" "$service" "$dispatcher"; do
	if [ ! -f "$file" ]; then
		echo "missing install input: $file" >&2
		exit 1
	fi
done

case "$(file -b "$binary")" in
	*"ARM aarch64"*) ;;
	*)
		echo "executor is not an ARM64 Linux binary: $binary" >&2
		exit 1
		;;
esac
case "$(file -b "$offboard_binary")" in
	*"ARM aarch64"*) ;;
	*)
		echo "Offboard executor is not an ARM64 Linux binary: $offboard_binary" >&2
		exit 1
		;;
esac

if ! getent group fftgyro >/dev/null 2>&1; then
	groupadd --system fftgyro
fi
if ! getent passwd fftgyro >/dev/null 2>&1; then
	useradd --system --gid fftgyro --home-dir /var/lib/fftgyro-executor \
		--shell /usr/sbin/nologin fftgyro
fi
usermod -a -G dialout fftgyro
if getent group gpio >/dev/null 2>&1; then
	usermod -a -G gpio fftgyro
fi

install -m 0755 "$binary" /usr/local/bin/fftgyro-executor
install -m 0755 "$offboard_binary" /usr/local/bin/fftgyro-offboard-executor
install -m 0755 "$dispatcher" /usr/local/bin/fftgyro-executor-dispatch
install -d -o root -g fftgyro -m 0750 /etc/fftgyro-executor
install -o root -g fftgyro -m 0640 "$config" /etc/fftgyro-executor/run.json
install -m 0644 "$service" /etc/systemd/system/fftgyro-executor.service

install -d -o root -g root -m 0755 /usr/local/share/fftgyro-executor
commit=${FFTGYRO_SOURCE_COMMIT:-$(git -C "$repo_dir" rev-parse HEAD 2>/dev/null || echo unknown)}
tree_state=${FFTGYRO_SOURCE_TREE_STATE:-clean}
if [ -z "${FFTGYRO_SOURCE_TREE_STATE:-}" ] \
	&& [ -n "$(git -C "$repo_dir" status --porcelain -- onboard_executor protocol 2>/dev/null || true)" ]; then
	tree_state=dirty
fi
binary_sha=$(sha256sum "$binary" | awk '{print $1}')
offboard_binary_sha=$(sha256sum "$offboard_binary" | awk '{print $1}')
dispatcher_sha=$(sha256sum "$dispatcher" | awk '{print $1}')
config_sha=$(sha256sum "$config" | awk '{print $1}')
service_sha=$(sha256sum "$service" | awk '{print $1}')
installed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

manifest=/usr/local/share/fftgyro-executor/install-manifest.json
{
	echo "{"
	echo "  \"git_commit\": \"${commit}\","
	echo "  \"git_tree_state\": \"${tree_state}\","
	echo "  \"installed_at_utc\": \"${installed_at}\","
	echo "  \"binary_sha256\": \"${binary_sha}\","
	echo "  \"offboard_binary_sha256\": \"${offboard_binary_sha}\","
	echo "  \"dispatcher_sha256\": \"${dispatcher_sha}\","
	echo "  \"config_sha256\": \"${config_sha}\","
	echo "  \"service_sha256\": \"${service_sha}\""
	echo "}"
} >"$manifest"
chmod 0644 "$manifest"

systemctl daemon-reload
systemctl disable fftgyro-executor.service >/dev/null 2>&1 || true

echo "installed; service remains disabled and stopped"
echo "manifest: $manifest"
