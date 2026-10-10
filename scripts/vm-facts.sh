#!/usr/bin/env bash
# Read-only facts about a VM, for sizing and the CPU benchmark (ADR-0020). Changes nothing.
#   ssh <host> 'bash -s' < scripts/vm-facts.sh
set -u
section() { printf '\n== %s\n' "$1"; }
section "host";    hostnamectl 2>/dev/null | grep -E "Static hostname|Operating System|Kernel|Virtualization" || uname -a
section "cpu";     lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)|L3 cache)"
                   printf 'AVX2: %s  AVX-512: %s\n' "$(grep -qw avx2 /proc/cpuinfo && echo yes || echo no)" \
                     "$(grep -qw avx512f /proc/cpuinfo && echo yes || echo no)"
section "memory";  free -h | head -2
section "disk";    df -h / /var/lib/docker 2>/dev/null | awk 'NR==1 || !seen[$1]++'
section "gpu";     (command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv) || echo "no NVIDIA GPU"
section "docker";  (docker --version && docker compose version) 2>/dev/null || echo "docker not installed"
section "tailscale"; (tailscale version | head -1 && tailscale ip -4 && tailscale status --self --peers=false) 2>/dev/null || echo "tailscale not installed"
section "network"; printf 'public egress IP: %s\n' "$(curl -s --max-time 5 https://ifconfig.me || echo unknown)"
                   ip -4 -brief addr | grep -v "^lo"
section "listening ports"; (ss -ltnp 2>/dev/null || ss -ltn) | awk 'NR==1 || /LISTEN/' | head -20
section "ollama";  (command -v ollama >/dev/null && ollama list) || docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | grep -i ollama || echo "no ollama"
