# Cloud Infer-Only (GPU) Setup

Goal: run ONLY the MiniCPM-o C++ inference service on a cloud GPU Linux box, while keeping LiveKit + Backend + Frontend on your Windows local machine.

This matches "方案 1": cloud does NOT register to backend. Your Windows backend manually registers a service that points to the SSH-tunneled localhost port.

## Cloud prerequisites

- Ubuntu 22.04/24.04
- NVIDIA driver working (`nvidia-smi`)
- Build tools: `git`, `cmake`, `make` (or `ninja`), `g++`
- Python 3.10+ (recommended 3.11)
- (Recommended for GPU build) CUDA toolkit available (`nvcc`) or `/usr/local/cuda`
  - If you only have driver but no toolkit, the script will likely build CPU-only.

## What runs on cloud

- `llama.cpp-omni` + compiled `llama-server`
- `WebRTC_Demo/cpp_server/minicpmo_cpp_http_server.py`
- Model files in GGUF repo: `openbmb/MiniCPM-o-4_5-gguf`

Ports:
- 9060: inference API
- 9061: health

## On the cloud machine

From the `WebRTC_Demo` directory:

```bash
bash cloud_infer_only.sh start
```

Optional env overrides:

```bash
HF_ENDPOINT=https://hf-mirror.com \
LLM_QUANT=Q4_K_M \
CPP_SERVER_PORT=9060 \
CPP_MODE=duplex \
bash cloud_infer_only.sh start
```

Verify on cloud:

```bash
curl -s http://127.0.0.1:9060/health
```

## SSH tunnel (Windows local -> cloud)

On Windows PowerShell (keep this terminal open):

```powershell
ssh -N -L 9060:127.0.0.1:9060 -L 9061:127.0.0.1:9061 ubuntu@<cloud_ip>
```

Verify on Windows:

```powershell
curl http://127.0.0.1:9060/health
```

## Manual registration (Windows backend)

Because the cloud inference service is reachable locally via the SSH tunnel, your Windows backend should register a service pointing to `127.0.0.1:9060`.

Example (adjust backend port if needed; default in oneclick is 8021):

```powershell
$backendPort = 8021
$inferPort = 9060
curl -X POST "http://127.0.0.1:$backendPort/api/inference/register" `
  -H "Content-Type: application/json" `
  -d "{\"ip\":\"127.0.0.1\",\"port\":$inferPort,\"model_port\":$inferPort,\"model_type\":\"duplex\",\"session_type\":\"release\",\"service_name\":\"o45-cpp-cloud\"}"
```

Then verify backend sees it:

```powershell
curl "http://127.0.0.1:$backendPort/api/inference/services"
```

If `services` becomes non-empty, `POST /api/login` should stop failing due to "no inference service".
