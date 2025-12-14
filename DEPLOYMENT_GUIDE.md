# 语音到语音模型服务端部署指南

## 目录
1. [环境要求](#环境要求)
2. [项目结构](#项目结构)
3. [安装步骤](#安装步骤)
4. [API服务代码](#api服务代码)
5. [Docker部署](#docker部署)
6. [测试API](#测试api)
7. [常见问题](#常见问题)

---

## 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐RTX 3060或更高)
- **显存**: 至少4GB (模型1GB + 推理开销)
- **内存**: 至少8GB RAM
- **存储**: 至少10GB可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04+ / CentOS 7+ / Windows 10+ (推荐Linux)
- **Python**: 3.8 - 3.11
- **CUDA**: 11.8 或 12.1 (与PyTorch版本匹配)
- **cuDNN**: 对应CUDA版本的cuDNN

---

## 项目结构

创建以下目录结构：

```
speech-to-speech-backend/
├── app.py                 # Flask/FastAPI主程序
├── model_loader.py        # 模型加载和推理
├── requirements.txt       # Python依赖
├── Dockerfile            # Docker配置
├── docker-compose.yml    # Docker Compose配置
├── models/               # 模型文件目录
│   └── your_model.pt     # 您的PyTorch模型
├── uploads/              # 临时上传文件
└── outputs/              # 处理后的音频输出
```

---

## 安装步骤

### 方法1: 直接安装 (推荐用于开发)

#### 1. 安装CUDA和cuDNN

```bash
# Ubuntu示例 - 安装CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 添加到环境变量
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
nvidia-smi
```

#### 2. 创建Python虚拟环境

```bash
# 创建项目目录
mkdir speech-to-speech-backend
cd speech-to-speech-backend

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 升级pip
pip install --upgrade pip
```

#### 3. 安装PyTorch (GPU版本)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证GPU可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

#### 4. 安装其他依赖

创建 `requirements.txt`:

```txt
flask==3.0.0
flask-cors==4.0.0
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
soundfile>=0.12.0
```

安装：

```bash
pip install -r requirements.txt
```

---

## API服务代码

### 1. 创建 `model_loader.py`

```python
import torch
import torchaudio
import numpy as np
from pathlib import Path

class SpeechToSpeechModel:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        初始化语音到语音模型
        
        Args:
            model_path: 模型文件路径 (.pt文件)
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        print("模型加载完成")
        
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000):
        """
        预处理音频文件
        
        Args:
            audio_path: 输入音频文件路径
            target_sr: 目标采样率
            
        Returns:
            处理后的音频张量
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 重采样
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 移到GPU
        waveform = waveform.to(self.device)
        
        return waveform
    
    def postprocess_audio(self, output_tensor, output_path: str, sample_rate: int = 16000):
        """
        后处理模型输出
        
        Args:
            output_tensor: 模型输出张量
            output_path: 输出文件路径
            sample_rate: 采样率
        """
        # 移到CPU
        output_tensor = output_tensor.cpu()
        
        # 保存音频
        torchaudio.save(output_path, output_tensor, sample_rate)
        
    @torch.no_grad()
    def process(self, input_audio_path: str, output_audio_path: str):
        """
        处理音频文件
        
        Args:
            input_audio_path: 输入音频路径
            output_audio_path: 输出音频路径
        """
        # 预处理
        input_tensor = self.preprocess_audio(input_audio_path)
        
        # 推理
        # 注意: 这里需要根据您的模型实际输入输出格式进行调整
        output_tensor = self.model(input_tensor)
        
        # 后处理
        self.postprocess_audio(output_tensor, output_audio_path)
        
        return output_audio_path


# 示例: 如果您的模型有特殊的推理方式，请修改这个类
class CustomSpeechModel(SpeechToSpeechModel):
    """
    自定义模型类 - 根据您的模型特点修改
    """
    
    @torch.no_grad()
    def process(self, input_audio_path: str, output_audio_path: str):
        """
        自定义推理流程
        """
        # 1. 加载和预处理
        waveform, sr = torchaudio.load(input_audio_path)
        waveform = waveform.to(self.device)
        
        # 2. 模型推理 - 根据您的模型调整
        # 例如: 如果模型需要特定的输入格式
        # input_features = self.extract_features(waveform)
        # output = self.model(input_features)
        
        output = self.model(waveform)
        
        # 3. 保存输出
        output = output.cpu()
        torchaudio.save(output_audio_path, output, sr)
        
        return output_audio_path
```

### 2. 创建 `app.py` (Flask版本)

```python
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path
from model_loader import SpeechToSpeechModel

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'models/your_model.pt'  # 修改为您的模型路径
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

# 创建必要的目录
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# 加载模型 (启动时加载一次)
print("正在加载模型...")
model = SpeechToSpeechModel(MODEL_PATH, device='cuda')
print("模型加载完成，API服务已启动")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available()
    })

@app.route('/api/process', methods=['POST'])
def process_audio():
    """
    处理音频的主接口
    
    请求: multipart/form-data
        - audio: 音频文件
    
    响应: 处理后的音频文件
    """
    try:
        # 检查文件
        if 'audio' not in request.files:
            return jsonify({'error': '没有上传音频文件'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        input_filename = secure_filename(f"{file_id}_input.wav")
        output_filename = f"{file_id}_output.wav"
        
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # 保存上传的文件
        file.save(input_path)
        print(f"接收到音频文件: {input_path}")
        
        # 处理音频
        print("开始处理...")
        model.process(input_path, output_path)
        print(f"处理完成: {output_path}")
        
        # 返回处理后的文件
        response = send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='processed_audio.wav'
        )
        
        # 清理临时文件 (可选)
        # os.remove(input_path)
        # os.remove(output_path)
        
        return response
        
    except Exception as e:
        print(f"处理错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    import torch
    return jsonify({
        'model_path': MODEL_PATH,
        'device': str(model.device),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

### 3. 创建 `app.py` (FastAPI版本 - 推荐)

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
import uuid
from pathlib import Path
from model_loader import SpeechToSpeechModel
import torch

app = FastAPI(title="Speech-to-Speech API")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_PATH = 'models/your_model.pt'

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

# 加载模型
print("正在加载模型...")
model = SpeechToSpeechModel(MODEL_PATH, device='cuda')
print("模型加载完成")

@app.get("/health")
async def health_check():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available()
    }

@app.post("/api/process")
async def process_audio(audio: UploadFile = File(...)):
    try:
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_input.wav")
        output_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_output.wav")
        
        # 保存上传文件
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        print(f"接收到音频: {input_path}")
        
        # 处理
        model.process(input_path, output_path)
        print(f"处理完成: {output_path}")
        
        # 返回文件
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="processed_audio.wav"
        )
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/info")
async def model_info():
    return {
        'model_path': MODEL_PATH,
        'device': str(model.device),
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Docker部署

### 1. 创建 `Dockerfile`

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 安装Python和系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p uploads outputs models

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "app.py"]
```

### 2. 创建 `docker-compose.yml`

```yaml
version: '3.8'

services:
  speech-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
```

### 3. 构建和运行

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 测试API

### 使用curl测试

```bash
# 健康检查
curl http://localhost:8000/health

# 获取模型信息
curl http://localhost:8000/api/info

# 处理音频
curl -X POST http://localhost:8000/api/process \
  -F "audio=@test_audio.wav" \
  -o output.wav
```

### Python测试脚本

创建 `test_api.py`:

```python
import requests

API_URL = "http://localhost:8000"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print("健康检查:", response.json())

def test_process(audio_file):
    with open(audio_file, 'rb') as f:
        files = {'audio': f}
        response = requests.post(f"{API_URL}/api/process", files=files)
        
        if response.status_code == 200:
            with open('output.wav', 'wb') as out:
                out.write(response.content)
            print("处理成功，输出保存为 output.wav")
        else:
            print(f"错误: {response.status_code}")
            print(response.json())

if __name__ == "__main__":
    test_health()
    test_process("test_audio.wav")
```

---

## 常见问题

### 1. CUDA Out of Memory

**问题**: GPU显存不足

**解决方案**:
```python
# 在model_loader.py中添加
torch.cuda.empty_cache()

# 或使用混合精度
with torch.cuda.amp.autocast():
    output = model(input_tensor)
```

### 2. 模型加载失败

**问题**: 无法加载.pt文件

**解决方案**:
```python
# 尝试不同的加载方式
model = torch.load(model_path, map_location=device)
# 或
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. 音频格式问题

**问题**: 不支持某些音频格式

**解决方案**:
```bash
# 安装ffmpeg
sudo apt-get install ffmpeg

# 或使用pydub转换
pip install pydub
```

### 4. 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8000

# 杀死进程
kill -9 <PID>

# 或更改端口
python app.py --port 8001
```

### 5. 性能优化

```python
# 使用TorchScript优化
model = torch.jit.script(model)

# 启用cudnn benchmark
torch.backends.cudnn.benchmark = True

# 批处理推理
def process_batch(audio_files):
    batch = torch.stack([preprocess(f) for f in audio_files])
    with torch.no_grad():
        outputs = model(batch)
    return outputs
```

---

## 生产环境建议

1. **使用HTTPS**: 配置SSL证书
2. **添加认证**: 实现API密钥或JWT认证
3. **限流**: 使用Redis实现请求限流
4. **日志**: 配置完善的日志系统
5. **监控**: 使用Prometheus + Grafana监控
6. **负载均衡**: 使用Nginx做反向代理

---

## 联系支持

如有问题，请参考:
- PyTorch文档: https://pytorch.org/docs/
- Flask文档: https://flask.palletsprojects.com/
- FastAPI文档: https://fastapi.tiangolo.com/

祝部署顺利！🚀