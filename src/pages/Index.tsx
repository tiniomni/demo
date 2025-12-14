import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Mic, Upload, Download, Play, Pause, AlertCircle, CheckCircle2, Sparkles } from 'lucide-react';

export default function Index() {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [processedAudioUrl, setProcessedAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement>(null);
  const sessionIdRef = useRef<string | null>(null);

  // API配置 - 使用服务端 FastAPI 的实际端口（start_api.sh 中为 1020）
  const API_BASE_URL = 'https://mnemic-trudie-waterlessly.ngrok-free.dev';

  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const drawWaveform = () => {
    if (!analyserRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      // 创建渐变背景 - 糖果色
      const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#FFF5F7');
      gradient.addColorStop(1, '#FFF0F5');
      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

      // 绘制波形 - 使用糖果色渐变
      const waveGradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
      waveGradient.addColorStop(0, '#FF6B9D');
      waveGradient.addColorStop(0.5, '#C084FC');
      waveGradient.addColorStop(1, '#60A5FA');
      
      canvasCtx.lineWidth = 3;
      canvasCtx.strokeStyle = waveGradient;
      canvasCtx.shadowBlur = 10;
      canvasCtx.shadowColor = '#FF6B9D';
      canvasCtx.beginPath();

      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;

        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
      canvasCtx.shadowBlur = 0;
    };

    draw();
  };

  const startRecording = async () => {
    try {
      setError(null);
      setSuccess(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      audioContextRef.current = new AudioContext();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      source.connect(analyserRef.current);

      drawWaveform();

      // 为本次录音生成一个会话 ID，用于流式上传
      sessionIdRef.current = crypto.randomUUID();

      // 使用浏览器支持的 webm 容器，由服务端统一转换为 WAV
      let mediaRecorder: MediaRecorder;
      try {
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      } catch {
        // 某些浏览器可能不支持指定 mimeType，退回默认配置
        mediaRecorder = new MediaRecorder(stream);
      }
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          // 在前端保留完整音频，便于显示大小或后续扩展
          audioChunksRef.current.push(event.data);

          // 录音过程中，持续将分片发送到服务端进行实时写入
          if (sessionIdRef.current) {
            try {
              await fetch(`${API_BASE_URL}/api/stream/chunk?session_id=${sessionIdRef.current}`, {
                method: 'POST',
                body: event.data,
                headers: {
                  'ngrok-skip-browser-warning': 'true'
                }
              });
            } catch (e) {
              console.error('上传音频分片失败:', e);
            }
          }
        }
      };

      mediaRecorder.onstop = async () => {
        // 在前端合成一份完整音频（仅用于显示“音频已准备就绪”的体积信息）
        const fullBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(fullBlob);
        stream.getTracks().forEach(track => track.stop());
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }

        // 可选：通知服务端本次录音结束，仅作标记，不触发推理
        if (sessionIdRef.current) {
          try {
            await fetch(
              `${API_BASE_URL}/api/stream/finish?session_id=${sessionIdRef.current}`,
              {
                method: 'POST',
                headers: {
                  'ngrok-skip-browser-warning': 'true'
                }
              }
            );
          } catch (e) {
            console.error('通知服务端录音结束失败:', e);
          }
        }
      };

      // 传入 timeslice（毫秒），让浏览器定期触发 ondataavailable，实现分片上传
      mediaRecorder.start(500);
      setIsRecording(true);
    } catch (err) {
      setError('无法访问麦克风，请检查权限设置');
      console.error('录音错误:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // 验证文件格式
      if (!file.name.toLowerCase().endsWith('.wav')) {
        setError('仅支持 WAV 格式文件');
        return;
      }
      setError(null);
      setSuccess(null);
      setAudioBlob(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.wav')) {
        setError('仅支持 WAV 格式文件');
        return;
      }
      setError(null);
      setSuccess(null);
      setAudioBlob(file);
    }
  };

  const processAudio = async () => {
    setIsProcessing(true);
    setError(null);
    setSuccess(null);
    setProgress(0);

    try {
      // 如果存在 sessionId，优先使用流式上传的音频，避免重复上传整段文件
      const useSession = !!sessionIdRef.current;

      // 模拟进度更新
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      let response: Response;

      if (useSession) {
        response = await fetch(
          `${API_BASE_URL}/api/process?session_id=${sessionIdRef.current}`,
          {
            method: 'POST',
            headers: {
              'ngrok-skip-browser-warning': 'true'
            }
          }
        );
      } else {
        if (!audioBlob) {
          setError('请先录音或上传音频文件');
          clearInterval(progressInterval);
          setIsProcessing(false);
          setTimeout(() => setProgress(0), 1000);
          return;
        }

        const formData = new FormData();
        formData.append('audio', audioBlob, 'input.wav');

        response = await fetch(`${API_BASE_URL}/api/process`, {
          method: 'POST',
          body: formData,
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        });
      }

      clearInterval(progressInterval);
      setProgress(100);

      if (!response.ok) {
        throw new Error(`服务器错误: ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setProcessedAudioUrl(url);
      setSuccess('音频处理完成！');
    } catch (err) {
      // 统一显示友好的错误提示
      const errorMessage = err instanceof Error && err.message.includes('服务器错误')
        ? err.message
        : '服务繁忙，请稍后重试';
      setError(errorMessage);
      console.error('处理错误:', err);
    } finally {
      setIsProcessing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const downloadAudio = () => {
    if (processedAudioUrl) {
      const a = document.createElement('a');
      a.href = processedAudioUrl;
      a.download = 'processed_audio.wav';
      a.click();
    }
  };

  const togglePlayback = () => {
    if (audioPlayerRef.current) {
      if (isPlaying) {
        audioPlayerRef.current.pause();
      } else {
        audioPlayerRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-blue-50 relative overflow-hidden">
      {/* 动态背景装饰 - 糖果色 */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-pink-300/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-300/20 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 right-1/3 w-80 h-80 bg-blue-300/15 rounded-full blur-3xl animate-pulse delay-500" />
      </div>

      {/* Hero Section */}
      <div className="relative h-72 overflow-hidden">
        <img 
          src="/assets/hero-ai-waveform.jpg" 
          alt="AI Waveform" 
          className="w-full h-full object-cover opacity-20"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-pink-50/50 to-pink-50" />
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center space-y-4 px-4">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-pink-100/80 border border-pink-200 backdrop-blur-sm mb-4">
              <Sparkles className="w-4 h-4 text-pink-500" />
              <span className="text-sm text-pink-700 font-medium">AI驱动的语音处理</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-pink-500 via-purple-500 to-blue-500 bg-clip-text text-transparent mb-4 animate-in fade-in slide-in-from-bottom-4 duration-700">
              Tini-Omni 语音助手
            </h1>
            <p className="text-xl text-gray-600 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-150">
              智能语音交互 · 实时对话 · 自然流畅
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-12 max-w-6xl relative z-10">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Input Section */}
          <Card className="bg-white/80 backdrop-blur-xl border-pink-200 shadow-xl hover:shadow-pink-200/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-gray-800 flex items-center gap-2">
                <div className="w-2 h-2 bg-pink-500 rounded-full animate-pulse" />
                输入音频
              </CardTitle>
              <CardDescription className="text-gray-600">
                录制或上传您的音频文件
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Tabs defaultValue="record" className="w-full">
                <TabsList className="grid w-full grid-cols-2 bg-pink-50">
                  <TabsTrigger 
                    value="record" 
                    className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-pink-400 data-[state=active]:to-pink-500 data-[state=active]:text-white"
                  >
                    <Mic className="w-4 h-4 mr-2" />
                    录音
                  </TabsTrigger>
                  <TabsTrigger 
                    value="upload" 
                    className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-pink-400 data-[state=active]:to-pink-500 data-[state=active]:text-white"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    上传
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="record" className="space-y-4">
                  <div className="relative">
                    <canvas
                      ref={canvasRef}
                      width={600}
                      height={150}
                      className="w-full rounded-xl bg-gradient-to-br from-pink-50 to-purple-50 border border-pink-200 shadow-inner"
                    />
                    {!isRecording && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <p className="text-gray-400 text-sm">点击下方按钮开始录音</p>
                          <p className="text-gray-400 text-sm mt-1">仅支持英文对话</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <Button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`w-full h-14 text-lg font-semibold transition-all duration-300 ${
                      isRecording 
                        ? 'bg-gradient-to-r from-red-400 to-red-500 hover:from-red-500 hover:to-red-600 text-white shadow-lg shadow-red-300/50' 
                        : 'bg-gradient-to-r from-pink-400 to-pink-500 hover:from-pink-500 hover:to-pink-600 text-white shadow-lg shadow-pink-300/50'
                    }`}
                  >
                    <Mic className={`mr-2 h-5 w-5 ${isRecording ? 'animate-pulse' : ''}`} />
                    {isRecording ? '停止录音' : '开始录音'}
                  </Button>
                </TabsContent>
                
                <TabsContent value="upload" className="space-y-4">
                  <div 
                    className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 backdrop-blur-sm ${
                      isDragging 
                        ? 'border-pink-500 bg-pink-100/50 scale-[1.02]' 
                        : 'border-pink-200 hover:border-pink-400 hover:bg-pink-50/50'
                    }`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                  >
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-br from-pink-100 to-purple-100 flex items-center justify-center">
                      <Upload className={`h-8 w-8 text-pink-500 transition-transform ${isDragging ? 'scale-110' : ''}`} />
                    </div>
                    <label htmlFor="file-upload" className="cursor-pointer">
                      <span className="text-pink-500 hover:text-pink-600 font-medium">
                        点击上传
                      </span>
                      <span className="text-gray-500"> 或拖拽文件到此处</span>
                      <input
                        id="file-upload"
                        type="file"
                        accept=".wav"
                        onChange={handleFileUpload}
                        className="hidden"
                      />
                    </label>
                    <p className="text-sm text-gray-400 mt-3">
                      仅支持 WAV 格式
                    </p>
                    <p className="text-gray-400 text-sm mt-1">仅支持英文对话</p>
                  </div>
                </TabsContent>
              </Tabs>

              {audioBlob && (
                <Alert className="bg-gradient-to-r from-green-50 to-emerald-50 border-green-300 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-700">
                    音频已准备就绪 ({(audioBlob.size / 1024 / 1024).toFixed(2)} MB)
                  </AlertDescription>
                </Alert>
              )}

              <Button
                onClick={processAudio}
                disabled={!audioBlob || isProcessing}
                className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 hover:from-pink-500 hover:via-purple-500 hover:to-blue-500 disabled:from-gray-300 disabled:to-gray-400 text-white shadow-lg shadow-purple-300/50 transition-all duration-300"
              >
                {isProcessing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                    处理中...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-5 w-5" />
                    开始处理
                  </>
                )}
              </Button>

              {isProcessing && (
                <div className="space-y-2">
                  <Progress value={progress} className="w-full h-2" />
                  <p className="text-sm text-gray-500 text-center">{progress}%</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Output Section */}
          <Card className="bg-white/80 backdrop-blur-xl border-purple-200 shadow-xl hover:shadow-purple-200/50 transition-all duration-300">
            <CardHeader>
              <CardTitle className="text-gray-800 flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                输出音频
              </CardTitle>
              <CardDescription className="text-gray-600">
                处理后的音频结果
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {processedAudioUrl ? (
                <>
                  <div className="bg-gradient-to-br from-pink-50 to-purple-50 rounded-xl p-8 text-center border border-purple-200 shadow-inner">
                    <div className="mb-6">
                      <div className="w-28 h-28 mx-auto bg-gradient-to-br from-pink-400 via-purple-400 to-blue-400 rounded-full flex items-center justify-center shadow-lg shadow-purple-300/50 animate-in zoom-in duration-500">
                        <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center">
                          {isPlaying ? (
                            <Pause className="h-12 w-12 text-purple-500" />
                          ) : (
                            <Play className="h-12 w-12 text-purple-500 ml-1" />
                          )}
                        </div>
                      </div>
                    </div>
                    
                    <audio
                      ref={audioPlayerRef}
                      src={processedAudioUrl}
                      onEnded={() => setIsPlaying(false)}
                      className="hidden"
                    />
                    
                    <Button
                      onClick={togglePlayback}
                      className="w-full h-14 bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 hover:from-pink-500 hover:via-purple-500 hover:to-blue-500 text-white mb-3 text-lg font-semibold shadow-lg shadow-purple-300/50 transition-all duration-300"
                    >
                      {isPlaying ? (
                        <>
                          <Pause className="mr-2 h-5 w-5" />
                          暂停播放
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-5 w-5" />
                          播放音频
                        </>
                      )}
                    </Button>
                    
                    <Button
                      onClick={downloadAudio}
                      variant="outline"
                      className="w-full h-14 border-purple-300 bg-white/50 text-purple-600 hover:bg-purple-50 hover:border-purple-400 text-lg font-semibold transition-all duration-300"
                    >
                      <Download className="mr-2 h-5 w-5" />
                      下载音频
                    </Button>
                  </div>
                </>
              ) : (
                <div className="bg-gradient-to-br from-pink-50 to-purple-50 rounded-xl p-16 text-center border border-purple-200 shadow-inner">
                  <div className="text-gray-400">
                    <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-purple-100 flex items-center justify-center">
                      <AlertCircle className="h-10 w-10 opacity-50" />
                    </div>
                    <p className="text-lg font-medium mb-2">等待处理...</p>
                    <p className="text-sm">处理完成后音频将显示在这里</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Alerts */}
        {error && (
          <Alert className="mt-6 bg-gradient-to-r from-red-50 to-rose-50 border-red-300 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <AlertDescription className="text-red-700">{error}</AlertDescription>
          </Alert>
        )}

        {success && (
          <Alert className="mt-6 bg-gradient-to-r from-green-50 to-emerald-50 border-green-300 backdrop-blur-sm animate-in fade-in slide-in-from-bottom-2">
            <CheckCircle2 className="h-4 w-4 text-green-600" />
            <AlertDescription className="text-green-700">{success}</AlertDescription>
          </Alert>
        )}

        {/* Instructions */}
        <Card className="mt-8 bg-white/80 backdrop-blur-xl border-pink-200 shadow-xl">
          <CardHeader>
            <CardTitle className="text-gray-800 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-pink-500" />
              使用说明
            </CardTitle>
          </CardHeader>
          <CardContent className="text-gray-700 space-y-3">
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 rounded-full bg-pink-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-pink-600 text-sm font-bold">1</span>
              </div>
              <p><strong className="text-gray-800">录音或上传</strong>：使用麦克风录制音频，或上传现有的 WAV 格式音频文件</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 rounded-full bg-purple-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-purple-600 text-sm font-bold">2</span>
              </div>
              <p><strong className="text-gray-800">处理音频</strong>：点击"开始处理"按钮，将音频发送到服务端进行 AI 模型推理</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                <span className="text-blue-600 text-sm font-bold">3</span>
              </div>
              <p><strong className="text-gray-800">播放和下载</strong>：处理完成后，可以播放或下载转换后的音频</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}