
import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { 
  Antenna, 
  Table as TableIcon, 
  Download, 
  Upload, 
  Cpu, 
  Activity, 
  Trash2, 
  Plus, 
  Layers, 
  Zap, 
  X, 
  BookOpen, 
  FileSpreadsheet,
  Boxes, 
  LayoutGrid,
  RefreshCw, 
  FileCode,
  Maximize,
  Minimize,
  Box,
  RotateCw,
  Navigation2,
  MoveRight,
  ChevronRight,
  Info,
  Settings2,
  ChevronDown,
  ListFilter,
  Eye, 
  EyeOff,
  Check,
  Target,
  FileDown,
  ArrowDownAz,
  ArrowDownWideNarrow,
  ArrowUpNarrowWide,
  SortAsc,
  Mic,
  MicOff,
  Sparkles,
  MessageSquare,
  Waves,
  PhoneOff,
  PhoneCall
} from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { GoogleGenAI, Type, Modality, LiveServerMessage } from "@google/genai";
import { 
  ICType, 
  ICConfig, 
  ArrayConfig, 
  ScanRange, 
  CalibrationMap, 
  BeamTableRow, 
  BeamMode,
  MappingOrder,
  PortMappingMap,
  CoordinateSystem
} from './types';
import { calculateBeamTable, parseCalibrationFile, parseMappingFile, getArrayFactorGain } from './services/physics';

// --- Base64 / PCM Utils for Live API ---
function encode(bytes: Uint8Array) {
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
  }
  return buffer;
}

// --- Constants ---
const CHART_COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', 
  '#8b5cf6', '#ec4899', '#06b6d4', '#f97316',
  '#6366f1', '#14b8a6', '#f43f5e', '#a855f7'
];

const INITIAL_ARRAY: ArrayConfig = {
  channelName: 'CH1', // 新增：預設通道名稱
  frequencyGHz: 19.825,
  nx: 16,
  ny: 2,
  dx_mm: 7.68,
  dy_mm: 36.43,
  mode: 'HYBRID',
  topology: {
    elementsPerChip: 4,
    mappingOrder: 'ROW_MAJOR'
  }
};

const INITIAL_SCAN: ScanRange = {
  system: 'AZ_EL',
  azStart: 0.0,
  azEnd: 30.0,
  azStep: 5.0,
  elStart: 30.0,
  elEnd: 45.0,
  elStep: 5.0
};

const INITIAL_ICS: ICConfig[] = [
  { id: '1', name: "TTD_STAGE", type: "TTD", lsb: 0.865, max: 54.5, bits: 6, offset: 0, polarity: 'NORMAL' },
  { id: '2', name: "PS_STAGE", type: "PS", lsb: 360.0/64, max: 360.0, bits: 6, offset: 0, polarity: 'NORMAL' },
];

const createTextSprite = (text: string, color: string) => {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  if (!context) return new THREE.Sprite();
  canvas.width = 256; canvas.height = 128;
  context.fillStyle = color; context.font = 'bold 84px Inter, sans-serif';
  context.textAlign = 'center'; context.textBaseline = 'middle';
  context.fillText(text, 128, 64);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material); sprite.scale.set(1.2, 0.6, 1);
  return sprite;
};

// --- AI Copilot Component (Live API Enhanced) ---
const AIAssistantModal = ({ 
  isOpen, 
  onClose, 
  currentConfig, 
  currentScan,
  onUpdateConfig 
}: { 
  isOpen: boolean, 
  onClose: () => void, 
  currentConfig: ArrayConfig, 
  currentScan: ScanRange,
  onUpdateConfig: (newConfig: Partial<ArrayConfig>, newScan: Partial<ScanRange>) => void 
}) => {
  const [input, setInput] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'ai', content: string }[]>([]);
  const [isLiveMode, setIsLiveMode] = useState(false);
  
  const sessionRef = useRef<any>(null);
  const inputAudioCtxRef = useRef<AudioContext | null>(null);
  const outputAudioCtxRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [chatHistory, isThinking]);

  useEffect(() => {
    return () => stopLiveSession();
  }, []);

  const stopLiveSession = () => {
    setIsLiveMode(false);
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (inputAudioCtxRef.current) {
      inputAudioCtxRef.current.close();
      inputAudioCtxRef.current = null;
    }
    if (outputAudioCtxRef.current) {
      outputAudioCtxRef.current.close();
      outputAudioCtxRef.current = null;
    }
    sourcesRef.current.forEach(s => s.stop());
    sourcesRef.current.clear();
  };

  const startLiveSession = async () => {
    setIsLiveMode(true);
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || "" });

    const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    
    inputAudioCtxRef.current = inputCtx;
    outputAudioCtxRef.current = outputCtx;

    const outputNode = outputCtx.createGain();
    outputNode.connect(outputCtx.destination);

    if (inputCtx.state === 'suspended') await inputCtx.resume();
    if (outputCtx.state === 'suspended') await outputCtx.resume();

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const lambda_mm = 299.79 / currentConfig.frequencyGHz;

    const sessionPromise = ai.live.connect({
      model: 'gemini-2.5-flash-native-audio-preview-12-2025',
      callbacks: {
        onopen: () => {
          const source = inputCtx.createMediaStreamSource(stream);
          const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
          scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const int16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) int16[i] = inputData[i] * 32768;
            sessionPromise.then(session => {
              if (session) {
                session.sendRealtimeInput({ media: { data: encode(new Uint8Array(int16.buffer)), mimeType: 'audio/pcm;rate=16000' } });
              }
            });
          };
          source.connect(scriptProcessor);
          scriptProcessor.connect(inputCtx.destination);
        },
        onmessage: async (message: LiveServerMessage) => {
          const audioBase64 = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
          if (audioBase64 && outputCtx) {
            nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
            const audioBuffer = await decodeAudioData(decode(audioBase64), outputCtx, 24000, 1);
            const source = outputCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(outputNode);
            source.start(nextStartTimeRef.current);
            nextStartTimeRef.current += audioBuffer.duration;
            sourcesRef.current.add(source);
            source.onended = () => sourcesRef.current.delete(source);
          }

          if (message.serverContent?.interrupted) {
            sourcesRef.current.forEach(s => { try { s.stop(); } catch(e){} });
            sourcesRef.current.clear();
            nextStartTimeRef.current = 0;
          }

          if (message.toolCall) {
            for (const fc of message.toolCall.functionCalls) {
              if (fc.name === 'update_antenna_params') {
                onUpdateConfig(fc.args, {});
                sessionPromise.then(s => s.sendToolResponse({ functionResponses: [{ id: fc.id, name: fc.name, response: { result: "Antenna parameters updated successfully." } }] }));
              }
              if (fc.name === 'update_scan_range') {
                onUpdateConfig({}, fc.args);
                sessionPromise.then(s => s.sendToolResponse({ functionResponses: [{ id: fc.id, name: fc.name, response: { result: "Scan range updated successfully." } }] }));
              }
            }
          }
        },
        onclose: () => setIsLiveMode(false),
        onerror: (e) => {
          console.error("Live API Error:", e);
          stopLiveSession();
        }
      },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } } },
        systemInstruction: `你是 "Beam Master Live Copilot"。
        你是一位頂尖的天線博士與波束成形專家。你可以透過語音即時指導使用者設計陣列天線。
        
        目前的物理環境資料：
        - 通道名稱：${currentConfig.channelName}
        - 頻率：${currentConfig.frequencyGHz} GHz
        - 波長 (λ)：${lambda_mm.toFixed(3)} mm
        - 半波長 (λ/2)：${(lambda_mm / 2).toFixed(3)} mm
        - 陣列大小：${currentConfig.nx} (X) x ${currentConfig.ny} (Y)
        - 單元間距：dx=${currentConfig.dx_mm}mm (${(currentConfig.dx_mm/lambda_mm).toFixed(2)}λ), dy=${currentConfig.dy_mm}mm (${(currentConfig.dy_mm/lambda_mm).toFixed(2)}λ)
        
        你的任務：
        1. 解釋公式：當使用者詢問運算方式時，解釋路徑差 Δd = x·sinθ·cosφ + y·sinθ·sinφ 如何決定相位，以及 Hybrid Steering 結合 TTD 與 PS 的優點。
        2. 物理診斷：若使用者問「為什麼旁瓣（sidelobes）這麼高？」，分析目前間距是否超過 λ/2。若超過，解釋這會導致光柵瓣（grating lobes）。
        3. 狀態控制：你可以主動調用工具來更新參數。
        
        可用工具：
        - update_antenna_params: 更改 nx, ny, frequencyGHz, dx_mm, dy_mm, channelName。
        - update_scan_range: 更改 azStart, azEnd, elStart, elEnd。`,
        tools: [
          {
            functionDeclarations: [
              {
                name: 'update_antenna_params',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    nx: { type: Type.NUMBER },
                    ny: { type: Type.NUMBER },
                    frequencyGHz: { type: Type.NUMBER },
                    dx_mm: { type: Type.NUMBER },
                    dy_mm: { type: Type.NUMBER },
                    channelName: { type: Type.STRING }
                  }
                }
              },
              {
                name: 'update_scan_range',
                parameters: {
                  type: Type.OBJECT,
                  properties: {
                    azStart: { type: Type.NUMBER },
                    azEnd: { type: Type.NUMBER },
                    elStart: { type: Type.NUMBER },
                    elEnd: { type: Type.NUMBER }
                  }
                }
              }
            ]
          }
        ]
      }
    });

    sessionRef.current = await sessionPromise;
  };

  const handleSendText = async () => {
    if (!input.trim()) return;
    const query = input;
    setInput("");
    setChatHistory(prev => [...prev, { role: 'user', content: query }]);
    setIsThinking(true);

    const lambda_mm = 299.79 / currentConfig.frequencyGHz;

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || "" });
      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: query,
        config: {
          systemInstruction: `你是 Beam Master 專家助理。
          
          目前狀態：
          - 通道：${currentConfig.channelName}
          - 頻率：${currentConfig.frequencyGHz} GHz (λ = ${lambda_mm.toFixed(3)} mm)
          - 陣列：${currentConfig.nx}x${currentConfig.ny}
          - 間距：dx=${currentConfig.dx_mm}mm (${(currentConfig.dx_mm/lambda_mm).toFixed(2)}λ), dy=${currentConfig.dy_mm}mm (${(currentConfig.dy_mm/lambda_mm).toFixed(2)}λ)
          
          能力：
          1. 解析使用者指令並回傳 JSON 格式的 updates 欄位。
          2. 解釋物理現象（旁瓣、增益、公式運算）。
          3. 若間距 > λ/2，解釋會有光柵瓣。
          
          回覆格式：請務必遵循 responseSchema 提供 message 與 updates。`,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              message: { type: Type.STRING, description: "給使用者的文字回覆或物理分析內容。" },
              updates: {
                type: Type.OBJECT,
                properties: {
                  config: {
                    type: Type.OBJECT,
                    properties: {
                      channelName: { type: Type.STRING },
                      nx: { type: Type.NUMBER },
                      ny: { type: Type.NUMBER },
                      frequencyGHz: { type: Type.NUMBER },
                      dx_mm: { type: Type.NUMBER },
                      dy_mm: { type: Type.NUMBER }
                    }
                  },
                  scan: {
                    type: Type.OBJECT,
                    properties: {
                      azStart: { type: Type.NUMBER },
                      azEnd: { type: Type.NUMBER },
                      elStart: { type: Type.NUMBER },
                      elEnd: { type: Type.NUMBER }
                    }
                  }
                }
              }
            },
            required: ["message"]
          }
        }
      });
      
      const data = JSON.parse(response.text);
      setChatHistory(prev => [...prev, { role: 'ai', content: data.message }]);
      
      if (data.updates) {
        onUpdateConfig(data.updates.config || {}, data.updates.scan || {});
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', content: "指令處理失敗，請確認內容或 API 金鑰。" }]);
    } finally {
      setIsThinking(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md" onClick={onClose} />
      <div className="bg-slate-900 border border-slate-700 w-full max-w-2xl h-[650px] rounded-3xl shadow-2xl relative z-10 flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        <div className="p-5 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl shadow-lg transition-all duration-500 ${isLiveMode ? 'bg-emerald-600 shadow-emerald-900/30 animate-pulse' : 'bg-blue-600 shadow-blue-600/20'}`}>
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-2">
                Gemini Beam Copilot 
                {isLiveMode && <span className="text-[9px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full animate-pulse border border-emerald-500/30">LIVE</span>}
              </h3>
              <p className="text-[10px] text-slate-500 font-bold uppercase tracking-tight">AI Physics & Control Assistant</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-800 rounded-xl transition-colors text-slate-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-5 custom-scrollbar bg-slate-950/20">
          {isLiveMode ? (
            <div className="h-full flex flex-col items-center justify-center space-y-8 animate-in fade-in">
              <div className="relative">
                <div className="absolute inset-0 bg-emerald-500/20 rounded-full blur-3xl animate-pulse" />
                <div className="w-32 h-32 rounded-full border-4 border-emerald-500/30 flex items-center justify-center relative bg-slate-900 shadow-2xl">
                  <Waves className="w-12 h-12 text-emerald-400 animate-bounce" />
                </div>
              </div>
              <div className="text-center space-y-2">
                <h4 className="text-lg font-black text-white uppercase tracking-widest">正在通話中...</h4>
                <p className="text-xs text-slate-500 font-medium">您可以直接開口詢問物理問題或下達指令</p>
                <div className="flex gap-1 justify-center mt-4">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="w-1 bg-emerald-500 rounded-full animate-wave" style={{ animationDelay: `${i * 100}ms` }} />
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="flex gap-4">
                <div className="w-9 h-9 rounded-full bg-blue-600 flex items-center justify-center shrink-0 shadow-lg shadow-blue-900/30">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                <div className="bg-slate-800 rounded-2xl rounded-tl-none p-4 text-xs leading-relaxed text-slate-300 border border-slate-700 max-w-[85%] shadow-sm">
                  您好，我是您的設計助理。您可以輸入「幫我把通道設為 CH1，陣列設為 8x8，掃描角度 -45 到 45 度」等指令。
                </div>
              </div>
              {chatHistory.map((msg, i) => (
                <div key={i} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 shadow-lg ${msg.role === 'user' ? 'bg-slate-700' : 'bg-blue-600'}`}>
                    {msg.role === 'user' ? <MessageSquare className="w-5 h-5 text-white" /> : <Sparkles className="w-5 h-5 text-white" />}
                  </div>
                  <div className={`rounded-2xl p-4 text-xs leading-relaxed border max-w-[85%] shadow-sm ${
                    msg.role === 'user' 
                    ? 'bg-blue-600/10 border-blue-500/30 text-slate-200 rounded-tr-none' 
                    : 'bg-slate-800 border-slate-700 text-slate-300 rounded-tl-none'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {isThinking && <div className="animate-pulse text-[10px] text-slate-500 uppercase font-black tracking-widest pl-12">Assistant is thinking...</div>}
            </>
          )}
        </div>

        <div className="p-6 bg-slate-900 border-t border-slate-800 space-y-4">
          <div className="flex items-center gap-3 bg-slate-950 p-2.5 rounded-2xl border border-slate-800 shadow-inner group transition-all">
            {isLiveMode ? (
              <button 
                onClick={stopLiveSession}
                className="flex-1 bg-red-600 hover:bg-red-500 text-white py-3 rounded-xl font-black uppercase text-[11px] tracking-widest flex items-center justify-center gap-2 shadow-lg shadow-red-900/40 active:scale-95 transition-all"
              >
                <PhoneOff className="w-4 h-4" /> 結束通話
              </button>
            ) : (
              <>
                <button 
                  onClick={startLiveSession}
                  className="p-3.5 bg-emerald-600/10 border border-emerald-500/30 text-emerald-400 rounded-xl hover:bg-emerald-600 hover:text-white transition-all group active:scale-95"
                  title="啟動即時對話"
                >
                  <PhoneCall className="w-5 h-5" />
                </button>
                <input 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendText()}
                  placeholder="輸入文字指令 (例: 設定 CH1 8x8 陣列)..."
                  className="flex-1 bg-transparent border-none outline-none text-sm font-medium text-slate-200 placeholder:text-slate-700"
                />
                <button 
                  onClick={handleSendText}
                  disabled={isThinking || !input.trim()}
                  className="p-3.5 bg-blue-600 hover:bg-blue-500 text-white rounded-xl shadow-lg shadow-blue-600/30 disabled:opacity-30 disabled:bg-slate-800 transition-all active:scale-95"
                >
                  <Zap className="w-5 h-5 fill-current" />
                </button>
              </>
            )}
          </div>
          <p className="text-[8px] text-slate-700 text-center uppercase font-black tracking-widest">Gemini AI Copilot • State Control & Physics Analysis Enabled</p>
        </div>
      </div>
      <style>{`
        @keyframes wave {
          0%, 100% { height: 4px; }
          50% { height: 16px; }
        }
        .animate-wave { animation: wave 1s ease-in-out infinite; }
      `}</style>
    </div>
  );
};

// --- Main UI Components ---
const CoordinateDiagram = ({ system, scan, selectedBeam }: { system: CoordinateSystem, scan: ScanRange, selectedBeam: any }) => {
  const cx = 75;
  const cy = 60;
  const project = (thetaDeg: number, phiDeg: number, r: number) => {
    const theta = (thetaDeg * Math.PI) / 180;
    const phi = (phiDeg * Math.PI) / 180;
    const px = r * Math.sin(theta) * Math.cos(phi);
    const py = r * Math.sin(theta) * Math.sin(phi);
    const pz = r * Math.cos(theta);
    const x = cx + (px * 0.8) - (py * 0.8);
    const y = cy - (pz * 0.9) + (px * 0.3) + (py * 0.3);
    return { x, y };
  };
  const getArcPath = (r: number, startT: number, startP: number, endT: number, endP: number) => {
    const steps = 20;
    const path = [];
    for (let i = 0; i <= steps; i++) {
      const t = startT + (endT - startT) * (i / steps);
      const p = startP + (endP - startP) * (i / steps);
      const pos = project(t, p, r);
      path.push(`${i === 0 ? 'M' : 'L'} ${pos.x} ${pos.y}`);
    }
    return path.join(' ');
  };
  const zAxisTip = project(0, 0, 50);
  const xAxisTip = project(90, 0, 50);
  const yAxisTip = project(90, 90, 50);
  const currentTheta = selectedBeam?.theta ?? 0;
  const currentPhi = selectedBeam?.phi ?? 0;
  const vectorTip = project(currentTheta, currentPhi, 45);
  const shadowTip = project(90, currentPhi, 45 * Math.sin(currentTheta * Math.PI / 180));
  const thetaArc = getArcPath(20, 0, 0, currentTheta, currentPhi);
  const phiArc = getArcPath(25, 90, 0, 90, currentPhi);
  const wedgePoints = useMemo(() => {
    const pts = [];
    const steps = 10;
    const pStart = scan.azStart;
    const pEnd = scan.azEnd;
    const tStart = system === 'AZ_EL' ? 90 - scan.elEnd : scan.elStart;
    const tEnd = system === 'AZ_EL' ? 90 - scan.elStart : scan.elEnd;
    for (let i = 0; i <= steps; i++) pts.push(project(tStart + (tEnd - tStart) * (i / steps), pStart, 40));
    for (let i = 0; i <= steps; i++) pts.push(project(tEnd, pStart + (pEnd - pStart) * (i / steps), 40));
    for (let i = 0; i <= steps; i++) pts.push(project(tEnd - (tEnd - tStart) * (i / steps), pEnd, 40));
    for (let i = 0; i <= steps; i++) pts.push(project(tStart, pEnd - (pEnd - pStart) * (i / steps), 40));
    return pts.map(p => `${p.x},${p.y}`).join(' ');
  }, [scan, system]);
  return (
    <div className="bg-slate-900/90 rounded-2xl p-4 border border-slate-800 shadow-xl mt-2 overflow-hidden">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-[10px] font-black uppercase text-blue-400 tracking-widest"><Navigation2 className="w-4 h-4" />Orientation: {system}</div>
      </div>
      <div className="flex items-center justify-center bg-slate-950/40 rounded-xl border border-slate-800/50 relative py-2">
        <svg width="220" height="160" viewBox="0 0 150 130" className="drop-shadow-2xl overflow-visible">
          <path d={getArcPath(45, 90, 0, 90, 360)} fill="none" stroke="#1e293b" strokeWidth="0.5" strokeDasharray="2 2" />
          <line x1={cx} y1={cy} x2={zAxisTip.x} y2={zAxisTip.y} stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" />
          <text x={zAxisTip.x} y={zAxisTip.y - 5} fill="#3b82f6" fontSize="5.5" fontWeight="900" textAnchor="middle">Z (Boresight)</text>
          <line x1={cx} y1={cy} x2={xAxisTip.x} y2={xAxisTip.y} stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
          <text x={xAxisTip.x + 8} y={xAxisTip.y + 4} fill="#ef4444" fontSize="5.5" fontWeight="900" textAnchor="start">X (φ=0°)</text>
          <line x1={cx} y1={cy} x2={yAxisTip.x} y2={yAxisTip.y} stroke="#10b981" strokeWidth="1.5" strokeLinecap="round" />
          <text x={yAxisTip.x - 5} y={yAxisTip.y + 4} fill="#10b981" fontSize="5.5" fontWeight="900" textAnchor="end">Y</text>
          <polygon points={wedgePoints} fill="#3b82f6" fillOpacity="0.1" stroke="#3b82f6" strokeWidth="0.5" strokeDasharray="2 2" />
          {selectedBeam && (
            <g className="transition-all duration-300">
              <defs><marker id="beamArrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#fbbf24" /></marker></defs>
              <line x1={cx} y1={cy} x2={shadowTip.x} y2={shadowTip.y} stroke="#facc15" strokeWidth="0.6" strokeDasharray="1 1" opacity="0.4" />
              <line x1={vectorTip.x} y1={vectorTip.y} x2={shadowTip.x} y2={shadowTip.y} stroke="#facc15" strokeWidth="0.6" strokeDasharray="2 2" opacity="0.4" />
              <path d={thetaArc} fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="1 1" />
              <text x={(cx + vectorTip.x)/2 + 8} y={(cy + vectorTip.y)/2 - 12} fill="#f59e0b" fontSize="5" fontWeight="900">θ</text>
              <path d={phiArc} fill="none" stroke="#ec4899" strokeWidth="1" strokeDasharray="1 1" />
              <text x={(xAxisTip.x + shadowTip.x)/2} y={(xAxisTip.y + shadowTip.y)/2 + 8} fill="#ec4899" fontSize="5" fontWeight="900" textAnchor="middle">φ</text>
              <line x1={cx} y1={cy} x2={vectorTip.x} y2={vectorTip.y} stroke="#fbbf24" strokeWidth="2.5" markerEnd="url(#beamArrow)" className="drop-shadow-lg" />
              <g transform={`translate(${vectorTip.x + 5}, ${vectorTip.y - 2})`}><rect x="-1" y="-12" width="45" height="11" rx="2" fill="#0f172a" opacity="0.9" stroke="#334155" strokeWidth="0.5" /><text x="1" y="-4" fill="#fbbf24" fontSize="5" fontWeight="900" fontFamily="monospace">{system === 'AZ_EL' ? `AZ:${selectedBeam.az}°` : `Φ:${selectedBeam.phi}°`}</text></g>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
};

const BeamVisualizer3D = ({ config, targetTheta, targetPhi }: { config: ArrayConfig, targetTheta: number, targetPhi: number }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<{ scene: THREE.Scene, camera: THREE.PerspectiveCamera, renderer: THREE.WebGLRenderer, beamMesh?: THREE.Mesh, arrayGroup?: THREE.Group, controls?: OrbitControls, labels?: THREE.Group } | null>(null);
  useEffect(() => {
    if (!mountRef.current) return;
    const scene = new THREE.Scene(); scene.background = new THREE.Color(0x020617);
    const camera = new THREE.PerspectiveCamera(40, mountRef.current.clientWidth / mountRef.current.clientHeight, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    mountRef.current.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
    camera.position.set(7, -7, 7); camera.up.set(0, 0, 1);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 1.5); dl.position.set(5, -5, 10); scene.add(dl);
    const axesHelper = new THREE.AxesHelper(6); scene.add(axesHelper);
    const labelsGroup = new THREE.Group();
    const xLabel = createTextSprite('X', '#ef4444'); xLabel.position.set(6.5, 0, 0); labelsGroup.add(xLabel);
    const yLabel = createTextSprite('Y', '#10b981'); yLabel.position.set(0, 6.5, 0); labelsGroup.add(yLabel);
    const zLabel = createTextSprite('Z', '#3b82f6'); zLabel.position.set(0, 0, 6.5); labelsGroup.add(zLabel);
    scene.add(labelsGroup);
    sceneRef.current = { scene, camera, renderer, controls, labels: labelsGroup };
    const animate = () => { requestAnimationFrame(animate); if (sceneRef.current) { sceneRef.current.controls?.update(); sceneRef.current.renderer.render(sceneRef.current.scene, sceneRef.current.camera); } };
    animate();
    return () => { renderer.dispose(); mountRef.current?.removeChild(renderer.domElement); };
  }, []);
  useEffect(() => {
    if (!sceneRef.current) return;
    const { scene } = sceneRef.current;
    if (sceneRef.current.arrayGroup) scene.remove(sceneRef.current.arrayGroup);
    const arrayGroup = new THREE.Group();
    const arrayScale = 0.02;
    const substrate = new THREE.Mesh(new THREE.BoxGeometry(config.nx * config.dx_mm * arrayScale + 0.4, config.ny * config.dy_mm * arrayScale + 0.4, 0.1), new THREE.MeshStandardMaterial({ color: 0x064e3b, roughness: 0.2, metalness: 0.8 }));
    substrate.position.z = -0.05; arrayGroup.add(substrate);
    const patchMat = new THREE.MeshStandardMaterial({ color: 0xb45309, metalness: 1, roughness: 0.1 });
    const patchGeom = new THREE.PlaneGeometry(0.12, 0.12);
    for (let y = 0; y < config.ny; y++) {
      for (let x = 0; x < config.nx; x++) {
        const p = new THREE.Mesh(patchGeom, patchMat);
        p.position.set((x - (config.nx-1)/2)*config.dx_mm*arrayScale, (y - (config.ny-1)/2)*config.dy_mm*arrayScale, 0.02);
        arrayGroup.add(p);
      }
    }
    scene.add(arrayGroup); sceneRef.current.arrayGroup = arrayGroup;
    if (sceneRef.current.beamMesh) scene.remove(sceneRef.current.beamMesh);
    const tr = (Math.PI / 180) * targetTheta; const pr = (Math.PI / 180) * targetPhi;
    const targetVec = new THREE.Vector3(Math.sin(tr) * Math.cos(pr), Math.sin(tr) * Math.sin(pr), Math.cos(tr)).normalize();
    const beamRes = 96; const geom = new THREE.SphereGeometry(1, beamRes, beamRes);
    const pos = geom.attributes.position; const colors = new Float32Array(pos.count * 3);
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      const dot = v.dot(targetVec); const gain = Math.max(0.05, Math.pow((dot + 1) / 2, 25)) * 6.0;
      v.normalize().multiplyScalar(gain); pos.setXYZ(i, v.x, v.y, v.z);
      const c = new THREE.Color(); const normalizedGain = (gain - 0.3) / 5.7;
      if (normalizedGain > 0.8) c.setHSL(0, 1, 0.5); else if (normalizedGain > 0.5) c.setHSL(0.1, 1, 0.5); else if (normalizedGain > 0.2) c.setHSL(0.3, 1, 0.4); else c.setHSL(0.6, 1, 0.4);
      colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
    }
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3)); geom.computeVertexNormals();
    const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.85, metalness: 0.3, roughness: 0.7 }));
    scene.add(mesh); sceneRef.current.beamMesh = mesh;
    const arrowHelper = new THREE.ArrowHelper(targetVec, new THREE.Vector3(0,0,0), 7, 0xfacc15, 0.5, 0.3); mesh.add(arrowHelper);
  }, [config, targetTheta, targetPhi]);
  return <div ref={mountRef} className="w-full h-full rounded-2xl overflow-hidden border border-slate-800 bg-slate-950" />;
};

type ColumnGroup = 'Hardware' | 'Theory' | 'Logic';
type SortPriority = 'BEAM' | 'ANTENNA';
type ElementSortKey = 'Element_ID' | 'Pos_X' | 'Pos_Y';

export default function App() {
  const [arrayConfig, setArrayConfig] = useState<ArrayConfig>(INITIAL_ARRAY);
  const [scanRange, setScanRange] = useState<ScanRange>(INITIAL_SCAN);
  const [ics, setIcs] = useState<ICConfig[]>(INITIAL_ICS);
  const [ttdCal, setTtdCal] = useState<CalibrationMap>({});
  const [psCal, setPsCal] = useState<CalibrationMap>({});
  const [isComputing, setIsComputing] = useState(false);
  const [previewData, setPreviewData] = useState<BeamTableRow[]>([]);
  const [showFormulaModal, setShowFormulaModal] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [selectedBeam, setSelectedBeam] = useState<{az: number, el: number, theta: number, phi: number} | null>(null);
  const [show3D, setShow3D] = useState(true);
  const [showColumnDropdown, setShowColumnDropdown] = useState(false);
  const [showSortDropdown, setShowSortDropdown] = useState(false);
  const [visibleGroups, setVisibleGroups] = useState<ColumnGroup[]>(['Hardware', 'Logic']);
  const [useSimplifiedHW, setUseSimplifiedHW] = useState(true);
  const [showAIModal, setShowAIModal] = useState(false);
  
  const [sortPriority, setSortPriority] = useState<SortPriority>('BEAM');
  const [elementSortKey, setElementSortKey] = useState<ElementSortKey>('Element_ID');

  const columnDefinitions = useMemo(() => ({
    Targeting: ['Channel', 'Beam_ID', 'Target_Az', 'Target_El', 'Target_Phi', 'Target_Theta'], 
    Hardware: useSimplifiedHW ? ['Element_ID', 'HW_Addr', 'Pos_X', 'Pos_Y'] : ['Element_ID', 'HW_Chip_ID', 'HW_Port_ID', 'Pos_X', 'Pos_Y'],
    Theory: ['Theory_Delay_ps', 'Theory_Phase_deg'],
    Logic: ['Cal_TTD_ps', 'Cal_PS_deg']
  }), [useSimplifiedHW]);

  const activeColumns = useMemo(() => {
    if (previewData.length === 0) return [];
    let baseCols: string[] = [...columnDefinitions.Targeting];
    visibleGroups.forEach(group => { baseCols = [...baseCols, ...columnDefinitions[group]]; });
    const icCols: string[] = [];
    if (visibleGroups.includes('Logic')) {
      ics.forEach(ic => {
        const suffix = ic.type === 'TTD' ? 'ps' : 'deg';
        // 將 Code_Dec 插入到 Code_Hex 之前
        icCols.push(
          `${ic.name}_Actual_${suffix}`, 
          `${ic.name}_Code_Dec`, 
          `${ic.name}_Code_Hex`, 
          `${ic.name}_Error_${suffix}`
        );
      });
    }
    return [...baseCols, ...icCols];
  }, [previewData, visibleGroups, columnDefinitions, ics]);

  const uniqueBeams = useMemo(() => {
    if (previewData.length === 0) return [];
    const seen = new Set<string>(); const beams: any[] = [];
    previewData.forEach(row => {
      const key = `${row['Target_Az']},${row['Target_El']}`;
      if (!seen.has(key)) { seen.add(key); beams.push({ az: Number(row['Target_Az']), el: Number(row['Target_El']), theta: Number(row['Target_Theta']), phi: Number(row['Target_Phi']) }); }
    });
    return beams;
  }, [previewData]);

  const currentBeamIndex = useMemo(() => !selectedBeam ? -1 : uniqueBeams.findIndex(b => b.az === selectedBeam.az && b.el === selectedBeam.el), [uniqueBeams, selectedBeam]);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().then(() => setIsFullscreen(true));
    else document.exitFullscreen().then(() => setIsFullscreen(false));
  };

  const handleCoordinateSwitch = (system: CoordinateSystem) => {
    if (system === scanRange.system) return;
    setScanRange(prev => ({ ...prev, system, azStart: prev.azStart, azEnd: prev.azEnd, elStart: 90 - prev.elEnd, elEnd: 90 - prev.elStart }));
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>, type: 'TTD' | 'PS') => {
    const file = e.target.files?.[0]; if (!file) return;
    const map = await parseCalibrationFile(file);
    if (type === 'TTD') setTtdCal(map); else setPsCal(map);
  };

  const handleMappingUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return;
    const map = await parseMappingFile(file);
    setArrayConfig(prev => ({ ...prev, topology: { ...prev.topology, customMapping: map, mappingOrder: 'CUSTOM' } }));
  };

  const downloadTemplate = (type: 'TTD_CAL' | 'PS_CAL' | 'MAPPING') => {
    let headers = ''; let rows: string[] = [];
    if (type === 'MAPPING') {
      headers = 'X_Idx,Y_Idx,Chip_ID,Port_ID,Element_ID';
      for (let y = 0; y < arrayConfig.ny; y++) {
        for (let x = 0; x < arrayConfig.nx; x++) {
          let linearIdx = (arrayConfig.topology.mappingOrder === 'COL_MAJOR') ? (x * arrayConfig.ny + y) : (y * arrayConfig.nx + x);
          rows.push(`${x},${y},${Math.floor(linearIdx/arrayConfig.topology.elementsPerChip)},${linearIdx%arrayConfig.topology.elementsPerChip},${linearIdx}`);
        }
      }
    } else {
      headers = `X_Idx,Y_Idx,${type === 'TTD_CAL' ? 'Cal_Delay_ps' : 'Cal_Phase_Deg'}`;
      for (let y = 0; y < arrayConfig.ny; y++) { for (let x = 0; x < arrayConfig.nx; x++) rows.push(`${x},${y},0.0`); }
    }
    const blob = new Blob([[headers, ...rows].join('\n')], { type: 'text/csv' });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `${type}_Template.csv`; link.click();
  };

  const compareElements = (a: BeamTableRow, b: BeamTableRow, key: ElementSortKey) => {
    const valA = Number(a[key]);
    const valB = Number(b[key]);
    if (valA !== valB) return valA - valB;
    if (key === 'Element_ID') return Number(a.Pos_X) - Number(b.Pos_X) || Number(a.Pos_Y) - Number(b.Pos_Y);
    return Number(a.Element_ID) - Number(b.Element_ID);
  };

  const sortData = useCallback((data: BeamTableRow[], priority: SortPriority, eKey: ElementSortKey) => {
    return [...data].sort((a, b) => {
      if (priority === 'BEAM') {
        if (Number(a.Target_Az) !== Number(b.Target_Az)) return Number(a.Target_Az) - Number(b.Target_Az);
        if (Number(a.Target_El) !== Number(b.Target_El)) return Number(a.Target_El) - Number(b.Target_El);
        return compareElements(a, b, eKey);
      } else {
        const eRes = compareElements(a, b, eKey);
        if (eRes !== 0) return eRes;
        if (Number(a.Target_Az) !== Number(b.Target_Az)) return Number(a.Target_Az) - Number(b.Target_Az);
        return Number(a.Target_El) - Number(b.Target_El);
      }
    });
  }, []);

  const handleCompute = useCallback(() => {
    setIsComputing(true);
    setTimeout(() => {
      try {
        const results = calculateBeamTable(arrayConfig, scanRange, ics, ttdCal, psCal);
        const enhancedResults: BeamTableRow[] = results.map(row => ({
          ...row,
          HW_Addr: `[E${String(row['Element_ID']).padStart(2, '0')}] C${row['HW_Chip_ID']}:P${row['HW_Port_ID']}`
        }));
        
        const sorted = sortData(enhancedResults, sortPriority, elementSortKey);

        setPreviewData(sorted);
        if (sorted.length > 0) {
          const first = sorted[0];
          setSelectedBeam({ az: Number(first['Target_Az']), el: Number(first['Target_El']), theta: Number(first['Target_Theta']), phi: Number(first['Target_Phi']) });
        }
      } catch (err) { alert("Calculation Error."); } finally { setIsComputing(false); }
    }, 10);
  }, [arrayConfig, scanRange, ics, ttdCal, psCal, sortPriority, elementSortKey, sortData]);

  useEffect(() => {
    if (previewData.length > 0) {
      setPreviewData(prev => sortData(prev, sortPriority, elementSortKey));
    }
  }, [sortPriority, elementSortKey, sortData]);

  const handleExport = () => {
    if (previewData.length === 0) return;
    const csvContent = [activeColumns.join(','), ...previewData.map(row => activeColumns.map(h => row[h]).join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `BeamTable_Export.csv`; link.click();
  };

  const updateIc = (id: string, updates: Partial<ICConfig>) => {
    setIcs(prev => prev.map(ic => {
      if (ic.id !== id) return ic;
      const next = { ...ic, ...updates };
      if (updates.bits !== undefined || updates.max !== undefined) {
        if (next.bits && next.bits > 0) {
          next.lsb = next.type === 'TTD' ? next.max / (Math.pow(2, next.bits) - 1) : next.max / Math.pow(2, next.bits);
        }
      }
      return next;
    }));
  };

  // --- AI Update Function ---
  const handleAICopilotUpdate = (configUpdates: Partial<ArrayConfig>, scanUpdates: Partial<ScanRange>) => {
    if (Object.keys(configUpdates).length > 0) setArrayConfig(prev => ({ ...prev, ...configUpdates }));
    if (Object.keys(scanUpdates).length > 0) setScanRange(prev => ({ ...prev, ...scanUpdates }));
    setTimeout(handleCompute, 100);
  };

  const renderTopologyGrid = () => {
    const { nx, ny, topology } = arrayConfig;
    const grid = [];
    for (let y = 0; y < ny; y++) {
      const row = [];
      for (let x = 0; x < nx; x++) {
        let chipId = 0; let portId = 0; let elementId = 0;
        const linearIdx = (topology.mappingOrder === 'COL_MAJOR') ? (x * ny + y) : (y * nx + x);
        elementId = linearIdx;
        if (topology.mappingOrder === 'CUSTOM' && topology.customMapping?.[`${x},${y}`]) {
          const custom = topology.customMapping[`${x},${y}`];
          chipId = custom.chipId; portId = custom.portId;
          if (custom.elementId !== undefined) elementId = custom.elementId;
        } else {
          chipId = Math.floor(linearIdx / topology.elementsPerChip);
          portId = linearIdx % topology.elementsPerChip;
        }
        const hwAddrTooltip = `[E${String(elementId).padStart(2, '0')}] Chip:${chipId} Port:${portId} (X:${x}, Y:${y})`;
        const color = CHART_COLORS[chipId % CHART_COLORS.length];
        row.push(
          <div key={`${x}-${y}`} className="w-4 h-4 rounded-sm border border-slate-700/50 relative group cursor-help transition-transform hover:scale-110 z-0 hover:z-10" style={{ backgroundColor: `${color}40`, borderColor: `${color}80` }} title={hwAddrTooltip}>
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 bg-white/20 z-10 transition-opacity rounded-sm shadow-sm" />
          </div>
        );
      }
      grid.push(<div key={y} className="flex gap-0.5">{row}</div>);
    }
    return <div className="flex flex-col gap-0.5 bg-slate-950 p-2 rounded-xl border border-slate-800 shadow-inner overflow-auto max-h-40">{grid}</div>;
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-100 selection:bg-blue-500/30">
      <header className="bg-slate-900 border-b border-slate-800 px-8 py-4 flex items-center justify-between shadow-2xl z-40 shrink-0">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-indigo-700 p-2.5 rounded-2xl shadow-lg shadow-blue-500/20"><Layers className="text-white w-6 h-6" /></div>
          <div>
            <h1 className="text-xl font-black tracking-tight flex items-center gap-2 uppercase">Beam Master <span className="text-blue-500 text-[10px] bg-blue-500/10 px-2 py-0.5 rounded-full">Pro</span></h1>
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest opacity-60">Hybrid Phase & Delay Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={() => setShowAIModal(true)} className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-blue-500/40 bg-blue-500/10 text-blue-400 hover:bg-blue-600 hover:text-white transition-all group active:scale-95">
            <Sparkles className="w-4 h-4 group-hover:animate-pulse" /><span className="text-[11px] font-black uppercase tracking-widest">AI Copilot</span>
          </button>
          <div className="w-px h-8 bg-slate-800 mx-1" />
          <div className="relative">
            <button onClick={() => { setShowSortDropdown(!showSortDropdown); setShowColumnDropdown(false); }} className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border transition-all text-[11px] font-black uppercase tracking-widest ${showSortDropdown ? 'bg-slate-700 border-blue-500 text-blue-400' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
              <SortAsc className="w-4 h-4" /> Element Order <ChevronDown className={`w-3 h-3 transition-transform ${showSortDropdown ? 'rotate-180' : ''}`} />
            </button>
            {showSortDropdown && (
              <div className="absolute left-0 mt-2 w-56 bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-50 overflow-hidden py-2 animate-in fade-in slide-in-from-top-2">
                <div className="px-4 py-2 text-[9px] font-black text-slate-500 uppercase tracking-widest border-b border-slate-800 mb-2">Priority Criteria</div>
                {(['Element_ID', 'Pos_X', 'Pos_Y'] as ElementSortKey[]).map(key => (
                  <button key={key} onClick={() => { setElementSortKey(key); setShowSortDropdown(false); }} className={`w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800 transition-colors text-left ${elementSortKey === key ? 'text-blue-400' : 'text-slate-500'}`}><span className="text-[11px] font-bold">{key.replace('_', ' ')}</span>{elementSortKey === key && <Check className="w-4 h-4" />}</button>
                ))}
                <div className="h-px bg-slate-800 my-2 mx-4" /><div className="px-4 py-1.5 flex gap-1 bg-slate-950/40 rounded-lg mx-2 border border-slate-800/50"><button onClick={() => setSortPriority('BEAM')} className={`flex-1 py-1.5 rounded-md text-[9px] font-black uppercase ${sortPriority === 'BEAM' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-600'}`}>Beam</button><button onClick={() => setSortPriority('ANTENNA')} className={`flex-1 py-1.5 rounded-md text-[9px] font-black uppercase ${sortPriority === 'ANTENNA' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-600'}`}>Element</button></div>
              </div>
            )}
          </div>
          <div className="relative">
            <button onClick={() => { setShowColumnDropdown(!showColumnDropdown); setShowSortDropdown(false); }} className={`flex items-center gap-2 px-4 py-2.5 rounded-xl border transition-all text-[11px] font-black uppercase tracking-widest ${showColumnDropdown ? 'bg-slate-700 border-blue-500 text-blue-400' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
              <ListFilter className="w-4 h-4" /> Views <ChevronDown className={`w-3 h-3 transition-transform ${showColumnDropdown ? 'rotate-180' : ''}`} />
            </button>
            {showColumnDropdown && (
              <div className="absolute right-0 mt-2 w-64 bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-50 overflow-hidden py-2 animate-in fade-in slide-in-from-top-2">
                <div className="px-4 py-2 text-[9px] font-black text-slate-500 uppercase tracking-widest border-b border-slate-800 mb-2 flex justify-between">Optional Groups <span className="text-blue-500/80 lowercase italic font-normal">(Targeting Locked)</span></div>
                {(['Hardware', 'Theory', 'Logic'] as ColumnGroup[]).map(group => (
                  <button key={group} onClick={() => setVisibleGroups(prev => prev.includes(group) ? prev.filter(g => g !== group) : [...prev, group])} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800 transition-colors text-left"><span className={`text-[11px] font-bold ${visibleGroups.includes(group) ? 'text-white' : 'text-slate-500'}`}>{group}</span>{visibleGroups.includes(group) ? <Eye className="w-4 h-4 text-blue-500" /> : <EyeOff className="w-4 h-4 text-slate-700" />}</button>
                ))}
                <div className="h-px bg-slate-800 my-2 mx-4" /><button onClick={() => setUseSimplifiedHW(!useSimplifiedHW)} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800 transition-colors text-left"><span className={`text-[11px] font-bold ${useSimplifiedHW ? 'text-emerald-400' : 'text-slate-500'}`}>Simplified HW Format</span>{useSimplifiedHW && <Check className="w-4 h-4 text-emerald-500" />}</button>
              </div>
            )}
          </div>
          <div className="w-px h-8 bg-slate-800 mx-2" />
          <button onClick={() => setShowFormulaModal(true)} className="p-2.5 rounded-xl bg-slate-800 text-slate-400 border border-slate-700 hover:text-white transition-all"><Info className="w-5 h-5" /></button>
          <button onClick={() => setShow3D(!show3D)} className={`p-2.5 rounded-xl border transition-all ${show3D ? 'bg-blue-600/20 text-blue-400 border-blue-500/50' : 'bg-slate-800 text-slate-400 border-slate-700'}`}><Box className="w-5 h-5" /></button>
          <button onClick={toggleFullscreen} className="p-2.5 rounded-xl bg-slate-800 text-slate-400 border border-slate-700"><Maximize className="w-5 h-5" /></button>
          <div className="w-px h-8 bg-slate-800 mx-2" />
          <button onClick={handleCompute} disabled={isComputing} className="flex items-center gap-2 px-8 py-2.5 rounded-xl font-black uppercase text-[12px] bg-blue-600 hover:bg-blue-500 shadow-lg shadow-blue-900/40 active:scale-95 disabled:opacity-20"><Zap className="w-4 h-4 fill-current" /> {isComputing ? 'Computing' : 'Generate'}</button>
          <button onClick={handleExport} disabled={previewData.length === 0} className="flex items-center gap-2 px-8 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 shadow-lg active:scale-95 disabled:opacity-20"><Download className="w-4 h-4" /> Export</button>
        </div>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <aside className="w-80 bg-slate-900/50 border-r border-slate-800 flex flex-col overflow-y-auto p-5 gap-8 shrink-0 scrollbar-hide">
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-slate-400 border-b border-slate-800 pb-2"><Settings2 className="w-4 h-4" /><h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-200">Array Setup</h2></div>
            <div className="space-y-4 bg-slate-800/20 p-4 rounded-2xl border border-slate-800/50">
              <div className="space-y-1.5 flex-1">
                <label className="text-[9px] font-black text-slate-500 uppercase block tracking-widest px-0.5">Channel Name</label>
                <input 
                  type="text" 
                  className="w-full bg-slate-950 border border-slate-700 rounded-xl py-2 px-3 text-[12px] text-slate-100 font-mono focus:ring-2 focus:ring-blue-500/50 outline-none transition-all" 
                  value={arrayConfig.channelName} 
                  onChange={(e) => setArrayConfig({...arrayConfig, channelName: e.target.value})} 
                  spellCheck={false} 
                />
              </div>
              <InputGroup label="Freq (GHz)" value={arrayConfig.frequencyGHz} onChange={v => setArrayConfig({...arrayConfig, frequencyGHz: v})} tiny compact />
              <div className="flex flex-col gap-2"><label className="text-[9px] font-black text-slate-500 uppercase tracking-widest px-0.5">Steering Mode</label><div className="grid grid-cols-3 gap-1 bg-slate-900 p-1 rounded-xl">{(['HYBRID', 'PS_ONLY', 'TTD_ONLY'] as BeamMode[]).map(m => (<button key={m} onClick={() => setArrayConfig({...arrayConfig, mode: m})} className={`py-1.5 text-[8px] font-black rounded-lg transition-all ${arrayConfig.mode === m ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}>{m.split('_')[0]}</button>))}</div></div>
              <div className="grid grid-cols-2 gap-3"><InputGroup label="Nx" value={arrayConfig.nx} onChange={v => setArrayConfig({...arrayConfig, nx: v})} tiny compact /><InputGroup label="Ny" value={arrayConfig.ny} onChange={v => setArrayConfig({...arrayConfig, ny: v})} tiny compact /></div>
              <div className="grid grid-cols-2 gap-3"><InputGroup label="Dx (mm)" value={arrayConfig.dx_mm} onChange={v => setArrayConfig({...arrayConfig, dx_mm: v})} tiny compact /><InputGroup label="Dy (mm)" value={arrayConfig.dy_mm} onChange={v => setArrayConfig({...arrayConfig, dy_mm: v})} tiny compact /></div>
            </div>
          </section>
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-emerald-500/80 border-b border-slate-800 pb-2"><RefreshCw className="w-4 h-4" /><h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-200">Calibration</h2></div>
            <div className="space-y-3 bg-slate-800/20 p-4 rounded-2xl border border-slate-800/50">
              <div className="space-y-2">
                <div className="flex items-center justify-between"><label className="text-[9px] font-black text-slate-500 uppercase">TTD Cal Map</label><button onClick={() => downloadTemplate('TTD_CAL')} className="text-[8px] text-blue-400 hover:underline flex items-center gap-1 uppercase"><FileDown className="w-2.5 h-2.5" /> Template</button></div>
                <label className="flex items-center gap-2 w-full cursor-pointer bg-slate-900 border border-slate-700 p-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-white transition-all"><Upload className="w-3.5 h-3.5" /> <span>Upload TTD Map</span><input type="file" className="hidden" accept=".csv" onChange={e => handleFileUpload(e, 'TTD')} /></label>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between"><label className="text-[9px] font-black text-slate-500 uppercase">PS Cal Map</label><button onClick={() => downloadTemplate('PS_CAL')} className="text-[8px] text-emerald-400 hover:underline flex items-center gap-1 uppercase"><FileDown className="w-2.5 h-2.5" /> Template</button></div>
                <label className="flex items-center gap-2 w-full cursor-pointer bg-slate-900 border border-slate-700 p-2.5 rounded-xl text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-white transition-all"><Upload className="w-3.5 h-3.5" /> <span>Upload PS Map</span><input type="file" className="hidden" accept=".csv" onChange={e => handleFileUpload(e, 'PS')} /></label>
              </div>
            </div>
          </section>
          <section className="space-y-4">
            <div className="flex items-center gap-2 text-amber-500/80 border-b border-slate-800 pb-2"><Boxes className="w-4 h-4" /><h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-200">Topology</h2></div>
            <div className="space-y-4 bg-slate-800/20 p-4 rounded-2xl border border-slate-800/50">
               <InputGroup label="Elements / IC" value={arrayConfig.topology.elementsPerChip} onChange={v => setArrayConfig({...arrayConfig, topology: {...arrayConfig.topology, elementsPerChip: v}})} tiny compact />
               <div className="flex flex-col gap-2"><label className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Mapping Order</label><div className="grid grid-cols-2 gap-1 bg-slate-900 p-1 rounded-xl">{(['ROW_MAJOR', 'COL_MAJOR'] as MappingOrder[]).map(m => (<button key={m} onClick={() => setArrayConfig({...arrayConfig, topology: {...arrayConfig.topology, mappingOrder: m}})} className={`py-1.5 text-[9px] font-bold rounded-lg transition-all ${arrayConfig.topology.mappingOrder === m ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-500 hover:text-slate-300'}`}>{m.replace('_', ' ')}</button>))}</div></div>
               {renderTopologyGrid()}
               <div className="flex gap-2"><button onClick={() => downloadTemplate('MAPPING')} className="p-2.5 bg-slate-800 border border-slate-700 rounded-xl hover:bg-slate-700 transition-colors"><FileDown className="w-4 h-4" /></button><label className="flex-1 cursor-pointer bg-slate-800 border border-slate-700 p-2.5 rounded-xl text-center text-[10px] font-black uppercase tracking-widest text-slate-400 hover:text-white transition-all"><input type="file" className="hidden" accept=".csv" onChange={handleMappingUpload} /> Upload Map</label></div>
            </div>
          </section>
          <section className="space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2"><div className="flex items-center gap-2 text-slate-400"><Target className="w-4 h-4 text-blue-500" /><h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-200">Scan Range</h2></div><button onClick={() => handleCoordinateSwitch(scanRange.system === 'AZ_EL' ? 'THETA_PHI' : 'AZ_EL')} className="text-[9px] bg-slate-800 p-1.5 rounded-lg border border-slate-700 text-blue-400 flex items-center gap-2 uppercase font-black"><RotateCw className="w-3 h-3" /> {scanRange.system === 'AZ_EL' ? 'AZ/EL' : 'Φ/Θ'}</button></div>
            <div className="space-y-3">
              <div className="p-4 bg-slate-800/20 rounded-2xl border border-slate-800/50">
                <label className="text-[10px] font-black text-slate-500 uppercase block mb-2">{scanRange.system === 'AZ_EL' ? 'Azimuth' : 'Phi (Φ)'}</label>
                <div className="grid grid-cols-3 gap-2"><InputGroup label="Start" value={scanRange.azStart} onChange={v => setScanRange({...scanRange, azStart: v})} tiny compact /><InputGroup label="End" value={scanRange.azEnd} onChange={v => setScanRange({...scanRange, azEnd: v})} tiny compact /><InputGroup label="Step" value={scanRange.azStep} onChange={v => setScanRange({...scanRange, azStep: v})} tiny compact /></div>
              </div>
              <div className="p-4 bg-slate-800/20 rounded-2xl border border-slate-800/50">
                <label className="text-[10px] font-black text-slate-500 uppercase block mb-2">{scanRange.system === 'AZ_EL' ? 'Elevation' : 'Theta (Θ)'}</label>
                <div className="grid grid-cols-3 gap-2"><InputGroup label="Start" value={scanRange.elStart} onChange={v => setScanRange({...scanRange, elStart: v})} tiny compact /><InputGroup label="End" value={scanRange.elEnd} onChange={v => setScanRange({...scanRange, elEnd: v})} tiny compact /><InputGroup label="Step" value={scanRange.elStep} onChange={v => setScanRange({...scanRange, elStep: v})} tiny compact /></div>
              </div>
            </div>
            <CoordinateDiagram system={scanRange.system} scan={scanRange} selectedBeam={selectedBeam} />
          </section>
        </aside>
        <main className="flex-1 flex flex-col overflow-hidden bg-slate-950">
          {show3D && (
            <div className="p-6 border-b border-slate-800 bg-slate-900/40 shrink-0">
              <div className="flex flex-col lg:flex-row gap-6 h-[420px]">
                <div className="flex-1 relative rounded-3xl border border-slate-800 overflow-hidden bg-slate-950 shadow-2xl">
                  {selectedBeam ? <BeamVisualizer3D config={arrayConfig} targetTheta={selectedBeam.theta} targetPhi={selectedBeam.phi} /> : <div className="w-full h-full flex flex-col items-center justify-center opacity-10"><Box className="w-16 h-16 mb-4" /><p className="font-black text-sm uppercase tracking-[0.3em]">No Simulation Active</p></div>}
                  {uniqueBeams.length > 1 && (
                    <div className="absolute top-4 left-4 w-60 bg-slate-900/80 backdrop-blur-md border border-slate-800/50 p-4 rounded-2xl shadow-2xl z-20">
                      <div className="flex justify-between items-center mb-2"><span className="text-[9px] font-black text-blue-400 uppercase tracking-widest">Active Sweep</span><span className="text-[9px] font-mono text-slate-500">{currentBeamIndex + 1} / {uniqueBeams.length}</span></div>
                      <input type="range" min="0" max={uniqueBeams.length - 1} value={currentBeamIndex} onChange={(e) => setSelectedBeam(uniqueBeams[parseInt(e.target.value)])} className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 mb-2 transition-all" />
                      <div className="flex justify-between items-center text-[10px] font-mono font-bold text-slate-300 bg-slate-950/50 px-2 py-1 rounded-lg border border-slate-800/50"><span>AZ: {selectedBeam?.az}°</span><span>EL: {selectedBeam?.el}°</span></div>
                    </div>
                  )}
                </div>
                <div className="w-80 bg-slate-900/60 border border-slate-800 rounded-3xl p-5 flex flex-col shadow-xl overflow-hidden">
                   <div className="flex items-center gap-2 mb-4 text-slate-500 border-b border-slate-800 pb-3"><LayoutGrid className="w-4 h-4 text-blue-500" /><span className="text-[11px] font-black uppercase tracking-widest text-slate-300">HW IC Logic</span></div>
                   <div className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar">
                    {ics.map(ic => (
                      <div key={ic.id} className="bg-slate-800/40 p-4 rounded-2xl border border-slate-700 relative group transition-all">
                        <button onClick={() => setIcs(ics.filter(i => i.id !== ic.id))} className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-red-500/20 rounded-lg"><Trash2 className="w-3.5 h-3.5 text-red-500" /></button>
                        <div className="space-y-3">
                          <div className="flex items-center gap-2"><input value={ic.name} onChange={(e) => updateIc(ic.id, { name: e.target.value.replace(/\s+/g, '_') })} className="bg-transparent border-b border-slate-700 hover:border-blue-500 focus:border-blue-500 outline-none text-[11px] font-black text-slate-200 uppercase w-24 transition-colors" placeholder="IC NAME" spellCheck={false} /><div className="flex bg-slate-950/80 p-0.5 rounded-lg border border-slate-700 ml-auto">{(['TTD', 'PS'] as ICType[]).map(type => (<button key={type} onClick={() => updateIc(ic.id, { type })} className={`px-2 py-0.5 text-[8px] font-black rounded-md transition-all ${ic.type === type ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:text-slate-300'}`}>{type}</button>))}</div></div>
                          <div className="grid grid-cols-2 gap-3"><InputGroup label="Bits" value={ic.bits || 0} onChange={v => updateIc(ic.id, { bits: v })} tiny compact /><InputGroup label={ic.type === 'TTD' ? 'Max (ps)' : 'Max (deg)'} value={ic.max} onChange={v => updateIc(ic.id, { max: v })} tiny compact /></div>
                          <div className="flex justify-between items-center bg-slate-900/50 p-2 rounded-xl border border-slate-800/50"><span className="text-[8px] font-black text-slate-500 uppercase tracking-widest">Calculated LSB</span><span className="text-[10px] font-mono text-blue-400 font-bold">{ic.lsb.toFixed(3)} {ic.type === 'TTD' ? 'ps' : '°'}</span></div>
                        </div>
                      </div>
                    ))}
                    <button onClick={() => setIcs([...ics, { id: Date.now().toString(), name: "NEW_STAGE_"+(ics.length+1), type: "PS", lsb: 5.625, max: 360, bits: 6, offset: 0, polarity: 'NORMAL' }])} className="w-full p-3 border border-dashed border-slate-700 rounded-2xl text-slate-500 hover:text-white transition-all text-[11px] uppercase font-black flex items-center justify-center gap-2 mt-2"><Plus className="w-4 h-4" /> Add Stage</button>
                   </div>
                </div>
              </div>
            </div>
          )}
          <div className="flex-1 overflow-auto bg-slate-950 scrollbar-thin scrollbar-thumb-slate-800">
            {previewData.length > 0 ? (
              <div className="inline-block min-w-full align-middle">
                <table className="min-w-full text-left text-xs border-collapse">
                  <thead className="sticky top-0 bg-slate-900 shadow-2xl z-20 border-b border-slate-800"><tr>{activeColumns.map(h => <th key={h} className="px-5 py-4 font-black text-slate-500 uppercase text-[9px] tracking-[0.15em] whitespace-nowrap bg-slate-900">{h.replace(/_/g, ' ')}</th>)}</tr></thead>
                  <tbody className="divide-y divide-slate-800/30">
                    {previewData.map((row, i) => (
                      <tr key={i} onClick={() => setSelectedBeam({ az: Number(row['Target_Az']), el: Number(row['Target_El']), theta: Number(row['Target_Theta']), phi: Number(row['Target_Phi'])})} className={`cursor-pointer transition-all ${selectedBeam?.az === Number(row['Target_Az']) && selectedBeam?.el === Number(row['Target_El']) ? 'bg-blue-600/15 border-l-2 border-l-blue-500' : 'hover:bg-slate-800/40'}`}>
                        {activeColumns.map((col, j) => {
                          const val = row[col]; if (col === 'HW_Addr') return <td key={j} className="px-5 py-2.5 font-mono text-[10px] whitespace-nowrap"><span className="text-emerald-400 font-bold">{val}</span></td>;
                          return <td key={j} className="px-5 py-2.5 font-mono text-[10px] whitespace-nowrap text-slate-400">{val}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="h-full flex flex-col items-center justify-center opacity-10"><TableIcon className="w-20 h-20 mb-6" /><p className="font-black uppercase tracking-[0.3em] text-lg">Engine Ready</p></div>}
          </div>
        </main>
      </div>
      {showFormulaModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md" onClick={() => setShowFormulaModal(false)}></div>
          <div className="bg-slate-900 border border-slate-700 w-full max-w-2xl rounded-3xl shadow-2xl relative z-10 overflow-hidden flex flex-col">
            <div className="p-6 border-b border-slate-800 flex items-center justify-between"><h2 className="text-lg font-black text-white uppercase tracking-widest">Physics Logic</h2><button onClick={() => setShowFormulaModal(false)} className="p-2 hover:bg-slate-800 rounded-xl transition-colors"><X className="w-6 h-6" /></button></div>
            <div className="p-8 space-y-6 text-sm text-slate-400"><p>1. <strong className="text-blue-400">Path Delay:</strong> Δd = x·sinθ·cosφ + y·sinθ·sinφ.</p><p>2. <strong className="text-amber-400">Hybrid Steering:</strong> TTD (ps) coarse, PS (deg) fine.</p><p>3. <strong className="text-emerald-400">Visualization:</strong> Radiation Pattern scaled for lobe analysis.</p></div>
          </div>
        </div>
      )}
      <AIAssistantModal isOpen={showAIModal} onClose={() => setShowAIModal(false)} currentConfig={arrayConfig} currentScan={scanRange} onUpdateConfig={handleAICopilotUpdate} />
    </div>
  );
}

function InputGroup({ label, value, onChange, compact = false, tiny = false }: { label: string, value: number, onChange: (v: number) => void, compact?: boolean, tiny?: boolean }) {
  const [localValue, setLocalValue] = useState(value.toString());
  useEffect(() => { if (parseFloat(localValue) !== value) setLocalValue(value.toString()); }, [value]);
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value; setLocalValue(v); if (v === '' || v === '-' || v === '.') return;
    const parsed = parseFloat(v); if (!isNaN(parsed)) onChange(parsed);
  };
  return (<div className="space-y-1.5 flex-1"><label className={`${tiny ? 'text-[8px]' : 'text-[9px]'} font-black text-slate-500 uppercase block tracking-widest px-0.5`}>{label}</label><input type="text" className={`w-full bg-slate-950 border border-slate-700 rounded-xl ${compact ? 'py-2 px-3' : 'p-3'} text-[12px] text-slate-100 font-mono focus:ring-2 focus:ring-blue-500/50 outline-none transition-all`} value={localValue} onChange={handleChange} spellCheck={false} /></div>);
}
