'use client';

import React, { useState, ChangeEvent, useEffect, useRef } from 'react';
import { Upload, Play, BarChart2, Box, Info, Loader2, Download, Zap, Activity } from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { saveAs } from 'file-saver';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import Link from 'next/link'

// 骨格結線の定義 (FastAPI側と完全に一貫)
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32]
];

// 3D空間における3点（関節）のなす角（度: 0〜180）をベクター計算する関数
function calculate3DAngle(
  p1: { x: number; y: number; z: number },
  p2: { x: number; y: number; z: number },
  p3: { x: number; y: number; z: number }
): number {
  const v1 = new THREE.Vector3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
  const v2 = new THREE.Vector3(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z);

  const angleRad = v1.angleTo(v2);
  const angleDeg = (angleRad * 180) / Math.PI;

  return Math.round(angleDeg * 10) / 10;
}

export default function Home() {
  // 状態管理（State）
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  
  // AI解析データ格納用
  const [motionData, setMotionData] = useState<any>(null);

  // 📊 タブ切り替え状態 ('velocity' | 'angle')
  const [activeTab, setActiveTab] = useState<'velocity' | 'angle'>('velocity');

  // 🔥 グラフ表示用の時系列データ
  const [velocityTimeSeries, setVelocityTimeSeries] = useState<any[]>([]);
  const [angleTimeSeries, setAngleTimeSeries] = useState<any[]>([]);

  // 🎯 現在再生中のフレームインデックス (1-based)
  const [currentFrame, setCurrentFrame] = useState<number>(1);

  // Three.js 用のDOM参照とアニメーション管理
  const canvasRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const jointsRef = useRef<{ [key: number]: THREE.Mesh }>({});
  const linesRef = useRef<THREE.Line[]>([]);
  const lineHeadRef = useRef<THREE.Line | null>(null);
  const lineRacketRef = useRef<THREE.Line | null>(null);

  // 💡 motionData が更新されたら、速度（Kinematic Chain）および角度の時系列データを生成
  useEffect(() => {
    if (!motionData || !motionData.frames) {
      setVelocityTimeSeries([]);
      setAngleTimeSeries([]);
      setCurrentFrame(1);
      return;
    }

    const vSeries: any[] = [];
    const aSeries: any[] = [];

    motionData.frames.forEach((frame: any, idx: number) => {
      const kpMap: { [key: number]: { x: number; y: number; z: number; velocity?: number; vis?: number } } = {};

      if (frame.skeletons && frame.skeletons[0]) {
        frame.skeletons[0].keypoints.forEach((kp: any) => {
          kpMap[kp.id] = {
            ...kp.rel_coord,
            velocity: kp.velocity ?? 0,
            vis: kp.visibility ?? 0
          };
        });
      }

      // --- 1. 速度データ ( Kinematic Chain: 腰 -> 肩 -> 肘 -> 手首 -> ラケット ) ---
      const roundV = (val: number | undefined) => (val !== undefined ? Math.round(val * 100) / 100 : 0);

      vSeries.push({
        frame: idx + 1,
        // ID: 99 (ラケット)。未検出フレーム (visibility=0) は手首位置の代用値なので描画しない
        racketHead: kpMap[99] && (kpMap[99].vis ?? 0) > 0 ? roundV(kpMap[99].velocity) : null,
        wrist: roundV(kpMap[16]?.velocity),       // ID: 16 (右手首)
        elbow: roundV(kpMap[14]?.velocity),       // ID: 14 (右肘)
        shoulder: roundV(kpMap[12]?.velocity),    // ID: 12 (右肩)
        hip: roundV(kpMap[24]?.velocity),         // ID: 24 (右腰)
      });

      // --- 2. 角度データ (右肘・右膝) ---
      let rightElbowAngle = null;
      if (kpMap[12] && kpMap[14] && kpMap[16]) {
        rightElbowAngle = calculate3DAngle(kpMap[12], kpMap[14], kpMap[16]);
      }

      let rightKneeAngle = null;
      if (kpMap[24] && kpMap[26] && kpMap[28]) {
        rightKneeAngle = calculate3DAngle(kpMap[24], kpMap[26], kpMap[28]);
      }

      aSeries.push({
        frame: idx + 1,
        rightElbow: rightElbowAngle,
        rightKnee: rightKneeAngle,
      });
    });

    setVelocityTimeSeries(vSeries);
    setAngleTimeSeries(aSeries);
  }, [motionData]);

  // 解析JSONファイルのダウンロード処理
  const handleDownloadJSON = () => {
    if (!motionData) return;
    
    const exportData = {
      ...motionData,
      analytics: {
        velocityTimeSeries,
        angleTimeSeries,
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json;charset=utf-8',
    });
    saveAs(blob, `tennis_motion_analysis_${Date.now()}.json`);
  };

  // ファイルが選択されたときの処理
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      setVideoUrl(URL.createObjectURL(file));
      setMotionData(null);
    }
  };

  // Cloud Run 解析ノードへ動画を送信する処理
  const handleUpload = async () => {
    if (!videoFile) return;
    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const baseUrl = (process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
      
      const response = await fetch(`${baseUrl}/tennis_motion/analysis_node/request`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`サーバーとの通信に失敗しました。Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Python解析ノードからの受信データ:', data);
      
      setMotionData(data.payload);
      alert(`解析成功！\n総フレーム数: ${data.summary.total_frames}\n推定最高スイング速度: ${data.summary.swing_speed_kmh} km/h`);

    } catch (error) {
      console.error('エラー内容:', error);
      alert('動画の送信中にエラーが発生しました。ネットワーク接続やCloud Runの設定を確認してください。');
    } finally {
      setIsAnalyzing(false);
    }
  };
  
  // Three.js 3D シミュレーター初期化 ＆ アニメーションループ
  useEffect(() => {
    if (!canvasRef.current) return;

    const width = canvasRef.current.clientWidth;
    const height = canvasRef.current.clientHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50);
    camera.position.set(0, 1.5, 5.0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    canvasRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 1.0, 0);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 4, 3);
    scene.add(dirLight);

    const gridHelper = new THREE.GridHelper(4, 20, 0x334155, 0x1e293b);
    scene.add(gridHelper);

    const jointGeometry = new THREE.SphereGeometry(0.03, 16, 16);
    const requiredIds = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 99, 100];
    const joints: { [key: number]: THREE.Mesh } = {};

    requiredIds.forEach((id) => {
      let color = 0x10b981;
      if (id === 100) color = 0xf97316;
      if (id === 99) color = 0x3b82f6;

      const mat = new THREE.MeshStandardMaterial({ color: color, roughness: 0.4 });
      const mesh = new THREE.Mesh(jointGeometry, mat);
      mesh.visible = false;
      scene.add(mesh);
      joints[id] = mesh;
    });
    jointsRef.current = joints;

    let currentFrameIdx = 0;
    let lastTime = performance.now();

    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      if (motionData && motionData.frames && motionData.frames.length > 0) {
        const fps = motionData.metadata.fps || 30;
        const now = performance.now();
        
        if (now - lastTime >= 1000 / fps) {
          setCurrentFrame(currentFrameIdx + 1);

          const frameData = motionData.frames[currentFrameIdx];
          
          if (frameData && frameData.skeletons && frameData.skeletons.length > 0) {
            const keypoints = frameData.skeletons[0].keypoints;
            const kpMap: { [key: number]: { x: number; y: number; z: number; vis: number } } = {};

            keypoints.forEach((kp: any) => {
              const id = kp.id;
              const { x, y, z } = kp.rel_coord;
              kpMap[id] = { x, y, z, vis: kp.visibility };

              if (joints[id]) {
                joints[id].position.set(x, y, z);
                joints[id].visible = id === 99 ? kp.visibility > 0 : true;
              }
            });

            linesRef.current.forEach((l) => scene.remove(l));
            linesRef.current = [];

            const lineMaterial = new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 });

            POSE_CONNECTIONS.forEach(([start, end]) => {
              if (kpMap[start] && kpMap[end]) {
                const points = [
                  new THREE.Vector3(kpMap[start].x, kpMap[start].y, kpMap[start].z),
                  new THREE.Vector3(kpMap[end].x, kpMap[end].y, kpMap[end].z)
                ];
                const geom = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geom, lineMaterial);
                scene.add(line);
                linesRef.current.push(line);
              }
            });

            if (kpMap[11] && kpMap[12] && kpMap[100]) {
              if (lineHeadRef.current) scene.remove(lineHeadRef.current);
              const midX = (kpMap[11].x + kpMap[12].x) / 2;
              const midY = (kpMap[11].y + kpMap[12].y) / 2;
              const midZ = (kpMap[11].z + kpMap[12].z) / 2;
              const hGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(midX, midY, midZ),
                new THREE.Vector3(kpMap[100].x, kpMap[100].y, kpMap[100].z)
              ]);
              lineHeadRef.current = new THREE.Line(hGeom, new THREE.LineBasicMaterial({ color: 0xf97316 }));
              scene.add(lineHeadRef.current);
            }

            if (kpMap[16] && kpMap[99] && kpMap[99].vis > 0) {
              if (lineRacketRef.current) scene.remove(lineRacketRef.current);
              const rGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(kpMap[16].x, kpMap[16].y, kpMap[16].z),
                new THREE.Vector3(kpMap[99].x, kpMap[99].y, kpMap[99].z)
              ]);
              lineRacketRef.current = new THREE.Line(rGeom, new THREE.LineBasicMaterial({ color: 0x3b82f6, linewidth: 3 }));
              scene.add(lineRacketRef.current);
            }
          }

          currentFrameIdx = (currentFrameIdx + 1) % motionData.frames.length;
          lastTime = now;
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (canvasRef.current) {
        renderer.dispose();
        // eslint-disable-next-line react-hooks/exhaustive-deps
        canvasRef.current.innerHTML = '';
      }
    };
  }, [motionData]);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* ヘッダー */}
      <header className="border-b border-gray-800 pb-4 mb-8 flex justify-between items-center">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-blue-400">Tennis Motion Analysis MVP</h1>
        <p className="text-sm text-gray-400">AI骨格検出を用いたスイング速度・運動連鎖（Kinematic Chain）解析システム</p>
      </div>
      <Link 
        href="/compare" 
        className="bg-purple-600 hover:bg-purple-500 text-white font-medium py-2 px-4 rounded-lg text-sm transition-colors flex items-center gap-2"
      >
        📊 分析結果比較画面へ
      </Link>
    </header>

      {/* イントロダクション ＆ 動画アップロード */}
      <section className="mb-8 bg-gray-900 border border-gray-800 rounded-xl p-6">
        <div className="space-y-4 max-w-3xl">
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Info className="w-5 h-5 text-blue-400" /> このアプリでできること
          </h2>
          <p className="text-gray-400 text-sm leading-relaxed">
            スイング動画をアップロードするだけで、身体の各部位からラケット先端へとどのように速度が伝達しているか（キネマティック・チェーン）を空間解析。最高速度のタイミングや加速のピークを一目で把握できます。
          </p>
          
          {/* 動画入力エリア */}
          <div className="pt-2">
            <div className="border-2 border-dashed border-gray-700 hover:border-blue-500 rounded-lg p-6 text-center cursor-pointer relative transition-colors bg-gray-950">
              <input 
                type="file" 
                accept="video/*" 
                onChange={handleFileChange} 
                disabled={isAnalyzing}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <Upload className="w-8 h-8 mx-auto text-gray-500 mb-2" />
              <p className="text-xs text-gray-400">
                {videoFile ? videoFile.name : "スイング動画を選択、またはドロップ"}
              </p>
            </div>

            {videoFile && (
              <button 
                onClick={handleUpload}
                disabled={isAnalyzing}
                className="w-full sm:w-auto mt-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 text-white font-medium py-2 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 text-sm"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    AIが空間速度・骨格を解析中...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> 解析を実行する
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </section>

      {/* 下段：ユーザー動画 ⇄ 3Dシミュレータ / パフォーマンスグラフ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 左側：ユーザー動画 ＆ 3Dモーション */}
        <div className="space-y-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Play className="w-5 h-5 text-blue-400" /> 選択されたスイング動画
            </h2>
            <div className="h-[280px] bg-gray-950 rounded-lg flex items-center justify-center border border-gray-800 overflow-hidden">
              {videoUrl ? (
                <video src={videoUrl} controls className="w-full h-full bg-black" />
              ) : (
                <p className="text-gray-500 text-sm">上のエリアから動画を読み込んでください</p>
              )}
            </div>
          </div>

          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-3">
              <Box className="w-5 h-5 text-emerald-400" /> 3D 骨格モーションシミュレータ
            </h2>
            <div 
              ref={canvasRef} 
              className="h-[280px] bg-gray-950 rounded-lg border border-gray-800 overflow-hidden relative"
            >
              {!motionData && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <p className="text-gray-500 text-sm">動画を解析すると、ここに補正済みの3D骨格が描画されます</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 右側：メインアナリティクス（スイング速度・運動連鎖） */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-gray-800">
              {/* タブ切り替えボタン */}
              <div className="flex items-center gap-2 bg-gray-950 p-1 rounded-lg border border-gray-800">
                <button
                  onClick={() => setActiveTab('velocity')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    activeTab === 'velocity'
                      ? 'bg-orange-500 text-white shadow'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" /> スイング速度・運動連鎖
                </button>
                <button
                  onClick={() => setActiveTab('angle')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    activeTab === 'angle'
                      ? 'bg-blue-600 text-white shadow'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" /> 関節角度詳細
                </button>
              </div>

              {/* JSON 保存ボタン */}
              {motionData && (
                <button
                  onClick={handleDownloadJSON}
                  className="bg-purple-600 hover:bg-purple-500 text-white text-xs px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1.5 font-medium shadow-sm"
                >
                  <Download className="w-3.5 h-3.5" /> 解析JSONを保存
                </button>
              )}
            </div>

            {/* メタ情報バッジ */}
            {motionData && (
              <div className="mb-4 grid grid-cols-3 gap-4 bg-gray-950 p-3 rounded-lg border border-gray-800 text-center">
                <div>
                  <p className="text-[10px] text-gray-400">推測最高ラケット速度</p>
                  <p className="text-sm font-bold text-orange-400">
                    {motionData.summary?.swing_speed_kmh || 0} km/h
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-400">FPS / 総解析コマ数</p>
                  <p className="text-xs font-semibold text-purple-400">
                    {motionData.metadata?.fps || 30} fps / {motionData.frames?.length || 0} frames
                  </p>
                </div>
                <div>
                  <p className="text-[10px] text-gray-400">骨格補正 (best.pt)</p>
                  {motionData.metadata?.skeleton_refined?.enabled ? (
                    <p className="text-xs font-semibold text-emerald-400">
                      有効{motionData.metadata.skeleton_refined.epoch != null && ` (epoch ${motionData.metadata.skeleton_refined.epoch})`}
                    </p>
                  ) : (
                    <p className="text-xs font-semibold text-gray-500">無効</p>
                  )}
                </div>
              </div>
            )}

            {/* 📈 メイングラフ：スイング速度・運動連鎖（Kinematic Chain） */}
            {activeTab === 'velocity' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
                    <BarChart2 className="w-4 h-4 text-orange-400" /> 速度伝達（腰 → 肩 → 肘 → 手首 → ラケット）
                  </h3>
                  <span className="text-[10px] text-gray-500">単位: m/s</span>
                </div>
                
                <div className="h-[480px] bg-gray-950 rounded-lg border border-gray-800 p-3 flex items-center justify-center">
                  {velocityTimeSeries.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={velocityTimeSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="frame" stroke="#64748b" fontSize={11} tickLine={false} />
                        <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                          formatter={(value: any) => [`${value} m/s`]}
                          labelFormatter={(label) => `Frame: ${label}`}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        
                        {motionData && (
                          <ReferenceLine 
                            x={currentFrame} 
                            stroke="#ef4444" 
                            strokeWidth={2} 
                            strokeDasharray="3 3"
                            label={{ value: 'PLAY', position: 'top', fill: '#ef4444', fontSize: 10, fontWeight: 'bold' }}
                          />
                        )}

                        <Line type="monotone" dataKey="racketHead" name="ラケット (ID:99)" stroke="#f97316" strokeWidth={3} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="wrist" name="右手首 (ID:16)" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="elbow" name="右肘 (ID:14)" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="shoulder" name="右肩 (ID:12)" stroke="#eab308" strokeWidth={1.5} strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="hip" name="右腰 (ID:24)" stroke="#a855f7" strokeWidth={1.5} strokeDasharray="3 3" dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-gray-500 text-sm">（解析完了後、ここに運動連鎖の速度伝達グラフが表示されます）</p>
                  )}
                </div>
              </div>
            )}

            {/* 📉 サブグラフ：関節角度詳細 */}
            {activeTab === 'angle' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
                    <Activity className="w-4 h-4 text-blue-400" /> 関節角度推移（フォーム詳細）
                  </h3>
                  <span className="text-[10px] text-gray-500">単位: 度(°)</span>
                </div>

                <div className="h-[480px] bg-gray-950 rounded-lg border border-gray-800 p-3 flex items-center justify-center">
                  {angleTimeSeries.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={angleTimeSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="frame" stroke="#64748b" fontSize={11} tickLine={false} />
                        <YAxis stroke="#64748b" domain={[0, 180]} unit="°" fontSize={11} tickLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#f8fafc', fontSize: '12px' }}
                          formatter={(value: any) => [`${value}°`]}
                          labelFormatter={(label) => `Frame: ${label}`}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />

                        {motionData && (
                          <ReferenceLine 
                            x={currentFrame} 
                            stroke="#ef4444" 
                            strokeWidth={2} 
                            strokeDasharray="3 3"
                            label={{ value: 'PLAY', position: 'top', fill: '#ef4444', fontSize: 10, fontWeight: 'bold' }}
                          />
                        )}

                        <Line type="monotone" dataKey="rightElbow" name="右肘角度" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
                        <Line type="monotone" dataKey="rightKnee" name="右膝角度" stroke="#10b981" strokeWidth={2} dot={false} isAnimationActive={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <p className="text-gray-500 text-sm">（解析完了後、ここに肘・膝の角度推移グラフが表示されます）</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}