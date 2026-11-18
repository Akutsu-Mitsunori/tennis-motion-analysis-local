'use client';

import React, { useState, ChangeEvent, useEffect, useRef } from 'react';
import Link from 'next/link';
import { Upload, ArrowLeft, BarChart2, Box, Zap, Activity, CheckCircle2, Play } from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
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

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32]
];

function calculate3DAngle(
  p1: { x: number; y: number; z: number },
  p2: { x: number; y: number; z: number },
  p3: { x: number; y: number; z: number }
): number {
  const v1 = new THREE.Vector3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
  const v2 = new THREE.Vector3(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z);
  const angleRad = v1.angleTo(v2);
  return Math.round(((angleRad * 180) / Math.PI) * 10) / 10;
}

export default function ComparePage() {
  // ファイル＆データ状態
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [dataA, setDataA] = useState<any>(null);
  const [dataB, setDataB] = useState<any>(null);

  // 比較実行フラグ
  const [isComparing, setIsComparing] = useState(false);

  // タブ管理
  const [activeTab, setActiveTab] = useState<'velocity' | 'angle'>('velocity');

  // 時系列グラフデータ
  const [velocityTimeSeries, setVelocityTimeSeries] = useState<any[]>([]);
  const [angleTimeSeries, setAngleTimeSeries] = useState<any[]>([]);

  // 再生制御
  const [currentFrame, setCurrentFrame] = useState<number>(1);

  // Three.js 参照
  const canvasRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);

  // JSON読み込み処理
  const handleJsonUpload = (e: ChangeEvent<HTMLInputElement>, target: 'A' | 'B') => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target?.result as string);
        if (target === 'A') {
          setFileA(file);
          setDataA(json);
        } else {
          setFileB(file);
          setDataB(json);
        }
      } catch (err) {
        alert('JSONファイルの読み込みに失敗しました。正しいフォーマットか確認してください。');
      }
    };
    reader.readAsText(file);
  };

  // 「比較開始」ボタン押下時のロジック
  const handleStartComparison = () => {
    if (!dataA || !dataB) return;

    // 1. グラフ用データの構築 (フレームごとにデータAとデータBのラケット速度・関節角度をマージ)
    const maxFrames = Math.max(dataA.frames?.length || 0, dataB.frames?.length || 0);
    const vSeries: any[] = [];
    const aSeries: any[] = [];

    for (let idx = 0; idx < maxFrames; idx++) {
      const frameA = dataA.frames?.[idx];
      const frameB = dataB.frames?.[idx];

      const kpMapA: any = {};
      if (frameA?.skeletons?.[0]) {
        frameA.skeletons[0].keypoints.forEach((kp: any) => {
          kpMapA[kp.id] = { ...kp.rel_coord, velocity: kp.velocity ?? 0, vis: kp.visibility ?? 0 };
        });
      }

      const kpMapB: any = {};
      if (frameB?.skeletons?.[0]) {
        frameB.skeletons[0].keypoints.forEach((kp: any) => {
          kpMapB[kp.id] = { ...kp.rel_coord, velocity: kp.velocity ?? 0, vis: kp.visibility ?? 0 };
        });
      }

      // 速度比較（ラケット ID:99）。未検出フレーム (visibility=0) は手首位置の代用値なので除外
      const racketSpeed = (m: any) =>
        m[99] && m[99].vis > 0 ? Math.round(m[99].velocity * 100) / 100 : null;
      vSeries.push({
        frame: idx + 1,
        racketSpeedA: racketSpeed(kpMapA),
        racketSpeedB: racketSpeed(kpMapB),
      });

      // 角度比較（右肘角度）
      let elbowA = null;
      if (kpMapA[12] && kpMapA[14] && kpMapA[16]) {
        elbowA = calculate3DAngle(kpMapA[12], kpMapA[14], kpMapA[16]);
      }
      let elbowB = null;
      if (kpMapB[12] && kpMapB[14] && kpMapB[16]) {
        elbowB = calculate3DAngle(kpMapB[12], kpMapB[14], kpMapB[16]);
      }

      aSeries.push({
        frame: idx + 1,
        elbowAngleA: elbowA,
        elbowAngleB: elbowB,
      });
    }

    setVelocityTimeSeries(vSeries);
    setAngleTimeSeries(aSeries);
    setIsComparing(true);
  };

  // 3D 重ね合わせアニメーション (Three.js)
  useEffect(() => {
    if (!isComparing || !canvasRef.current || !dataA || !dataB) return;

    const width = canvasRef.current.clientWidth;
    const height = canvasRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 50);
    camera.position.set(0, 1.5, 5.0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    canvasRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 1.0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const grid = new THREE.GridHelper(4, 20, 0x334155, 0x1e293b);
    scene.add(grid);

    // GroupA (対象① - 青系) / GroupB (対象② - 緑系)
    const groupA = new THREE.Group();
    const groupB = new THREE.Group();
    scene.add(groupA);
    scene.add(groupB);

    let currentFrameIdx = 0;
    let lastTime = performance.now();

    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      const fps = dataA.metadata?.fps || 30;
      const now = performance.now();

      if (now - lastTime >= 1000 / fps) {
        setCurrentFrame(currentFrameIdx + 1);

        // --- Group A 描画 ---
        while (groupA.children.length > 0) groupA.remove(groupA.children[0]);
        const frameA = dataA.frames?.[currentFrameIdx];
        if (frameA?.skeletons?.[0]) {
          const kpMapA: any = {};
          frameA.skeletons[0].keypoints.forEach((kp: any) => {
            kpMapA[kp.id] = kp.rel_coord;
            // 関節点 (青)
            const mesh = new THREE.Mesh(
              new THREE.SphereGeometry(0.025, 8, 8),
              new THREE.MeshBasicMaterial({ color: 0x3b82f6 })
            );
            mesh.position.set(kp.rel_coord.x, kp.rel_coord.y, kp.rel_coord.z);
            groupA.add(mesh);
          });
          // ボーン結線
          const lineMatA = new THREE.LineBasicMaterial({ color: 0x60a5fa, linewidth: 2 });
          POSE_CONNECTIONS.forEach(([s, e]) => {
            if (kpMapA[s] && kpMapA[e]) {
              const geom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(kpMapA[s].x, kpMapA[s].y, kpMapA[s].z),
                new THREE.Vector3(kpMapA[e].x, kpMapA[e].y, kpMapA[e].z)
              ]);
              groupA.add(new THREE.Line(geom, lineMatA));
            }
          });
        }

        // --- Group B 描画 ---
        while (groupB.children.length > 0) groupB.remove(groupB.children[0]);
        const frameB = dataB.frames?.[currentFrameIdx];
        if (frameB?.skeletons?.[0]) {
          const kpMapB: any = {};
          frameB.skeletons[0].keypoints.forEach((kp: any) => {
            kpMapB[kp.id] = kp.rel_coord;
            // 関節点 (緑)
            const mesh = new THREE.Mesh(
              new THREE.SphereGeometry(0.025, 8, 8),
              new THREE.MeshBasicMaterial({ color: 0x10b981 })
            );
            mesh.position.set(kp.rel_coord.x, kp.rel_coord.y, kp.rel_coord.z);
            groupB.add(mesh);
          });
          // ボーン結線
          const lineMatB = new THREE.LineBasicMaterial({ color: 0x34d399, linewidth: 2 });
          POSE_CONNECTIONS.forEach(([s, e]) => {
            if (kpMapB[s] && kpMapB[e]) {
              const geom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(kpMapB[s].x, kpMapB[s].y, kpMapB[s].z),
                new THREE.Vector3(kpMapB[e].x, kpMapB[e].y, kpMapB[e].z)
              ]);
              groupB.add(new THREE.Line(geom, lineMatB));
            }
          });
        }

        const maxLen = Math.max(dataA.frames?.length || 0, dataB.frames?.length || 0);
        currentFrameIdx = (currentFrameIdx + 1) % (maxLen || 1);
        lastTime = now;
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (canvasRef.current) {
        renderer.dispose();
        canvasRef.current.innerHTML = '';
      }
    };
  }, [isComparing, dataA, dataB]);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-6">
      {/* ヘッダー */}
      <header className="border-b border-gray-800 pb-4 mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-purple-400">Tennis Motion Analysis - Compare Mode</h1>
          <p className="text-sm text-gray-400">2つの動作解析（JSON）データを読み込み、フォームや速度伝達を比較分析</p>
        </div>
        <Link 
          href="/" 
          className="bg-gray-800 hover:bg-gray-700 text-gray-200 font-medium py-2 px-4 rounded-lg text-sm transition-colors flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" /> メイン画面へ戻る
        </Link>
      </header>

      {/* ファイルアップロードエリア（対象① vs 対象②） */}
      <section className="mb-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 対象① (左) */}
        <div className="bg-gray-900 border border-blue-900/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-blue-400 mb-3 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-blue-500 inline-block"></span> 対象①（基準スイング / 自分）
          </h2>
          <div className="border-2 border-dashed border-gray-700 hover:border-blue-500 rounded-lg p-6 text-center cursor-pointer relative bg-gray-950">
            <input 
              type="file" 
              accept=".json" 
              onChange={(e) => handleJsonUpload(e, 'A')}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="w-8 h-8 mx-auto text-gray-500 mb-2" />
            <p className="text-xs text-gray-400">
              {fileA ? fileA.name : "解析済み JSON ファイルを選択 (Target A)"}
            </p>
          </div>
          {dataA && (
            <div className="mt-3 text-xs text-blue-400 flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" /> 読み込み完了 ({dataA.frames?.length || 0} frames)
            </div>
          )}
        </div>

        {/* 対象② (右) */}
        <div className="bg-gray-900 border border-emerald-900/50 rounded-xl p-6">
          <h2 className="text-lg font-bold text-emerald-400 mb-3 flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block"></span> 対象②（比較対象 / プロなど）
          </h2>
          <div className="border-2 border-dashed border-gray-700 hover:border-emerald-500 rounded-lg p-6 text-center cursor-pointer relative bg-gray-950">
            <input 
              type="file" 
              accept=".json" 
              onChange={(e) => handleJsonUpload(e, 'B')}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="w-8 h-8 mx-auto text-gray-500 mb-2" />
            <p className="text-xs text-gray-400">
              {fileB ? fileB.name : "解析済み JSON ファイルを選択 (Target B)"}
            </p>
          </div>
          {dataB && (
            <div className="mt-3 text-xs text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-4 h-4" /> 読み込み完了 ({dataB.frames?.length || 0} frames)
            </div>
          )}
        </div>
      </section>

      {/* 真ん中の比較開始ボタン */}
      <div className="text-center mb-8">
        <button
          onClick={handleStartComparison}
          disabled={!dataA || !dataB}
          className="bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 disabled:from-gray-800 disabled:to-gray-800 disabled:text-gray-600 text-white font-bold py-3 px-10 rounded-xl transition-all shadow-lg flex items-center gap-2 mx-auto text-base"
        >
          <Play className="w-5 h-5 fill-current" /> 2つのスイングデータを比較開始
        </button>
      </div>

      {/* 比較結果表示エリア */}
      {isComparing && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 左側：3D 骨格重ね合わせシミュレータ */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-3">
              <Box className="w-5 h-5 text-purple-400" /> 3D 骨格重ね合わせ（Overlay）
            </h2>
            <div className="flex gap-4 mb-2 text-xs">
              <span className="flex items-center gap-1 text-blue-400 font-semibold">● 対象① (青)</span>
              <span className="flex items-center gap-1 text-emerald-400 font-semibold">● 対象② (緑)</span>
            </div>
            <div 
              ref={canvasRef} 
              className="h-[480px] bg-gray-950 rounded-lg border border-gray-800 overflow-hidden relative"
            />
          </div>

          {/* 右側：時系列比較グラフ */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 bg-gray-950 p-1 rounded-lg border border-gray-800 mb-4">
                <button
                  onClick={() => setActiveTab('velocity')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    activeTab === 'velocity' ? 'bg-orange-500 text-white' : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" /> ラケット速度比較
                </button>
                <button
                  onClick={() => setActiveTab('angle')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all ${
                    activeTab === 'angle' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" /> 肘関節角度比較
                </button>
              </div>

              {/* 速度比較グラフ */}
              {activeTab === 'velocity' && (
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
                    <BarChart2 className="w-4 h-4 text-orange-400" /> ラケットヘッド速度比較 (m/s)
                  </h3>
                  <div className="h-[420px] bg-gray-950 rounded-lg border border-gray-800 p-3">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={velocityTimeSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="frame" stroke="#64748b" fontSize={11} />
                        <YAxis stroke="#64748b" fontSize={11} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        <ReferenceLine x={currentFrame} stroke="#ef4444" strokeWidth={2} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="racketSpeedA" name="対象① (青)" stroke="#3b82f6" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="racketSpeedB" name="対象② (緑)" stroke="#10b981" strokeWidth={2.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* 角度比較グラフ */}
              {activeTab === 'angle' && (
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-gray-300 flex items-center gap-1">
                    <Activity className="w-4 h-4 text-blue-400" /> 右肘角度推移比較 (°)
                  </h3>
                  <div className="h-[420px] bg-gray-950 rounded-lg border border-gray-800 p-3">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={angleTimeSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="frame" stroke="#64748b" fontSize={11} />
                        <YAxis stroke="#64748b" domain={[0, 180]} fontSize={11} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
                        <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                        <ReferenceLine x={currentFrame} stroke="#ef4444" strokeWidth={2} strokeDasharray="3 3" />
                        <Line type="monotone" dataKey="elbowAngleA" name="対象① 右肘 (青)" stroke="#3b82f6" strokeWidth={2.5} dot={false} />
                        <Line type="monotone" dataKey="elbowAngleB" name="対象② 右肘 (緑)" stroke="#10b981" strokeWidth={2.5} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}