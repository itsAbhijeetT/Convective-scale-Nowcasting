import React, { useEffect, useState, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Play, Pause, AlertTriangle, CloudRain, Zap, Wind } from 'lucide-react';

export default function App() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [leadTime, setLeadTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    if (map.current) return;
    
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
      center: [75.0, 20.0],
      zoom: 5
    });

    map.current.on('load', () => {
      fetchAlerts();
      fetchMetrics();
    });
  }, []);

  const fetchAlerts = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/alerts');
      const data = await res.json();
      setAlerts(data);
    } catch (e) {
      console.error(e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/metrics');
      const data = await res.json();
      
      const chartData = data.lead_times.map((lt, i) => ({
        lead_time: lt,
        CSI: data.CSI[i],
        POD: data.POD[i],
        FAR: data.FAR[i]
      }));
      setMetrics(chartData);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setLeadTime(prev => (prev >= 35 ? 0 : prev + 1));
      }, 500);
    } else if (!isPlaying && interval) {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  return (
    <div className="flex h-screen w-full bg-gray-900 text-white overflow-hidden">
      {/* Map Area */}
      <div className="flex-1 relative">
        <div ref={mapContainer} className="absolute inset-0" />
        
        {/* Controls Overlay */}
        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-gray-800/90 backdrop-blur-md p-4 rounded-xl border border-gray-700 shadow-2xl flex items-center space-x-6">
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            className="p-3 bg-blue-600 hover:bg-blue-500 rounded-full transition-colors"
          >
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>
          
          <div className="flex flex-col">
            <span className="text-sm font-semibold mb-1">Lead Time: T+{leadTime * 10} mins</span>
            <input 
              type="range" 
              min="0" 
              max="35" 
              value={leadTime} 
              onChange={(e) => setLeadTime(parseInt(e.target.value))}
              className="w-64 accent-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Right Panel */}
      <div className="w-96 bg-gray-900 border-l border-gray-800 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            SIH Nowcasting 26084
          </h1>
          <p className="text-sm text-gray-400 mt-1">Convective Storm Prediction</p>
        </div>

        {/* Alerts Section */}
        <div className="flex-1 overflow-y-auto p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4 flex items-center">
            <AlertTriangle size={16} className="mr-2 text-yellow-500" /> Active Alerts
          </h2>
          
          <div className="space-y-4">
            {alerts.length === 0 ? (
              <p className="text-gray-500 text-sm">No active alerts.</p>
            ) : (
              alerts.map((alert, i) => (
                <div key={i} className="bg-gray-800 rounded-lg p-4 border-l-4 border-red-500 shadow-lg relative overflow-hidden group hover:bg-gray-750 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-red-400 flex items-center">
                      {alert.type === 'Cloudburst' ? <CloudRain size={16} className="mr-1"/> : <Zap size={16} className="mr-1"/>}
                      {alert.type}
                    </span>
                    <span className="text-xs bg-red-500/20 text-red-300 px-2 py-1 rounded-full">{alert.severity}</span>
                  </div>
                  <p className="text-sm text-gray-300">ETA: T+{alert.eta_minutes} mins</p>
                  <p className="text-xs text-gray-400 mt-2 p-2 bg-gray-900/50 rounded">{alert.action}</p>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Metrics Chart */}
        <div className="h-64 border-t border-gray-800 p-4">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Model Skill (CSI)</h2>
          {metrics && (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="lead_time" stroke="#9CA3AF" fontSize={10} />
                <YAxis stroke="#9CA3AF" fontSize={10} domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' }} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '10px' }} />
                <Line type="monotone" dataKey="CSI" stroke="#3B82F6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="POD" stroke="#10B981" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
