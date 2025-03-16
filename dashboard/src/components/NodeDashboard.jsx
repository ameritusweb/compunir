import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, AreaChart, Area } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { AlertCircle, Check, Clock, Activity, CreditCard, Server } from 'lucide-react';

const NodeDashboard = () => {
  // Mock data - would be fetched from your node's API
  const [stats, setStats] = useState({
    earnings: {
      total: "1.234 XMR",
      today: "0.023 XMR",
      pending: "0.012 XMR"
    },
    gpu: {
      utilization: 87,
      temperature: 65,
      memory: {
        used: 8.2,
        total: 12
      }
    },
    jobs: {
      active: 2,
      completed: 45,
      failed: 3,
      verification_rate: 98
    }
  });

  const [earningHistory] = useState([
    { date: '2024-03-10', earnings: 0.021 },
    { date: '2024-03-11', earnings: 0.019 },
    { date: '2024-03-12', earnings: 0.024 },
    { date: '2024-03-13', earnings: 0.022 },
    { date: '2024-03-14', earnings: 0.023 },
  ]);

  const [gpuHistory] = useState([
    { time: '12:00', utilization: 85, temperature: 62 },
    { time: '12:10', utilization: 88, temperature: 64 },
    { time: '12:20', utilization: 90, temperature: 65 },
    { time: '12:30', utilization: 87, temperature: 63 },
    { time: '12:40', utilization: 89, temperature: 65 },
  ]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Node Dashboard</h1>

      {/* Earnings Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Earnings</CardTitle>
            <CreditCard className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.earnings.total}</div>
            <p className="text-xs text-muted-foreground">
              +{stats.earnings.today} today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Earnings</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.earnings.pending}</div>
            <p className="text-xs text-muted-foreground">
              Awaiting verification
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Verification Rate</CardTitle>
            <Check className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.jobs.verification_rate}%</div>
            <p className="text-xs text-muted-foreground">
              Last 30 days
            </p>
          </CardContent>
        </Card>
      </div>

      {/* GPU Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">GPU Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Utilization</span>
                  <span className="text-sm font-semibold">{stats.gpu.utilization}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full" 
                    style={{ width: `${stats.gpu.utilization}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Temperature</span>
                  <span className="text-sm font-semibold">{stats.gpu.temperature}°C</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-orange-500 h-2 rounded-full" 
                    style={{ width: `${(stats.gpu.temperature / 100) * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm">Memory</span>
                  <span className="text-sm font-semibold">
                    {stats.gpu.memory.used}/{stats.gpu.memory.total} GB
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-purple-600 h-2 rounded-full" 
                    style={{ width: `${(stats.gpu.memory.used / stats.gpu.memory.total) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-lg">GPU History</CardTitle>
          </CardHeader>
          <CardContent>
            <LineChart
              width={400}
              height={200}
              data={gpuHistory}
              margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="utilization" stroke="#2563eb" name="Utilization %" />
              <Line type="monotone" dataKey="temperature" stroke="#f97316" name="Temperature °C" />
            </LineChart>
          </CardContent>
        </Card>
      </div>

      {/* Earnings Chart */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-lg">Earnings History</CardTitle>
        </CardHeader>
        <CardContent>
          <AreaChart
            width={800}
            height={200}
            data={earningHistory}
            margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="earnings" stroke="#2563eb" fill="#93c5fd" />
          </AreaChart>
        </CardContent>
      </Card>

      {/* Job Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.jobs.active}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed Jobs</CardTitle>
            <Check className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.jobs.completed}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed Jobs</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.jobs.failed}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">GPU Status</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Online</div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default NodeDashboard;