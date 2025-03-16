import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from 'recharts';
import { 
  Shield, 
  AlertTriangle, 
  Users, 
  TrendingUp,
  MapPin,
  Network,
  Clock,
  AlertCircle
} from 'lucide-react';

// Helper function to format timestamps for charts
const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp * 1000);
  return `${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
};

// Convert reputation distribution object to array for chart
const prepareReputationData = (distribution) => {
  return Object.entries(distribution).map(([range, count]) => ({
    range,
    count
  }));
};

const SybilProtectionStats = ({ stats }) => {
  // If no stats yet, show loading
  if (!stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Shield className="h-12 w-12 mx-auto mb-4 text-muted-foreground animate-pulse" />
          <p className="text-muted-foreground">Loading Sybil protection metrics...</p>
        </div>
      </div>
    );
  }

  // Prepare data for charts
  const reputationData = prepareReputationData(stats.reputationDistribution || {});
  const reputationHistory = stats.history?.reputationHistory || [];
  const trustHistory = stats.history?.trustHistory || [];
  const nodeCountHistory = stats.history?.nodeCountHistory || [];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Network Trust</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.trustScore || 0}%</div>
            <Progress value={stats.trustScore || 0} className="h-2 mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Nodes</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.activeNodes || 0}</div>
            <p className="text-xs text-muted-foreground">
              {stats.suspiciousNodes || 0} suspicious
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Reputation</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgReputation?.toFixed(2) || '0.00'}</div>
            <Progress 
              value={(stats.avgReputation || 0) * 100} 
              className="h-2 mt-2" 
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Stake</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalStake || '0'} XMR</div>
            <p className="text-xs text-muted-foreground">
              Avg {stats.avgStake || '0'} XMR per node
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Reputation History Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-md">Reputation History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reputationHistory.map(item => ({
                ...item,
                formattedTime: formatTimestamp(item.timestamp)
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="formattedTime" />
                <YAxis domain={[0, 1]} />
                <Tooltip 
                  formatter={(value) => [value.toFixed(2), "Reputation"]}
                  labelFormatter={(time) => `Time: ${time}`}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="avgReputation" 
                  stroke="#2563eb" 
                  name="Average Reputation" 
                />
                <Line 
                  type="monotone" 
                  dataKey="minReputation" 
                  stroke="#dc2626" 
                  name="Minimum Reputation" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Trust Score + Node Count Chart */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Trust Score History */}
        <Card>
          <CardHeader>
            <CardTitle className="text-md">Trust Score History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trustHistory.map(item => ({
                  ...item,
                  formattedTime: formatTimestamp(item.timestamp)
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="formattedTime" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value) => [`${value}%`, "Trust Score"]}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="trustScore" 
                    stroke="#059669" 
                    name="Trust Score" 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Reputation Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-md">Reputation Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[250px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={reputationData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#2563eb" name="Node Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Geographic Distribution */}
      <Card>
        <CardHeader>
          <CardTitle className="text-md">Geographic Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {(stats.geographicClusters || []).map((cluster, index) => (
              <div key={index} className="flex items-center justify-between border-b pb-2">
                <div className="flex items-center space-x-2">
                  <MapPin className="h-4 w-4 text-muted-foreground" />
                  <span>{cluster.region}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge>{cluster.nodeCount} nodes</Badge>
                  {cluster.suspicious && (
                    <Badge variant="destructive">Suspicious Cluster</Badge>
                  )}
                </div>
              </div>
            ))}
            
            {(stats.geographicClusters || []).length === 0 && (
              <div className="text-center py-6 text-muted-foreground">
                No geographic data available
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      <div className="space-y-4">
        {(stats.alerts || []).map((alert, index) => (
          <Alert 
            key={index}
            variant={alert.severity === 'high' ? 'destructive' : 'warning'}
          >
            <AlertCircle className="h-4 w-4" />
            <div className="ml-2">
              <AlertTitle>{alert.title}</AlertTitle>
              <AlertDescription>{alert.message}</AlertDescription>
            </div>
          </Alert>
        ))}
        
        {(stats.alerts || []).length === 0 && (
          <div className="flex items-center p-4 border rounded-md bg-muted/50">
            <Shield className="h-5 w-5 mr-2 text-green-500" />
            <p className="text-sm">No active Sybil threats detected</p>
          </div>
        )}
      </div>

      {/* Node Statistics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-md">Node Activity History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={nodeCountHistory.map(item => ({
                ...item,
                formattedTime: formatTimestamp(item.timestamp)
              }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="formattedTime" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="activeNodes" 
                  stroke="#2563eb" 
                  name="Active Nodes" 
                />
                <Line 
                  type="monotone" 
                  dataKey="suspiciousNodes" 
                  stroke="#dc2626" 
                  name="Suspicious Nodes" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SybilProtectionStats;