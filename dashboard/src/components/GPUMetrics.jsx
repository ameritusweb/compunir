import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const GPUMetrics = ({ metrics, history }) => {
  const isTemperatureHigh = metrics?.temperature > 80;
  const isMemoryHigh = (metrics?.memory.used / metrics?.memory.total) > 0.9;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="text-sm font-medium">GPU Status</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {/* Utilization */}
          <div className="space-y-3">
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Utilization</span>
                <span className="text-sm font-semibold">{metrics?.utilization}%</span>
              </div>
              <Progress value={metrics?.utilization} className="h-2" />
            </div>

            {/* Temperature */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Temperature</span>
                <span className="text-sm font-semibold">{metrics?.temperature}°C</span>
              </div>
              <Progress 
                value={metrics?.temperature} 
                className="h-2"
                indicatorClassName={isTemperatureHigh ? 'bg-red-500' : 'bg-blue-500'} 
              />
            </div>

            {/* Memory */}
            <div>
              <div className="flex justify-between mb-1">
                <span className="text-sm">Memory Usage</span>
                <span className="text-sm font-semibold">
                  {metrics?.memory.used.toFixed(1)}GB / {metrics?.memory.total}GB
                </span>
              </div>
              <Progress 
                value={(metrics?.memory.used / metrics?.memory.total) * 100} 
                className="h-2"
                indicatorClassName={isMemoryHigh ? 'bg-red-500' : 'bg-blue-500'}
              />
            </div>
          </div>

          {/* Alerts */}
          {isTemperatureHigh && (
            <Alert variant="destructive" className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                GPU temperature is critically high! Consider reducing workload.
              </AlertDescription>
            </Alert>
          )}

          {isMemoryHigh && (
            <Alert variant="warning" className="mt-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                GPU memory usage is near capacity.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Historical Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Performance History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="utilization" 
                  stroke="#2563eb" 
                  name="Utilization %" 
                />
                <Line 
                  type="monotone" 
                  dataKey="temperature" 
                  stroke="#f97316" 
                  name="Temperature °C" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default GPUMetrics;