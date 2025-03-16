import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

const PerformanceCharts = ({ data, timeframe, onTimeframeChange }) => {
  const formatValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(2);
    }
    return value;
  };

  return (
    <div className="space-y-6">
      {/* GPU Performance Chart */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="text-sm font-medium">GPU Performance</CardTitle>
            <div className="flex gap-2">
              {['1h', '24h', '7d', '30d'].map((period) => (
                <Button
                  key={period}
                  variant={timeframe === period ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => onTimeframeChange(period)}
                >
                  {period}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.performance}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="timestamp" 
                  className="text-xs" 
                />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--background))',
                    borderColor: 'hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line
                  type="monotone"
                  dataKey="utilization"
                  stroke="hsl(var(--primary))"
                  name="Utilization %"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  stroke="hsl(var(--destructive))"
                  name="Temperature °C"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Memory Usage Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data.memory}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="timestamp" 
                  className="text-xs" 
                />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--background))',
                    borderColor: 'hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Area
                  type="monotone"
                  dataKey="used"
                  stackId="1"
                  stroke="hsl(var(--primary))"
                  fill="hsl(var(--primary))"
                  fillOpacity={0.2}
                  name="Used Memory (GB)"
                />
                <Area
                  type="monotone"
                  dataKey="available"
                  stackId="1"
                  stroke="hsl(var(--muted-foreground))"
                  fill="hsl(var(--muted))"
                  fillOpacity={0.1}
                  name="Available Memory (GB)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Verification Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Verification Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.verification}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="timestamp" 
                  className="text-xs" 
                />
                <YAxis className="text-xs" />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--background))',
                    borderColor: 'hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                  labelStyle={{ color: 'hsl(var(--foreground))' }}
                />
                <Line
                  type="monotone"
                  dataKey="successRate"
                  stroke="hsl(var(--success))"
                  name="Success Rate %"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="verificationTime"
                  stroke="hsl(var(--info))"
                  name="Verification Time (s)"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PerformanceCharts;