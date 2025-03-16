import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { CreditCard, Clock, TrendingUp, Download } from 'lucide-react';

const EarningsOverview = ({ earnings, history, onTimeframeChange }) => {
  const timeframes = ['24h', '7d', '30d', '90d'];

  const downloadEarningsReport = () => {
    // Implementation for downloading earnings report
    console.log('Downloading earnings report...');
  };

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Earnings</CardTitle>
            <CreditCard className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{earnings.total} XMR</div>
            <p className="text-xs text-muted-foreground">
              +{earnings.today} XMR today
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Earnings</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{earnings.pending} XMR</div>
            <p className="text-xs text-muted-foreground">
              From {earnings.pendingJobs} active jobs
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Daily</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{earnings.dailyAverage} XMR</div>
            <p className="text-xs text-muted-foreground">
              Last 30 days
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Earnings Chart */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <CardTitle className="text-sm font-medium">Earnings History</CardTitle>
            <div className="space-x-2">
              {timeframes.map(timeframe => (
                <Button
                  key={timeframe}
                  variant={earnings.timeframe === timeframe ? "default" : "outline"}
                  size="sm"
                  onClick={() => onTimeframeChange(timeframe)}
                >
                  {timeframe}
                </Button>
              ))}
              <Button
                variant="outline"
                size="sm"
                onClick={downloadEarningsReport}
              >
                <Download className="h-4 w-4 mr-2" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip 
                  formatter={(value) => `${value} XMR`}
                  labelFormatter={(label) => `Date: ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="earnings"
                  stroke="#2563eb"
                  fill="#93c5fd"
                  name="Earnings"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div>
              <p className="text-sm text-muted-foreground">Highest Day</p>
              <p className="text-lg font-semibold">{earnings.stats.highestDay} XMR</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Average per Job</p>
              <p className="text-lg font-semibold">{earnings.stats.averagePerJob} XMR</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Total Jobs</p>
              <p className="text-lg font-semibold">{earnings.stats.totalJobs}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Success Rate</p>
              <p className="text-lg font-semibold">{earnings.stats.successRate}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EarningsOverview;