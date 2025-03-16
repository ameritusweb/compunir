import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Shield, Clock, CheckCircle, AlertCircle } from 'lucide-react';

const VerificationStats = ({ stats, history }) => {
  // Calculate success rate color
  const getSuccessRateColor = (rate) => {
    if (rate >= 90) return 'bg-green-500';
    if (rate >= 75) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-4">
      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.successRate}%</div>
            <Progress 
              value={stats.successRate} 
              className="h-2 mt-2"
              indicatorClassName={getSuccessRateColor(stats.successRate)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Verifications</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.activeVerifications}</div>
            <p className="text-xs text-muted-foreground">
              {stats.pendingVerifications} pending
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Verified</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalVerified}</div>
            <p className="text-xs text-muted-foreground">
              Last 30 days
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Failed Verifications</CardTitle>
            <AlertCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.failedVerifications}</div>
            <p className="text-xs text-muted-foreground">
              {stats.failureRate}% failure rate
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Verification History Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Verification History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="successRate" 
                  stroke="#22c55e" 
                  name="Success Rate" 
                />
                <Line 
                  type="monotone" 
                  dataKey="verificationTime" 
                  stroke="#3b82f6" 
                  name="Verification Time (s)" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Recent Verifications */}
          <div className="mt-4">
            <h3 className="text-sm font-medium mb-2">Recent Verifications</h3>
            <div className="space-y-2">
              {stats.recentVerifications.map((verification) => (
                <div 
                  key={verification.id} 
                  className="flex items-center justify-between py-2 border-b last:border-0"
                >
                  <div className="flex items-center space-x-2">
                    {verification.success ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    )}
                    <span className="text-sm font-medium">Job #{verification.jobId}</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <Badge variant={verification.success ? "success" : "destructive"}>
                      {verification.success ? "Success" : "Failed"}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {verification.time}s
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div>
              <p className="text-sm text-muted-foreground">Avg. Time</p>
              <p className="text-lg font-semibold">{stats.averageTime}s</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Verifiers/Job</p>
              <p className="text-lg font-semibold">{stats.verifiersPerJob}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Consensus Rate</p>
              <p className="text-lg font-semibold">{stats.consensusRate}%</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Network Health</p>
              <p className="text-lg font-semibold">{stats.networkHealth}%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default VerificationStats;