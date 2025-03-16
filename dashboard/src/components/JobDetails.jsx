import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  X, 
  AlertTriangle, 
  RefreshCw, 
  Server,
  Clock,
  Activity,
  Database,
  Shield,
  CreditCard
} from 'lucide-react';

const MetricCard = ({ title, value, icon: Icon }) => (
  <div className="flex items-center gap-2 p-3 rounded-lg border">
    <Icon className="h-4 w-4 text-muted-foreground" />
    <div>
      <p className="text-sm text-muted-foreground">{title}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  </div>
);

const JobDetails = ({ job, onClose, onRetry }) => {
  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm">
      <div className="fixed inset-y-0 right-0 w-full max-w-xl border-l bg-background p-6 shadow-lg">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold">Job Details</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-6">
          {/* Basic Info */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Job Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard 
                  title="Status" 
                  value={job.status}
                  icon={Activity}
                />
                <MetricCard 
                  title="Duration" 
                  value={job.duration}
                  icon={Clock}
                />
                <MetricCard 
                  title="GPU Usage" 
                  value={`${job.gpuUtilization}%`}
                  icon={Server}
                />
                <MetricCard 
                  title="Memory" 
                  value={`${job.memoryUsage} GB`}
                  icon={Database}
                />
              </div>

              <div className="mt-4">
                <div className="text-sm text-muted-foreground mb-2">Progress</div>
                <Progress value={job.progress * 100} className="h-2" />
                <div className="text-xs text-muted-foreground mt-1">
                  {(job.progress * 100).toFixed(1)}% complete
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Verification Status */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Verification Status</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard 
                  title="Verifiers" 
                  value={job.verifiersCount}
                  icon={Shield}
                />
                <MetricCard 
                  title="Success Rate" 
                  value={`${job.verificationRate}%`}
                  icon={RefreshCw}
                />
              </div>

              {job.verificationHistory && (
                <div className="mt-4 space-y-2">
                  <div className="text-sm text-muted-foreground">Recent Verifications</div>
                  {job.verificationHistory.map((verification, index) => (
                    <div 
                      key={index}
                      className="flex items-center justify-between py-2 border-b last:border-0"
                    >
                      <div className="flex items-center gap-2">
                        <Badge variant={verification.success ? "success" : "destructive"}>
                          {verification.success ? "Success" : "Failed"}
                        </Badge>
                        <span className="text-sm">{verification.verifier}</span>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {verification.time}s
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Earnings */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Earnings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <MetricCard 
                  title="Total Earned" 
                  value={`${job.earnings} XMR`}
                  icon={CreditCard}
                />
                <MetricCard 
                  title="Pending" 
                  value={`${job.pendingEarnings} XMR`}
                  icon={Clock}
                />
              </div>
            </CardContent>
          </Card>

          {/* Error Information */}
          {job.error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{job.error}</AlertDescription>
            </Alert>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end gap-2 mt-6">
            {job.status === 'failed' && (
              <Button onClick={() => onRetry(job.id)}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry Job
              </Button>
            )}
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default JobDetails;