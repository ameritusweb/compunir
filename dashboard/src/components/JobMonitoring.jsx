import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Activity,
  Clock,
  AlertTriangle,
  Shield,
  Server
} from 'lucide-react';

const JobProgressCard = ({ job }) => {
  // Calculate time remaining based on progress and elapsed time
  const getTimeRemaining = () => {
    const elapsedTime = job.elapsedTime;
    const progress = job.progress;
    if (progress > 0) {
      const totalEstimatedTime = elapsedTime / progress;
      const remaining = totalEstimatedTime - elapsedTime;
      return formatDuration(remaining);
    }
    return 'Calculating...';
  };

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ${Math.round(seconds % 60)}s`;
    }
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-sm font-medium">
              Job #{job.id.substring(0, 8)}
            </CardTitle>
            <p className="text-xs text-muted-foreground">{job.type}</p>
          </div>
          <Badge variant={job.status === 'active' ? 'default' : 'secondary'}>
            {job.status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Progress Bar */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-sm">Progress</span>
              <span className="text-sm font-medium">
                {(job.progress * 100).toFixed(1)}%
              </span>
            </div>
            <Progress value={job.progress * 100} className="h-2" />
            <div className="flex justify-between mt-1">
              <span className="text-xs text-muted-foreground">
                <Clock className="h-3 w-3 inline mr-1" />
                {formatDuration(job.elapsedTime)} elapsed
              </span>
              <span className="text-xs text-muted-foreground">
                ~{getTimeRemaining()} remaining
              </span>
            </div>
          </div>

          {/* Resource Usage */}
          <div className="grid grid-cols-3 gap-2">
            <div className="text-center p-2 rounded-lg border">
              <Server className="h-4 w-4 mx-auto mb-1" />
              <div className="text-xs font-medium">{job.gpuUtilization}%</div>
              <div className="text-xs text-muted-foreground">GPU</div>
            </div>
            <div className="text-center p-2 rounded-lg border">
              <Shield className="h-4 w-4 mx-auto mb-1" />
              <div className="text-xs font-medium">{job.verifiersCount}</div>
              <div className="text-xs text-muted-foreground">Verifiers</div>
            </div>
            <div className="text-center p-2 rounded-lg border">
              <Activity className="h-4 w-4 mx-auto mb-1" />
              <div className="text-xs font-medium">{job.memoryUsage}GB</div>
              <div className="text-xs text-muted-foreground">Memory</div>
            </div>
          </div>

          {/* Warnings */}
          {job.warnings && job.warnings.length > 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-2 rounded-lg">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                <span className="text-xs text-yellow-600 dark:text-yellow-400">
                  {job.warnings[0]}
                </span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

const JobMonitoring = ({ activeJobs }) => {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {activeJobs.map(job => (
          <JobProgressCard key={job.id} job={job} />
        ))}
      </div>

      {activeJobs.length === 0 && (
        <Card>
          <CardContent className="py-8">
            <div className="text-center text-muted-foreground">
              No active jobs at the moment
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default JobMonitoring;