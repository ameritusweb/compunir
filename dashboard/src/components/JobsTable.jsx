import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  AlertCircle,
  CheckCircle,
  Clock,
  RefreshCw,
  Play,
  Pause,
  MoreVertical,
  ChevronRight
} from 'lucide-react';

const JobStatus = ({ status }) => {
  const statusConfig = {
    active: { icon: Play, color: 'text-green-500', badge: 'default' },
    pending: { icon: Clock, color: 'text-yellow-500', badge: 'secondary' },
    completed: { icon: CheckCircle, color: 'text-blue-500', badge: 'outline' },
    failed: { icon: AlertCircle, color: 'text-red-500', badge: 'destructive' },
    verifying: { icon: RefreshCw, color: 'text-purple-500', badge: 'secondary' },
    paused: { icon: Pause, color: 'text-orange-500', badge: 'warning' }
  };

  const config = statusConfig[status] || statusConfig.pending;
  const Icon = config.icon;

  return (
    <div className="flex items-center gap-2">
      <Icon className={`h-4 w-4 ${config.color}`} />
      <Badge variant={config.badge}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    </div>
  );
};

const JobsTable = ({ jobs, onJobSelect, onJobAction }) => {
  const [selectedStatus, setSelectedStatus] = useState('all');

  const filteredJobs = selectedStatus === 'all' 
    ? jobs 
    : jobs.filter(job => job.status === selectedStatus);

  const statusCounts = jobs.reduce((acc, job) => {
    acc[job.status] = (acc[job.status] || 0) + 1;
    return acc;
  }, {});

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">Jobs</CardTitle>
          <div className="flex gap-2">
            {Object.entries(statusCounts).map(([status, count]) => (
              <Button
                key={status}
                variant={selectedStatus === status ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedStatus(status)}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)} ({count})
              </Button>
            ))}
            <Button
              variant={selectedStatus === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setSelectedStatus('all')}
            >
              All ({jobs.length})
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto rounded-md border">
          <table className="w-full">
            <thead className="bg-muted">
              <tr className="border-b">
                <th className="text-left p-4 font-medium">Job ID</th>
                <th className="text-left p-4 font-medium">Type</th>
                <th className="text-left p-4 font-medium">Status</th>
                <th className="text-left p-4 font-medium">Progress</th>
                <th className="text-left p-4 font-medium">Verification</th>
                <th className="text-left p-4 font-medium">Earnings</th>
                <th className="text-left p-4 font-medium">Time</th>
                <th className="text-right p-4 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.map((job) => (
                <tr key={job.id} className="border-b hover:bg-muted/50">
                  <td className="p-4 font-mono text-xs">
                    {job.id.substring(0, 8)}...
                  </td>
                  <td className="p-4">
                    <Badge variant="outline">{job.type}</Badge>
                  </td>
                  <td className="p-4">
                    <JobStatus status={job.status} />
                  </td>
                  <td className="p-4 w-[200px]">
                    <div className="space-y-1">
                      <Progress value={job.progress * 100} className="h-2" />
                      <p className="text-xs text-muted-foreground">
                        {(job.progress * 100).toFixed(1)}%
                      </p>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">
                        {job.verificationStatus}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        ({job.verifiersCount} verifiers)
                      </span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="text-sm font-medium">{job.earnings} XMR</div>
                    {job.pendingEarnings > 0 && (
                      <div className="text-xs text-muted-foreground">
                        +{job.pendingEarnings} XMR pending
                      </div>
                    )}
                  </td>
                  <td className="p-4">
                    <div className="text-sm">{job.duration}</div>
                    <div className="text-xs text-muted-foreground">
                      Started {job.startTime}
                    </div>
                  </td>
                  <td className="p-4 text-right">
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onJobSelect(job)}
                      >
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onJobAction(job)}
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default JobsTable;